"""
Post-flop discard engine for the 27-card keep-2 poker variant.

Shared by METAV4 and ALPHANiTV5.  Evaluates all C(5,2)=10 possible keep-pairs
from a 5-card hand against a 3-card flop using a 6-step scoring pipeline:

  1. Immediate made-hand classification (flush / FH / straight / trips / 2P / pair / nothing)
  2. Full-house potential  – live boat outs for trips and two-pair candidates
  3. Flush-draw potential  – 4-to-a-flush when both kept cards are suited with 2+ on flop
  4. Straight-draw potential – OESD / gutshot / double-gutter via the 6 valid windows
  5. Second-to-discard adjustment – dead-out removal + lightweight opp-discard inference
  6. Fallback structural heuristics – suitedness, connectivity, high-card, flop interaction

BB (blind_position=1) discards first with no opponent information.
SB (blind_position=0) discards second and can see the BB's 3 discarded cards.
"""

from collections import Counter
from itertools import combinations

# ── Deck geometry ─────────────────────────────────────────────────────────────
# 27-card deck: ranks 2-9,A  (encoded 0-8), suits d,h,s (encoded 0-2)
# card_id = suit * NUM_RANKS + rank

NUM_RANKS = 9
DECK_SIZE = 27
RANK_A = 8   # Ace
RANK_9 = 7
RANK_8 = 6

# The six legal 5-card straight windows (rank indices)
VALID_STRAIGHTS = [
    (8, 0, 1, 2, 3),   # A-2-3-4-5  (wheel)
    (0, 1, 2, 3, 4),   # 2-3-4-5-6
    (1, 2, 3, 4, 5),   # 3-4-5-6-7
    (2, 3, 4, 5, 6),   # 4-5-6-7-8
    (3, 4, 5, 6, 7),   # 5-6-7-8-9
    (4, 5, 6, 7, 8),   # 6-7-8-9-A  (broadway)
]

_WHEEL_SET = frozenset({8, 0, 1, 2, 3})


def _rank(c):
    return c % NUM_RANKS


def _suit(c):
    return c // NUM_RANKS


# ═════════════════════════════════════════════════════════════════════════════
#  Step 1 — Classify made hand (keep2 + flop3 = 5 cards)
# ═════════════════════════════════════════════════════════════════════════════

def classify_made_hand(keep2, flop3):
    """Return (category, details) for the 5-card hand.

    Categories in priority order (reflects 27-card variant where flushes
    are scarce and therefore ranked highest):
      flush > full_house > straight > trips > two_pair > pair > nothing

    ``details`` carries sub-ranking info: quality, trips_rank, pair_rank, etc.
    """
    all5 = list(keep2) + list(flop3)
    ranks = [_rank(c) for c in all5]
    suits = [_suit(c) for c in all5]
    rc = Counter(ranks)
    sc = Counter(suits)

    is_flush = (sc.most_common(1)[0][1] == 5)

    # Straight requires five distinct ranks forming a valid window
    unique_ranks = sorted(set(ranks))
    is_straight = False
    straight_high = -1
    if len(unique_ranks) == 5:
        ur_set = set(unique_ranks)
        for window in VALID_STRAIGHTS:
            if set(window) == ur_set:
                is_straight = True
                straight_high = 3 if ur_set == _WHEEL_SET else max(window)
                break

    # ── Flush (highest tier) ──────────────────────────────────────────────
    if is_flush:
        flush_suit = sc.most_common(1)[0][0]
        flush_ranks = sorted(
            [_rank(c) for c in all5 if _suit(c) == flush_suit], reverse=True
        )
        return ('flush', {
            'quality': flush_ranks[0],
            'ranks': flush_ranks,
            'is_straight_flush': is_straight,
        })

    # ── Identify trips / pairs ────────────────────────────────────────────
    trips_rank = None
    pair_ranks = []
    for r, cnt in rc.most_common():
        if cnt >= 3 and trips_rank is None:
            trips_rank = r
        elif cnt == 2:
            pair_ranks.append(r)

    # ── Full house ────────────────────────────────────────────────────────
    if trips_rank is not None and pair_ranks:
        return ('full_house', {
            'trips_rank': trips_rank,
            'pair_rank': max(pair_ranks),
        })

    # ── Straight ──────────────────────────────────────────────────────────
    if is_straight:
        return ('straight', {'high': straight_high})

    # ── Trips (no pair beside it) ─────────────────────────────────────────
    if trips_rank is not None:
        kickers = sorted([r for r in ranks if r != trips_rank], reverse=True)
        return ('trips', {
            'trips_rank': trips_rank,
            'kicker_ranks': kickers,
        })

    # ── Two pair ──────────────────────────────────────────────────────────
    if len(pair_ranks) >= 2:
        pr_sorted = sorted(pair_ranks, reverse=True)
        kicker = [r for r in ranks if r not in pair_ranks]
        return ('two_pair', {
            'pair_ranks': pr_sorted,
            'kicker': max(kicker) if kicker else 0,
        })

    # ── One pair ──────────────────────────────────────────────────────────
    if pair_ranks:
        pr = pair_ranks[0]
        kickers = sorted([r for r in ranks if r != pr], reverse=True)
        return ('pair', {'pair_rank': pr, 'kickers': kickers})

    # ── Nothing ───────────────────────────────────────────────────────────
    return ('nothing', {'high_cards': sorted(ranks, reverse=True)})


# ═════════════════════════════════════════════════════════════════════════════
#  Step 2 — Full-house potential (for trips / two-pair tiers only)
# ═════════════════════════════════════════════════════════════════════════════

def compute_full_house_potential(keep2, flop3, known_cards):
    """Count live cards that would improve trips→FH or two-pair→FH.

    For trips:  any unseen card matching either kicker rank pairs it up → boat.
    For two pair: the third copy of either pair rank → trips half of a boat.

    Each rank has exactly 3 copies in the 27-card deck.  Cards already in
    ``known_cards`` (our keep + flop + our discards + visible opp discards)
    are removed from the out pool.

    Returns (live_boat_outs, best_boat_quality) where
        quality = trips_rank * 10 + pair_rank   (of the resulting full house).
    """
    all5 = list(keep2) + list(flop3)
    ranks = [_rank(c) for c in all5]
    rc = Counter(ranks)

    trips_rank = None
    pair_ranks = []
    for r, cnt in rc.items():
        if cnt >= 3:
            trips_rank = r
        elif cnt == 2:
            pair_ranks.append(r)

    live_outs = 0
    best_quality = 0

    if trips_rank is not None and not pair_ranks:
        # Trips without a pair → need a kicker to pair up
        kicker_ranks = set(r for r in ranks if r != trips_rank)
        for target in kicker_ranks:
            for s in range(3):
                cid = s * NUM_RANKS + target
                if cid not in known_cards:
                    live_outs += 1
            best_quality = max(best_quality, trips_rank * 10 + target)

    elif len(pair_ranks) >= 2:
        # Two pair → need either pair rank to become trips
        for pr in pair_ranks:
            for s in range(3):
                cid = s * NUM_RANKS + pr
                if cid not in known_cards:
                    live_outs += 1
            other = max(r for r in pair_ranks if r != pr)
            best_quality = max(best_quality, pr * 10 + other)

    return live_outs, best_quality


# ═════════════════════════════════════════════════════════════════════════════
#  Step 3 — Flush-draw potential
# ═════════════════════════════════════════════════════════════════════════════

def compute_flush_draw_potential(keep2, flop3, known_cards):
    """Detect a 4-to-a-flush draw (both kept cards suited, 2+ on flop).

    Only fires when both cards in ``keep2`` share a suit that also appears
    at least twice on the flop, yielding 4 of the suit across 5 cards.
    (5-of-suit is a made flush and handled by classify_made_hand first.)

    Returns (has_draw, live_flush_outs, flush_quality).
    ``flush_quality`` is the highest rank of the draw suit in our kept cards
    (determines how strong the resulting flush would be).
    """
    k_suits = [_suit(c) for c in keep2]
    if k_suits[0] != k_suits[1]:
        return False, 0, 0

    draw_suit = k_suits[0]
    flop_of_suit = sum(1 for c in flop3 if _suit(c) == draw_suit)
    total = 2 + flop_of_suit
    if total < 4:
        return False, 0, 0

    live_outs = 0
    for r in range(NUM_RANKS):
        cid = draw_suit * NUM_RANKS + r
        if cid not in known_cards:
            live_outs += 1

    flush_quality = max(_rank(c) for c in keep2)
    return True, live_outs, flush_quality


# ═════════════════════════════════════════════════════════════════════════════
#  Step 4 — Straight-draw potential
# ═════════════════════════════════════════════════════════════════════════════

def compute_straight_draw_potential(keep2, flop3, known_cards):
    """Identify OESD / gutshot / double-gutter straight draws.

    Scans all 6 valid straight windows.  A window needing exactly 1 card is a
    single-window draw; 2+ such windows sharing the same hand make an OESD.
    Dead cards in ``known_cards`` are excluded from live-out counts.

    Returns (draw_type, live_straight_outs, best_straight_high).
    draw_type ∈ {'oesd', 'gutshot', 'double_gutter', 'none'}.
    """
    all5 = list(keep2) + list(flop3)
    have_ranks = set(_rank(c) for c in all5)

    completing_cards = set()
    best_high = -1
    windows_needing_one = 0

    for window in VALID_STRAIGHTS:
        w_set = set(window)
        need = w_set - have_ranks

        if len(need) == 1:
            windows_needing_one += 1
            needed_rank = next(iter(need))
            for s in range(3):
                cid = s * NUM_RANKS + needed_rank
                if cid not in known_cards:
                    completing_cards.add(cid)
            high = 3 if w_set == _WHEEL_SET else max(window)
            best_high = max(best_high, high)

        elif len(need) == 2:
            # Backdoor / double-gutter contributor
            for needed_rank in need:
                for s in range(3):
                    cid = s * NUM_RANKS + needed_rank
                    if cid not in known_cards:
                        completing_cards.add(cid)

    live_outs = len(completing_cards)

    if windows_needing_one >= 2:
        draw_type = 'oesd'
    elif windows_needing_one == 1:
        draw_type = 'gutshot'
    elif live_outs >= 4:
        draw_type = 'double_gutter'
    else:
        draw_type = 'none'

    return draw_type, live_outs, best_high


# ═════════════════════════════════════════════════════════════════════════════
#  Step 6 — Fallback structural heuristic (no strong made hand or draw)
# ═════════════════════════════════════════════════════════════════════════════

def fallback_structural_score(keep2, flop3, known_cards):
    """Score 0-99 based on structural quality of keep2 relative to the flop.

    Considers: suitedness, connectivity (gap), high-card value, flop pairing
    potential, and whether improvement cards are dead.

    Used only when no higher-tier made hand or draw is present.
    """
    r0, r1 = _rank(keep2[0]), _rank(keep2[1])
    s0, s1 = _suit(keep2[0]), _suit(keep2[1])
    suited = (s0 == s1)
    high_rank = max(r0, r1)

    gap = abs(r0 - r1)
    if gap > 4:
        gap = NUM_RANKS - gap   # A-2 wraps around

    score = 0.0

    # ── Suitedness ────────────────────────────────────────────────────────
    if suited:
        score += 30
        flop_suit_count = sum(1 for c in flop3 if _suit(c) == s0)
        score += 8 * flop_suit_count

    # ── Connectivity ──────────────────────────────────────────────────────
    if gap == 0:
        score += 25 + high_rank * 2   # pocket pair
    elif gap == 1:
        score += 20                    # connector
    elif gap == 2:
        score += 12                    # one-gap
    elif gap == 3:
        score += 5                     # two-gap

    # ── High-card value ───────────────────────────────────────────────────
    score += high_rank * 2.5

    # ── Flop interaction (pairing potential) ──────────────────────────────
    flop_ranks = [_rank(c) for c in flop3]
    if r0 in flop_ranks or r1 in flop_ranks:
        score += 10

    # ── Penalize dead improvement cards ───────────────────────────────────
    keep_set = set(keep2)
    flop_set = set(flop3)
    dead_penalty = 0
    for r in (r0, r1):
        for s in range(3):
            cid = s * NUM_RANKS + r
            if cid in known_cards and cid not in keep_set and cid not in flop_set:
                dead_penalty += 2
    score -= dead_penalty

    return max(0.0, min(99.0, score))


# ═════════════════════════════════════════════════════════════════════════════
#  Step 5 — Second-to-discard inference (SB only)
# ═════════════════════════════════════════════════════════════════════════════

def _second_discard_bonus(keep2, flop3, opp_discards, known_cards):
    """Lightweight inference from opponent's visible discards.

    Applied only when we are the SB (second to discard).  The primary
    benefit—dead-out removal—is already baked into the live-out counts
    throughout the pipeline.  This function adds two small secondary signals:

    1. Suit-abandon inference: if opponent threw ≥2 cards of a suit that
       appears ≥2 times on the flop, they probably *don't* have that flush
       draw.  If we hold cards of that suit, we face less competition.
    2. High-card dump: if opponent discarded ≥2 high cards (rank ≥ 8),
       their kept hand skews weaker on average.
    """
    bonus = 0.0
    opp_suit_counts = Counter(_suit(c) for c in opp_discards)
    keep_suits = [_suit(c) for c in keep2]
    flop_suits = [_suit(c) for c in flop3]

    for s, cnt in opp_suit_counts.items():
        flop_of_suit = sum(1 for fs in flop_suits if fs == s)
        if cnt >= 2 and flop_of_suit >= 2:
            if any(ks == s for ks in keep_suits):
                bonus += 3.0

    opp_high = sum(1 for c in opp_discards if _rank(c) >= RANK_8)
    if opp_high >= 2:
        bonus += 2.0

    return bonus


# ═════════════════════════════════════════════════════════════════════════════
#  Scoring orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def rank_keep_candidate(keep2, flop3, my_discards, opp_discards):
    """Produce a single comparable score for a keep-pair candidate.

    Scoring tiers (higher = better):
        900-999   Made flush        (sub-ranked by quality + straight-flush bump)
        800-899   Made full house   (trips_rank * 10 + pair_rank)
        700-799   Made straight     (high-card * 10)
        500-699   Trips             (500 + rank*10 + boat-out bonus up to 80)
        400-499   Two pair          (400 + rank sub-score + boat-out bonus up to 60)
        200-399   Pair              (200 + pair_rank*12 + draw bonus capped at 99)
          0-199   Nothing           (draw bonus up to 180, else fallback 0-99)
    """
    dead = set(my_discards) | set(opp_discards)
    known = set(keep2) | set(flop3) | dead

    cat, details = classify_made_hand(keep2, flop3)

    # ── Flush ─────────────────────────────────────────────────────────────
    if cat == 'flush':
        q = details['quality']
        sf_bump = 10 if details.get('is_straight_flush') else 0
        return 900.0 + q * 8 + sf_bump

    # ── Full house ────────────────────────────────────────────────────────
    if cat == 'full_house':
        return 800.0 + details['trips_rank'] * 10 + details['pair_rank']

    # ── Straight ──────────────────────────────────────────────────────────
    if cat == 'straight':
        return 700.0 + details['high'] * 10

    # ── Trips (with boat-out bonus) ───────────────────────────────────────
    if cat == 'trips':
        live_boat, _ = compute_full_house_potential(keep2, flop3, known)
        return 500.0 + details['trips_rank'] * 10 + min(live_boat * 8, 80)

    # ── Two pair (with boat-out bonus) ────────────────────────────────────
    if cat == 'two_pair':
        live_boat, _ = compute_full_house_potential(keep2, flop3, known)
        pr = details['pair_ranks']
        return 400.0 + pr[0] * 5 + pr[1] * 2 + min(live_boat * 10, 60)

    # ── Draw bonuses (apply to pair and nothing tiers) ────────────────────
    draw_bonus = 0.0

    has_fd, fd_outs, fd_q = compute_flush_draw_potential(keep2, flop3, known)
    if has_fd and fd_outs >= 2:
        draw_bonus = max(draw_bonus, 80.0 + fd_outs * 4 + fd_q * 2)

    sd_type, sd_outs, sd_high = compute_straight_draw_potential(keep2, flop3, known)
    sd_h = max(sd_high, 0)
    if sd_type == 'oesd' and sd_outs >= 3:
        draw_bonus = max(draw_bonus, 50.0 + sd_outs * 3 + sd_h)
    elif sd_type in ('gutshot', 'double_gutter') and sd_outs >= 2:
        draw_bonus = max(draw_bonus, 20.0 + sd_outs * 3 + sd_h)

    # ── Pair ──────────────────────────────────────────────────────────────
    if cat == 'pair':
        base = 200.0 + details['pair_rank'] * 12
        return base + min(draw_bonus, 99.0)

    # ── Nothing ───────────────────────────────────────────────────────────
    if draw_bonus > 0:
        return min(draw_bonus, 180.0)

    return fallback_structural_score(keep2, flop3, known)


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def choose_keep_postflop(my_cards_5, flop3, opp_discards, blind_position):
    """Evaluate all 10 keep-pair candidates and return the best (i, j).

    Args
    ----
    my_cards_5 :     list[int]  – 5 hole-card IDs
    flop3 :          list[int]  – 3 community flop-card IDs
    opp_discards :   list[int]  – opponent's discarded card IDs ([] if BB)
    blind_position : int        – 1 = BB (first, no opp info),
                                  0 = SB (second, sees opp discards)

    Returns
    -------
    (i, j) : tuple[int, int]  – indices into my_cards_5 of the cards to keep
    """
    best_score = -1.0
    best_ij = (0, 1)

    for i, j in combinations(range(len(my_cards_5)), 2):
        keep = [my_cards_5[i], my_cards_5[j]]
        discards = [my_cards_5[k] for k in range(len(my_cards_5))
                    if k != i and k != j]

        score = rank_keep_candidate(keep, flop3, discards, opp_discards)

        # SB inference bonus when opponent discards are visible (Step 5)
        if opp_discards and blind_position == 0:
            all_known = (set(keep) | set(flop3)
                         | set(discards) | set(opp_discards))
            score += _second_discard_bonus(keep, flop3, opp_discards,
                                           all_known)

        if score > best_score:
            best_score = score
            best_ij = (i, j)

    return best_ij