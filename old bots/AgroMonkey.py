# AgroMonkey: Range-Pressure Exploit Bot
#
# Core philosophy: Does NOT ask "Do I have the best hand?"
# Asks: "Is villain capped? Does the board favor a story I can credibly rep?
#        Do I have enough real backup equity / blockers to keep firing?"
#
# AggroScore = 0.45 * CapScore + 0.35 * StoryScore + 0.20 * BackupScore
# + hard trigger bonuses - hard brakes
# * villain profile multiplier
# + seeded noise

import random
from collections import Counter
from itertools import combinations

from agents.agent import Agent
from gym_env import PokerEnv

# ── Constants ─────────────────────────────────────────────────────────────────

RANKS = "23456789A"
SUITS = "dhs"
NUM_RANKS = 9
DECK_SIZE = 27
RANK_A = 8
RANK_9 = 7
RANK_8 = 6

FOLD    = PokerEnv.ActionType.FOLD.value
RAISE   = PokerEnv.ActionType.RAISE.value
CHECK   = PokerEnv.ActionType.CHECK.value
CALL    = PokerEnv.ActionType.CALL.value
DISCARD = PokerEnv.ActionType.DISCARD.value

_int_to_card = PokerEnv.int_to_card

# ── Card helpers ──────────────────────────────────────────────────────────────

def _rank(c):
    return c % NUM_RANKS

def _suit(c):
    return c // NUM_RANKS

def _effective_gap(c1, c2):
    g = abs(_rank(c1) - _rank(c2))
    return g if g <= 4 else NUM_RANKS - g

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _max_connectivity(ranks):
    """Return the longest run of consecutive ranks in a sorted list of rank ints."""
    unique = sorted(set(ranks))
    if not unique:
        return 0
    best = cur = 1
    for i in range(1, len(unique)):
        if unique[i] - unique[i - 1] == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    # Ace-low wrap (A-2-3...)
    if RANK_A in unique and 0 in unique:
        best = max(best, 2)
    return best


# ── PlayerAgent ───────────────────────────────────────────────────────────────

class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self._env = PokerEnv()
        self.evaluator = self._env.evaluator

        self.hand_number = 0
        self._rng = random.Random(42)

        # Opponent stat tracking
        self._stats = {
            "fold_flop_bet":    [0, 0],  # [folds, total_faced]
            "fold_turn_bet":    [0, 0],
            "fold_river_bet":   [0, 0],
            "fold_to_raise":    [0, 0],
            "check_raise":      [0, 0],  # [times raised after we checked, total_checks]
            "call_down":        [0, 0],  # [postflop calls vs our bet, total postflop bets]
            "showdown_win":     [0, 0],  # [opp won at showdown, total showdowns reached]
            "suit_attack_fold": [0, 0],  # [folds to suit-cap attack, total suit attacks]
            "opp_aggression":   [0, 0],  # [opp raises, total opp actions observed]
        }

        # Per-hand state
        self._betting_history = {"bet_flop": False, "bet_turn": False}
        self._last_was_bet = False
        self._last_street = 0
        self._last_suit_attack = False

    def __name__(self):
        return "AgroMonkey"

    # ── RNG ───────────────────────────────────────────────────────────────────

    def _seed_rng(self, my_cards, street):
        seed = hash((self.hand_number, street, tuple(sorted(my_cards))))
        self._rng = random.Random(seed)

    # ── MC Equity ─────────────────────────────────────────────────────────────

    def _mc_equity(self, my2, community, dead, num_sims=200):
        known = set(my2) | set(community) | dead
        remaining = [c for c in range(DECK_SIZE) if c not in known]
        board_needed = 5 - len(community)
        sample_needed = 2 + board_needed

        if sample_needed > len(remaining):
            return 0.5

        wins = 0.0
        total = 0
        for _ in range(num_sims):
            sample = self._rng.sample(remaining, sample_needed)
            opp = sample[:2]
            full_board = list(community) + sample[2:]
            my_hand  = [_int_to_card(c) for c in my2]
            opp_hand = [_int_to_card(c) for c in opp]
            board    = [_int_to_card(c) for c in full_board]
            mr   = self.evaluator.evaluate(my_hand, board)
            orr  = self.evaluator.evaluate(opp_hand, board)
            if mr < orr:
                wins += 1.0
            elif mr == orr:
                wins += 0.5
            total += 1
        return wins / total if total > 0 else 0.5

    # ── Cap Score ─────────────────────────────────────────────────────────────

    def _cap_score(self, opp_discards, community, my_discards):
        """How capped is the opponent. 0.0–1.0."""
        if not opp_discards or len(opp_discards) < 3:
            return 0.20  # no info

        score = 0.0
        opp_suit_counts = Counter(_suit(c) for c in opp_discards)
        opp_rank_list   = sorted(_rank(c) for c in opp_discards)
        my_suit_d       = Counter(_suit(c) for c in my_discards) if my_discards else Counter()
        board_suit_c    = Counter(_suit(c) for c in community) if community else Counter()

        # Suit-cap attack
        for suit, opp_count in opp_suit_counts.items():
            if opp_count < 2:
                continue

            base = 0.6 if opp_count == 3 else 0.3

            # Board amplifier: board heavy in the capped suit
            board_in_suit = board_suit_c.get(suit, 0)
            if board_in_suit >= 3:
                base += 0.35
            elif board_in_suit >= 2:
                base += 0.15

            # Safety reduction: we also tossed that suit
            my_d_suit = my_suit_d.get(suit, 0)
            if my_d_suit >= 2:
                base *= 0.20
            elif my_d_suit == 1:
                base *= 0.70

            score += base

        # Connectivity cap: opp tossed connected ranks
        opp_conn = sum(
            1 for i in range(len(opp_rank_list) - 1)
            if opp_rank_list[i + 1] - opp_rank_list[i] <= 1
        )
        if opp_conn >= 2 and community and len(community) >= 3:
            b_conn = _max_connectivity([_rank(c) for c in community])
            if b_conn >= 3:
                score += 0.20
            elif b_conn >= 2:
                score += 0.10
        elif opp_conn >= 2:
            score += 0.08

        # High-value structure cap: 2+ high cards discarded
        high_count = sum(1 for c in opp_discards if _rank(c) >= RANK_8)
        if high_count >= 2:
            score += 0.10

        return min(score, 1.0)

    # ── Story Score ───────────────────────────────────────────────────────────

    def _story_score(self, my_cards, community, my_discards, street):
        """How credible our strong-hand story is. 0.0–1.0."""
        if not community or len(community) < 3:
            return 0.40

        b_suits = [_suit(c) for c in community]
        b_ranks = [_rank(c) for c in community]
        b_suit_counts = Counter(b_suits)
        dominant_suit, dom_count = b_suit_counts.most_common(1)[0]

        # Board flush texture
        flush_base = 0.0
        if dom_count >= 3:
            flush_base = 0.45
        elif dom_count >= 2:
            flush_base = 0.20

        # Board straight texture
        b_conn = _max_connectivity(b_ranks)
        straight_base = 0.0
        if b_conn >= 3:
            straight_base = 0.35
        elif b_conn >= 2:
            straight_base = 0.15

        score = max(flush_base, straight_base)

        # Our keep supports the story
        if my_cards and len(my_cards) == 2:
            k_suits = [_suit(c) for c in my_cards]
            k_ranks = [_rank(c) for c in my_cards]

            # Flush story support
            if flush_base > 0:
                our_in_dom = sum(1 for s in k_suits if s == dominant_suit)
                if our_in_dom >= 2:
                    score += 0.20
                elif our_in_dom >= 1:
                    score += 0.10

            # Straight story support
            if straight_base > 0:
                all_conn = _max_connectivity(k_ranks + b_ranks)
                if all_conn >= 5:
                    score += 0.20  # made straight
                elif all_conn >= 4:
                    score += 0.10  # strong draw

        # Brake: we discarded the suit we're trying to represent
        if my_discards and flush_base > 0:
            my_d_suit = Counter(_suit(c) for c in my_discards)
            discarded_dom = my_d_suit.get(dominant_suit, 0)
            if discarded_dom >= 2:
                score -= 0.30
            elif discarded_dom >= 1:
                score -= 0.10

        # Betting history coherence (multi-street story)
        if street >= 2 and self._betting_history.get("bet_flop"):
            score += 0.10
        if street == 3 and self._betting_history.get("bet_turn"):
            score += 0.10

        return max(0.0, min(score, 1.0))

    # ── Backup Score ──────────────────────────────────────────────────────────

    def _backup_score(self, my_cards, community, dead, equity, opp_discards):
        """Our real backup: equity + draws + blockers. 0.0–1.0."""
        score = equity * 0.6

        if not my_cards or len(my_cards) != 2:
            return min(score, 1.0)

        k_suits = [_suit(c) for c in my_cards]
        k_ranks = [_rank(c) for c in my_cards]

        # Pocket pair bonus
        if k_ranks[0] == k_ranks[1]:
            if k_ranks[0] >= RANK_9:
                score += 0.15
            elif k_ranks[0] >= RANK_8:
                score += 0.08
            else:
                score += 0.04

        if community:
            b_suits = [_suit(c) for c in community]
            b_ranks = [_rank(c) for c in community]

            # Flush draws
            for s in set(k_suits):
                total_in_suit = (
                    sum(1 for cs in b_suits if cs == s) +
                    sum(1 for cs in k_suits if cs == s)
                )
                if total_in_suit >= 4:
                    score += 0.15  # made flush
                elif total_in_suit == 3:
                    score += 0.08  # flush draw

            # Straight draws
            all_conn = _max_connectivity(k_ranks + b_ranks)
            if all_conn >= 5:
                score += 0.15
            elif all_conn >= 4:
                score += 0.08

            # Blocker value: hold cards in dominant board suit
            b_suit_counts = Counter(b_suits)
            if b_suit_counts:
                dom_suit, dom_cnt = b_suit_counts.most_common(1)[0]
                if dom_cnt >= 2:
                    our_in_suit = sum(1 for s in k_suits if s == dom_suit)
                    score += 0.05 * our_in_suit

        return min(score, 1.0)

    # ── Hard Triggers / Brakes ────────────────────────────────────────────────

    def _hard_triggers(self, my_cards, community, opp_discards, my_discards):
        """Returns (bonus, brake) to add/subtract from AggroScore."""
        bonus = 0.0
        brake = 0.0

        if not opp_discards or len(opp_discards) < 3:
            return bonus, brake

        opp_suit_c = Counter(_suit(c) for c in opp_discards)
        my_suit_d  = Counter(_suit(c) for c in my_discards) if my_discards else Counter()
        board_suit_c = Counter(_suit(c) for c in community) if community else Counter()

        for suit, opp_count in opp_suit_c.items():
            board_count = board_suit_c.get(suit, 0)
            my_d_count  = my_suit_d.get(suit, 0)

            # Trigger A: opp discarded 3 of suit X, board has 2+ of X, we didn't dump X
            if opp_count == 3 and board_count >= 2 and my_d_count <= 1:
                bonus += 0.35

            # Trigger B: opp discarded 2 of suit X, board monotone/near-mono in X
            elif opp_count == 2:
                if board_count >= 3:
                    bonus += 0.20
                elif board_count >= 2:
                    our_has = sum(1 for c in my_cards if _suit(c) == suit) if my_cards else 0
                    if our_has >= 1 or my_d_count == 0:
                        bonus += 0.12

        # Trigger C: opp discarded 2+ connected ranks, board is also connected
        opp_ranks = sorted(_rank(c) for c in opp_discards)
        opp_conn = sum(
            1 for i in range(len(opp_ranks) - 1)
            if opp_ranks[i + 1] - opp_ranks[i] <= 1
        )
        if opp_conn >= 2 and community and len(community) >= 3:
            b_conn = _max_connectivity([_rank(c) for c in community])
            if b_conn >= 2:
                bonus += 0.15

        # Trigger E: strong draw density + villain folds enough
        if community and len(community) >= 3 and my_cards and len(my_cards) == 2:
            k_suits_e = [_suit(c) for c in my_cards]
            k_ranks_e = [_rank(c) for c in my_cards]
            b_suits_e = [_suit(c) for c in community]
            best_flush_total = max(
                sum(1 for cs in b_suits_e if cs == s) + sum(1 for cs in k_suits_e if cs == s)
                for s in set(k_suits_e)
            )
            st_conn_e = _max_connectivity(k_ranks_e + [_rank(c) for c in community])
            has_strong_draw = best_flush_total >= 3 or st_conn_e >= 4
            if has_strong_draw and self._safe_rate("fold_flop_bet") > 0.45:
                bonus += 0.12

        # Adaptive suit attack: scale bonus based on historical suit attack success
        suit_attack_rate = self._safe_rate("suit_attack_fold")
        if bonus > 0.0:
            if suit_attack_rate > 0.60:
                bonus = min(1.0, bonus * 1.20)  # working well — press harder
            elif suit_attack_rate < 0.25:
                bonus *= 0.70                    # not working — back off

        # Brake A: we discarded the dominant board suit we're trying to rep
        if community and my_discards:
            b_suit_counts = Counter(_suit(c) for c in community)
            if b_suit_counts:
                dom_suit, dom_cnt = b_suit_counts.most_common(1)[0]
                if dom_cnt >= 2:
                    if my_suit_d.get(dom_suit, 0) >= 2:
                        brake += 0.30

        # Brake B: villain is a calling station
        fold_flop = self._safe_rate("fold_flop_bet")
        if fold_flop < 0.20:
            brake += 0.20

        # Brake C: villain check-raises often
        cr_rate = self._safe_rate("check_raise")
        if cr_rate > 0.15:
            brake += 0.15

        return bonus, brake

    # ── Villain Profile ───────────────────────────────────────────────────────

    # Sensible priors before we have enough data
    _STAT_PRIORS = {
        "fold_flop_bet":    0.35,
        "fold_turn_bet":    0.35,
        "fold_river_bet":   0.35,
        "fold_to_raise":    0.35,
        "check_raise":      0.05,  # most players rarely check-raise
        "call_down":        0.40,
        "showdown_win":     0.50,
        "suit_attack_fold": 0.40,
        "opp_aggression":   0.20,  # most players aren't maniacs
    }

    def _safe_rate(self, key):
        folds, total = self._stats.get(key, [0, 0])
        if total < 8:
            return self._STAT_PRIORS.get(key, 0.35)
        return folds / total

    def _villain_profile(self):
        """Returns (aggro_multiplier, bluff_enable)."""
        fold_flop   = self._safe_rate("fold_flop_bet")
        fold_turn   = self._safe_rate("fold_turn_bet")
        cr_rate     = self._safe_rate("check_raise")
        call_down   = self._safe_rate("call_down")
        opp_aggro   = self._safe_rate("opp_aggression")
        showdown_w  = self._safe_rate("showdown_win")  # opp win rate at showdown

        # Aggro maniac: raises very frequently — don't out-spew them, shift to trap-value
        if opp_aggro > 0.50:
            return 0.80, False

        # Overfolder: easy target — increase all pressure
        if fold_flop > 0.50 or fold_turn > 0.55:
            return 1.4, True

        # Calling station: can't bluff them out — play equity only
        if fold_flop < 0.25 or call_down > 0.60:
            return 0.7, False

        # Trapper / check-raiser: be careful with thin bluffs
        if cr_rate > 0.15:
            return 0.85, False

        # Honest strong: folds bluffs but wins at showdown when they call
        # — reduce pure bluff frequency, still value-bet hard
        if showdown_w > 0.65:
            return 0.90, False

        return 1.0, True

    # ── Discard Phase: KeepScore Components ───────────────────────────────────

    def _pressure_potential(self, keep2, community, opp_discards):
        score = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]

        # Suited: can represent flush
        if k_suits[0] == k_suits[1]:
            score += 0.40

        # Connected: can represent straights
        eg = _effective_gap(keep2[0], keep2[1])
        if eg <= 1:
            score += 0.30
        elif eg <= 2:
            score += 0.20
        elif eg <= 3:
            score += 0.10

        # Board flush interaction
        if community:
            b_suits = [_suit(c) for c in community]
            for s in set(k_suits):
                if sum(1 for bs in b_suits if bs == s) >= 2:
                    score += 0.15
                    break

        # High rank blockers
        max_rank = max(k_ranks)
        if max_rank >= RANK_A:
            score += 0.15
        elif max_rank >= RANK_9:
            score += 0.08

        return min(score, 1.0)

    def _draw_density(self, keep2, community):
        score = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]

        if not community:
            # Pre-board: structural only
            if k_suits[0] == k_suits[1]:
                score += 0.30
            eg = _effective_gap(keep2[0], keep2[1])
            if eg <= 1:
                score += 0.25
            elif eg <= 2:
                score += 0.15
            return min(score, 1.0)

        b_suits = [_suit(c) for c in community]
        b_ranks = [_rank(c) for c in community]

        # Flush draws
        for s in set(k_suits):
            total = (sum(1 for cs in b_suits if cs == s) +
                     sum(1 for cs in k_suits if cs == s))
            if total >= 4:
                score += 0.40  # made flush
            elif total == 3:
                score += 0.25  # flush draw
            elif total == 2:
                score += 0.08  # backdoor

        # Straight draws
        all_conn = _max_connectivity(k_ranks + b_ranks)
        if all_conn >= 5:
            score += 0.35
        elif all_conn >= 4:
            score += 0.20
        elif all_conn >= 3:
            score += 0.08

        return min(score, 1.0)

    def _made_hand_value(self, keep2, community):
        score = 0.0
        k_ranks = [_rank(c) for c in keep2]

        # Pocket pair
        if k_ranks[0] == k_ranks[1]:
            r = k_ranks[0]
            if r >= RANK_9:
                score += 0.80
            elif r >= RANK_8:
                score += 0.55
            elif r >= 4:
                score += 0.35
            else:
                score += 0.20
            return min(score, 1.0)

        if not community:
            return 0.0

        b_ranks = [_rank(c) for c in community]
        rank_counts = Counter(k_ranks + b_ranks)
        max_count = rank_counts.most_common(1)[0][1]
        pairs = [r for r, v in rank_counts.items() if v >= 2]

        if max_count >= 3:
            score += 0.70  # trips
        elif len(pairs) >= 2:
            score += 0.50  # two pair
        elif len(pairs) == 1:
            pr = pairs[0]
            if pr in k_ranks:  # our card paired
                if pr >= RANK_A:
                    score += 0.35
                elif pr >= RANK_8:
                    score += 0.20
                else:
                    score += 0.10

        return min(score, 1.0)

    def _blocker_value(self, keep2, community, opp_discards):
        score = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]

        # Ace blocks premium hands
        if RANK_A in k_ranks:
            score += 0.15
        if RANK_9 in k_ranks:
            score += 0.08

        if not community:
            return min(score, 1.0)

        b_suit_counts = Counter(_suit(c) for c in community)
        dom_suit, dom_cnt = b_suit_counts.most_common(1)[0]

        # Block flush combos in dominant board suit
        if dom_cnt >= 2:
            our_in_suit = sum(1 for s in k_suits if s == dom_suit)
            score += 0.20 * our_in_suit

        return min(score, 1.0)

    def _story_flexibility(self, keep2, community):
        score = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]

        # Suited: flush story
        if k_suits[0] == k_suits[1]:
            score += 0.35

        # Connected: straight story
        eg = _effective_gap(keep2[0], keep2[1])
        if eg <= 1:
            score += 0.30
        elif eg <= 2:
            score += 0.20

        # High cards: top pair / kicker
        max_r = max(k_ranks)
        if max_r >= RANK_A:
            score += 0.20
        elif max_r >= RANK_9:
            score += 0.10

        # Premium pair: set story on many boards
        if k_ranks[0] == k_ranks[1] and k_ranks[0] >= RANK_8:
            score += 0.30

        return min(score, 1.0)

    def _keep_score(self, keep2, community, opp_discards, toss_dead):
        """Full KeepScore = 2.8*WinEq + 2.2*PP + 1.7*DD + 1.0*MHV + 0.8*BV + 0.6*SF"""
        eq = self._mc_equity(keep2, community, toss_dead, num_sims=150)
        pp = self._pressure_potential(keep2, community, opp_discards)
        dd = self._draw_density(keep2, community)
        mhv = self._made_hand_value(keep2, community)
        bv = self._blocker_value(keep2, community, opp_discards)
        sf = self._story_flexibility(keep2, community)
        score = 2.8*eq + 2.2*pp + 1.7*dd + 1.0*mhv + 0.8*bv + 0.6*sf
        return score, eq

    def _choose_keep(self, my_cards, community, opp_discards):
        """Returns (i, j) indices into my_cards of the best 2 cards to keep."""
        best_score = -999.0
        best_ij = (0, 1)
        candidates = []

        for i, j in combinations(range(len(my_cards)), 2):
            keep = [my_cards[i], my_cards[j]]
            toss = [my_cards[k] for k in range(len(my_cards)) if k not in (i, j)]
            toss_dead = set(toss)
            if opp_discards:
                toss_dead |= set(opp_discards)

            sc, eq = self._keep_score(keep, community, opp_discards, toss_dead)
            candidates.append((i, j, sc))
            if sc > best_score:
                best_score = sc
                best_ij = (i, j)

        # Near-tie randomization (mixed strategy)
        ties = [(i, j) for i, j, sc in candidates if best_score - sc < 0.08]
        if len(ties) > 1:
            best_ij = self._rng.choice(ties)

        return best_ij

    # ── Preflop ───────────────────────────────────────────────────────────────

    def _preflop_equity(self, my5, dead):
        """Estimate preflop equity: average of top-3 2-card keeps."""
        scores = []
        for i, j in combinations(range(len(my5)), 2):
            keep = [my5[i], my5[j]]
            toss = [my5[k] for k in range(len(my5)) if k not in (i, j)]
            d = set(toss) | dead
            eq = self._mc_equity(keep, [], d, num_sims=80)
            scores.append(eq)
        scores.sort(reverse=True)
        top = scores[:3]
        return sum(top) / len(top) if top else 0.45

    def _choose_preflop(self, my_cards, valid, min_raise, max_raise,
                        my_bet, opp_bet, pot_size, blind_pos, equity):
        to_call = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        noise = self._rng.uniform(0.90, 1.10)

        # Detect premium
        k_ranks = sorted([_rank(c) for c in my_cards[:2]], reverse=True)
        is_premium = (
            equity > 0.65
            or (k_ranks[0] == k_ranks[1] and k_ranks[0] >= RANK_8)
        )

        if blind_pos == 0:  # SB
            if is_premium:
                # Slow-play 15%
                if self._rng.random() < 0.15:
                    if valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[CHECK]:
                        return (CHECK, 0, 0, 0)
                if valid[RAISE] and max_raise >= min_raise:
                    amt = _clamp(int(10 * noise), min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
            elif equity > 0.45:
                if self._rng.random() < 0.65 and valid[RAISE] and max_raise >= min_raise:
                    amt = _clamp(int(8 * noise), min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
                if valid[CALL]:
                    return (CALL, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
            else:
                if self._rng.random() < 0.30 and valid[RAISE] and max_raise >= min_raise:
                    amt = _clamp(int(7 * noise), min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
                if valid[FOLD]:
                    return (FOLD, 0, 0, 0)

        else:  # BB
            if is_premium:
                if to_call > 0:
                    if self._rng.random() < 0.70 and valid[RAISE] and max_raise >= min_raise:
                        amt = _clamp(int(to_call * 3 * noise), min_raise, max_raise)
                        return (RAISE, amt, 0, 0)
                    if valid[CALL]:
                        return (CALL, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
            elif equity > 0.40:
                if to_call > 0:
                    if self._rng.random() < 0.15 and valid[RAISE] and max_raise >= min_raise:
                        amt = _clamp(int(to_call * 2.5 * noise), min_raise, max_raise)
                        return (RAISE, amt, 0, 0)
                    if valid[CALL]:
                        return (CALL, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
            else:
                if to_call > 0:
                    if equity > pot_odds and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[FOLD]:
                        return (FOLD, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)

        # Fallback
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        if valid[CALL]:
            return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    # ── Postflop Betting ──────────────────────────────────────────────────────

    def _aggro_to_action(self, aggro, equity, street, valid, min_raise,
                         max_raise, pot_size, to_call, bluff_enable):
        pot_ref = max(pot_size, 1)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0

        # If calling station: only fire real equity, not pure bluffs
        eff_aggro = aggro if bluff_enable else aggro * (0.5 + 0.5 * equity)

        # River: polarized
        if street == 3:
            if eff_aggro > 0.70:
                if valid[RAISE] and max_raise >= min_raise:
                    if eff_aggro > 0.85:
                        return (RAISE, max_raise, 0, 0)  # jam
                    frac = self._rng.uniform(0.80, 1.00)
                    amt = _clamp(int(pot_ref * frac), min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
                if valid[CALL]:
                    return (CALL, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
            elif eff_aggro < 0.30:
                # Give up
                if to_call > 0:
                    if equity > pot_odds and to_call <= pot_ref * 0.25 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[FOLD]:
                        return (FOLD, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
            else:
                # Medium: cheap call or check
                if to_call > 0:
                    if equity > pot_odds and to_call <= pot_ref * 0.35 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[FOLD]:
                        return (FOLD, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)

        # Flop / Turn
        if eff_aggro > 0.70:
            if valid[RAISE] and max_raise >= min_raise:
                frac = self._rng.uniform(0.70, 1.10) if street == 2 else self._rng.uniform(0.60, 0.85)
                amt = _clamp(int(pot_ref * frac), min_raise, max_raise)
                return (RAISE, amt, 0, 0)
        elif eff_aggro > 0.55:
            if valid[RAISE] and max_raise >= min_raise:
                frac = self._rng.uniform(0.45, 0.70)
                amt = _clamp(int(pot_ref * frac), min_raise, max_raise)
                return (RAISE, amt, 0, 0)
        elif eff_aggro > 0.40:
            if self._rng.random() < 0.55 and valid[RAISE] and max_raise >= min_raise:
                frac = self._rng.uniform(0.25, 0.45)
                amt = _clamp(int(pot_ref * frac), min_raise, max_raise)
                return (RAISE, amt, 0, 0)
            if to_call > 0 and equity > pot_odds and valid[CALL]:
                return (CALL, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)
        elif eff_aggro > 0.25:
            if to_call > 0:
                if equity > pot_odds and valid[CALL]:
                    return (CALL, 0, 0, 0)
                if valid[FOLD]:
                    return (FOLD, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
        else:
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if valid[FOLD]:
                return (FOLD, 0, 0, 0)

        # Fallback
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        if to_call > 0 and equity > pot_odds and valid[CALL]:
            return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _choose_bet(self, street, my_cards, community, opp_discards, my_discards,
                    valid, min_raise, max_raise, my_bet, opp_bet, pot_size,
                    blind_pos, opp_last_action):
        to_call = max(0, opp_bet - my_bet)

        dead = set()
        if my_discards:
            dead |= set(my_discards)
        if opp_discards:
            dead |= set(opp_discards)

        # Compute equity
        if len(my_cards) == 2 and len(community) >= 3:
            equity = self._mc_equity(my_cards, community, dead, num_sims=250)
        else:
            equity = 0.45

        # Three scores
        cap    = self._cap_score(opp_discards, community, my_discards)
        story  = self._story_score(my_cards, community, my_discards, street)
        backup = self._backup_score(my_cards, community, dead, equity, opp_discards)

        aggro = 0.45 * cap + 0.35 * story + 0.20 * backup

        # Hard triggers and brakes
        bonus, brake = self._hard_triggers(my_cards, community, opp_discards, my_discards)

        # Trigger D: opponent passive line on scary board
        if opp_last_action in ("check", "call") and story >= 0.40 and cap >= 0.30:
            bonus += 0.10

        # Track whether this decision is a suit-cap attack (for adaptive learning)
        self._last_suit_attack = (bonus >= 0.12)

        # Brake D: backup score very low + we've already built a story → don't over-fire
        if backup < 0.20 and (self._betting_history.get("bet_flop") or
                               self._betting_history.get("bet_turn")):
            brake += 0.15

        # Brake E: line incoherent by river (never bet flop or turn) → give up more
        if street == 3:
            bet_flop = self._betting_history.get("bet_flop", False)
            bet_turn = self._betting_history.get("bet_turn", False)
            if not bet_flop and not bet_turn:
                brake += 0.20
            elif not bet_turn:
                brake += 0.10

        aggro = min(1.0, max(0.0, aggro + bonus - brake))

        # Villain profile multiplier
        vm, bluff_enable = self._villain_profile()
        aggro = min(1.0, aggro * vm)

        # Seeded noise
        noise = self._rng.uniform(-0.05, 0.05)
        aggro = min(1.0, max(0.0, aggro + noise))

        return self._aggro_to_action(
            aggro, equity, street, valid, min_raise, max_raise,
            pot_size, to_call, bluff_enable
        )

    # ── Observe: Track Opponent Stats ─────────────────────────────────────────

    def observe(self, observation, reward, terminated, truncated, info):
        opp_action = observation.get("opp_last_action", "")

        # Showdown tracking: when the hand ends, record whether opp won
        if terminated:
            self._stats["showdown_win"][1] += 1
            if reward < 0:  # we lost — opp had a strong hand or we folded to a bluff
                self._stats["showdown_win"][0] += 1

        if not opp_action:
            return

        street_map = {1: "fold_flop_bet", 2: "fold_turn_bet", 3: "fold_river_bet"}
        street_key = street_map.get(self._last_street)

        if self._last_was_bet:
            if opp_action == "fold":
                if street_key:
                    self._stats[street_key][0] += 1
                    self._stats[street_key][1] += 1
                self._stats["fold_to_raise"][0] += 1
                self._stats["fold_to_raise"][1] += 1
                # Suit-attack tracking: did a suit-cap attack cause a fold?
                if self._last_suit_attack:
                    self._stats["suit_attack_fold"][0] += 1
                    self._stats["suit_attack_fold"][1] += 1
            elif opp_action in ("call", "check", "raise"):
                if street_key:
                    self._stats[street_key][1] += 1
                self._stats["fold_to_raise"][1] += 1
                # Suit-attack: opp did NOT fold
                if self._last_suit_attack:
                    self._stats["suit_attack_fold"][1] += 1
                # Call-down tracking: opp called our postflop bet
                if opp_action == "call" and self._last_street >= 1:
                    self._stats["call_down"][0] += 1
                    self._stats["call_down"][1] += 1
                elif self._last_street >= 1:
                    self._stats["call_down"][1] += 1

        # Track check-raise: opp raised after we checked/called
        if not self._last_was_bet and opp_action == "raise":
            self._stats["check_raise"][0] += 1
            self._stats["check_raise"][1] += 1
        elif not self._last_was_bet and opp_action in ("check", "call", "fold"):
            self._stats["check_raise"][1] += 1

        # Track overall opponent aggression
        if opp_action in ("raise", "call", "check", "fold"):
            self._stats["opp_aggression"][1] += 1
            if opp_action == "raise":
                self._stats["opp_aggression"][0] += 1

    # ── Main act ──────────────────────────────────────────────────────────────

    def act(self, observation, reward, terminated, truncated, info):
        my_cards    = [c for c in observation["my_cards"] if c != -1]
        community   = [c for c in observation["community_cards"] if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards  = [c for c in observation["my_discarded_cards"] if c != -1]
        valid       = observation["valid_actions"]
        street      = observation["street"]
        min_raise   = observation["min_raise"]
        max_raise   = observation["max_raise"]
        my_bet      = observation["my_bet"]
        opp_bet     = observation["opp_bet"]
        pot_size    = observation.get("pot_size", my_bet + opp_bet)
        blind_pos   = observation.get("blind_position", 0)
        opp_last    = observation.get("opp_last_action", "")

        # New hand detection
        if street == 0 and my_bet <= 2 and opp_bet <= 2:
            self.hand_number += 1
            self._betting_history = {"bet_flop": False, "bet_turn": False}
            self._last_was_bet = False

        self._seed_rng(my_cards, street)

        # ── Discard phase ────────────────────────────────────────────────────
        if valid[DISCARD]:
            i, j = self._choose_keep(my_cards, community, opp_discards)
            self._last_was_bet = False
            self._last_street = street
            return (DISCARD, 0, i, j)

        # ── Preflop betting ──────────────────────────────────────────────────
        if street == 0:
            dead = set()
            if len(my_cards) == 5:
                equity = self._preflop_equity(my_cards, dead)
            else:
                equity = self._mc_equity(my_cards, [], dead, num_sims=120)
            result = self._choose_preflop(
                my_cards, valid, min_raise, max_raise,
                my_bet, opp_bet, pot_size, blind_pos, equity
            )
            self._last_was_bet = (result[0] == RAISE)
            self._last_street = street
            return result

        # ── Post-flop betting ────────────────────────────────────────────────
        result = self._choose_bet(
            street, my_cards, community, opp_discards, my_discards,
            valid, min_raise, max_raise, my_bet, opp_bet, pot_size,
            blind_pos, opp_last
        )

        # Track betting history for story coherence
        if result[0] == RAISE:
            if street == 1:
                self._betting_history["bet_flop"] = True
            elif street == 2:
                self._betting_history["bet_turn"] = True

        self._last_was_bet = (result[0] == RAISE)
        self._last_street = street
        return result
