# MetaBot v4: Adaptive Decision-Tree Poker Agent
#
# Routes between four strategy modes based on live opponent profiling:
#   GTO_DEFAULT  — Libratus policy tables  (unknown villain, first ~5 hands)
#   AGRO_EXPLOIT — Cap/Story/Backup scoring (Overfolders and default)
#   VALUE_GRIND  — Equity ladder, no bluffs (Calling Stations)
#   TRAP_MODE    — Check-call heavy        (Maniacs and Trappers)
#
# Decision tree (per hand, priority order):
#   total_obs < 5            -> GTO_DEFAULT
#   opp_aggression > 0.45    -> TRAP_MODE    (Maniac)
#   fold_to_bet > 0.48       -> AGRO + mult  (Overfolder)
#   fold_to_bet < 0.30 or
#     call_down > 0.55       -> VALUE_GRIND  (Station)
#   check_raise > 0.10       -> TRAP_MODE    (Trapper)
#   opp_aggression < 0.15    -> AGRO + mult  (Passive)
#   opp_preflop_raise > 0.50 -> AGRO         (Loose PF)
#   opp_preflop_raise < 0.12 -> VALUE_GRIND  (Limper)
#   default                  -> AGRO x 1.0

import random
from collections import Counter
from itertools import combinations

from agents.agent import Agent
from gym_env import PokerEnv

try:
    from submission.libratus_tables import POLICY, KEEP_EQUITY, POSTERIOR, MATCHUPS
except ImportError:
    from libratus_tables import POLICY, KEEP_EQUITY, POSTERIOR, MATCHUPS

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_RANKS = 9
DECK_SIZE = 27
RANK_A    = 8
RANK_9    = 7
RANK_8    = 6

FOLD    = PokerEnv.ActionType.FOLD.value
RAISE   = PokerEnv.ActionType.RAISE.value
CHECK   = PokerEnv.ActionType.CHECK.value
CALL    = PokerEnv.ActionType.CALL.value
DISCARD = PokerEnv.ActionType.DISCARD.value

_int_to_card = PokerEnv.int_to_card

MODE_GTO   = "GTO"
MODE_AGRO  = "AGRO"
MODE_VALUE = "VALUE"
MODE_TRAP  = "TRAP"

# ── Discard scoring weights ──────────────────────────────────────────────────
# The old discard_engine.py hard-tiered pairs above all draws (pair base=200,
# max flush draw=116).  These rebalanced weights let strong draws beat weak
# pairs, matching the empirical finding that MetaV5 kept flush draws only 498
# times vs AlphaNiTV5's 1303 times.

W_FLUSH_MADE       = 900
W_FH_MADE          = 800
W_STRAIGHT_MADE    = 700
W_TRIPS_BASE       = 500
W_TWO_PAIR_BASE    = 370
W_PAIR_BASE        = 70       # heavily reduced — mid pairs were massive EV drain
W_PAIR_RANK_MULT   = 14       # steeper rank curve so AA stays viable, low pairs drop
PAIR_DRAW_PENALTY  = 60       # stronger penalty when flush/straight draws available

W_FLUSH_DRAW       = 210      # up from ~80 — flush draws are the biggest leak
W_FLUSH_OUT_MULT   = 12       # up from 4
W_FLUSH_QUAL       = 5        # up from 2
W_OESD_BASE        = 155      # up from ~50
W_OESD_OUT_MULT    = 8        # up from 3
W_GUTSHOT_BASE     = 60
W_GUTSHOT_OUT_MULT = 5
W_BOAT_OUT_MULT    = 12
W_BOAT_QUALITY     = 3
W_THREAT_SUIT      = 10       # up from 3 — was too small to ever matter
W_OPP_HIGH_DUMP    = 6
W_STRUCTURE_MAX    = 60
TOP_K_MC           = 5        # wider net so rule-based ranker can't bury the best hand
MC_TIEBREAK_SIMS   = 200      # SE ~3.5% — reliable enough to influence close decisions
MC_TIEBREAK_W      = 80       # bridges gaps up to ~80 pts without overriding structural winners
TOTAL_HANDS        = 1000    # match length for bleed-out lock

# Six legal 5-card straight windows in the 27-card deck (rank indices)
_VALID_STRAIGHTS = [
    (8, 0, 1, 2, 3),   # A-2-3-4-5  (wheel)
    (0, 1, 2, 3, 4),   # 2-3-4-5-6
    (1, 2, 3, 4, 5),   # 3-4-5-6-7
    (2, 3, 4, 5, 6),   # 4-5-6-7-8
    (3, 4, 5, 6, 7),   # 5-6-7-8-9
    (4, 5, 6, 7, 8),   # 6-7-8-9-A  (broadway)
]
_WHEEL_SET = frozenset({8, 0, 1, 2, 3})

# ── Premium hand definitions (single source of truth) ─────────────────────────

PREM_PAIRS = frozenset([
    frozenset([RANK_A, RANK_A]),
    frozenset([RANK_9, RANK_9]),
    frozenset([RANK_8, RANK_8]),
])

PREM_ANY = frozenset([
    frozenset([RANK_A, RANK_9]),
    frozenset([RANK_A, RANK_8]),
])

PREM_SUITED = frozenset([
    frozenset([RANK_9, RANK_8]),
    frozenset([RANK_8, 5]),
    frozenset([5, 4]),
    frozenset([5, RANK_9]),
])

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
    """Longest run of consecutive ranks in a list."""
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
    if RANK_A in unique and 0 in unique:
        best = max(best, 2)
    return best

def _is_premium(c1, c2):
    r1, r2 = _rank(c1), _rank(c2)
    s1, s2 = _suit(c1), _suit(c2)
    ranks = frozenset([r1, r2])
    if r1 == r2 and frozenset([r1, r1]) in PREM_PAIRS:
        return True
    if ranks in PREM_ANY:
        return True
    if s1 == s2 and ranks in PREM_SUITED:
        return True
    return False

def _has_any_premium(cards):
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _is_premium(cards[i], cards[j]):
                return True
    return False

def _is_premium_pair(c1, c2):
    r1, r2 = _rank(c1), _rank(c2)
    return r1 == r2 and frozenset([r1, r1]) in PREM_PAIRS

def _has_premium_pair(cards):
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _is_premium_pair(cards[i], cards[j]):
                return True
    return False

def _board_paired_and_we_weak(my_cards, community):
    if len(my_cards) < 2 or len(community) < 3:
        return False
    b_ranks = [_rank(c) for c in community]
    rc = Counter(b_ranks)
    bp_rank = next((r for r, cnt in rc.items() if cnt >= 2), None)
    if bp_rank is None:
        return False
    my_ranks = [_rank(c) for c in my_cards[:2]]
    if bp_rank in my_ranks:
        return False
    if my_ranks[0] == my_ranks[1] and frozenset([my_ranks[0], my_ranks[0]]) in PREM_PAIRS:
        return False
    return True

def _normalize_action(raw):
    """Normalize opp_last_action from match.py (uppercase enum names or 'None')."""
    if not raw:
        return ""
    s = raw.strip().lower()
    if s == "none":
        return ""
    return s

# ── Libratus bucket helpers ────────────────────────────────────────────────────

# ── Hand rank and out-counting helpers ────────────────────────────────────────

def _hand_rank_category(my_cards, community):
    """
    Determine hand category from our 2 cards + community.
    Returns: 'trips_plus', 'two_pair', 'one_pair', 'draw_only', 'nothing'
    'trips_plus' covers trips, straight, flush, full house, straight flush.
    """
    if len(my_cards) < 2 or len(community) < 3:
        return "nothing"
    all_ranks = [_rank(c) for c in my_cards[:2]] + [_rank(c) for c in community]
    all_suits = [_suit(c) for c in my_cards[:2]] + [_suit(c) for c in community]
    rc = Counter(all_ranks)
    sc = Counter(all_suits)

    # Check flush (5 of same suit)
    if sc.most_common(1)[0][1] >= 5:
        return "trips_plus"
    # Check straight (5 consecutive)
    unique_r = sorted(set(all_ranks))
    # Add ace-low wrap
    if RANK_A in unique_r:
        unique_r_ext = [-1] + unique_r  # ace as -1 (below 0=rank2)
    else:
        unique_r_ext = unique_r
    best_run = 1
    cur_run = 1
    for i in range(1, len(unique_r_ext)):
        if unique_r_ext[i] - unique_r_ext[i-1] == 1:
            cur_run += 1
            best_run = max(best_run, cur_run)
        else:
            cur_run = 1
    if best_run >= 5:
        return "trips_plus"
    # Check trips (3 of a kind)
    if rc.most_common(1)[0][1] >= 3:
        return "trips_plus"
    # Check two pair
    pairs = [r for r, cnt in rc.items() if cnt >= 2]
    if len(pairs) >= 2:
        return "two_pair"
    if len(pairs) == 1:
        return "one_pair"
    return "nothing"


def _count_flush_outs(my_cards, community, opp_discards, my_discards):
    """
    Count live flush outs. Returns (best_suit_count, live_outs, total_unknowns).
    best_suit_count = how many of the best suit we have (hand + board).
    """
    if len(my_cards) < 2 or len(community) < 3:
        return 0, 0, 27
    known = set(my_cards[:2]) | set(community) | set(opp_discards) | set(my_discards)
    total_unknowns = 27 - len(known)

    best_suit = -1
    best_count = 0
    best_live = 0
    for s in range(3):  # 3 suits
        in_hand = sum(1 for c in my_cards[:2] if _suit(c) == s)
        on_board = sum(1 for c in community if _suit(c) == s)
        count = in_hand + on_board
        if count >= best_count:
            # Count dead cards of this suit
            dead_of_suit = sum(1 for c in known if _suit(c) == s)
            live = 9 - dead_of_suit  # 9 cards per suit in 27-card deck
            if count > best_count or live > best_live:
                best_suit = s
                best_count = count
                best_live = live
    return best_count, best_live, total_unknowns


def _count_straight_outs(my_cards, community, opp_discards, my_discards):
    """
    Count live straight outs. Returns (cards_in_run, live_outs, total_unknowns).
    Checks how close we are to a 5-card straight and how many outs complete it.
    """
    if len(my_cards) < 2 or len(community) < 3:
        return 0, 0, 27
    known = set(my_cards[:2]) | set(community) | set(opp_discards) | set(my_discards)
    total_unknowns = 27 - len(known)
    have_ranks = set(_rank(c) for c in my_cards[:2]) | set(_rank(c) for c in community)

    best_in_run = 0
    best_outs = 0

    # Check all possible 5-card straights: A-2-3-4-5 through 6-7-8-9-A
    straight_windows = []
    for low in range(0, NUM_RANKS):  # 0=rank2 through 8=rankA
        window = [(low + i) % NUM_RANKS for i in range(5)]
        # Validate: must be consecutive (handle ace-high: 5-6-7-8-A where A=8)
        # Actually straights are: A2345, 23456, 34567, 45678, 56789, 6789A
        straight_windows.append(window)
    # Specific valid straights for this game:
    valid_straights = [
        [RANK_A, 0, 1, 2, 3],  # A-2-3-4-5
        [0, 1, 2, 3, 4],       # 2-3-4-5-6
        [1, 2, 3, 4, 5],       # 3-4-5-6-7
        [2, 3, 4, 5, 6],       # 4-5-6-7-8
        [3, 4, 5, 6, 7],       # 5-6-7-8-9
        [4, 5, 6, 7, RANK_A],  # 6-7-8-9-A
    ]

    for window in valid_straights:
        have = sum(1 for r in window if r in have_ranks)
        need_ranks = [r for r in window if r not in have_ranks]
        if have >= 3 and len(need_ranks) <= 2:
            # Count live outs for needed ranks
            live = 0
            for nr in need_ranks:
                # Each rank has 3 cards (3 suits); count how many are still unknown
                for s in range(3):
                    card_id = s * NUM_RANKS + nr
                    if card_id not in known:
                        live += 1
            if have > best_in_run or (have == best_in_run and live > best_outs):
                best_in_run = have
                best_outs = live

    return best_in_run, best_outs, total_unknowns


def _has_live_draw(my_cards, community, opp_discards, my_discards):
    """Return True if the hand has a meaningful draw (4-to-flush or OESD)."""
    suit_count, flush_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
    if suit_count >= 4 and flush_outs >= 2:
        return True
    run_count, straight_outs, _ = _count_straight_outs(my_cards, community, opp_discards, my_discards)
    if run_count >= 4 and straight_outs >= 3:
        return True
    return False


def _bucket_keep(keep2):
    r1, r2 = _rank(keep2[0]), _rank(keep2[1])
    s1, s2 = _suit(keep2[0]), _suit(keep2[1])
    suited = (s1 == s2)
    if r1 == r2:
        if r1 >= RANK_9:   return "premium_pair"
        elif r1 >= RANK_8: return "medium_pair"
        else:              return "low_pair"
    eg = _effective_gap(keep2[0], keep2[1])
    if suited:
        if eg <= 1:   return "suited_connector"
        elif eg <= 3: return "suited_semi"
        else:         return "suited_gapper"
    else:
        if eg <= 1: return "offsuit_connector"
        else:       return "offsuit_other"

def _bucket_flop_simple(community):
    if not community or len(community) < 3:
        return "medium"
    suits  = [_suit(c) for c in community]
    ranks  = [_rank(c) for c in community]
    sc     = Counter(suits)
    rc     = Counter(ranks)
    score  = 0
    if sc.most_common(1)[0][1] >= 3:   score += 3
    elif sc.most_common(1)[0][1] >= 2: score += 1
    conn = _max_connectivity(ranks)
    if conn >= 3:   score += 2
    elif conn >= 2: score += 1
    if rc.most_common(1)[0][1] >= 2: score += 1
    if score >= 3: return "wet"
    if score >= 1: return "medium"
    return "dry"

def _bucket_strength(equity):
    if equity > 0.80:   return "monster"
    elif equity > 0.65: return "strong"
    elif equity > 0.50: return "good"
    elif equity > 0.35: return "marginal"
    return "weak"

def _bucket_to_call(to_call, pot_size):
    if to_call <= 0: return "none"
    ratio = to_call / max(pot_size + to_call, 1)
    if ratio <= 0.15: return "small"
    elif ratio <= 0.40: return "medium"
    return "large"

def _bucket_opp_discard(opp_discards):
    if len(opp_discards) < 3:
        return "unknown"
    ranks    = [_rank(c) for c in opp_discards]
    suits    = [_suit(c) for c in opp_discards]
    sc       = Counter(suits)
    rc       = Counter(ranks)
    sorted_r = sorted(ranks)
    conn = sum(1 for i in range(len(sorted_r) - 1)
               if sorted_r[i + 1] - sorted_r[i] == 1)
    if RANK_A in sorted_r and 0 in sorted_r:
        conn += 1
    if rc.most_common(1)[0][1] >= 2:  return "discarded_pair"
    if sc.most_common(1)[0][1] >= 2:  return "suited_cluster"
    if conn >= 2:                      return "connected_cluster"
    if RANK_A in ranks:                return "high_junk"
    if max(ranks) <= 5:                return "low_junk"
    return "mixed_discard"

# ── PlayerAgent ───────────────────────────────────────────────────────────────

class PlayerAgent(Agent):

    _STAT_PRIORS = {
        "fold_to_bet":       0.35,
        "fold_to_raise":     0.35,
        "check_raise":       0.05,
        "call_down":         0.40,
        "opp_aggression":    0.25,
        "opp_avg_bet_frac":  0.50,
        "opp_preflop_raise": 0.30,
    }

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self._env      = PokerEnv()
        self.evaluator = self._env.evaluator
        self.hand_number = 0
        self._rng        = random.Random(42)

        self._stats = {
            "fold_to_bet":       [0, 0],
            "fold_to_raise":     [0, 0],
            "check_raise":       [0, 0],
            "call_down":         [0, 0],
            "opp_aggression":    [0, 0],
            "opp_avg_bet_frac":  [0.0, 0],
            "opp_preflop_raise": [0, 0],
        }

        self._hand_mode       = MODE_GTO
        self._hand_override   = None
        self._agro_mult       = 1.0
        self._betting_history = {"bet_flop": False, "bet_turn": False}
        self._last_was_bet    = False
        self._last_street     = 0
        self._opp_folded       = False
        self._we_folded        = False
        self._prev_opp_bet     = 0
        self._prev_my_bet      = 0

        self._running_pnl  = 0
        self._mode_reason  = "warmup"
        self._prev_mode    = MODE_GTO
        self._last_equity  = 0.5
        self._last_aggro   = 0.5
        self._opp_hand_aggr = 0.0
        self._opp_archetype = "default"

    def __name__(self):
        return "MetaBot"

    # ── Time-aware sim budget ─────────────────────────────────────────────────

    def _sim_budget(self, time_left):
        """Returns (discard_sims_per_combo, postflop_sims)."""
        if time_left > 300:   return 200, 400
        elif time_left > 150: return 120, 250
        elif time_left > 60:  return 60,  150
        return 30, 80

    # ── MC Equity ─────────────────────────────────────────────────────────────

    def _mc_equity(self, my2, community, dead, num_sims=200):
        known     = set(my2) | set(community) | dead
        remaining = [c for c in range(DECK_SIZE) if c not in known]
        needed    = 2 + (5 - len(community))
        if needed > len(remaining):
            return 0.5
        wins  = 0.0
        total = 0
        for _ in range(num_sims):
            sample     = self._rng.sample(remaining, needed)
            opp        = sample[:2]
            full_board = list(community) + sample[2:]
            my_hand    = [_int_to_card(c) for c in my2]
            opp_hand   = [_int_to_card(c) for c in opp]
            board      = [_int_to_card(c) for c in full_board]
            mr         = self.evaluator.evaluate(my_hand, board)
            orr        = self.evaluator.evaluate(opp_hand, board)
            if mr < orr:    wins += 1.0
            elif mr == orr: wins += 0.5
            total += 1
        return wins / total if total > 0 else 0.5

    def _mc_equity_ranged(self, my2, community, dead, opp_signal, num_sims=200):
        """MC equity with opponent range narrowing via rejection sampling."""
        known     = set(my2) | set(community) | dead
        remaining = [c for c in range(DECK_SIZE) if c not in known]
        needed    = 2 + (5 - len(community))
        if needed > len(remaining):
            return 0.5

        reject_nothing  = opp_signal >= 2.0
        reject_one_pair = opp_signal >= 3.5
        max_retries = 3 if reject_nothing else 0

        wins  = 0.0
        total = 0
        for _ in range(num_sims):
            sample = self._rng.sample(remaining, needed)
            opp    = sample[:2]

            if max_retries > 0 and len(community) >= 3:
                for retry in range(max_retries):
                    full_board_check = list(community) + sample[2:]
                    cat = _hand_rank_category(list(opp), full_board_check[:3])
                    if cat == "nothing" or (reject_one_pair and cat == "one_pair"):
                        sample = self._rng.sample(remaining, needed)
                        opp    = sample[:2]
                    else:
                        break

            full_board = list(community) + sample[2:]
            my_hand    = [_int_to_card(c) for c in my2]
            opp_hand   = [_int_to_card(c) for c in opp]
            board      = [_int_to_card(c) for c in full_board]
            mr         = self.evaluator.evaluate(my_hand, board)
            orr        = self.evaluator.evaluate(opp_hand, board)
            if mr < orr:    wins += 1.0
            elif mr == orr: wins += 0.5
            total += 1
        return wins / total if total > 0 else 0.5

    def _street_adjust(self, equity, street, has_draw, flush_outs, straight_outs,
                        hand_cat="unknown"):
        """Adjust raw MC equity for street-dependent implied odds and draw death."""
        outs = max(flush_outs, straight_outs)
        if street == 1 and has_draw and outs > 0:
            equity += min(0.10, outs * 0.025)
        elif street == 3 and has_draw and hand_cat in ("nothing", "one_pair"):
            equity -= 0.10
        return _clamp(equity, 0.0, 0.98)

    def _range_equity(self, my2, community, dead, opp_b, num_sims, opp_signal=0.0):
        """Blend MC equity (70%) with Libratus range-vs-range table (30%)."""
        mc_eq = self._mc_equity_ranged(my2, community, dead, opp_signal, num_sims)
        if opp_b == "unknown" or len(community) < 3:
            return mc_eq
        board_b  = _bucket_flop_simple(community)
        opp_dist = POSTERIOR.get(f"{opp_b}|{board_b}", {})
        if not opp_dist:
            return mc_eq
        my_kb      = _bucket_keep(my2)
        matchup_eq = 0.0
        wsum       = 0.0
        for opp_kb, prob in opp_dist.items():
            matchup_eq += prob * MATCHUPS.get(f"{my_kb}|{opp_kb}", 0.5)
            wsum += prob
        if wsum > 0:
            return 0.7 * mc_eq + 0.3 * (matchup_eq / wsum)
        return mc_eq

    def _preflop_equity(self, my5, p_sims):
        """Best 2-card keep equity from a 5-card preflop hand."""
        if _has_premium_pair(my5):
            return 0.75
        sims = max(p_sims // 10, 15)
        best = 0.0
        for i, j in combinations(range(len(my5)), 2):
            keep = [my5[i], my5[j]]
            toss = set(my5[k] for k in range(len(my5)) if k not in (i, j))
            eq = self._mc_equity(keep, [], toss, num_sims=sims)
            if eq > best:
                best = eq
        return best if best > 0 else 0.45

    # ── Stat helpers ──────────────────────────────────────────────────────────

    def _safe_rate(self, key):
        folds, total = self._stats.get(key, [0, 0])
        if total < 3:
            return self._STAT_PRIORS.get(key, 0.35)
        return folds / total

    def _fmt_stat(self, key):
        num, den = self._stats.get(key, [0, 0])
        rate = self._safe_rate(key)
        return f"{rate:.2f}({num}/{den})"

    def _avg_bet_frac(self):
        total_sum, count = self._stats["opp_avg_bet_frac"]
        if count < 5:
            return self._STAT_PRIORS["opp_avg_bet_frac"]
        return total_sum / count

    def _total_obs(self):
        return self.hand_number

    # ── Opponent action processing (called from act()) ────────────────────────

    def _process_opponent_action(self, observation, opp_action, last_was_bet, last_street):
        """Process the opponent's action and update stats. Called from act()."""
        if not opp_action:
            return

        if opp_action == "fold":
            self._opp_folded = True

        if last_was_bet:
            if opp_action == "fold":
                self._stats["fold_to_bet"][0] += 1
                self._stats["fold_to_bet"][1] += 1
                self._stats["fold_to_raise"][0] += 1
                self._stats["fold_to_raise"][1] += 1
            elif opp_action in ("call", "check", "raise"):
                self._stats["fold_to_bet"][1] += 1
                self._stats["fold_to_raise"][1] += 1
                if opp_action == "call" and last_street >= 1:
                    self._stats["call_down"][0] += 1
                    self._stats["call_down"][1] += 1
                elif last_street >= 1:
                    self._stats["call_down"][1] += 1

        if not last_was_bet and opp_action == "raise":
            self._stats["check_raise"][0] += 1
            self._stats["check_raise"][1] += 1
            if self._hand_override is None:
                self._hand_override = MODE_VALUE
                self.logger.info(
                    f"  [OVERRIDE H{self.hand_number} S{last_street}] "
                    f"check_raise -> VALUE (cr={self._fmt_stat('check_raise')})"
                )
        elif not last_was_bet and opp_action in ("check", "call", "fold"):
            self._stats["check_raise"][1] += 1
        elif last_was_bet and opp_action == "raise":
            if self._hand_override is None:
                self._hand_override = MODE_VALUE
                self.logger.info(
                    f"  [OVERRIDE H{self.hand_number} S{last_street}] "
                    f"reraise -> VALUE"
                )

        if opp_action in ("raise", "call", "check"):
            self._stats["opp_aggression"][1] += 1
            if opp_action == "raise":
                self._stats["opp_aggression"][0] += 1
                opp_bet_obs = observation.get("opp_bet", 0)
                my_bet_obs  = observation.get("my_bet", 0)
                pot_obs     = observation.get("pot_size", opp_bet_obs + my_bet_obs)
                raise_size  = max(0, opp_bet_obs - my_bet_obs)
                if pot_obs > 0 and raise_size > 0:
                    frac = raise_size / pot_obs
                    self._stats["opp_avg_bet_frac"][0] += frac
                    self._stats["opp_avg_bet_frac"][1] += 1

        if last_street == 0 and opp_action in ("raise", "call", "check", "fold"):
            self._stats["opp_preflop_raise"][1] += 1
            if opp_action == "raise":
                self._stats["opp_preflop_raise"][0] += 1

        current_street = observation.get("street", 0)
        if current_street > last_street:
            self._opp_hand_aggr *= 0.7

        if opp_action == "raise":
            self._opp_hand_aggr += 0.7
        elif opp_action == "call" and last_was_bet:
            self._opp_hand_aggr += 0.2

    # ── Villain classifier ────────────────────────────────────────────────────

    def _select_mode(self):
        """Returns (mode, agro_mult). Called once per hand at hand start."""
        if self._total_obs() < 5:
            self._mode_reason = "warmup"
            self._opp_archetype = "default"
            return MODE_GTO, 1.0

        ramp = min(1.0, (self.hand_number - 5) / 7.0)

        fold_to_bet = self._safe_rate("fold_to_bet")
        cr_rate     = self._safe_rate("check_raise")
        call_down   = self._safe_rate("call_down")
        opp_aggro   = self._safe_rate("opp_aggression")
        pf_raise    = self._safe_rate("opp_preflop_raise")

        if opp_aggro > 0.45:
            self._mode_reason = f"maniac(agg={opp_aggro:.2f})"
            self._opp_archetype = "maniac"
            return MODE_TRAP, 1.0
        if fold_to_bet > 0.48:
            mult = round(1.0 + 0.4 * ramp, 2)
            self._mode_reason = f"overfold(fb={fold_to_bet:.2f})"
            self._opp_archetype = "overfolder"
            return MODE_AGRO, mult
        if fold_to_bet < 0.30 or call_down > 0.55:
            self._mode_reason = f"station(fb={fold_to_bet:.2f},cd={call_down:.2f})"
            self._opp_archetype = "station"
            return MODE_VALUE, 1.0
        if cr_rate > 0.10:
            self._mode_reason = f"checkraise({cr_rate:.2f})"
            self._opp_archetype = "maniac"
            return MODE_TRAP, 1.0
        if opp_aggro < 0.15 and self._stats["opp_aggression"][1] >= 8:
            mult = round(1.0 + 0.3 * ramp, 2)
            self._mode_reason = f"passive(agg={opp_aggro:.2f})"
            self._opp_archetype = "overfolder"
            return MODE_AGRO, mult
        if pf_raise > 0.50:
            mult = round(1.0 + 0.2 * ramp, 2)
            self._mode_reason = f"loose_pf({pf_raise:.2f})"
            self._opp_archetype = "default"
            return MODE_AGRO, mult
        if pf_raise < 0.12:
            self._mode_reason = f"limper({pf_raise:.2f})"
            self._opp_archetype = "station"
            return MODE_VALUE, 1.0
        self._mode_reason = "default"
        self._opp_archetype = "default"
        return MODE_AGRO, 1.0

    # =========================================================================
    # DISCARD — Improved keep-2 engine  (fixes flush-draw / boat / pair bias)
    # =========================================================================
    #
    # The old discard_engine.py hard-tiered pairs (base 200) above all draws
    # (max flush draw ~116), so MetaV5 almost never kept flush draws.  The new
    # engine scores every candidate as a weighted sum of six components so that
    # strong live draws can beat naked pairs.
    # =========================================================================

    # ── Step 1: classify the immediate 5-card made hand ──────────────────────

    @staticmethod
    def _classify_made(keep2, flop3):
        """Classify keep2 + flop3 into (category, details).

        Priority: flush > full_house > straight > trips > two_pair > pair > nothing.
        In the 27-card variant flushes are the rarest and strongest hands.
        """
        all5 = list(keep2) + list(flop3)
        ranks = [_rank(c) for c in all5]
        suits = [_suit(c) for c in all5]
        rc = Counter(ranks)
        sc = Counter(suits)

        is_flush = (sc.most_common(1)[0][1] == 5)

        unique_ranks = sorted(set(ranks))
        is_straight = False
        straight_high = -1
        if len(unique_ranks) == 5:
            ur_set = set(unique_ranks)
            for window in _VALID_STRAIGHTS:
                if set(window) == ur_set:
                    is_straight = True
                    straight_high = 3 if ur_set == _WHEEL_SET else max(window)
                    break

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

        trips_rank = None
        pair_ranks = []
        for r, cnt in rc.most_common():
            if cnt >= 3 and trips_rank is None:
                trips_rank = r
            elif cnt == 2:
                pair_ranks.append(r)

        if trips_rank is not None and pair_ranks:
            return ('full_house', {
                'trips_rank': trips_rank,
                'pair_rank': max(pair_ranks),
            })

        if is_straight:
            return ('straight', {'high': straight_high})

        if trips_rank is not None:
            kickers = sorted([r for r in ranks if r != trips_rank], reverse=True)
            return ('trips', {
                'trips_rank': trips_rank,
                'kicker_ranks': kickers,
            })

        if len(pair_ranks) >= 2:
            pr_sorted = sorted(pair_ranks, reverse=True)
            kicker = [r for r in ranks if r not in pair_ranks]
            return ('two_pair', {
                'pair_ranks': pr_sorted,
                'kicker': max(kicker) if kicker else 0,
            })

        if pair_ranks:
            pr = pair_ranks[0]
            kickers = sorted([r for r in ranks if r != pr], reverse=True)
            return ('pair', {'pair_rank': pr, 'kickers': kickers})

        return ('nothing', {'high_cards': sorted(ranks, reverse=True)})

    # ── Step 2: full-house / boat potential ───────────────────────────────────

    @staticmethod
    def _compute_boat_potential(keep2, flop3, known):
        """Score boat potential for trips / two-pair / paired-board keeps.

        In the 27-card game paired boards are common (only 9 ranks, 3 copies
        each).  Boat potential is a first-class scoring term — not a tiny bonus
        — because full houses are extremely strong when they hit.

        Returns (boat_score, live_boat_outs).
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
            kicker_ranks = set(r for r in ranks if r != trips_rank)
            for target in kicker_ranks:
                for s in range(3):
                    cid = s * NUM_RANKS + target
                    if cid not in known:
                        live_outs += 1
                best_quality = max(best_quality, trips_rank * 10 + target)

        elif len(pair_ranks) >= 2:
            for pr in pair_ranks:
                for s in range(3):
                    cid = s * NUM_RANKS + pr
                    if cid not in known:
                        live_outs += 1
                other = max(r for r in pair_ranks if r != pr)
                best_quality = max(best_quality, pr * 10 + other)

        boat_score = live_outs * W_BOAT_OUT_MULT + best_quality * W_BOAT_QUALITY
        return boat_score, live_outs

    # ── Step 3: flush-draw potential ─────────────────────────────────────────

    @staticmethod
    def _compute_flush_potential(keep2, flop3, known):
        """Score flush-draw potential when both kept cards are suited and the
        flop contains 2+ of that suit (4-to-a-flush on the flop).

        This was the single biggest leak in the old engine: flush draws were
        scored so low they could never beat a naked pair of 2s.  Now a strong
        live flush draw scores 250-320, beating most naked pairs.

        Opponent discards are already folded into *known* so dead outs are
        automatically excluded from live_outs.

        Returns (flush_score, is_draw, live_outs, quality_rank).
        """
        k_suits = [_suit(c) for c in keep2]
        if k_suits[0] != k_suits[1]:
            return 0.0, False, 0, 0

        draw_suit = k_suits[0]
        flop_of_suit = sum(1 for c in flop3 if _suit(c) == draw_suit)
        total = 2 + flop_of_suit
        if total < 4:
            return 0.0, False, 0, 0

        live_outs = 0
        for r in range(NUM_RANKS):
            cid = draw_suit * NUM_RANKS + r
            if cid not in known:
                live_outs += 1

        quality_rank = max(_rank(c) for c in keep2)
        flush_score = W_FLUSH_DRAW + live_outs * W_FLUSH_OUT_MULT + quality_rank * W_FLUSH_QUAL
        return flush_score, True, live_outs, quality_rank

    # ── Step 4: straight-draw potential ──────────────────────────────────────

    @staticmethod
    def _compute_straight_potential(keep2, flop3, known):
        """Score straight-draw potential (OESD / gutshot / double-gutter).

        Scans all 6 valid straight windows.  OESD = 2+ windows each needing
        exactly 1 card.  Open-ended beats gutshot; more live outs beats fewer;
        higher straight beats lower.

        Returns (straight_score, draw_type, live_outs, straight_high).
        """
        all5 = list(keep2) + list(flop3)
        have_ranks = set(_rank(c) for c in all5)

        completing_cards = set()
        best_high = -1
        windows_needing_one = 0

        for window in _VALID_STRAIGHTS:
            w_set = set(window)
            need = w_set - have_ranks

            if len(need) == 1:
                windows_needing_one += 1
                needed_rank = next(iter(need))
                for s in range(3):
                    cid = s * NUM_RANKS + needed_rank
                    if cid not in known:
                        completing_cards.add(cid)
                high = 3 if w_set == _WHEEL_SET else max(window)
                best_high = max(best_high, high)

            elif len(need) == 2:
                for needed_rank in need:
                    for s in range(3):
                        cid = s * NUM_RANKS + needed_rank
                        if cid not in known:
                            completing_cards.add(cid)

        live_outs = len(completing_cards)

        if windows_needing_one >= 2:
            draw_type = 'oesd'
            straight_score = W_OESD_BASE + live_outs * W_OESD_OUT_MULT + max(best_high, 0)
        elif windows_needing_one == 1:
            draw_type = 'gutshot'
            straight_score = W_GUTSHOT_BASE + live_outs * W_GUTSHOT_OUT_MULT + max(best_high, 0)
        elif live_outs >= 4:
            draw_type = 'double_gutter'
            straight_score = W_GUTSHOT_BASE + live_outs * W_GUTSHOT_OUT_MULT + max(best_high, 0)
        else:
            draw_type = 'none'
            straight_score = 0.0

        return straight_score, draw_type, live_outs, best_high

    # ── Step 5: opponent discard threat-suit inference ────────────────────────

    @staticmethod
    def _compute_threat_bonus(keep2, flop3, opp_discards, known):
        """Lightweight inference from visible opponent discards.

        Two signals:
        1. Suit-abandon: if opponent threw 0 cards of suit S and the flop has
           2+ of S, suit S is a "threat suit" they may have kept.  Holding
           blockers of that suit is valuable.
        2. High-card dump: if opponent tossed 2+ high cards (rank >= 8) their
           kept hand likely skews weaker.

        The old engine used +3 per match on a 0-999 scale — too small to ever
        change a decision.  Now W_THREAT_SUIT = 10.
        """
        if not opp_discards:
            return 0.0

        bonus = 0.0
        opp_suit_counts = Counter(_suit(c) for c in opp_discards)
        keep_suits = [_suit(c) for c in keep2]
        flop_suits = Counter(_suit(c) for c in flop3)

        for s in range(3):
            if opp_suit_counts.get(s, 0) == 0 and flop_suits.get(s, 0) >= 2:
                if any(ks == s for ks in keep_suits):
                    bonus += W_THREAT_SUIT

        # Also credit when opponent abandons high cards of a suit we draw in
        opp_suit_abandon = Counter(_suit(c) for c in opp_discards)
        for s, cnt in opp_suit_abandon.items():
            flop_of_s = flop_suits.get(s, 0)
            if cnt >= 2 and flop_of_s >= 2:
                if any(ks == s for ks in keep_suits):
                    bonus += W_THREAT_SUIT * 0.5

        opp_high = sum(1 for c in opp_discards if _rank(c) >= RANK_8)
        if opp_high >= 2:
            bonus += W_OPP_HIGH_DUMP

        return bonus

    # ── Step 6: structural fallback ──────────────────────────────────────────

    @staticmethod
    def _structural_fallback(keep2, flop3, known):
        """Score 0-60 based on raw card quality for close tiebreaks.

        Considers suitedness, connectivity, high-card value, and flop
        interaction.  Only matters when higher-tier components are tied.
        """
        r0, r1 = _rank(keep2[0]), _rank(keep2[1])
        s0, s1 = _suit(keep2[0]), _suit(keep2[1])
        suited = (s0 == s1)
        high_rank = max(r0, r1)

        gap = abs(r0 - r1)
        if gap > 4:
            gap = NUM_RANKS - gap

        score = 0.0

        if suited:
            flop_suit_count = sum(1 for c in flop3 if _suit(c) == s0)
            if flop_suit_count >= 2:
                score += 18
            elif flop_suit_count == 1:
                score += 3
            # 0 matching flop cards = worthless backdoor in 27-card game

        if gap == 0:
            score += 12 + high_rank
        elif gap == 1:
            score += 12
        elif gap == 2:
            score += 7
        elif gap == 3:
            score += 3

        score += high_rank * 1.5

        flop_ranks = [_rank(c) for c in flop3]
        if r0 in flop_ranks or r1 in flop_ranks:
            score += 6

        keep_set = set(keep2)
        flop_set = set(flop3)
        for r in (r0, r1):
            for s in range(3):
                cid = s * NUM_RANKS + r
                if cid in known and cid not in keep_set and cid not in flop_set:
                    score -= 1.5

        return max(0.0, min(float(W_STRUCTURE_MAX), score))

    # ── Orchestrator: score one candidate and pick the best ──────────────────

    def _choose_keep(self, my_cards, community, opp_discards, d_sims, mode=MODE_AGRO):
        """Evaluate all 10 keep-pairs and return the best (i, j).

        Uses a weighted sum of made-hand / boat / flush / straight / blocker /
        structure scores, then applies a small MC-equity tiebreak on the top-K
        candidates only (runtime safety).
        """
        if not community or len(community) < 3:
            # Safety fallback — discard only fires on the flop
            return self._choose_keep_fallback(my_cards, community, opp_discards, d_sims, mode)

        flop3 = list(community[:3])
        opp_d = list(opp_discards) if opp_discards else []

        candidates = []
        any_flush_draw = False

        for i, j in combinations(range(len(my_cards)), 2):
            keep = [my_cards[i], my_cards[j]]
            toss = [my_cards[k] for k in range(len(my_cards)) if k not in (i, j)]

            known = set(keep) | set(flop3) | set(toss) | set(opp_d)

            # ── Component scores ────────────────────────────────────────
            cat, details = self._classify_made(keep, flop3)

            # Made-hand base score
            if cat == 'flush':
                q = details['quality']
                sf_bump = 10 if details.get('is_straight_flush') else 0
                made_score = W_FLUSH_MADE + q * 8 + sf_bump
            elif cat == 'full_house':
                made_score = W_FH_MADE + details['trips_rank'] * 10 + details['pair_rank']
            elif cat == 'straight':
                made_score = W_STRAIGHT_MADE + details['high'] * 10
            elif cat == 'trips':
                made_score = W_TRIPS_BASE + details['trips_rank'] * 10
            elif cat == 'two_pair':
                pr = details['pair_ranks']
                made_score = W_TWO_PAIR_BASE + pr[0] * 5 + pr[1] * 2
            elif cat == 'pair':
                made_score = W_PAIR_BASE + details['pair_rank'] * W_PAIR_RANK_MULT
            else:
                made_score = 0.0

            boat_score, boat_outs = self._compute_boat_potential(keep, flop3, known)
            flush_score, is_fd, fd_outs, fd_qual = self._compute_flush_potential(keep, flop3, known)
            straight_score, sd_type, sd_outs, sd_high = self._compute_straight_potential(keep, flop3, known)
            blocker_score = self._compute_threat_bonus(keep, flop3, opp_d, known)
            structure_score = self._structural_fallback(keep, flop3, known)

            if is_fd:
                any_flush_draw = True

            total = made_score + boat_score + flush_score + straight_score + blocker_score + structure_score

            candidates.append((i, j, total, keep, toss, known, cat, is_fd, sd_type))

        # ── Pair penalty: downgrade naked pairs when live draws exist ────
        if any_flush_draw:
            for idx, (i, j, total, keep, toss, known, cat, is_fd, sd_type) in enumerate(candidates):
                if cat == 'pair' and not is_fd and sd_type == 'none':
                    candidates[idx] = (i, j, total - PAIR_DRAW_PENALTY, keep, toss, known, cat, is_fd, sd_type)

        # ── Top-K MC tiebreak ────────────────────────────────────────────
        candidates.sort(key=lambda c: c[2], reverse=True)
        top_k = candidates[:TOP_K_MC]

        best_score = -1.0
        best_ij = (candidates[0][0], candidates[0][1])

        for i, j, total, keep, toss, known, cat, is_fd, sd_type in top_k:
            dead = set(toss) | set(opp_d)
            eq = self._mc_equity(keep, flop3, dead, num_sims=MC_TIEBREAK_SIMS)
            final = total + MC_TIEBREAK_W * eq
            if final > best_score:
                best_score = final
                best_ij = (i, j)

        return best_ij

    def _choose_keep_fallback(self, my_cards, community, opp_discards, d_sims, mode=MODE_AGRO):
        """Original MC-based fallback for edge cases (pre-flop / missing community)."""
        best_score    = -999.0
        best_ij       = (0, 1)
        best_eq_ij    = (0, 1)
        best_raw_eq   = -1.0
        chosen_raw_eq = 0.0
        candidates    = []
        for i, j in combinations(range(len(my_cards)), 2):
            keep      = [my_cards[i], my_cards[j]]
            toss      = [my_cards[k] for k in range(len(my_cards)) if k not in (i, j)]
            toss_dead = set(toss) | (set(opp_discards) if opp_discards else set())
            sc, eq    = self._keep_score(keep, community, opp_discards, toss_dead, d_sims, mode)
            candidates.append((i, j, sc, eq))
            if eq > best_raw_eq:
                best_raw_eq = eq
                best_eq_ij  = (i, j)
            if sc > best_score:
                best_score    = sc
                best_ij       = (i, j)
                chosen_raw_eq = eq
        ties = [(i, j) for i, j, sc, _ in candidates if best_score - sc < 0.03]
        if len(ties) > 1:
            best_ij = self._rng.choice(ties)
            chosen_raw_eq = next(eq for i, j, _, eq in candidates if (i, j) == best_ij)
        if best_raw_eq - chosen_raw_eq > 0.10:
            best_ij = best_eq_ij
        return best_ij

    # =========================================================================
    # DISCARD — Legacy helpers (kept for reference / fallback path)
    # =========================================================================

    def _pressure_potential(self, keep2, community, opp_discards):
        score   = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]
        if k_suits[0] == k_suits[1]: score += 0.40
        eg = _effective_gap(keep2[0], keep2[1])
        if eg <= 1:   score += 0.30
        elif eg <= 2: score += 0.20
        elif eg <= 3: score += 0.10
        if community:
            b_suits = [_suit(c) for c in community]
            for s in set(k_suits):
                if sum(1 for bs in b_suits if bs == s) >= 2:
                    score += 0.15
                    break
        max_rank = max(k_ranks)
        if max_rank >= RANK_A:   score += 0.15
        elif max_rank >= RANK_9: score += 0.08
        return min(score, 1.0)

    def _draw_density(self, keep2, community):
        score   = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]
        if not community:
            if k_suits[0] == k_suits[1]: score += 0.30
            eg = _effective_gap(keep2[0], keep2[1])
            if eg <= 1:   score += 0.25
            elif eg <= 2: score += 0.15
            return min(score, 1.0)
        b_suits = [_suit(c) for c in community]
        b_ranks = [_rank(c) for c in community]
        for s in set(k_suits):
            total = (sum(1 for cs in b_suits if cs == s) +
                     sum(1 for cs in k_suits if cs == s))
            if total >= 4:   score += 0.40
            elif total == 3: score += 0.25
            elif total == 2: score += 0.08
        conn = _max_connectivity(k_ranks + b_ranks)
        if conn >= 5:   score += 0.35
        elif conn >= 4: score += 0.20
        elif conn >= 3: score += 0.08
        return min(score, 1.0)

    def _made_hand_value(self, keep2, community):
        score   = 0.0
        k_ranks = [_rank(c) for c in keep2]
        if k_ranks[0] == k_ranks[1]:
            r = k_ranks[0]
            if r >= RANK_9:   score += 0.80
            elif r >= RANK_8: score += 0.55
            elif r >= 4:      score += 0.35
            else:             score += 0.20
            return min(score, 1.0)
        if not community:
            return 0.0
        b_ranks     = [_rank(c) for c in community]
        rank_counts = Counter(k_ranks + b_ranks)
        max_count   = rank_counts.most_common(1)[0][1]
        pairs       = [r for r, v in rank_counts.items() if v >= 2]
        if max_count >= 3:    score += 0.70
        elif len(pairs) >= 2: score += 0.50
        elif len(pairs) == 1:
            pr = pairs[0]
            if pr in k_ranks:
                if pr >= RANK_A:   score += 0.35
                elif pr >= RANK_8: score += 0.20
                else:              score += 0.10
        return min(score, 1.0)

    def _blocker_value(self, keep2, community, opp_discards):
        score   = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]
        if RANK_A in k_ranks: score += 0.15
        if RANK_9 in k_ranks: score += 0.08
        if not community:
            return min(score, 1.0)
        b_suit_c = Counter(_suit(c) for c in community)
        dom_suit, dom_cnt = b_suit_c.most_common(1)[0]
        if dom_cnt >= 2:
            score += 0.20 * sum(1 for s in k_suits if s == dom_suit)
        return min(score, 1.0)

    def _story_flexibility(self, keep2, community):
        score   = 0.0
        k_suits = [_suit(c) for c in keep2]
        k_ranks = [_rank(c) for c in keep2]
        if k_suits[0] == k_suits[1]: score += 0.35
        eg = _effective_gap(keep2[0], keep2[1])
        if eg <= 1:   score += 0.30
        elif eg <= 2: score += 0.20
        max_r = max(k_ranks)
        if max_r >= RANK_A:   score += 0.20
        elif max_r >= RANK_9: score += 0.10
        if k_ranks[0] == k_ranks[1] and k_ranks[0] >= RANK_8:
            score += 0.30
        return min(score, 1.0)

    def _keep_score(self, keep2, community, opp_discards, toss_dead, d_sims, mode=MODE_AGRO):
        eq  = self._mc_equity(keep2, community, toss_dead, num_sims=d_sims)
        pp  = self._pressure_potential(keep2, community, opp_discards)
        dd  = self._draw_density(keep2, community)
        mhv = self._made_hand_value(keep2, community)
        bv  = self._blocker_value(keep2, community, opp_discards)
        sf  = self._story_flexibility(keep2, community)
        if mode in (MODE_VALUE, MODE_TRAP):
            return 3.5*eq + 0.6*pp + 1.7*dd + 2.0*mhv + 0.8*bv + 0.2*sf, eq
        return 2.8*eq + 2.2*pp + 1.7*dd + 1.0*mhv + 0.8*bv + 0.6*sf, eq

    # =========================================================================
    # GTO_DEFAULT — Libratus policy table + range-vs-range equity
    # =========================================================================

    def _gto_equity(self, my_cards, community, opp_discards, my_discards, p_sims):
        dead   = set()
        if my_discards:  dead |= set(my_discards)
        if opp_discards: dead |= set(opp_discards)
        if len(my_cards) > 2 and not community:
            return self._preflop_equity(my_cards, p_sims)
        cards2 = my_cards[:2] if len(my_cards) > 2 else my_cards
        _range_sig = 0.0 if self._opp_archetype == "maniac" else self._opp_hand_aggr
        if len(cards2) == 2 and len(community) >= 3:
            opp_b  = _bucket_opp_discard(opp_discards) if len(opp_discards) >= 3 else "unknown"
            equity = self._range_equity(cards2, community, dead, opp_b, p_sims,
                                        opp_signal=_range_sig)
        else:
            equity = self._mc_equity_ranged(cards2, community, dead, _range_sig,
                                            num_sims=max(p_sims // 3, 30))
        if _board_paired_and_we_weak(cards2, community):
            equity -= 0.08
        if len(opp_discards) >= 3:
            opp_b = _bucket_opp_discard(opp_discards)
            adj   = {"low_junk": -0.04, "suited_cluster": -0.03,
                     "connected_cluster": -0.02, "high_junk": 0.02, "discarded_pair": 0.03}
            equity += adj.get(opp_b, 0.0)
        return _clamp(equity, 0.0, 0.98)

    def _gto_action(self, street, my_cards, community, opp_discards, my_discards,
                    valid, min_raise, max_raise, my_bet, opp_bet, pot_size,
                    blind_pos, p_sims):
        to_call  = max(0, opp_bet - my_bet)
        position = "sb" if blind_pos == 0 else "bb"
        equity   = self._gto_equity(my_cards, community, opp_discards, my_discards, p_sims)
        if street >= 1 and len(my_cards) == 2 and len(community) >= 3:
            _, f_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
            _, s_outs, _ = _count_straight_outs(my_cards, community, opp_discards, my_discards)
            has_draw = _has_live_draw(my_cards, community, opp_discards, my_discards)
            _hcat = _hand_rank_category(my_cards, community)
            equity = self._street_adjust(equity, street, has_draw, f_outs, s_outs, hand_cat=_hcat)
        self._last_equity = equity
        strength = _bucket_strength(equity)
        board_b  = _bucket_flop_simple(community) if street >= 1 else "any"
        tc_b     = _bucket_to_call(to_call, pot_size)

        key    = str((street, position, strength, board_b, tc_b))
        policy = POLICY.get(key)
        if not policy:
            key2   = str((street, position, strength, "medium", "none"))
            policy = POLICY.get(key2, {
                "fold": 0.20, "check_call": 0.40, "small_bet": 0.25,
                "medium_bet": 0.10, "large_bet": 0.05, "jam": 0.0
            })
        policy = dict(policy)

        fold_r = self._safe_rate("fold_to_raise")
        agg_r  = self._safe_rate("opp_aggression")
        if fold_r > 0.50:
            policy["small_bet"]  = policy.get("small_bet", 0) + 0.12
            policy["fold"]       = max(0.0, policy.get("fold", 0) - 0.12)
        elif fold_r < 0.20:
            policy["small_bet"]  = max(0.0, policy.get("small_bet", 0) - 0.10)
            policy["check_call"] = policy.get("check_call", 0) + 0.10
        if agg_r > 0.55 and strength in ("marginal", "weak"):
            policy["check_call"] = max(0.0, policy.get("check_call", 0) - 0.12)
            policy["fold"]       = policy.get("fold", 0) + 0.12

        action = self._sample_policy(policy)
        return self._policy_to_tuple(action, valid, min_raise, max_raise,
                                     pot_size, to_call, equity)

    def _sample_policy(self, policy):
        r     = self._rng.random()
        cumul = 0.0
        for act, prob in policy.items():
            cumul += prob
            if r < cumul:
                return act
        return "check_call"

    def _policy_to_tuple(self, action, valid, min_raise, max_raise,
                         pot_size, to_call, equity):
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        if action == "fold":
            if valid[FOLD]:  return (FOLD,  0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)
        if action == "check_call":
            if to_call > 0 and valid[CALL]: return (CALL,  0, 0, 0)
            if valid[CHECK]:                return (CHECK, 0, 0, 0)
            if valid[CALL]:                 return (CALL,  0, 0, 0)
            return (CHECK, 0, 0, 0)
        if action == "jam":
            if valid[RAISE] and max_raise > 0: return (RAISE, max_raise, 0, 0)
            if valid[CALL]:                     return (CALL,  0, 0, 0)
            if valid[CHECK]:                    return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)
        pot_ref = max(pot_size, 1)
        if action == "small_bet":    frac = self._rng.uniform(0.30, 0.45)
        elif action == "medium_bet": frac = self._rng.uniform(0.55, 0.75)
        elif action == "large_bet":  frac = self._rng.uniform(0.85, 1.10)
        else:                        frac = 0.50
        amount = _clamp(int(pot_ref * frac), min_raise, max_raise)
        if valid[RAISE] and max_raise >= min_raise: return (RAISE, amount, 0, 0)
        if valid[CALL]:                              return (CALL,  0, 0, 0)
        if valid[CHECK]:                             return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    # =========================================================================
    # AGRO_EXPLOIT — Cap / Story / Backup scoring
    # =========================================================================

    def _cap_score(self, opp_discards, community, my_discards):
        if not opp_discards or len(opp_discards) < 3:
            return 0.20
        score      = 0.0
        opp_suit_c = Counter(_suit(c) for c in opp_discards)
        opp_ranks  = sorted(_rank(c) for c in opp_discards)
        my_suit_d  = Counter(_suit(c) for c in my_discards) if my_discards else Counter()
        b_suit_c   = Counter(_suit(c) for c in community)   if community   else Counter()
        for suit, opp_cnt in opp_suit_c.items():
            if opp_cnt < 2: continue
            base = 0.6 if opp_cnt == 3 else 0.3
            board_in = b_suit_c.get(suit, 0)
            if board_in >= 3:   base += 0.35
            elif board_in >= 2: base += 0.15
            my_d = my_suit_d.get(suit, 0)
            if my_d >= 2:   base *= 0.20
            elif my_d == 1: base *= 0.70
            score += base
        opp_conn = sum(1 for i in range(len(opp_ranks) - 1)
                       if opp_ranks[i + 1] - opp_ranks[i] <= 1)
        if opp_conn >= 2 and community and len(community) >= 3:
            b_conn = _max_connectivity([_rank(c) for c in community])
            if b_conn >= 3:   score += 0.20
            elif b_conn >= 2: score += 0.10
        elif opp_conn >= 2:
            score += 0.08
        if sum(1 for c in opp_discards if _rank(c) >= RANK_8) >= 2:
            score += 0.10
        return min(score, 1.0)

    def _story_score(self, my_cards, community, my_discards, street):
        if not community or len(community) < 3:
            return 0.40
        b_suits  = [_suit(c) for c in community]
        b_ranks  = [_rank(c) for c in community]
        b_suit_c = Counter(b_suits)
        dom_suit, dom_cnt = b_suit_c.most_common(1)[0]
        flush_base    = 0.45 if dom_cnt >= 3 else (0.20 if dom_cnt >= 2 else 0.0)
        b_conn        = _max_connectivity(b_ranks)
        straight_base = 0.35 if b_conn >= 3 else (0.15 if b_conn >= 2 else 0.0)
        score = max(flush_base, straight_base)
        if my_cards and len(my_cards) == 2:
            k_suits = [_suit(c) for c in my_cards]
            k_ranks = [_rank(c) for c in my_cards]
            if flush_base > 0:
                n_dom = sum(1 for s in k_suits if s == dom_suit)
                if n_dom >= 2:   score += 0.20
                elif n_dom >= 1: score += 0.10
            if straight_base > 0:
                all_conn = _max_connectivity(k_ranks + b_ranks)
                if all_conn >= 5:   score += 0.20
                elif all_conn >= 4: score += 0.10
        if my_discards and flush_base > 0:
            d_dom = Counter(_suit(c) for c in my_discards).get(dom_suit, 0)
            if d_dom >= 2:   score -= 0.30
            elif d_dom >= 1: score -= 0.10
        if street >= 2 and self._betting_history.get("bet_flop"):  score += 0.10
        if street == 3 and self._betting_history.get("bet_turn"):   score += 0.10
        if self._hand_override is not None:
            score = max(0.0, score - 0.30)
        return _clamp(score, 0.0, 1.0)

    def _backup_score(self, my_cards, community, dead, equity, opp_discards):
        score = equity * 0.6
        if not my_cards or len(my_cards) != 2:
            return min(score, 1.0)
        k_suits = [_suit(c) for c in my_cards]
        k_ranks = [_rank(c) for c in my_cards]
        if k_ranks[0] == k_ranks[1]:
            if k_ranks[0] >= RANK_9:   score += 0.15
            elif k_ranks[0] >= RANK_8: score += 0.08
            else:                      score += 0.04
        if community:
            b_suits = [_suit(c) for c in community]
            b_ranks = [_rank(c) for c in community]
            for s in set(k_suits):
                total = (sum(1 for cs in b_suits if cs == s) +
                         sum(1 for cs in k_suits if cs == s))
                if total >= 4:   score += 0.15
                elif total == 3: score += 0.08
            conn = _max_connectivity(k_ranks + b_ranks)
            if conn >= 5:   score += 0.15
            elif conn >= 4: score += 0.08
            b_suit_c = Counter(b_suits)
            if b_suit_c:
                dom_s, dom_c = b_suit_c.most_common(1)[0]
                if dom_c >= 2:
                    score += 0.05 * sum(1 for s in k_suits if s == dom_s)
        return min(score, 1.0)

    def _hard_triggers(self, my_cards, community, opp_discards, my_discards):
        bonus = 0.0
        brake = 0.0
        if not opp_discards or len(opp_discards) < 3:
            return bonus, brake
        opp_suit_c = Counter(_suit(c) for c in opp_discards)
        my_suit_d  = Counter(_suit(c) for c in my_discards) if my_discards else Counter()
        b_suit_c   = Counter(_suit(c) for c in community)   if community   else Counter()

        for suit, opp_cnt in opp_suit_c.items():
            board_cnt = b_suit_c.get(suit, 0)
            my_d_cnt  = my_suit_d.get(suit, 0)
            if opp_cnt == 3 and board_cnt >= 2 and my_d_cnt <= 1:
                bonus += 0.35
            elif opp_cnt == 2:
                if board_cnt >= 3:
                    bonus += 0.20
                elif board_cnt >= 2:
                    our = sum(1 for c in my_cards if _suit(c) == suit) if my_cards else 0
                    if our >= 1 or my_d_cnt == 0:
                        bonus += 0.12

        opp_ranks = sorted(_rank(c) for c in opp_discards)
        opp_conn  = sum(1 for i in range(len(opp_ranks) - 1)
                        if opp_ranks[i + 1] - opp_ranks[i] <= 1)
        if opp_conn >= 2 and community and len(community) >= 3:
            if _max_connectivity([_rank(c) for c in community]) >= 2:
                bonus += 0.15

        if community and len(community) >= 3 and my_cards and len(my_cards) == 2:
            k_suits_e = [_suit(c) for c in my_cards]
            k_ranks_e = [_rank(c) for c in my_cards]
            b_suits_e = [_suit(c) for c in community]
            best_ft   = max(
                sum(1 for cs in b_suits_e if cs == s) + sum(1 for cs in k_suits_e if cs == s)
                for s in set(k_suits_e)
            )
            if (best_ft >= 3 or
                    _max_connectivity(k_ranks_e + [_rank(c) for c in community]) >= 4):
                if self._safe_rate("fold_to_bet") > 0.45:
                    bonus += 0.12

        if community and my_discards:
            b_sc2 = Counter(_suit(c) for c in community)
            if b_sc2:
                dom_s2, dom_c2 = b_sc2.most_common(1)[0]
                if dom_c2 >= 2 and my_suit_d.get(dom_s2, 0) >= 2:
                    brake += 0.30
        if self._safe_rate("fold_to_bet") < 0.20: brake += 0.20
        if self._safe_rate("check_raise")  > 0.15: brake += 0.15

        brake = min(brake, 0.45)
        return bonus, brake

    def _aggro_to_action(self, aggro, equity, street, valid, min_raise,
                         max_raise, pot_size, to_call, bluff_enable):
        pot_ref  = max(pot_size, 1)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        eff      = aggro if bluff_enable else aggro * (0.5 + 0.5 * equity)

        if street == 3:
            if eff > 0.70:
                if valid[RAISE] and max_raise >= min_raise:
                    if eff > 0.85:
                        return (RAISE, max_raise, 0, 0)
                    amt = _clamp(int(pot_ref * self._rng.uniform(0.80, 1.00)), min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            elif eff < 0.30:
                if to_call > 0:
                    if equity > pot_odds + 0.15 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if equity > pot_odds and to_call <= pot_ref * 0.25 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[FOLD]: return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            else:
                if to_call > 0:
                    if equity > pot_odds and to_call <= pot_ref * 0.35 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if equity > pot_odds + 0.15 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[FOLD]: return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
        else:
            if eff > 0.70:
                if valid[RAISE] and max_raise >= min_raise:
                    frac = self._rng.uniform(0.70, 1.10) if street == 2 \
                           else self._rng.uniform(0.60, 0.85)
                    return (RAISE, _clamp(int(pot_ref * frac), min_raise, max_raise), 0, 0)
                if to_call > 0 and valid[CALL]: return (CALL,  0, 0, 0)
                if valid[CHECK]:                return (CHECK, 0, 0, 0)
            elif eff > 0.55:
                if valid[RAISE] and max_raise >= min_raise:
                    amt = _clamp(int(pot_ref * self._rng.uniform(0.45, 0.70)), min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
            elif eff > 0.40:
                if self._rng.random() < 0.55 and valid[RAISE] and max_raise >= min_raise:
                    amt = _clamp(int(pot_ref * self._rng.uniform(0.25, 0.45)), min_raise, max_raise)
                    return (RAISE, amt, 0, 0)
                if to_call > 0 and equity > pot_odds and valid[CALL]: return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            elif eff > 0.25:
                if to_call > 0:
                    if equity > pot_odds + 0.15 and valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if equity > pot_odds and valid[CALL]: return (CALL, 0, 0, 0)
                    if valid[FOLD]: return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            else:
                if to_call > 0 and equity > pot_odds + 0.15 and valid[CALL]:
                    return (CALL, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                if valid[FOLD]:  return (FOLD,  0, 0, 0)

        if valid[CHECK]:  return (CHECK, 0, 0, 0)
        if to_call > 0 and equity > pot_odds and valid[CALL]: return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _agro_preflop(self, my_cards, valid, min_raise, max_raise,
                      my_bet, opp_bet, pot_size, blind_pos, p_sims):
        is_prem = _has_any_premium(my_cards)
        if len(my_cards) > 2:
            equity = self._preflop_equity(my_cards, p_sims)
        else:
            equity = self._mc_equity(my_cards, [], set(), num_sims=max(p_sims // 3, 30))

        to_call  = max(0, opp_bet - my_bet)
        if to_call > 30 and not is_prem:
            if valid[FOLD]:  return (FOLD,  0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        noise    = self._rng.uniform(0.90, 1.10)
        pot_ref_pf = max(pot_size, 1)

        if blind_pos == 0:  # SB
            if is_prem:
                if self._rng.random() < 0.15:
                    if valid[CALL]:  return (CALL,  0, 0, 0)
                    if valid[CHECK]: return (CHECK, 0, 0, 0)
                if valid[RAISE] and max_raise >= min_raise:
                    return (RAISE, _clamp(int(pot_ref_pf * 2.5 * noise), min_raise, max_raise), 0, 0)
            elif equity > 0.50:
                if self._rng.random() < 0.65 and valid[RAISE] and max_raise >= min_raise:
                    return (RAISE, _clamp(int(pot_ref_pf * 2.0 * noise), min_raise, max_raise), 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            else:
                if self._rng.random() < 0.15 and valid[RAISE] and max_raise >= min_raise:
                    return (RAISE, _clamp(int(pot_ref_pf * 1.5 * noise), min_raise, max_raise), 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                if valid[FOLD]:  return (FOLD,  0, 0, 0)
        else:  # BB
            if is_prem:
                if to_call > 0:
                    if self._rng.random() < 0.70 and valid[RAISE] and max_raise >= min_raise:
                        return (RAISE, _clamp(int(to_call * 3 * noise), min_raise, max_raise), 0, 0)
                    if valid[CALL]: return (CALL, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            elif equity > 0.45:
                if to_call > 0:
                    if self._rng.random() < 0.15 and valid[RAISE] and max_raise >= min_raise:
                        return (RAISE, _clamp(int(to_call * 2.5 * noise), min_raise, max_raise), 0, 0)
                    if valid[CALL]: return (CALL, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            else:
                if to_call > 0:
                    if equity > pot_odds + 0.05 and valid[CALL]: return (CALL, 0, 0, 0)
                    if valid[FOLD]: return (FOLD, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
        if valid[CHECK]: return (CHECK, 0, 0, 0)
        if valid[CALL]:  return (CALL,  0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _agro_postflop(self, street, my_cards, community, opp_discards, my_discards,
                       valid, min_raise, max_raise, my_bet, opp_bet, pot_size,
                       opp_last, p_sims):
        to_call = max(0, opp_bet - my_bet)
        dead    = set()
        if my_discards:  dead |= set(my_discards)
        if opp_discards: dead |= set(opp_discards)
        _range_sig = 0.0 if self._opp_archetype == "maniac" else self._opp_hand_aggr
        equity  = self._mc_equity_ranged(my_cards, community, dead, _range_sig, num_sims=p_sims) \
                  if len(my_cards) == 2 and len(community) >= 3 else 0.45

        if _board_paired_and_we_weak(my_cards, community):
            equity -= 0.08

        _, f_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
        _, s_outs, _ = _count_straight_outs(my_cards, community, opp_discards, my_discards)
        has_draw = _has_live_draw(my_cards, community, opp_discards, my_discards)
        _hcat = _hand_rank_category(my_cards, community)
        equity = self._street_adjust(equity, street, has_draw, f_outs, s_outs, hand_cat=_hcat)

        cap    = self._cap_score(opp_discards, community, my_discards)
        story  = self._story_score(my_cards, community, my_discards, street)
        backup = self._backup_score(my_cards, community, dead, equity, opp_discards)
        aggro  = 0.35 * cap + 0.25 * story + 0.40 * backup

        bonus, brake = self._hard_triggers(my_cards, community, opp_discards, my_discards)
        if opp_last in ("check", "call") and story >= 0.40 and cap >= 0.30:
            bonus += 0.10

        if backup < 0.20 and (self._betting_history.get("bet_flop") or
                               self._betting_history.get("bet_turn")):
            brake += 0.15
        if street == 3:
            bf = self._betting_history.get("bet_flop", False)
            bt = self._betting_history.get("bet_turn", False)
            if not bf and not bt: brake += 0.20
            elif not bt:          brake += 0.10

        brake = min(brake, 0.45)
        aggro = _clamp(aggro + bonus - brake, 0.0, 1.0)
        aggro = min(1.0, aggro * self._agro_mult)

        if street == 3 and opp_last == "check":
            if (self._betting_history.get("bet_flop") or
                    self._betting_history.get("bet_turn")):
                aggro = min(1.0, aggro + 0.10)

        aggro = _clamp(aggro + self._rng.uniform(-0.05, 0.05), 0.0, 1.0)

        if equity > 0.55:
            aggro = max(aggro, 0.30)

        self._last_equity = equity
        self._last_aggro  = aggro

        return self._aggro_to_action(aggro, equity, street, valid, min_raise,
                                     max_raise, pot_size, to_call, True)

    # =========================================================================
    # VALUE_GRIND — Equity ladder, no bluffing
    # =========================================================================

    def _value_preflop(self, my_cards, valid, min_raise, max_raise,
                       my_bet, opp_bet, pot_size, blind_pos):
        to_call  = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        is_prem      = _has_any_premium(my_cards)
        is_prem_pair = _has_premium_pair(my_cards)
        if to_call > 30 and not is_prem:
            if valid[FOLD]:  return (FOLD,  0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
        noise = self._rng.uniform(0.90, 1.10)
        if is_prem:
            if is_prem_pair and to_call >= 15 and valid[RAISE] and max_raise >= min_raise:
                return (RAISE, max_raise, 0, 0)
            if is_prem_pair and self._rng.random() < 0.20:
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
            open_size = _clamp(int(10 * noise), min_raise, max_raise)
            if valid[RAISE] and max_raise >= min_raise: return (RAISE, open_size, 0, 0)
            if valid[CALL]:  return (CALL,  0, 0, 0)
            return (CHECK, 0, 0, 0)
        else:
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

    def _value_postflop(self, street, my_cards, community, opp_discards, my_discards,
                        valid, min_raise, max_raise, my_bet, opp_bet, pot_size,
                        blind_pos, p_sims):
        to_call  = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        dead     = set()
        if my_discards:  dead |= set(my_discards)
        if opp_discards: dead |= set(opp_discards)
        _range_sig = 0.0 if self._opp_archetype == "maniac" else self._opp_hand_aggr
        equity = self._mc_equity_ranged(my_cards, community, dead, _range_sig, num_sims=p_sims) \
                 if len(my_cards) == 2 and len(community) >= 3 else 0.45

        if _board_paired_and_we_weak(my_cards, community):
            equity -= 0.08

        _, f_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
        _, s_outs, _ = _count_straight_outs(my_cards, community, opp_discards, my_discards)
        has_draw = _has_live_draw(my_cards, community, opp_discards, my_discards)
        _hcat = _hand_rank_category(my_cards, community)
        equity = self._street_adjust(equity, street, has_draw, f_outs, s_outs, hand_cat=_hcat)

        pos_adj = 0.04 if blind_pos == 1 else -0.04

        avg_bf = self._avg_bet_frac()
        size_adj = -0.04 if avg_bf > 0.80 else (0.04 if avg_bf < 0.30 else 0.0)

        if len(opp_discards) >= 3:
            opp_b = _bucket_opp_discard(opp_discards)
            adj   = {"suited_cluster": -0.03, "connected_cluster": -0.02,
                     "high_junk": 0.02, "discarded_pair": 0.03}
            equity = _clamp(equity + adj.get(opp_b, 0.0), 0.0, 0.98)

        pot_ref = max(pot_size, 1)
        noise   = self._rng.uniform(0.92, 1.08)
        self._last_equity = equity

        bet_thresh_hi  = 0.82 - pos_adj
        bet_thresh_mid = 0.68 - pos_adj
        bet_thresh_lo  = 0.52 - pos_adj
        call_limit     = 0.35 - size_adj

        if street == 3:
            if equity > bet_thresh_hi:
                if valid[RAISE] and max_raise >= min_raise: return (RAISE, max_raise, 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                return (CHECK, 0, 0, 0)
            elif equity > bet_thresh_mid:
                amt = _clamp(int(pot_ref * self._rng.uniform(0.70, 0.90) * noise), min_raise, max_raise)
                if valid[RAISE] and max_raise >= min_raise: return (RAISE, amt, 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                return (CHECK, 0, 0, 0)
            elif equity > pot_odds and to_call <= pot_ref * call_limit:
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            else:
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
        else:
            if equity > bet_thresh_hi:
                amt = _clamp(int(pot_ref * self._rng.uniform(0.65, 0.90) * noise), min_raise, max_raise)
                if valid[RAISE] and max_raise >= min_raise: return (RAISE, amt, 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                return (CHECK, 0, 0, 0)
            elif equity > bet_thresh_mid:
                amt = _clamp(int(pot_ref * self._rng.uniform(0.55, 0.75) * noise), min_raise, max_raise)
                if valid[RAISE] and max_raise >= min_raise: return (RAISE, amt, 0, 0)
                if valid[CALL]:  return (CALL,  0, 0, 0)
                return (CHECK, 0, 0, 0)
            elif equity > bet_thresh_lo:
                amt = _clamp(int(pot_ref * self._rng.uniform(0.30, 0.48) * noise), min_raise, max_raise)
                if valid[RAISE] and max_raise >= min_raise: return (RAISE, amt, 0, 0)
                if to_call > 0 and equity >= pot_odds and valid[CALL]: return (CALL, 0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            elif equity >= pot_odds and to_call <= pot_ref * (0.30 - size_adj):
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            else:
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)

    # =========================================================================
    # TRAP_MODE — Check-call heavy, river value-bet
    # =========================================================================

    def _trap_preflop(self, my_cards, valid, min_raise, max_raise,
                      my_bet, opp_bet, pot_size, blind_pos):
        to_call  = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        is_prem = _has_any_premium(my_cards)
        if to_call > 30 and not is_prem:
            if valid[FOLD]:  return (FOLD,  0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
        if is_prem:
            if blind_pos == 0 and self._rng.random() < 0.30 and valid[RAISE] and max_raise >= min_raise:
                amt = _clamp(int(pot_size * 2.5), min_raise, max_raise)
                return (RAISE, amt, 0, 0)
            if valid[CALL]:  return (CALL,  0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
        if valid[CHECK]: return (CHECK, 0, 0, 0)
        if to_call > 0 and pot_odds < 0.35 and valid[CALL]: return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _trap_postflop(self, street, my_cards, community, opp_discards, my_discards,
                       valid, min_raise, max_raise, my_bet, opp_bet, pot_size, p_sims):
        to_call  = max(0, opp_bet - my_bet)
        pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0
        dead     = set()
        if my_discards:  dead |= set(my_discards)
        if opp_discards: dead |= set(opp_discards)
        _range_sig = 0.0 if self._opp_archetype == "maniac" else self._opp_hand_aggr
        equity  = self._mc_equity_ranged(my_cards, community, dead, _range_sig, num_sims=p_sims) \
                  if len(my_cards) == 2 and len(community) >= 3 else 0.45

        _, f_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
        _, s_outs, _ = _count_straight_outs(my_cards, community, opp_discards, my_discards)
        has_draw = _has_live_draw(my_cards, community, opp_discards, my_discards)
        _hcat = _hand_rank_category(my_cards, community)
        equity = self._street_adjust(equity, street, has_draw, f_outs, s_outs, hand_cat=_hcat)

        self._last_equity = equity
        pot_ref = max(pot_size, 1)

        avg_bf = self._avg_bet_frac()
        call_adj = -0.05 if avg_bf < 0.30 else (0.05 if avg_bf > 0.80 else 0.0)

        if to_call > pot_ref * 0.50 and equity < 0.40:
            if valid[FOLD]:  return (FOLD,  0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)

        if street == 3:
            if equity > 0.65 and valid[RAISE] and max_raise >= min_raise:
                amt = _clamp(int(pot_ref * self._rng.uniform(0.60, 0.80)), min_raise, max_raise)
                return (RAISE, amt, 0, 0)
            if equity > pot_odds and to_call <= pot_ref * (0.45 - call_adj):
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)
        else:
            if equity > 0.58 and valid[RAISE] and max_raise >= min_raise:
                amt = _clamp(int(pot_ref * 0.75), min_raise, max_raise)
                return (RAISE, amt, 0, 0)
            if equity > pot_odds and to_call <= pot_ref * 0.40:
                if valid[CALL]:  return (CALL,  0, 0, 0)
                if valid[CHECK]: return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)
            if valid[CHECK]: return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

    # =========================================================================
    # Dynamic bet sizing
    # =========================================================================

    @staticmethod
    def _cat_to_strength(hand_cat, has_draw):
        if hand_cat in ("trips_plus", "two_pair"):
            return "monster"
        if hand_cat == "one_pair" and has_draw:
            return "strong"
        if hand_cat == "one_pair":
            return "medium"
        if has_draw:
            return "draw"
        return "weak"

    def _dynamic_sizing(self, base_amount, strength, street, is_semi_bluff):
        """Apply opponent-archetype-aware sizing multiplier to a raise amount."""
        arch = self._opp_archetype
        mult = 1.0

        if arch == "overfolder":
            if is_semi_bluff or strength in ("draw", "weak"):
                mult = 1.25
            elif strength == "monster":
                mult = 0.85
            elif strength in ("strong", "medium"):
                mult = 0.75
        elif arch == "station":
            if strength == "monster":
                mult = 1.30 if street == 3 else 1.20
            elif strength in ("strong", "medium"):
                mult = 1.15
            elif is_semi_bluff or strength == "draw":
                mult = 0.80
        elif arch == "maniac":
            if strength == "monster":
                mult = 0.70
            elif strength in ("strong", "medium"):
                mult = 0.80

        return max(1, int(base_amount * mult))

    # =========================================================================
    # Semi-bluff engine
    # =========================================================================

    def _semi_bluff_check(self, my_cards, community, opp_discards, my_discards,
                          pot_size, to_call, street, valid, min_raise, max_raise):
        """Compute semi-bluff EV and return (should_fire, action_tuple)."""
        if street not in (1, 2):
            return False, None
        if not (valid[RAISE] and max_raise >= min_raise):
            return False, None
        if not _has_live_draw(my_cards, community, opp_discards, my_discards):
            return False, None

        _, flush_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
        _, straight_outs, _ = _count_straight_outs(my_cards, community, opp_discards, my_discards)
        outs = max(flush_outs, straight_outs)
        if outs < 2:
            return False, None

        if to_call > pot_size * 0.40:
            return False, None

        b_ranks = [_rank(c) for c in community]
        if len(b_ranks) != len(set(b_ranks)):
            return False, None

        known = set(my_cards[:2]) | set(community) | set(opp_discards) | set(my_discards)
        remaining = max(1, DECK_SIZE - len(known))
        cards_to_come = 2 if street == 1 else 1

        if cards_to_come == 1:
            draw_eq = min(1.0, outs / remaining)
        else:
            miss_one = max(0, remaining - outs) / remaining
            miss_two = max(0, remaining - outs - 1) / max(1, remaining - 1)
            draw_eq = 1.0 - miss_one * miss_two

        fold_prob = self._safe_rate("fold_to_bet")

        pot = max(pot_size, 1)
        if fold_prob > 0.48:
            sizing_frac = self._rng.uniform(0.65, 0.80)
        elif fold_prob < 0.25:
            sizing_frac = self._rng.uniform(0.45, 0.60)
        else:
            sizing_frac = self._rng.uniform(0.55, 0.70)

        bet_size = pot * sizing_frac

        ev = (fold_prob * pot
              + (1 - fold_prob) * (draw_eq * (pot + 2 * bet_size) - bet_size))

        if ev > 0:
            amt = _clamp(int(bet_size), min_raise, max_raise)
            return True, (RAISE, amt, 0, 0)
        return False, None

    # =========================================================================
    # Main act / observe
    # =========================================================================

    def act(self, observation, reward, terminated, truncated, info):
        my_cards     = [c for c in observation["my_cards"]           if c != -1]
        community    = [c for c in observation["community_cards"]     if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards  = [c for c in observation["my_discarded_cards"]  if c != -1]
        valid        = observation["valid_actions"]
        street       = observation["street"]
        min_raise    = observation["min_raise"]
        max_raise    = observation["max_raise"]
        my_bet       = observation["my_bet"]
        opp_bet      = observation["opp_bet"]
        pot_size     = observation.get("pot_size",        my_bet + opp_bet)
        blind_pos    = observation.get("blind_position",  0)
        time_left    = observation.get("time_left",       400.0)

        opp_last = _normalize_action(observation.get("opp_last_action", ""))

        # ── Save prev-turn context before new-hand detection ──────────────────
        prev_was_bet = self._last_was_bet
        prev_street  = self._last_street

        # ── New hand detection ────────────────────────────────────────────────
        is_new_hand = (street == 0 and my_bet <= 2 and opp_bet <= 2)
        if is_new_hand:
            self.hand_number += 1
            self._hand_mode, self._agro_mult = self._select_mode()
            self._hand_override    = None
            self._betting_history  = {"bet_flop": False, "bet_turn": False}
            self._last_was_bet     = False
            self._last_street      = 0
            self._opp_folded       = False
            self._we_folded        = False
            self._opp_hand_aggr    = 0.0

            mode_change = (
                f" [WAS {self._prev_mode}]" if self._hand_mode != self._prev_mode else ""
            )
            self.logger.info(
                f"Hand {self.hand_number}: mode={self._hand_mode}{mode_change}"
                f"[{self._mode_reason}], mult={self._agro_mult:.1f}, "
                f"pnl={self._running_pnl:+d}, "
                f"aggro={self._fmt_stat('opp_aggression')}, "
                f"fb={self._fmt_stat('fold_to_bet')}, "
                f"cd={self._fmt_stat('call_down')}"
            )
            self._prev_mode = self._hand_mode

            if self.hand_number % 50 == 0:
                parts = []
                for k, (num, den) in self._stats.items():
                    if k == "opp_avg_bet_frac":
                        val = f"{num/den:.2f}" if den > 0 else "prior"
                        parts.append(f"bet_frac={val}({den})")
                    else:
                        parts.append(f"{k}={num}/{den}")
                self.logger.info(f"[STATS H{self.hand_number}] " + " ".join(parts))

        # ── Process opponent action (stat updates happen here) ────────────────
        if opp_last and not is_new_hand:
            self._process_opponent_action(observation, opp_last, prev_was_bet, prev_street)
        elif is_new_hand and opp_last:
            self._process_opponent_action(observation, opp_last, prev_was_bet, prev_street)

        self._prev_opp_bet = opp_bet
        self._prev_my_bet  = my_bet

        d_sims, p_sims = self._sim_budget(time_left)

        # ── Resolve effective mode ────────────────────────────────────────────
        mode = self._hand_override or self._hand_mode

        # ── Bleed-out lock: if folding every remaining hand still wins, do it
        if not valid[DISCARD]:
            rounds_left = max(0, TOTAL_HANDS - self.hand_number)
            sb_left = (rounds_left + 1) // 2
            bb_left = rounds_left - sb_left
            max_bleed = sb_left * 1 + bb_left * 2
            if self._running_pnl > max_bleed:
                if valid[FOLD]:
                    return (FOLD, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)

        # ── Discard (mode-aware KeepScore) ────────────────────────────────────
        if valid[DISCARD]:
            i, j = self._choose_keep(my_cards, community, opp_discards, d_sims, mode)
            self._last_was_bet = False
            self._last_street  = street
            return (DISCARD, 0, i, j)

        # ── Preflop ──────────────────────────────────────────────────────────
        if street == 0:
            if mode == MODE_VALUE:
                result = self._value_preflop(my_cards, valid, min_raise, max_raise,
                                             my_bet, opp_bet, pot_size, blind_pos)
            elif mode == MODE_TRAP:
                result = self._trap_preflop(my_cards, valid, min_raise, max_raise,
                                            my_bet, opp_bet, pot_size, blind_pos)
            elif mode == MODE_GTO:
                result = self._gto_action(street, my_cards, community, opp_discards,
                                          my_discards, valid, min_raise, max_raise,
                                          my_bet, opp_bet, pot_size, blind_pos, p_sims)
            else:  # AGRO
                result = self._agro_preflop(my_cards, valid, min_raise, max_raise,
                                            my_bet, opp_bet, pot_size, blind_pos, p_sims)
            if result[0] == FOLD:
                self._we_folded = True
            self._last_was_bet = (result[0] == RAISE)
            self._last_street  = street
            return result

        # ── Post-flop ─────────────────────────────────────────────────────────
        if mode == MODE_VALUE:
            result = self._value_postflop(street, my_cards, community, opp_discards,
                                          my_discards, valid, min_raise, max_raise,
                                          my_bet, opp_bet, pot_size, blind_pos, p_sims)
        elif mode == MODE_TRAP:
            result = self._trap_postflop(street, my_cards, community, opp_discards,
                                         my_discards, valid, min_raise, max_raise,
                                         my_bet, opp_bet, pot_size, p_sims)
        elif mode == MODE_GTO:
            result = self._gto_action(street, my_cards, community, opp_discards,
                                      my_discards, valid, min_raise, max_raise,
                                      my_bet, opp_bet, pot_size, blind_pos, p_sims)
        else:  # AGRO
            result = self._agro_postflop(street, my_cards, community, opp_discards,
                                         my_discards, valid, min_raise, max_raise,
                                         my_bet, opp_bet, pot_size, opp_last, p_sims)

        # ── Semi-bluff override ──────────────────────────────────────────
        _is_semi_bluff = False
        if result[0] in (CHECK, FOLD) and street in (1, 2):
            to_call_sb = max(0, opp_bet - my_bet)
            sb_fire, sb_action = self._semi_bluff_check(
                my_cards, community, opp_discards, my_discards,
                pot_size, to_call_sb, street, valid, min_raise, max_raise)
            if sb_fire:
                result = sb_action
                _is_semi_bluff = True

        # ── Dynamic bet sizing ───────────────────────────────────────────
        if result[0] == RAISE and len(my_cards) == 2 and len(community) >= 3:
            _has_d = _has_live_draw(my_cards, community, opp_discards, my_discards)
            _hcat = _hand_rank_category(my_cards, community)
            _str = self._cat_to_strength(_hcat, _has_d)
            adj_amt = self._dynamic_sizing(result[1], _str, street, _is_semi_bluff)
            result = (RAISE, _clamp(adj_amt, min_raise, max_raise), 0, 0)

        # ── One-pair / nothing cap (draw-aware) ─────────────────────────
        # Draws are now checked: hands with live flush or straight draws
        # are allowed to semi-bluff raise and call larger bets.
        if len(my_cards) == 2 and len(community) >= 3:
            hand_cat = _hand_rank_category(my_cards, community)
            to_call_now = max(0, opp_bet - my_bet)
            pot_ref_now = max(pot_size, 1)
            has_draw = _has_live_draw(my_cards, community, opp_discards, my_discards)

            if hand_cat == "one_pair":
                if has_draw:
                    # Pair + draw: allow raising, fold only to >70% pot
                    if result[0] == CALL and to_call_now > pot_ref_now * 0.70:
                        if valid[FOLD]:
                            result = (FOLD, 0, 0, 0)
                else:
                    # Naked pair: no raising, fold to >25% pot
                    if result[0] == RAISE:
                        if to_call_now > 0 and valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        elif valid[CHECK]:
                            result = (CHECK, 0, 0, 0)
                    if result[0] == CALL and to_call_now > pot_ref_now * 0.25:
                        if valid[FOLD]:
                            result = (FOLD, 0, 0, 0)

            elif hand_cat == "nothing":
                if has_draw:
                    # Draw without a pair: allow calling, fold only to >50% pot
                    if result[0] == CALL and to_call_now > pot_ref_now * 0.50:
                        if valid[FOLD]:
                            result = (FOLD, 0, 0, 0)
                else:
                    # True nothing: fold to any bet, check if free
                    if to_call_now > 0:
                        if valid[FOLD]:
                            result = (FOLD, 0, 0, 0)
                    elif valid[CHECK]:
                        result = (CHECK, 0, 0, 0)

        if result[0] == RAISE:
            if street == 1:   self._betting_history["bet_flop"] = True
            elif street == 2: self._betting_history["bet_turn"] = True

        if result[0] == FOLD:
            self._we_folded = True

        if street >= 1 and result[0] in (RAISE, FOLD):
            act_name = "RAISE" if result[0] == RAISE else "FOLD"
            agro_tag = f" agg={self._last_aggro:.2f}" if mode == MODE_AGRO else ""
            self.logger.info(
                f"  [{act_name} H{self.hand_number} S{street} {mode}] "
                f"eq={self._last_equity:.2f}{agro_tag}"
                + (f" amt={result[1]}" if result[0] == RAISE else "")
            )

        self._last_was_bet = (result[0] == RAISE)
        self._last_street  = street
        return result

    def observe(self, observation, reward, terminated, truncated, info):
        raw = observation.get("opp_last_action", "")
        opp_action = _normalize_action(raw)

        if opp_action == "fold":
            self._opp_folded = True

        if terminated:
            self._running_pnl += int(reward)

            # Terminal fold tracking: when opponent folds to our bet, act() is
            # never called again, so we must update fold stats here.
            if self._opp_folded and self._last_was_bet:
                self._stats["fold_to_bet"][0] += 1
                self._stats["fold_to_bet"][1] += 1
                self._stats["fold_to_raise"][0] += 1
                self._stats["fold_to_raise"][1] += 1