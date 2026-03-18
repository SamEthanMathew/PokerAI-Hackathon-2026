"""
GenesisV2 — Targeted fixes over GenesisV1.

Root causes confirmed from scrimmage log analysis (2026-03-18, -125 chips vs Attention Is All You Need):
  Check-raise overcount → false maniac detection at hand 13 (all -100 losses are arch=maniac)
  Maniac + premium pair → all-in every hand; medium pairs have only 55-65% equity → loss loop
  _cat_to_strength ignores equity (two_pair with 34% equity → "monster")
  River calls too loose (GOOD hands call any pot-odds bet)

Fixes applied over GenesisV1:
  1. Fix check-raise counting: only count when WE explicitly checked last
  2. Maniac guard: only AA max-raises vs maniac; medium pairs raise 2-3x max
  3. Equity-aware strength classification: equity caps strength label
  4. River-specific logic: separate _act_river() with tighter call thresholds
  5. Preflop equity: best-keep (not avg-top-3) + per-hand cache
  6. Position awareness: track SB/BB, BB defends wider
  7. Maniac counter-strategy: check trap monsters, don't bet weak vs maniac
  8. Draw equity bonus by street: bigger on flop (2 cards to come)
"""

import random
from itertools import combinations

from treys import Card, Evaluator

from agents.agent import Agent
from gym_env import PokerEnv

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_RANKS = 9
DECK_SIZE = 27
RANK_A = 8
RANK_9 = 7
RANK_8 = 6

MONSTER_THRESHOLD = 0.82
STRONG_THRESHOLD  = 0.65
GOOD_THRESHOLD    = 0.48

PREFLOP_COMMIT_THRESHOLD = 20   # raised from 15; less trigger-happy all-in
TOTAL_HANDS = 1000
EARLY_PHASE_HANDS = 50
EARLY_PREFLOP_MIN_EQUITY = 0.48
NORMAL_PREFLOP_MIN_EQUITY = 0.45
EARLY_OPEN_MULTIPLIER = 1.20
SLOW_PLAY_CHANCE = 0.20
STANDARD_OPEN = 8

# Action enum values
FOLD    = PokerEnv.ActionType.FOLD.value
RAISE   = PokerEnv.ActionType.RAISE.value
CHECK   = PokerEnv.ActionType.CHECK.value
CALL    = PokerEnv.ActionType.CALL.value
DISCARD = PokerEnv.ActionType.DISCARD.value

# ── Card caches ───────────────────────────────────────────────────────────────

_INT_TO_TREYS = [PokerEnv.int_to_card(i) for i in range(DECK_SIZE)]
_INT_TO_TREYS_ALT = []
for _tc in _INT_TO_TREYS:
    _s = Card.int_to_str(_tc)
    _INT_TO_TREYS_ALT.append(Card.new(_s.replace("A", "T")))

_base_eval = Evaluator()

_RANK = [i % NUM_RANKS for i in range(DECK_SIZE)]
_SUIT = [i // NUM_RANKS for i in range(DECK_SIZE)]

# ── Premium hand definitions ──────────────────────────────────────────────────

PREMIUM_PAIRS = frozenset([
    frozenset([RANK_A, RANK_A]),
    frozenset([RANK_9, RANK_9]),
    frozenset([RANK_8, RANK_8]),
])
PREMIUM_ANY_SUIT = frozenset([
    frozenset([RANK_A, RANK_9]),
    frozenset([RANK_A, RANK_8]),
])
PREMIUM_SUITED_ONLY = frozenset([
    frozenset([RANK_9, RANK_8]),
    frozenset([RANK_8, 5]),
    frozenset([5, 4]),
    frozenset([5, RANK_9]),
])

# ── Helpers ───────────────────────────────────────────────────────────────────


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _fast_eval(hand, board, alt_hand, alt_board):
    r = _base_eval.evaluate(hand, board)
    a = _base_eval.evaluate(alt_hand, alt_board)
    return a if a < r else r


def _max_connectivity(ranks):
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


def _normalize_action(raw):
    if not raw:
        return ""
    s = str(raw).strip().lower()
    return "" if s == "none" else s


# ── Premium hand helpers ──────────────────────────────────────────────────────


def _is_premium(c1, c2):
    r1, r2 = _RANK[c1], _RANK[c2]
    s1, s2 = _SUIT[c1], _SUIT[c2]
    ranks = frozenset([r1, r2])
    if r1 == r2 and frozenset([r1, r1]) in PREMIUM_PAIRS:
        return True
    if ranks in PREMIUM_ANY_SUIT:
        return True
    if s1 == s2 and ranks in PREMIUM_SUITED_ONLY:
        return True
    return False


def _has_any_premium(cards):
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _is_premium(cards[i], cards[j]):
                return True
    return False


def _is_premium_pair(c1, c2):
    r1, r2 = _RANK[c1], _RANK[c2]
    return r1 == r2 and frozenset([r1, r1]) in PREMIUM_PAIRS


def _has_premium_pair(cards):
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _is_premium_pair(cards[i], cards[j]):
                return True
    return False


def _has_pair_of_rank(cards, rank):
    """True if cards contains at least two cards of the given rank."""
    return sum(1 for c in cards if _RANK[c] == rank) >= 2


# ── Board texture penalties ───────────────────────────────────────────────────


def _board_monotone_penalty(my_cards, community):
    if len(my_cards) < 2 or len(community) < 3:
        return 0.0
    sc = [0, 0, 0]
    for c in community:
        sc[_SUIT[c]] += 1
    dom_cnt = max(sc)
    if dom_cnt < 3:
        return 0.0
    dom_suit = sc.index(dom_cnt)
    my_in_suit = (_SUIT[my_cards[0]] == dom_suit) + (_SUIT[my_cards[1]] == dom_suit)
    return -0.18 if my_in_suit == 0 else (-0.06 if my_in_suit == 1 else 0.0)


def _board_connected_penalty(my_cards, community):
    if len(my_cards) < 2 or len(community) < 3:
        return 0.0
    b_ranks = [_RANK[c] for c in community]
    if _max_connectivity(b_ranks) < 3:
        return 0.0
    hcat = _hand_rank_category(my_cards, community)
    if hcat == "trips_plus":
        return 0.0
    if hcat == "two_pair":
        return -0.10
    if hcat == "one_pair":
        return -0.05
    return 0.0


def _opp_flush_inference(community, opp_discards):
    if not opp_discards or len(opp_discards) < 3 or len(community) < 3:
        return 0.0
    b_sc = [0, 0, 0]
    for c in community:
        b_sc[_SUIT[c]] += 1
    opp_sc = [0, 0, 0]
    for c in opp_discards:
        opp_sc[_SUIT[c]] += 1
    for s in range(3):
        if b_sc[s] >= 2 and opp_sc[s] == 0:
            return -0.08
    return 0.0


# ── Hand classification & draw detection ──────────────────────────────────────


def _hand_rank_category(my_cards, community):
    if len(my_cards) < 2 or len(community) < 3:
        return "nothing"
    rc = [0] * NUM_RANKS
    sc = [0, 0, 0]
    for c in my_cards[:2]:
        rc[_RANK[c]] += 1
        sc[_SUIT[c]] += 1
    for c in community:
        rc[_RANK[c]] += 1
        sc[_SUIT[c]] += 1
    if max(sc) >= 5:
        return "trips_plus"
    unique_r = sorted(r for r in range(NUM_RANKS) if rc[r] > 0)
    ext = ([-1] + unique_r) if RANK_A in unique_r else unique_r
    best_run = cur_run = 1
    for i in range(1, len(ext)):
        if ext[i] - ext[i - 1] == 1:
            cur_run += 1
            best_run = max(best_run, cur_run)
        else:
            cur_run = 1
    if best_run >= 5 or max(rc) >= 3:
        return "trips_plus"
    pairs = sum(1 for r in range(NUM_RANKS) if rc[r] >= 2)
    if pairs >= 2:
        return "two_pair"
    if pairs == 1:
        return "one_pair"
    return "nothing"


def _count_flush_outs(my_cards, community, opp_discards, my_discards):
    if len(my_cards) < 2 or len(community) < 3:
        return 0, 0, DECK_SIZE
    known = set(my_cards[:2]) | set(community) | set(opp_discards) | set(my_discards)
    best_count = best_live = 0
    for s in range(3):
        in_hand = sum(1 for c in my_cards[:2] if _SUIT[c] == s)
        on_board = sum(1 for c in community if _SUIT[c] == s)
        count = in_hand + on_board
        if count >= best_count:
            dead_of_suit = sum(1 for c in known if _SUIT[c] == s)
            live = 9 - dead_of_suit
            if count > best_count or live > best_live:
                best_count, best_live = count, live
    return best_count, best_live, DECK_SIZE - len(known)


def _count_straight_outs(my_cards, community, opp_discards, my_discards):
    if len(my_cards) < 2 or len(community) < 3:
        return 0, 0, DECK_SIZE
    known = set(my_cards[:2]) | set(community) | set(opp_discards) | set(my_discards)
    have_ranks = set(_RANK[c] for c in my_cards[:2]) | set(_RANK[c] for c in community)
    best_in_run = best_outs = 0
    windows = [
        [RANK_A, 0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, RANK_A],
    ]
    for window in windows:
        have = sum(1 for r in window if r in have_ranks)
        if have >= 3:
            need = [r for r in window if r not in have_ranks]
            live = sum(1 for nr in need for s in range(3)
                      if s * NUM_RANKS + nr not in known)
            if have > best_in_run or (have == best_in_run and live > best_outs):
                best_in_run, best_outs = have, live
    return best_in_run, best_outs, DECK_SIZE - len(known)


# ── Adaptive Opponent Discard Model ───────────────────────────────────────────


class _OppDiscardModel:
    """Adaptive linear model that learns opponent's discard feature weights."""

    _DEFAULTS = [10.0, 8.0, 6.0, 3.0, 2.0, 4.0, 3.0]
    _ALPHA = 0.15
    _WEIGHT_FLOOR = 0.5

    def __init__(self):
        self.weights = list(self._DEFAULTS)

    @staticmethod
    def _features(keep, flop, fr=None, fs=None, bsc=None):
        r0, r1 = _RANK[keep[0]], _RANK[keep[1]]
        s0, s1 = _SUIT[keep[0]], _SUIT[keep[1]]
        flop_ranks = fr if fr is not None else [_RANK[c] for c in flop]
        flop_suits = fs if fs is not None else [_SUIT[c] for c in flop]

        sc = [0, 0, 0]
        sc[s0] += 1; sc[s1] += 1
        for fsuit in flop_suits:
            sc[fsuit] += 1

        if bsc is None:
            bsc_arr = [0, 0, 0]
            for fsuit in flop_suits:
                bsc_arr[fsuit] += 1
        else:
            bsc_arr = bsc

        bsm = sum(1 for c in keep if bsc_arr[_SUIT[c]] >= 2) / 2.0

        return [
            1.0 if (r0 in flop_ranks or r1 in flop_ranks) else 0.0,  # pair_with_board
            1.0 if r0 == r1 else 0.0,                                  # pocket_pair
            1.0 if max(sc) >= 4 else 0.0,                              # flush_draw
            1.0 if s0 == s1 else 0.0,                                  # suited
            1.0 if abs(r0 - r1) <= 2 else 0.0,                        # connected
            max(r0, r1) / 8.0,                                         # high_card
            bsm,                                                        # board_suit_match
        ]

    def score(self, keep, flop):
        return sum(w * f for w, f in zip(self.weights, self._features(keep, flop)))

    def score_pre(self, keep, fr, fs, bsc):
        return sum(w * f for w, f in zip(self.weights, self._features(keep, None, fr, fs, bsc)))

    def update(self, opp_kept, flop, opp_discards):
        if len(opp_kept) < 2 or len(opp_discards) < 3 or len(flop) < 3:
            return
        original5 = list(opp_kept) + list(opp_discards)
        if len(original5) < 5:
            return
        scored = [(cand, self.score(cand, flop))
                  for cand in ([original5[i], original5[j]]
                               for i, j in combinations(range(len(original5)), 2))]
        scored.sort(key=lambda x: x[1], reverse=True)
        predicted = scored[0][0]
        if set(predicted) == set(opp_kept):
            return
        cf = self._features(list(opp_kept), flop)
        pf = self._features(predicted, flop)
        for i in range(len(self.weights)):
            self.weights[i] += self._ALPHA * (cf[i] - pf[i])
            self.weights[i] = max(self._WEIGHT_FLOOR, self.weights[i])


# ── MC equity helpers ─────────────────────────────────────────────────────────


def _compute_equity(my_cards, community, dead, n_sims=300):
    """Basic MC equity vs random opponent range."""
    remaining = [i for i in range(DECK_SIZE) if i not in dead]
    board_needed = 5 - len(community)
    sample_size = 2 + board_needed
    if sample_size > len(remaining):
        return 0.5

    my_h  = [_INT_TO_TREYS[c] for c in my_cards]
    my_ha = [_INT_TO_TREYS_ALT[c] for c in my_cards]
    comm  = [_INT_TO_TREYS[c] for c in community]
    comm_a = [_INT_TO_TREYS_ALT[c] for c in community]

    wins = 0.0
    for _ in range(n_sims):
        samp = random.sample(remaining, sample_size)
        oh  = [_INT_TO_TREYS[c] for c in samp[:2]]
        oha = [_INT_TO_TREYS_ALT[c] for c in samp[:2]]
        board   = comm   + [_INT_TO_TREYS[c]     for c in samp[2:]]
        board_a = comm_a + [_INT_TO_TREYS_ALT[c] for c in samp[2:]]
        mr = _fast_eval(my_h, board, my_ha, board_a)
        orr = _fast_eval(oh, board, oha, board_a)
        wins += (1.0 if mr < orr else (0.5 if mr == orr else 0.0))
    return wins / n_sims


def _compute_equity_ranged(my2, community, dead, opp_model, opp_discards,
                           opp_signal, n_sims=300):
    """Range-weighted MC equity conditioned on opponent's discard preference."""
    remaining = [i for i in range(DECK_SIZE) if i not in dead]
    board_needed = 5 - len(community)
    sample_size = 2 + board_needed
    if sample_size > len(remaining):
        return 0.5

    have_disc = len(opp_discards) >= 3
    flop = community[:3] if len(community) >= 3 else []
    flop_cache = None
    opp_weight_map = {}

    if have_disc and flop:
        fr  = [_RANK[c] for c in flop]
        fs  = [_SUIT[c] for c in flop]
        bsc = [0, 0, 0]
        for s in fs:
            bsc[s] += 1
        flop_cache = (fr, fs, bsc)
        for pair in combinations(remaining, 2):
            original5 = list(pair) + list(opp_discards)
            if len(original5) < 5:
                w = 1.0
            else:
                scored = [(opp_model.score_pre([original5[i], original5[j]], fr, fs, bsc),
                           frozenset([original5[i], original5[j]]))
                          for i, j in combinations(range(len(original5)), 2)]
                scored.sort(key=lambda x: x[0], reverse=True)
                opp_set = frozenset(pair)
                w = 1.0
                for rank_idx, (_, kset) in enumerate(scored):
                    if kset == opp_set:
                        w = [1.0, 0.3, 0.1, 0.02][min(rank_idx, 3)]
                        break
            if w >= 0.01:
                opp_weight_map[pair] = w

    reject_nothing  = opp_signal >= 2.0
    reject_one_pair = opp_signal >= 3.5
    max_retries = 3 if reject_nothing else 0

    my_h   = [_INT_TO_TREYS[c]     for c in my2]
    my_ha  = [_INT_TO_TREYS_ALT[c] for c in my2]
    comm   = [_INT_TO_TREYS[c]     for c in community]
    comm_a = [_INT_TO_TREYS_ALT[c] for c in community]

    wins = 0.0
    total_w = 0.0

    for _ in range(n_sims):
        samp = random.sample(remaining, sample_size)
        opp = samp[:2]
        runout = samp[2:]

        if max_retries > 0 and len(community) >= 3:
            for _ in range(max_retries):
                cat = _hand_rank_category(list(opp), community)
                if cat == "nothing" or (reject_one_pair and cat == "one_pair"):
                    samp = random.sample(remaining, sample_size)
                    opp = samp[:2]; runout = samp[2:]
                else:
                    break

        if have_disc and flop_cache:
            key = (opp[0], opp[1]) if opp[0] < opp[1] else (opp[1], opp[0])
            w = opp_weight_map.get(key, 0.0)
            if w < 0.01:
                continue
        else:
            w = 1.0

        oh  = [_INT_TO_TREYS[c]     for c in opp]
        oha = [_INT_TO_TREYS_ALT[c] for c in opp]
        board   = comm   + [_INT_TO_TREYS[c]     for c in runout]
        board_a = comm_a + [_INT_TO_TREYS_ALT[c] for c in runout]

        mr  = _fast_eval(my_h, board, my_ha, board_a)
        orr = _fast_eval(oh, board, oha, board_a)
        wins   += w * (1.0 if mr < orr else (0.5 if mr == orr else 0.0))
        total_w += w

    return wins / total_w if total_w > 0 else 0.5


def _exact_discard_equity(my_keep, community, dead_cards):
    """Brute-force exact equity vs all opponent hands and all runouts."""
    remaining = [c for c in range(DECK_SIZE) if c not in dead_cards]
    board_needed = 5 - len(community)

    my_h   = [_INT_TO_TREYS[c]     for c in my_keep]
    my_ha  = [_INT_TO_TREYS_ALT[c] for c in my_keep]
    comm   = [_INT_TO_TREYS[c]     for c in community]
    comm_a = [_INT_TO_TREYS_ALT[c] for c in community]

    opp_treys = {}
    for pair in combinations(remaining, 2):
        opp_treys[pair] = (
            [_INT_TO_TREYS[pair[0]],     _INT_TO_TREYS[pair[1]]],
            [_INT_TO_TREYS_ALT[pair[0]], _INT_TO_TREYS_ALT[pair[1]]],
        )

    wins = 0.0
    total = 0
    for runout in combinations(remaining, board_needed):
        board   = comm   + [_INT_TO_TREYS[c]     for c in runout]
        board_a = comm_a + [_INT_TO_TREYS_ALT[c] for c in runout]
        mr = _fast_eval(my_h, board, my_ha, board_a)
        runout_set = set(runout)
        opp_pool = [c for c in remaining if c not in runout_set]
        for opp in combinations(opp_pool, 2):
            oh, oha = opp_treys[opp]
            orr = _fast_eval(oh, board, oha, board_a)
            wins += (1.0 if mr < orr else (0.5 if mr == orr else 0.0))
            total += 1

    return wins / total if total > 0 else 0.5


def _exact_discard_equity_weighted(my_keep, community, dead_cards, opp_model, opp_discards):
    """Brute-force exact equity weighted by opponent's discard model."""
    remaining = [c for c in range(DECK_SIZE) if c not in dead_cards]
    board_needed = 5 - len(community)

    my_h   = [_INT_TO_TREYS[c]     for c in my_keep]
    my_ha  = [_INT_TO_TREYS_ALT[c] for c in my_keep]
    comm   = [_INT_TO_TREYS[c]     for c in community]
    comm_a = [_INT_TO_TREYS_ALT[c] for c in community]

    flop = community[:3]
    fr  = [_RANK[c] for c in flop]
    fs  = [_SUIT[c] for c in flop]
    bsc = [0, 0, 0]
    for s in fs:
        bsc[s] += 1

    opp_data = {}
    for opp in combinations(remaining, 2):
        original5 = list(opp) + list(opp_discards)
        if len(original5) < 5:
            w = 1.0
        else:
            scored = [(opp_model.score_pre([original5[i], original5[j]], fr, fs, bsc),
                       frozenset([original5[i], original5[j]]))
                      for i, j in combinations(range(len(original5)), 2)]
            scored.sort(key=lambda x: x[0], reverse=True)
            opp_set = frozenset(opp)
            w = 1.0
            for rank_idx, (_, kset) in enumerate(scored):
                if kset == opp_set:
                    w = [1.0, 0.3, 0.1, 0.02][min(rank_idx, 3)]
                    break
        if w >= 0.01:
            opp_data[opp] = (
                w,
                [_INT_TO_TREYS[opp[0]],     _INT_TO_TREYS[opp[1]]],
                [_INT_TO_TREYS_ALT[opp[0]], _INT_TO_TREYS_ALT[opp[1]]],
            )

    wins = 0.0
    total_w = 0.0
    for runout in combinations(remaining, board_needed):
        board   = comm   + [_INT_TO_TREYS[c]     for c in runout]
        board_a = comm_a + [_INT_TO_TREYS_ALT[c] for c in runout]
        mr = _fast_eval(my_h, board, my_ha, board_a)
        runout_set = set(runout)
        opp_pool = [c for c in remaining if c not in runout_set]
        for opp in combinations(opp_pool, 2):
            entry = opp_data.get(opp)
            if entry is None:
                continue
            w, oh, oha = entry
            orr = _fast_eval(oh, board, oha, board_a)
            wins   += w * (1.0 if mr < orr else (0.5 if mr == orr else 0.0))
            total_w += w

    return wins / total_w if total_w > 0 else 0.5


# ── GenesisV1Agent ────────────────────────────────────────────────────────────


class GenesisV2Agent(Agent):
    """
    GenesisV2: targeted fixes over V1 — check-raise bug, maniac guard,
    equity-aware strength, river tightening, position awareness.
    """

    _STAT_PRIORS = {
        "fold_to_bet":       0.35,
        "fold_to_raise":     0.35,
        "check_raise":       0.05,
        "call_down":         0.40,
        "opp_aggression":    0.25,
        "opp_avg_bet_frac":  0.50,
        "opp_preflop_raise": 0.30,
    }

    def __init__(self, stream: bool = True, entry_point: str = "genesisV2"):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self._entry_point = entry_point

        # Opponent model
        self._opp_model = _OppDiscardModel()
        self._stats = {
            "fold_to_bet":       [0, 0],
            "fold_to_raise":     [0, 0],
            "check_raise":       [0, 0],
            "call_down":         [0, 0],
            "opp_aggression":    [0, 0],
            "opp_avg_bet_frac":  [0.0, 0],
            "opp_preflop_raise": [0, 0],
        }
        self._opp_archetype = "default"
        self._opp_hand_aggr = 0.0
        self._last_was_bet = False
        self._last_was_check = False   # Fix: track explicit CHECK to count check-raises correctly
        self._last_street_seen = 0
        self._opp_folded = False

        # Per-match state
        self._running_pnl = 0
        self._hands_completed = 0

        # Per-hand state (reset on new hand)
        self._last_hand_number = -1
        self._last_community = []
        self._last_opp_discards = []
        self._last_my_cards = []
        self._discard_class = ""
        self._cached_preflop_eq = None   # Fix: cache preflop equity per hand
        self._position = "BB"            # Fix: track SB/BB position

    def __name__(self):
        return "GenesisV2Agent"

    # ── Opponent profiling helpers ────────────────────────────────────────────

    def _safe_rate(self, key):
        folds, total = self._stats.get(key, [0, 0])
        if total < 3:
            return self._STAT_PRIORS.get(key, 0.35)
        return folds / total

    def _total_obs(self):
        return sum(v[1] for v in self._stats.values() if isinstance(v[1], int))

    def _select_mode(self):
        if self._total_obs() < 5:
            self._opp_archetype = "default"
            return
        fold_to_bet  = self._safe_rate("fold_to_bet")
        cr_rate      = self._safe_rate("check_raise")
        call_down    = self._safe_rate("call_down")
        opp_aggro    = self._safe_rate("opp_aggression")
        pf_raise     = self._safe_rate("opp_preflop_raise")

        if opp_aggro > 0.45 or cr_rate > 0.10:
            self._opp_archetype = "maniac"
        elif fold_to_bet > 0.48 or (opp_aggro < 0.15 and self._stats["opp_aggression"][1] >= 8):
            self._opp_archetype = "overfolder"
        elif fold_to_bet < 0.30 or call_down > 0.55 or pf_raise < 0.12:
            self._opp_archetype = "station"
        else:
            self._opp_archetype = "default"

    def _dynamic_sizing(self, base_amount, strength, street):
        arch = self._opp_archetype
        mult = 1.0
        if arch == "overfolder":
            if strength in ("draw", "weak"):
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
            elif strength == "draw":
                mult = 0.80
        elif arch == "maniac":
            if strength == "monster":
                mult = 0.70
            elif strength in ("strong", "medium"):
                mult = 0.80
        return max(1, int(base_amount * mult))

    def _process_opp_action(self, observation, opp_action):
        if not opp_action:
            return
        if opp_action == "fold":
            self._opp_folded = True
        street = observation.get("street", 0)
        last_was_bet = self._last_was_bet

        if last_was_bet:
            if opp_action == "fold":
                self._stats["fold_to_bet"][0]   += 1
                self._stats["fold_to_bet"][1]   += 1
                self._stats["fold_to_raise"][0] += 1
                self._stats["fold_to_raise"][1] += 1
            elif opp_action in ("call", "check", "raise"):
                self._stats["fold_to_bet"][1]   += 1
                self._stats["fold_to_raise"][1] += 1
                if opp_action == "call" and street >= 1:
                    self._stats["call_down"][0] += 1
                    self._stats["call_down"][1] += 1
                elif street >= 1:
                    self._stats["call_down"][1] += 1

        # Fix: only count as check-raise opportunity when WE explicitly checked last action
        last_was_check = self._last_was_check
        if last_was_check and opp_action == "raise":
            self._stats["check_raise"][0] += 1
            self._stats["check_raise"][1] += 1
        elif last_was_check and opp_action in ("check", "call", "fold"):
            self._stats["check_raise"][1] += 1

        if opp_action in ("raise", "call", "check"):
            self._stats["opp_aggression"][1] += 1
            if opp_action == "raise":
                self._stats["opp_aggression"][0] += 1
                opp_bet_obs = observation.get("opp_bet", 0)
                my_bet_obs  = observation.get("my_bet", 0)
                pot_obs = opp_bet_obs + my_bet_obs
                raise_size = max(0, opp_bet_obs - my_bet_obs)
                if pot_obs > 0 and raise_size > 0:
                    frac = raise_size / pot_obs
                    self._stats["opp_avg_bet_frac"][0] += frac
                    self._stats["opp_avg_bet_frac"][1] += 1

        if self._last_street_seen == 0 and opp_action in ("raise", "call", "check", "fold"):
            self._stats["opp_preflop_raise"][1] += 1
            if opp_action == "raise":
                self._stats["opp_preflop_raise"][0] += 1

        if street > self._last_street_seen:
            self._opp_hand_aggr *= 0.7
        if opp_action == "raise":
            self._opp_hand_aggr += 0.7
        elif opp_action == "call" and last_was_bet:
            self._opp_hand_aggr += 0.2

    # ── observe() ────────────────────────────────────────────────────────────

    def observe(self, observation, reward, terminated, truncated, info):
        raw = observation.get("opp_last_action", "")
        opp_action = _normalize_action(raw)
        if opp_action == "fold":
            self._opp_folded = True

        if terminated:
            self._running_pnl += int(reward)
            self._hands_completed += 1

            if self._opp_folded and self._last_was_bet:
                self._stats["fold_to_bet"][0]   += 1
                self._stats["fold_to_bet"][1]   += 1
                self._stats["fold_to_raise"][0] += 1
                self._stats["fold_to_raise"][1] += 1

            # Learn from showdown (update discard model)
            self._learn_from_showdown(observation, info)

            self._log_hand_result(observation, info, reward, truncated)

            self._opp_folded = False
            self._last_was_bet = False
            self._opp_hand_aggr = 0.0
            self._last_street_seen = 0

    def _learn_from_showdown(self, observation, info):
        p0_cards = info.get("player_0_cards")
        p1_cards = info.get("player_1_cards")
        if not p0_cards or not p1_cards:
            return
        if len(self._last_opp_discards) < 3 or len(self._last_community) < 3:
            return
        try:
            _RANKS_STR = "23456789A"
            _SUITS_STR = "dhs"
            def _s2i(cs):
                return _RANKS_STR.index(cs[0]) + _SUITS_STR.index(cs[1]) * NUM_RANKS
            p0_ints = [_s2i(c) for c in p0_cards]
            p1_ints = [_s2i(c) for c in p1_cards]
        except (ValueError, IndexError):
            return
        my_set = set(self._last_my_cards[:2])
        if set(p0_ints) == my_set:
            opp_kept = p1_ints
        elif set(p1_ints) == my_set:
            opp_kept = p0_ints
        else:
            return
        flop = self._last_community[:3]
        self._opp_model.update(opp_kept, flop, self._last_opp_discards)

    # ── act() ─────────────────────────────────────────────────────────────────

    def act(self, observation, reward, terminated, truncated, info):
        my_cards    = [c for c in observation["my_cards"]            if c != -1]
        community   = [c for c in observation["community_cards"]     if c != -1]
        opp_discards= [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards = [c for c in observation["my_discarded_cards"]  if c != -1]
        valid       = observation["valid_actions"]
        street      = observation["street"]
        min_raise   = int(observation.get("min_raise", 2))
        max_raise   = int(observation.get("max_raise", 0))
        my_bet      = observation.get("my_bet", 0)
        opp_bet     = observation.get("opp_bet", 0)
        pot_size    = my_bet + opp_bet
        hand_number = info.get("hand_number", 0)

        # Track state for discard learning
        self._last_my_cards = list(my_cards)
        if opp_discards:
            self._last_opp_discards = list(opp_discards)
        if community:
            self._last_community = list(community)

        # New hand reset
        if hand_number != self._last_hand_number:
            self._last_hand_number = hand_number
            self._discard_class = ""
            self._opp_folded = False
            self._opp_hand_aggr = 0.0
            self._last_was_bet = False
            self._last_was_check = False
            self._last_street_seen = 0
            self._last_opp_discards = []
            self._cached_preflop_eq = None

        # Track position (SB = blind_position 0, BB = blind_position 1)
        blind_pos = observation.get("blind_position", 1)
        self._position = "SB" if blind_pos == 0 else "BB"

        # Update opponent stats
        raw = observation.get("opp_last_action", "")
        opp_action = _normalize_action(raw)
        if opp_action:
            self._process_opp_action(observation, opp_action)

        # Refresh archetype every preflop act
        if street == 0 and not valid[DISCARD]:
            self._select_mode()

        # ── Bleed-out lock (early check) ─────────────────────────────────────
        if not valid[DISCARD]:
            hands_remaining = max(0, TOTAL_HANDS - self._hands_completed)
            sb_left   = (hands_remaining + 1) // 2
            bb_left   = hands_remaining // 2
            max_bleed = sb_left + bb_left * 2
            if self._running_pnl > max_bleed:
                if valid[FOLD]:
                    return (FOLD, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)

        # ── Discard phase ─────────────────────────────────────────────────────
        if valid[DISCARD]:
            result = self._act_discard(my_cards, community, opp_discards, my_discards)
            self._last_street_seen = street
            return result

        # ── Street 0: Preflop ─────────────────────────────────────────────────
        if street == 0:
            result = self._act_street0(my_cards, my_bet, opp_bet, valid, min_raise, max_raise)
            self._last_was_bet   = result[0] == RAISE
            self._last_was_check = result[0] == CHECK
            self._last_street_seen = street
            return result

        # ── Streets 1-3: Postflop ─────────────────────────────────────────────
        result = self._act_postflop(
            my_cards, community, opp_discards, my_discards,
            valid, street, pot_size, my_bet, opp_bet, min_raise, max_raise,
        )

        # ── Safety-net bleed-out ──────────────────────────────────────────────
        hands_remaining = max(0, TOTAL_HANDS - self._hands_completed)
        sb_left   = (hands_remaining + 1) // 2
        bb_left   = hands_remaining // 2
        max_bleed = sb_left + bb_left * 2
        if self._running_pnl > max_bleed and result[0] in (RAISE, CALL):
            if valid[FOLD]:
                result = (FOLD, 0, 0, 0)
            elif valid[CHECK]:
                result = (CHECK, 0, 0, 0)

        self._last_was_bet   = result[0] == RAISE
        self._last_was_check = result[0] == CHECK
        self._last_street_seen = street
        return result

    # ── Discard ───────────────────────────────────────────────────────────────

    def _act_discard(self, my_cards, community, opp_discards, my_discards):
        dead_base = set(my_cards) | set(community)
        have_disc = len(opp_discards) >= 3

        all_keeps = []
        for i, j in combinations(range(len(my_cards)), 2):
            keep = [my_cards[i], my_cards[j]]
            toss = [my_cards[k] for k in range(len(my_cards)) if k not in (i, j)]
            all_keeps.append((i, j, keep, toss))

        if have_disc:
            best_eq = -1.0
            best_ij = (0, 1)
            for i, j, keep, toss in all_keeps:
                dead = dead_base | set(toss) | set(opp_discards)
                eq = _exact_discard_equity_weighted(keep, community, dead,
                                                    self._opp_model, opp_discards)
                if eq > best_eq:
                    best_eq = eq
                    best_ij = (i, j)
        else:
            # Screen with MC, then exact on top 5
            opp_disc_set = set(opp_discards)
            board_needed = 5 - len(community)
            screen_size  = 2 + board_needed
            comm_t = [_INT_TO_TREYS[c]     for c in community]
            comm_a = [_INT_TO_TREYS_ALT[c] for c in community]

            candidates = []
            for i, j, keep, toss in all_keeps:
                per_dead = dead_base | set(toss) | opp_disc_set
                remaining = [c for c in range(DECK_SIZE) if c not in per_dead]
                if screen_size > len(remaining):
                    candidates.append((i, j, 0.5, keep, toss))
                    continue
                kh  = [_INT_TO_TREYS[c]     for c in keep]
                kha = [_INT_TO_TREYS_ALT[c] for c in keep]
                w = t = 0
                for _ in range(500):
                    samp = random.sample(remaining, screen_size)
                    oh  = [_INT_TO_TREYS[c]     for c in samp[:2]]
                    oha = [_INT_TO_TREYS_ALT[c] for c in samp[:2]]
                    bd   = comm_t + [_INT_TO_TREYS[c]     for c in samp[2:]]
                    bda  = comm_a + [_INT_TO_TREYS_ALT[c] for c in samp[2:]]
                    mr   = _fast_eval(kh, bd, kha, bda)
                    orr  = _fast_eval(oh, bd, oha, bda)
                    w   += 1 if mr < orr else (0.5 if mr == orr else 0)
                    t   += 1
                candidates.append((i, j, w / t if t else 0.5, keep, toss))

            candidates.sort(key=lambda c: c[2], reverse=True)
            best_eq = -1.0
            best_ij = (0, 1)
            for i, j, _, keep, toss in candidates[:5]:
                dead = dead_base | set(toss) | opp_disc_set
                eq = _exact_discard_equity(keep, community, dead)
                if eq > best_eq:
                    best_eq = eq
                    best_ij = (i, j)

        # Classify discard for logging
        ki0, ki1 = best_ij
        kept = [my_cards[ki0], my_cards[ki1]]
        discarded = [c for i, c in enumerate(my_cards) if i not in best_ij]
        self._discard_class = self._classify_discard(kept, discarded, community)

        self.logger.info(
            f"DISCARD | entry={self._entry_point} | keep_idx={ki0},{ki1}"
            f" | class={self._discard_class} | best_eq={best_eq:.4f}"
            f" | have_disc={have_disc}"
        )
        return (DISCARD, 0, ki0, ki1)

    @staticmethod
    def _classify_discard(kept, discarded, community):
        """Simple discard class for logging (mirrors Genesis categories)."""
        if len(kept) < 2:
            return "unknown"
        r0, r1 = _RANK[kept[0]], _RANK[kept[1]]
        s0, s1 = _SUIT[kept[0]], _SUIT[kept[1]]
        board_ranks = [_RANK[c] for c in community] if community else []
        if r0 == r1:
            return "pair_transparent"
        if s0 == s1:
            sc = [0, 0, 0]
            sc[s0] += 2
            for c in community:
                sc[_SUIT[c]] += 1
            if max(sc) >= 4:
                return "flush_transparent"
        # Straight check: kept + board has 3+ consecutive
        all_ranks = sorted(set([r0, r1] + board_ranks))
        best_run = cur_run = 1
        for i in range(1, len(all_ranks)):
            if all_ranks[i] - all_ranks[i-1] == 1:
                cur_run += 1
                best_run = max(best_run, cur_run)
            else:
                cur_run = 1
        if best_run >= 4:
            return "straight_transparent"
        # pair with board
        if r0 in board_ranks or r1 in board_ranks:
            return "pair_transparent"
        return "ambiguous"

    # ── Street 0: Preflop ─────────────────────────────────────────────────────

    def _act_street0(self, my_cards, my_bet, opp_bet, valid, min_raise, max_raise):
        if len(my_cards) != 5:
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        premium      = _has_any_premium(my_cards)
        premium_pair = _has_premium_pair(my_cards)
        to_call      = max(0, opp_bet - my_bet)
        in_early     = self._hands_completed < EARLY_PHASE_HANDS

        result = None

        if in_early:
            if premium:
                if premium_pair and opp_bet >= PREFLOP_COMMIT_THRESHOLD and valid[RAISE]:
                    # Fix: vs maniac, only AA goes all-in; medium pairs raise controlled
                    if self._opp_archetype == "maniac" and not _has_pair_of_rank(my_cards, RANK_A):
                        amt = _clamp(int(to_call * 2.5 * random.uniform(0.9, 1.1)),
                                     min_raise, max(min_raise, max_raise // 3))
                        result = (RAISE, amt, 0, 0)
                    else:
                        result = (RAISE, max_raise, 0, 0)
                elif to_call > 0 and valid[RAISE] and random.random() < 0.70:
                    amt = _clamp(int(to_call * 2.5 * random.uniform(0.9, 1.1)),
                                 min_raise, max_raise)
                    result = (RAISE, amt, 0, 0)
                elif valid[RAISE]:
                    noise = random.uniform(0.85, 1.15)
                    open_sz = _clamp(int(STANDARD_OPEN * EARLY_OPEN_MULTIPLIER * noise),
                                     min_raise, max_raise)
                    result = (RAISE, open_sz, 0, 0)
                elif valid[CALL]:
                    result = (CALL, 0, 0, 0)
                else:
                    result = (CHECK, 0, 0, 0)
            else:
                pf_eq = self._preflop_equity(my_cards)
                if pf_eq >= EARLY_PREFLOP_MIN_EQUITY:
                    if valid[RAISE] and random.random() < 0.65:
                        amt = _clamp(int(9 * random.uniform(0.9, 1.1)), min_raise, max_raise)
                        result = (RAISE, amt, 0, 0)
                    elif valid[CALL]:
                        result = (CALL, 0, 0, 0)
                    elif valid[CHECK]:
                        result = (CHECK, 0, 0, 0)
        else:
            if premium:
                if premium_pair and opp_bet >= PREFLOP_COMMIT_THRESHOLD and valid[RAISE]:
                    # Fix: vs maniac, only AA goes all-in; medium pairs raise controlled
                    if self._opp_archetype == "maniac" and not _has_pair_of_rank(my_cards, RANK_A):
                        amt = _clamp(int(to_call * 2.5 * random.uniform(0.9, 1.1)),
                                     min_raise, max(min_raise, max_raise // 3))
                        result = (RAISE, amt, 0, 0)
                    else:
                        result = (RAISE, max_raise, 0, 0)
                elif premium_pair and random.random() < SLOW_PLAY_CHANCE:
                    result = (CALL if valid[CALL] else CHECK, 0, 0, 0)
                else:
                    noise = random.uniform(0.85, 1.15)
                    open_sz = _clamp(int(STANDARD_OPEN * noise), min_raise, max_raise)
                    if valid[RAISE]:
                        result = (RAISE, open_sz, 0, 0)
                    elif valid[CALL]:
                        result = (CALL, 0, 0, 0)
                    else:
                        result = (CHECK, 0, 0, 0)
            else:
                pf_eq = self._preflop_equity(my_cards)
                # Fix: position-aware BB defend (already in pot, call slightly wider)
                if self._position == "BB" and to_call <= 4 and pf_eq >= 0.40 and valid[CALL]:
                    result = (CALL, 0, 0, 0)
                elif pf_eq >= NORMAL_PREFLOP_MIN_EQUITY:
                    if to_call <= 0 and valid[RAISE] and random.random() < 0.25:   # was 0.40
                        amt = _clamp(int(8 * random.uniform(0.9, 1.1)), min_raise, max_raise)
                        result = (RAISE, amt, 0, 0)
                    elif valid[CALL] and to_call <= 4:
                        result = (CALL, 0, 0, 0)
                    elif valid[CHECK]:
                        result = (CHECK, 0, 0, 0)

        if result is None:
            result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

        self.logger.info(
            f"STREET0 | entry={self._entry_point} | hand={self._last_hand_number}"
            f" | premium={premium} | early={in_early}"
            f" | arch={self._opp_archetype} | action={result[0]}"
        )
        return result

    def _preflop_equity(self, my5):
        # Fix: use BEST keep equity (not avg top-3) + cache per hand
        if self._cached_preflop_eq is not None:
            return self._cached_preflop_eq
        best_eq = 0.0
        dead = set(my5)
        for i, j in combinations(range(len(my5)), 2):
            keep = [my5[i], my5[j]]
            eq = _compute_equity(keep, [], dead, n_sims=80)
            best_eq = max(best_eq, eq)
        self._cached_preflop_eq = best_eq
        return best_eq

    # ── Streets 1-3: Postflop ─────────────────────────────────────────────────

    def _act_postflop(self, my_cards, community, opp_discards, my_discards,
                      valid, street, pot_size, my_bet, opp_bet, min_raise, max_raise):
        if len(my_cards) > 2:
            my_cards = my_cards[:2]

        dead = set(my_cards) | set(community) | set(opp_discards) | set(my_discards)

        opp_signal = 0.0 if self._opp_archetype == "maniac" else self._opp_hand_aggr
        equity = _compute_equity_ranged(
            my_cards, community, dead, self._opp_model,
            opp_discards, opp_signal, n_sims=300,
        )

        hand_cat = _hand_rank_category(my_cards, community)
        suit_count, flush_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
        run_count,  str_outs, _   = _count_straight_outs(my_cards, community, opp_discards, my_discards)
        has_draw = (suit_count >= 4 and flush_outs >= 2) or (run_count >= 4 and str_outs >= 3)

        # Apply board texture adjustments (fix: draw bonus scales by street)
        outs = max(flush_outs, str_outs)
        if street == 1 and has_draw and outs > 0:
            equity += min(0.14, outs * 0.035)   # flop: 2 cards to come, more draw value
        elif street == 2 and has_draw and outs > 0:
            equity += min(0.10, outs * 0.025)   # turn: 1 card to come
        elif street == 3 and has_draw and hand_cat in ("nothing", "one_pair"):
            equity -= 0.10   # missed draw on river: penalize
        equity += _board_monotone_penalty(my_cards, community)
        equity += _board_connected_penalty(my_cards, community)
        equity += _opp_flush_inference(community, opp_discards)
        equity = _clamp(equity, 0.0, 0.98)

        # Fix: equity-aware strength (equity caps the strength label)
        strength = self._cat_to_strength(hand_cat, has_draw, equity)

        # Fix: river gets its own tighter logic
        if street == 3:
            return self._act_river(
                my_cards, community, opp_discards, my_discards,
                valid, pot_size, my_bet, opp_bet, min_raise, max_raise,
                equity, hand_cat, strength, has_draw,
            )

        to_call  = max(0, opp_bet - my_bet)
        pot_ref  = max(pot_size, 1)
        pot_odds = to_call / (pot_ref + to_call) if to_call > 0 else 0.0

        result = None

        # Fix: maniac counter-strategy — trap strong hands, don't bet weak hands
        if self._opp_archetype == "maniac":
            if equity > MONSTER_THRESHOLD and to_call <= 0:
                # 40% check-trap to induce raise, then re-raise
                if valid[CHECK] and random.random() < 0.40:
                    self.logger.info(
                        f"POSTFLOP | entry={self._entry_point} | hand={self._last_hand_number}"
                        f" | street={street} | eq={equity:.4f} | cat={hand_cat}"
                        f" | strength={strength} | arch={self._opp_archetype}"
                        f" | to_call={to_call} | pot={pot_ref} | action={CHECK} (maniac_trap)"
                    )
                    return (CHECK, 0, 0, 0)
            elif equity <= GOOD_THRESHOLD and to_call <= 0:
                # Don't bluff into maniac (they call/raise everything)
                self.logger.info(
                    f"POSTFLOP | entry={self._entry_point} | hand={self._last_hand_number}"
                    f" | street={street} | eq={equity:.4f} | cat={hand_cat}"
                    f" | strength={strength} | arch={self._opp_archetype}"
                    f" | to_call={to_call} | pot={pot_ref} | action={CHECK} (maniac_no_bluff)"
                )
                return (CHECK if valid[CHECK] else FOLD, 0, 0, 0)
            elif equity <= GOOD_THRESHOLD and to_call > pot_ref * 0.30:
                # Marginal hand facing maniac aggression: fold
                self.logger.info(
                    f"POSTFLOP | entry={self._entry_point} | hand={self._last_hand_number}"
                    f" | street={street} | eq={equity:.4f} | cat={hand_cat}"
                    f" | strength={strength} | arch={self._opp_archetype}"
                    f" | to_call={to_call} | pot={pot_ref} | action={FOLD} (maniac_marginal_fold)"
                )
                return (FOLD if valid[FOLD] else (CHECK if valid[CHECK] else CALL), 0, 0, 0)

        # Semi-bluff opportunity (streets 1-2, when no facing bet)
        if to_call <= 0 and street in (1, 2):
            fire, sb_result = self._semi_bluff_check(
                my_cards, community, opp_discards, my_discards,
                pot_size, to_call, street, valid, min_raise, max_raise,
                has_draw=has_draw, flush_outs_v=flush_outs, str_outs_v=str_outs,
            )
            if fire:
                result = sb_result

        if result is None:
            if equity > MONSTER_THRESHOLD:
                if street == 1:
                    bet_frac = random.uniform(0.55, 0.72)
                else:   # street == 2
                    bet_frac = random.uniform(0.70, 0.90)
                base_amt  = _clamp(int(pot_ref * bet_frac), min_raise, max_raise)
                raise_amt = _clamp(self._dynamic_sizing(base_amt, strength, street),
                                   min_raise, max_raise)
                if valid[RAISE]:
                    result = (RAISE, raise_amt, 0, 0)
                elif valid[CALL]:
                    result = (CALL, 0, 0, 0)
                else:
                    result = (CHECK, 0, 0, 0)

            elif equity > STRONG_THRESHOLD:
                if street == 1:
                    bet_frac = random.uniform(0.55, 0.72)
                else:   # street == 2
                    bet_frac = random.uniform(0.65, 0.80)
                base_amt  = _clamp(int(pot_ref * bet_frac), min_raise, max_raise)
                raise_amt = _clamp(self._dynamic_sizing(base_amt, strength, street),
                                   min_raise, max_raise)
                if valid[RAISE]:
                    result = (RAISE, raise_amt, 0, 0)
                elif valid[CALL]:
                    result = (CALL, 0, 0, 0)
                else:
                    result = (CHECK, 0, 0, 0)

            elif equity > GOOD_THRESHOLD:
                bet_frac  = random.uniform(0.30, 0.50)
                base_amt  = _clamp(int(pot_ref * bet_frac), min_raise, max_raise)
                raise_amt = _clamp(self._dynamic_sizing(base_amt, strength, street),
                                   min_raise, max_raise)
                if to_call <= 0 and valid[RAISE]:
                    result = (RAISE, raise_amt, 0, 0)
                elif to_call > 0 and equity >= pot_odds and valid[CALL]:
                    result = (CALL, 0, 0, 0)
                elif valid[CHECK]:
                    result = (CHECK, 0, 0, 0)
                else:
                    result = (FOLD, 0, 0, 0)

            elif equity >= pot_odds and to_call > 0 and to_call <= pot_ref * 0.35:
                result = (CALL if valid[CALL] else (CHECK if valid[CHECK] else FOLD), 0, 0, 0)

            else:
                result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

        # Decision guards
        if result[0] == FOLD and to_call <= 0 and valid[CHECK]:
            result = (CHECK, 0, 0, 0)
        if result[0] == FOLD and my_bet > 0 and pot_ref > 0 and my_bet >= pot_ref * 0.40:
            if valid[CALL]:
                result = (CALL, 0, 0, 0)
        if (result[0] == FOLD and len(my_cards) == 2
                and _is_premium_pair(my_cards[0], my_cards[1])
                and to_call > 0 and to_call <= pot_ref * 0.20 and valid[CALL]):
            result = (CALL, 0, 0, 0)

        self.logger.info(
            f"POSTFLOP | entry={self._entry_point} | hand={self._last_hand_number}"
            f" | street={street} | eq={equity:.4f} | cat={hand_cat}"
            f" | strength={strength} | arch={self._opp_archetype}"
            f" | to_call={to_call} | pot={pot_ref} | action={result[0]}"
        )
        return result

    @staticmethod
    def _cat_to_strength(hand_cat, has_draw, equity=None):
        # Fix: equity takes precedence — prevents two_pair at 34% equity → "monster"
        if equity is not None:
            if equity > MONSTER_THRESHOLD:      # > 0.82
                return "monster"
            if equity > STRONG_THRESHOLD:       # > 0.65
                return "strong"
            if equity > GOOD_THRESHOLD:         # > 0.48
                # trips_plus gets "strong" not "good" (category above two_pair)
                if hand_cat == "trips_plus":
                    return "strong"
                return "good"
            if has_draw:
                return "draw"
            return "weak"
        # Fallback when no equity provided
        if hand_cat in ("trips_plus", "two_pair"):
            return "monster"
        if hand_cat == "one_pair" and has_draw:
            return "strong"
        if hand_cat == "one_pair":
            return "medium"
        if has_draw:
            return "draw"
        return "weak"

    def _semi_bluff_check(self, my_cards, community, opp_discards, my_discards,
                          pot_size, to_call, street, valid, min_raise, max_raise,
                          has_draw=None, flush_outs_v=None, str_outs_v=None):
        if not (valid[RAISE] and max_raise >= min_raise):
            return False, None
        if to_call > pot_size * 0.40:
            return False, None
        if has_draw is not None:
            if not has_draw:
                return False, None
            f_outs = flush_outs_v or 0
            s_outs = str_outs_v or 0
        else:
            sc, f_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
            _, s_outs, _  = _count_straight_outs(my_cards, community, opp_discards, my_discards)
            if not ((sc >= 4 and f_outs >= 2) or (s_outs >= 3)):
                return False, None

        outs = max(f_outs, s_outs)
        if outs < 2:
            return False, None

        b_ranks = [_RANK[c] for c in community]
        if len(b_ranks) != len(set(b_ranks)):
            return False, None

        known = set(my_cards[:2]) | set(community) | set(opp_discards) | set(my_discards)
        remaining = max(1, DECK_SIZE - len(known))
        cards_to_come = 2 if street == 1 else 1

        if cards_to_come == 1:
            draw_eq = min(1.0, outs / remaining)
        else:
            miss1 = max(0, remaining - outs) / remaining
            miss2 = max(0, remaining - outs - 1) / max(1, remaining - 1)
            draw_eq = 1.0 - miss1 * miss2

        fold_prob = self._safe_rate("fold_to_bet")
        pot = max(pot_size, 1)

        if fold_prob > 0.48:
            sizing_frac = random.uniform(0.65, 0.80)
        elif fold_prob < 0.25:
            sizing_frac = random.uniform(0.45, 0.60)
        else:
            sizing_frac = random.uniform(0.55, 0.70)

        bet_size = pot * sizing_frac
        ev = (fold_prob * pot + (1 - fold_prob) * (draw_eq * (pot + 2 * bet_size) - bet_size))

        if ev > 0:
            amt = _clamp(int(bet_size), min_raise, max_raise)
            return True, (RAISE, amt, 0, 0)
        return False, None

    # ── River (tighter logic) ─────────────────────────────────────────────────

    def _act_river(self, my_cards, community, opp_discards, my_discards,
                   valid, pot_size, my_bet, opp_bet, min_raise, max_raise,
                   equity, hand_cat, strength, has_draw):
        """Separate river decision with tighter call thresholds and archetype-aware sizing."""
        RIVER_CALL_PREMIUM = 0.08   # equity must exceed pot_odds by this margin to call

        to_call  = max(0, opp_bet - my_bet)
        pot_ref  = max(pot_size, 1)
        pot_odds = to_call / (pot_ref + to_call) if to_call > 0 else 0.0
        arch     = self._opp_archetype
        fold_rate = self._safe_rate("fold_to_bet")

        result = None

        if to_call <= 0:   # Opening bet (no one bet yet)
            if equity > MONSTER_THRESHOLD:
                # Size based on archetype: stations call big bets; overfolders fold to large
                if arch == "station":
                    frac = random.uniform(0.85, 1.00)
                elif arch == "overfolder":
                    frac = random.uniform(0.60, 0.75)
                else:
                    frac = random.uniform(0.70, 0.88)
                amt = _clamp(int(pot_ref * frac), min_raise, max_raise)
                if valid[RAISE]:
                    result = (RAISE, _clamp(self._dynamic_sizing(amt, strength, 3), min_raise, max_raise), 0, 0)
                elif valid[CALL]:
                    result = (CALL, 0, 0, 0)
                else:
                    result = (CHECK, 0, 0, 0)

            elif equity > STRONG_THRESHOLD:
                if arch == "station":
                    frac = random.uniform(0.65, 0.80)
                elif arch == "overfolder":
                    frac = random.uniform(0.45, 0.60)
                else:
                    frac = random.uniform(0.55, 0.70)
                amt = _clamp(int(pot_ref * frac), min_raise, max_raise)
                if valid[RAISE]:
                    result = (RAISE, _clamp(self._dynamic_sizing(amt, strength, 3), min_raise, max_raise), 0, 0)
                else:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

            elif equity > GOOD_THRESHOLD:
                # One pair on river: thin bet only vs overfolders; check otherwise
                if arch == "overfolder" and valid[RAISE]:
                    frac = random.uniform(0.35, 0.50)
                    amt = _clamp(int(pot_ref * frac), min_raise, max_raise)
                    result = (RAISE, _clamp(self._dynamic_sizing(amt, strength, 3), min_raise, max_raise), 0, 0)
                else:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

            else:
                # Weak / missed draw: bluff only vs overfolders with high fold rate
                if arch == "overfolder" and fold_rate > 0.55 and valid[RAISE]:
                    if random.random() < 0.25:
                        amt = _clamp(int(pot_ref * random.uniform(0.55, 0.70)), min_raise, max_raise)
                        result = (RAISE, amt, 0, 0)
                if result is None:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

        else:   # Facing a bet
            if equity > MONSTER_THRESHOLD:
                # Re-raise or call — never fold monsters
                if valid[RAISE]:
                    amt = _clamp(int(pot_ref * 1.0), min_raise, max_raise)
                    result = (RAISE, amt, 0, 0)
                elif valid[CALL]:
                    result = (CALL, 0, 0, 0)
                else:
                    result = (CHECK, 0, 0, 0)

            elif equity > STRONG_THRESHOLD:
                # Call (don't fold strong hands to any bet)
                if valid[CALL]:
                    result = (CALL, 0, 0, 0)
                else:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

            elif equity > GOOD_THRESHOLD:
                # GOOD hands (one pair): call only if equity well exceeds pot_odds AND bet is small
                if equity >= pot_odds + RIVER_CALL_PREMIUM and to_call <= pot_ref * 0.45:
                    if valid[CALL]:
                        result = (CALL, 0, 0, 0)
                if result is None:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

            else:
                # Weak/missed draw: call only on tiny bets with very favorable odds
                if equity >= pot_odds + RIVER_CALL_PREMIUM + 0.05 and to_call <= pot_ref * 0.20:
                    if valid[CALL]:
                        result = (CALL, 0, 0, 0)
                if result is None:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

        # Safety guard: never fold when no bet to call
        if result[0] == FOLD and to_call <= 0 and valid[CHECK]:
            result = (CHECK, 0, 0, 0)
        # Never fold on river when call is tiny relative to chips already in pot
        # (only protect when to_call <= 15% of pot — truly tiny bets are never fold)
        if result[0] == FOLD and my_bet > 0 and to_call > 0 and to_call <= pot_ref * 0.15:
            if valid[CALL]:
                result = (CALL, 0, 0, 0)

        self.logger.info(
            f"POSTFLOP | entry={self._entry_point} | hand={self._last_hand_number}"
            f" | street=3 | eq={equity:.4f} | cat={hand_cat}"
            f" | strength={strength} | arch={self._opp_archetype}"
            f" | to_call={to_call} | pot={pot_ref} | action={result[0]}"
        )
        return result

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_hand_result(self, observation, info, reward, truncated):
        hand_number = info.get("hand_number", self._last_hand_number)
        street = observation.get("street", -1)
        my_bet  = int(observation.get("my_bet", 0) or 0)
        opp_bet = int(observation.get("opp_bet", 0) or 0)
        pot = my_bet + opp_bet
        blind_pos = observation.get("blind_position", -1)
        position = "SB" if blind_pos == 0 else "BB" if blind_pos == 1 else str(blind_pos)
        opp_last = (observation.get("opp_last_action") or "").strip()
        is_showdown = street > 3

        def card_str(c):
            if c is None or c == -1:
                return None
            return PokerEnv.int_card_to_str(int(c))

        my_cards_raw = observation.get("my_cards", []) or []
        my_cards_str = " ".join(s for c in my_cards_raw if (s := card_str(c)))
        community = observation.get("community_cards", []) or []
        board_str = " ".join(s for c in community if (s := card_str(c)))

        parts = [
            "HAND_RESULT",
            f"entry={self._entry_point}",
            f"hand={hand_number}",
            f"reward={reward}",
            f"won={reward > 0}",
            f"lost={reward < 0}",
            f"tie={reward == 0}",
            f"running_pnl={self._running_pnl}",
            f"street_ended={street}",
            f"is_showdown={is_showdown}",
            f"end_type={'showdown' if is_showdown else ('they_fold' if opp_last.upper()=='FOLD' else 'we_fold')}",
            f"position={position}",
            f"pot={pot}",
            f"my_bet={my_bet}",
            f"opp_bet={opp_bet}",
            f"discard_class={self._discard_class}",
            f"arch={self._opp_archetype}",
            f"fold_to_bet={self._safe_rate('fold_to_bet'):.3f}",
            f"opp_aggr={self._safe_rate('opp_aggression'):.3f}",
            f"my_cards={my_cards_str}",
            f"board={board_str}",
            f"truncated={truncated}",
        ]
        self.logger.info(" | ".join(parts))
