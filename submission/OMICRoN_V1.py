# OMICRoN V1: Fork of ALPHANiTV8 — exact subgame solver for discard + adaptive
#             opponent modeling + full postflop engine (range-weighted equity,
#             board texture, semi-bluff, dynamic sizing, opponent profiling).
#
# Speed-optimized: lookup arrays, no Counter in hot paths, precomputed treys
# pairs, precomputed opponent weight maps, shared dead-sets.

import json
import logging
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
from treys import Card, Evaluator

from agents.agent import Agent
from gym_env import PokerEnv

_PROFILE = {}
_profile_path = os.path.join(os.path.dirname(__file__), "..", "logs", "bot_profile.json")
if os.path.isfile(_profile_path):
    try:
        with open(_profile_path, "r", encoding="utf-8") as f:
            _PROFILE = json.load(f)
    except (json.JSONDecodeError, OSError):
        pass

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_RANKS = 9
DECK_SIZE = 27
RANK_A = 8
RANK_9 = 7
RANK_8 = 6

SLOW_PLAY_CHANCE = 0.20
STANDARD_OPEN = _PROFILE.get("standard_open", 8)

MONSTER_THRESHOLD = 0.82
STRONG_THRESHOLD = 0.65
GOOD_THRESHOLD = 0.48
PREFLOP_COMMIT_THRESHOLD = 15
TOTAL_HANDS = _PROFILE.get("total_hands", 1000)

EARLY_PHASE_HANDS = _PROFILE.get("early_phase_hands", 50)
EARLY_PREFLOP_MIN_EQUITY = _PROFILE.get("early_preflop_min_equity", 0.48)
NORMAL_PREFLOP_MIN_EQUITY = 0.45
EARLY_OPEN_MULTIPLIER = 1.20

RANKS = "23456789A"

int_to_card = PokerEnv.int_to_card

# ── Card caches for fast evaluation ──────────────────────────────────────────

_INT_TO_TREYS = [PokerEnv.int_to_card(i) for i in range(DECK_SIZE)]
_INT_TO_TREYS_ALT = []
for _tc in _INT_TO_TREYS:
    _s = Card.int_to_str(_tc)
    _INT_TO_TREYS_ALT.append(Card.new(_s.replace("A", "T")))

_base_eval = Evaluator()

# OPT4: module-level rank/suit lookup arrays — replaces function calls with
# array indexing throughout
_RANK = [i % NUM_RANKS for i in range(DECK_SIZE)]
_SUIT = [i // NUM_RANKS for i in range(DECK_SIZE)]

# ── Evaluation Lookup Table ──────────────────────────────────────────────────

_C = [[0] * 28 for _ in range(28)]
for _n in range(28):
    _C[_n][0] = 1
    for _k in range(1, _n + 1):
        _C[_n][_k] = _C[_n - 1][_k - 1] + _C[_n - 1][_k]

_EVAL_LUT = np.load(os.path.join(os.path.dirname(__file__), "eval_table.npy"))


def _lut_eval_7(cards_7):
    s = sorted(cards_7)
    return int(_EVAL_LUT[
        _C[s[0]][1] + _C[s[1]][2] + _C[s[2]][3] + _C[s[3]][4] +
        _C[s[4]][5] + _C[s[5]][6] + _C[s[6]][7]])


def _fast_evaluate(hand, board, alt_hand, alt_board):
    r = _base_eval.evaluate(hand, board)
    a = _base_eval.evaluate(alt_hand, alt_board)
    return a if a < r else r


# ── Action constants ─────────────────────────────────────────────────────────

FOLD = PokerEnv.ActionType.FOLD.value
RAISE = PokerEnv.ActionType.RAISE.value
CHECK = PokerEnv.ActionType.CHECK.value
CALL = PokerEnv.ActionType.CALL.value
DISCARD = PokerEnv.ActionType.DISCARD.value

# ── Premium hand definitions ────────────────────────────────────────────────

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

# ── Card helpers ─────────────────────────────────────────────────────────────


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


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


# ── Board texture functions ──────────────────────────────────────────────────


def _board_paired_and_we_weak(my_cards, community):
    if len(my_cards) < 2 or len(community) < 3:
        return False
    rc = [0] * NUM_RANKS
    for c in community:
        rc[_RANK[c]] += 1
    bp_rank = -1
    for r in range(NUM_RANKS):
        if rc[r] >= 2:
            bp_rank = r
    if bp_rank < 0:
        return False
    mr0, mr1 = _RANK[my_cards[0]], _RANK[my_cards[1]]
    if bp_rank == mr0 or bp_rank == mr1:
        return False
    if mr0 == mr1 and frozenset([mr0, mr0]) in PREMIUM_PAIRS:
        return False
    return True


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
    if my_in_suit == 0:
        return -0.18
    if my_in_suit == 1:
        return -0.06
    return 0.0


def _board_connected_penalty(my_cards, community):
    if len(my_cards) < 2 or len(community) < 3:
        return 0.0
    b_ranks = [_RANK[c] for c in community]
    if _max_connectivity(b_ranks) < 3:
        return 0.0
    hcat = _hand_rank_category(my_cards, community)
    if hcat == "trips_plus":
        all_ranks = [_RANK[c] for c in my_cards[:2]] + b_ranks
        sc = [0, 0, 0]
        for c in my_cards[:2]:
            sc[_SUIT[c]] += 1
        for c in community:
            sc[_SUIT[c]] += 1
        if max(sc) >= 5:
            return 0.0
        unique_r = sorted(set(all_ranks))
        if RANK_A in unique_r:
            unique_r = [-1] + unique_r
        best_run = 1
        cur_run = 1
        for i in range(1, len(unique_r)):
            if unique_r[i] - unique_r[i - 1] == 1:
                cur_run += 1
                best_run = max(best_run, cur_run)
            else:
                cur_run = 1
        if best_run >= 5:
            return 0.0
        return -0.10
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


def _normalize_action(raw):
    # Canonical opponent action vocabulary for this bot:
    # {"fold", "check", "call", "bet", "raise"}.
    # Environment-facing values are often {FOLD,CHECK,CALL,RAISE}; when "BET"
    # is not explicitly emitted, "raise" may include no-prior-bet initiative.
    if not raw:
        return ""
    s = raw.strip().lower()
    if s == "none":
        return ""
    aliases = {
        "r": "raise",
        "raise_to": "raise",
        "c": "call",
        "x": "check",
        "f": "fold",
        "b": "bet",
    }
    if s in aliases:
        return aliases[s]
    return s


# ── Hand classification and draw detection ───────────────────────────────────


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
    all_ranks = set()
    for r in range(NUM_RANKS):
        if rc[r] > 0:
            all_ranks.add(r)
    unique_r = sorted(all_ranks)
    if RANK_A in unique_r:
        unique_r_ext = [-1] + unique_r
    else:
        unique_r_ext = unique_r
    best_run = 1
    cur_run = 1
    for i in range(1, len(unique_r_ext)):
        if unique_r_ext[i] - unique_r_ext[i - 1] == 1:
            cur_run += 1
            best_run = max(best_run, cur_run)
        else:
            cur_run = 1
    if best_run >= 5:
        return "trips_plus"
    if max(rc) >= 3:
        return "trips_plus"
    pairs = sum(1 for r in range(NUM_RANKS) if rc[r] >= 2)
    if pairs >= 2:
        return "two_pair"
    if pairs == 1:
        return "one_pair"
    return "nothing"


def _count_flush_outs(my_cards, community, opp_discards, my_discards):
    if len(my_cards) < 2 or len(community) < 3:
        return 0, 0, 27
    known = set(my_cards[:2]) | set(community) | set(opp_discards) | set(my_discards)
    best_count = 0
    best_live = 0
    for s in range(3):
        in_hand = sum(1 for c in my_cards[:2] if _SUIT[c] == s)
        on_board = sum(1 for c in community if _SUIT[c] == s)
        count = in_hand + on_board
        if count >= best_count:
            dead_of_suit = sum(1 for c in known if _SUIT[c] == s)
            live = 9 - dead_of_suit
            if count > best_count or live > best_live:
                best_count = count
                best_live = live
    return best_count, best_live, DECK_SIZE - len(known)


def _count_straight_outs(my_cards, community, opp_discards, my_discards):
    if len(my_cards) < 2 or len(community) < 3:
        return 0, 0, 27
    known = set(my_cards[:2]) | set(community) | set(opp_discards) | set(my_discards)
    have_ranks = set(_RANK[c] for c in my_cards[:2]) | set(_RANK[c] for c in community)
    best_in_run = 0
    best_outs = 0
    valid_straights = [
        [RANK_A, 0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, RANK_A],
    ]
    for window in valid_straights:
        have = sum(1 for r in window if r in have_ranks)
        need_ranks = [r for r in window if r not in have_ranks]
        if have >= 3 and len(need_ranks) <= 2:
            live = 0
            for nr in need_ranks:
                for s in range(3):
                    card_id = s * NUM_RANKS + nr
                    if card_id not in known:
                        live += 1
            if have > best_in_run or (have == best_in_run and live > best_outs):
                best_in_run = have
                best_outs = live
    return best_in_run, best_outs, DECK_SIZE - len(known)


# ── String/int conversion ────────────────────────────────────────────────────

_RANKS_STR = "23456789A"
_SUITS_STR = "dhs"


def _str_to_int(card_str):
    rank_ch = card_str[0]
    suit_ch = card_str[1]
    return _RANKS_STR.index(rank_ch) + _SUITS_STR.index(suit_ch) * NUM_RANKS


# ── Adaptive Opponent Discard Model ──────────────────────────────────────────


class _OppDiscardModel:
    """Adaptive linear model that learns opponent's discard feature weights."""

    _FEATURE_NAMES = [
        "pair_with_board", "pocket_pair", "flush_draw",
        "suited", "connected", "high_card", "board_suit_match",
    ]
    _NUM_FEATURES = len(_FEATURE_NAMES)
    _DEFAULTS = [10.0, 8.0, 6.0, 3.0, 2.0, 4.0, 3.0]
    _ALPHA = 0.15
    _WEIGHT_FLOOR = 0.5

    def __init__(self):
        self.weights = list(self._DEFAULTS)

    @staticmethod
    def _extract_features(keep, flop, _fr=None, _fs=None, _bsc=None):
        r0, r1 = _RANK[keep[0]], _RANK[keep[1]]
        s0, s1 = _SUIT[keep[0]], _SUIT[keep[1]]
        flop_ranks = _fr if _fr is not None else [_RANK[c] for c in flop]
        flop_suits = _fs if _fs is not None else [_SUIT[c] for c in flop]

        pair_with_board = 1.0 if (r0 in flop_ranks or r1 in flop_ranks) else 0.0
        pocket_pair = 1.0 if r0 == r1 else 0.0

        sc = [0, 0, 0]
        sc[s0] += 1
        sc[s1] += 1
        for fs in flop_suits:
            sc[fs] += 1
        flush_draw = 1.0 if max(sc) >= 4 else 0.0

        suited = 1.0 if s0 == s1 else 0.0
        gap = abs(r0 - r1)
        connected = 1.0 if gap <= 2 else 0.0
        high_card = max(r0, r1) / 8.0

        if _bsc is not None:
            bsc_arr = _bsc
        else:
            bsc_arr = [0, 0, 0]
            for fs in flop_suits:
                bsc_arr[fs] += 1
        bsm = 0
        for c in keep:
            if bsc_arr[_SUIT[c]] >= 2:
                bsm += 1
        bsm /= 2.0

        return [pair_with_board, pocket_pair, flush_draw,
                suited, connected, high_card, bsm]

    def score(self, keep, flop):
        feats = self._extract_features(keep, flop)
        return sum(w * f for w, f in zip(self.weights, feats))

    def score_precomputed(self, keep, flop_ranks, flop_suits, bsc):
        feats = self._extract_features(keep, None, flop_ranks, flop_suits, bsc)
        return sum(w * f for w, f in zip(self.weights, feats))

    def update(self, opp_kept, flop, opp_discards):
        if len(opp_kept) < 2 or len(opp_discards) < 3 or len(flop) < 3:
            return
        original5 = list(opp_kept) + list(opp_discards)
        if len(original5) < 5:
            return
        keeps_and_scores = []
        for i, j in combinations(range(len(original5)), 2):
            cand = [original5[i], original5[j]]
            sc = self.score(cand, flop)
            keeps_and_scores.append((cand, sc))
        keeps_and_scores.sort(key=lambda x: x[1], reverse=True)
        predicted_best = keeps_and_scores[0][0]
        chosen_set = set(opp_kept)
        predicted_set = set(predicted_best)
        if chosen_set == predicted_set:
            return
        chosen_feats = self._extract_features(list(opp_kept), flop)
        predicted_feats = self._extract_features(predicted_best, flop)
        for i in range(self._NUM_FEATURES):
            diff = chosen_feats[i] - predicted_feats[i]
            self.weights[i] += self._ALPHA * diff
            self.weights[i] = max(self._WEIGHT_FLOOR, self.weights[i])


# ── Module-level solver functions (for ProcessPoolExecutor pickling) ──────────


def _opp_keep_weight_lut(opp_hand, community, opp_discards, model_weights,
                         _flop_cache=None):
    original5 = list(opp_hand) + list(opp_discards)
    if len(original5) < 5:
        return 1.0
    if _flop_cache is not None:
        fr, fs, bsc = _flop_cache
    else:
        flop = community[:3]
        fr = [_RANK[c] for c in flop]
        fs = [_SUIT[c] for c in flop]
        bsc = [0, 0, 0]
        for s in fs:
            bsc[s] += 1
    keeps_and_scores = []
    for i, j in combinations(range(len(original5)), 2):
        cand = [original5[i], original5[j]]
        feats = _OppDiscardModel._extract_features(cand, None, fr, fs, bsc)
        sc = sum(w * f for w, f in zip(model_weights, feats))
        keeps_and_scores.append((frozenset([original5[i], original5[j]]), sc))
    keeps_and_scores.sort(key=lambda x: x[1], reverse=True)
    opp_set = frozenset(opp_hand)
    for rank_idx, (kset, _) in enumerate(keeps_and_scores):
        if kset == opp_set:
            if rank_idx == 0:
                return 1.0
            elif rank_idx == 1:
                return 0.3
            elif rank_idx == 2:
                return 0.1
            else:
                return 0.02
    return 0.02


def _exact_discard_equity_lut(my_keep, community, dead_cards):
    remaining = [c for c in range(DECK_SIZE) if c not in dead_cards]
    board_needed = 5 - len(community)
    community_l = list(community)
    wins = 0.0
    total = 0
    for runout in combinations(remaining, board_needed):
        board_5 = community_l + list(runout)
        mr = _lut_eval_7(list(my_keep) + board_5)
        runout_set = set(runout)
        opp_pool = [c for c in remaining if c not in runout_set]
        for opp in combinations(opp_pool, 2):
            orr = _lut_eval_7(list(opp) + board_5)
            if mr < orr:
                wins += 1.0
            elif mr == orr:
                wins += 0.5
            total += 1
    return wins / total if total > 0 else 0.5


def _exact_discard_equity_weighted_lut(my_keep, community, dead_cards,
                                       opp_discards, model_weights):
    remaining = [c for c in range(DECK_SIZE) if c not in dead_cards]
    board_needed = 5 - len(community)
    community_l = list(community)
    flop = community_l[:3]
    fr = [_RANK[c] for c in flop]
    fs = [_SUIT[c] for c in flop]
    bsc = [0, 0, 0]
    for s in fs:
        bsc[s] += 1
    flop_cache = (fr, fs, bsc)

    opp_weights = {}
    for opp in combinations(remaining, 2):
        w = _opp_keep_weight_lut(list(opp), community, opp_discards,
                                 model_weights, flop_cache)
        if w >= 0.01:
            opp_weights[opp] = w

    wins = 0.0
    total_weight = 0.0
    for runout in combinations(remaining, board_needed):
        board_5 = community_l + list(runout)
        mr = _lut_eval_7(list(my_keep) + board_5)
        runout_set = set(runout)
        opp_pool = [c for c in remaining if c not in runout_set]
        for opp in combinations(opp_pool, 2):
            entry = opp_weights.get(opp)
            if entry is None:
                continue
            orr = _lut_eval_7(list(opp) + board_5)
            if mr < orr:
                wins += entry
            elif mr == orr:
                wins += 0.5 * entry
            total_weight += entry
    return wins / total_weight if total_weight > 0 else 0.5


_NUM_CPUS = min(4, os.cpu_count() or 1)
_DISCARD_POOL_ENABLED = os.getenv("OMICRON_USE_DISCARD_POOL", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
_POOL = None
_discard_pool_fallback_logged = False


def _discard_mp_context():
    import multiprocessing as _mp

    if sys.platform == "win32":
        return _mp.get_context("spawn")
    try:
        return _mp.get_context("fork")
    except ValueError:
        return _mp.get_context("spawn")


def _discard_pool_reset() -> None:
    global _POOL
    p = _POOL
    _POOL = None
    if p is None:
        return
    try:
        p.shutdown(wait=False, cancel_futures=True)
    except (BrokenProcessPool, OSError, RuntimeError):
        pass


def _discard_pool() -> ProcessPoolExecutor | None:
    global _POOL
    if not _DISCARD_POOL_ENABLED:
        return None
    if _POOL is None:
        import multiprocessing as _mp

        _POOL = ProcessPoolExecutor(
            max_workers=_NUM_CPUS,
            mp_context=_discard_mp_context(),
        )
    return _POOL


def _run_discard_equities_parallel(fn, jobs: list[tuple]):
    global _discard_pool_fallback_logged
    pool = _discard_pool()
    if pool is None:
        return [fn(*a) for a in jobs]
    try:
        futs = [pool.submit(fn, *a) for a in jobs]
        return [f.result() for f in futs]
    except BrokenProcessPool:
        _discard_pool_reset()
        if not _discard_pool_fallback_logged:
            logging.getLogger(__name__).warning(
                "Discard solver process pool broke (worker OOM/crash); "
                "using in-process fallback. Set OMICRON_USE_DISCARD_POOL=0 to "
                "disable the pool."
            )
            _discard_pool_fallback_logged = True
        return [fn(*a) for a in jobs]


# ── PlayerAgent ──────────────────────────────────────────────────────────────


class PlayerAgent(Agent):

    _LIVE_STAGE = int(os.getenv("OMICRON_LIVE_STAGE", "1"))
    _SHADOW_ONLY = os.getenv("OMICRON_SHADOW_ONLY", "0") == "1"
    _DEBUG_VERBOSE = os.getenv("OMICRON_DEBUG_VERBOSE", "0") == "1"
    _LOG_EVENTS = os.getenv("OMICRON_LOG_EVENTS", "0") == "1"
    # Default low-volume logging for external validators with strict timeouts.
    _LOG_DECISIONS = os.getenv("OMICRON_LOG_DECISIONS", "0") == "1"
    _LOG_NODE_UPDATES = os.getenv("OMICRON_LOG_NODE_UPDATES", "off").strip().lower()
    _LOG_DISCARD_CANDIDATES = os.getenv("OMICRON_LOG_DISCARD_CANDIDATES", "0") == "1"
    _EVENT_BUFFER_LIMIT = 420

    _FACE_BET_NODES = tuple(
        f"face_bet_{s}_{b}"
        for s in ("flop", "turn", "river")
        for b in ("small", "medium", "large")
    )
    _BET_INIT_NODES = tuple(
        f"bet_initiative_{s}_{b}"
        for s in ("flop", "turn", "river")
        for b in ("small", "medium", "large")
    )
    _STRUCT_BINARY_NODES = (
        "raise_vs_bet_flop", "raise_vs_bet_turn", "raise_vs_bet_river",
        "check_raise_flop", "check_raise_turn", "check_raise_river",
        "stab_after_check_flop", "stab_after_check_turn", "stab_after_check_river",
        "barrel_turn_after_flop_aggr", "barrel_river_after_turn_aggr",
        "call_down_turn", "call_down_river",
    )
    _SHIFT_KEYS = ("shift_fold_pressure", "shift_aggression", "shift_sizing", "shift_showdown_honesty")
    _COEFF_CAPS = {
        "bluff_freq_adj": 0.12,
        "value_bet_size_adj": 0.18,
        "semi_bluff_freq_adj": 0.10,
        "bluff_catch_adj": 0.10,
        "hero_fold_adj": 0.10,
        "probe_freq_adj": 0.08,
        "delayed_barrel_adj": 0.08,
        "trap_freq_adj": 0.08,
        "preflop_pressure_adj": 0.08,
        "preflop_defense_adj": 0.08,
    }
    _COEFF_RELEVANCE = {
        "bluff_freq_adj": frozenset(["flop", "turn", "river"]),
        "value_bet_size_adj": frozenset(["flop", "turn", "river"]),
        "semi_bluff_freq_adj": frozenset(["flop", "turn"]),
        "bluff_catch_adj": frozenset(["turn", "river"]),
        "hero_fold_adj": frozenset(["turn", "river"]),
        "probe_freq_adj": frozenset(["flop", "turn"]),
        "delayed_barrel_adj": frozenset(["turn", "river"]),
        "trap_freq_adj": frozenset(["flop", "turn", "river"]),
        "preflop_pressure_adj": frozenset(["preflop"]),
        "preflop_defense_adj": frozenset(["preflop"]),
    }

    class BinaryNodeEstimator:
        def __init__(self, prior=0.5, alpha0=2.0, beta0=2.0, ema_decay=0.90, reliability=1.0):
            self.prior = prior
            self.alpha = alpha0 * prior
            self.beta = beta0 * (1.0 - prior)
            self.ema_decay = ema_decay
            self.reliability = reliability
            self.recent = prior
            self.recency_mass = 0.0
            self.samples = 0.0
            self.vol_ema = 0.0

        def observe(self, outcome):
            x = 1.0 if outcome else 0.0
            self.alpha += x
            self.beta += (1.0 - x)
            self.recent = self.ema_decay * self.recent + (1.0 - self.ema_decay) * x
            self.recency_mass = min(60.0, self.recency_mass * self.ema_decay + 1.0)
            self.samples += 1.0
            self.vol_ema = 0.85 * self.vol_ema + 0.15 * abs(x - self.recent)

        def metrics(self):
            life = self.alpha / max(1e-6, self.alpha + self.beta)
            rw = self.recency_mass / (self.recency_mass + 8.0)
            blend = life * (1.0 - rw) + self.recent * rw
            disagreement = abs(self.recent - life)
            sample_conf = self.samples / (self.samples + 10.0)
            recency_conf = min(1.0, self.recency_mass / 14.0)
            consistency = max(0.0, 1.0 - 2.0 * disagreement)
            conf = _clamp(0.45 * sample_conf + 0.30 * recency_conf + 0.25 * consistency, 0.0, 1.0)
            conf = conf * _clamp(self.reliability, 0.20, 1.0)
            vol = _clamp(0.65 * self.vol_ema + 0.35 * disagreement, 0.0, 1.0)
            return life, self.recent, blend, self.recency_mass, disagreement, vol, conf

    class ContinuousNodeEstimator:
        def __init__(self, prior=0.5, ema_decay=0.90, reliability=1.0):
            self.prior = prior
            self.ema_decay = ema_decay
            self.reliability = reliability
            self.sum_v = 0.0
            self.sum_sq = 0.0
            self.samples = 0.0
            self.recent = prior
            self.recency_mass = 0.0
            self.disp_ema = 0.0

        def observe(self, value):
            x = float(value)
            self.sum_v += x
            self.sum_sq += x * x
            self.samples += 1.0
            self.recent = self.ema_decay * self.recent + (1.0 - self.ema_decay) * x
            self.recency_mass = min(60.0, self.recency_mass * self.ema_decay + 1.0)
            self.disp_ema = 0.85 * self.disp_ema + 0.15 * abs(x - self.recent)

        def metrics(self):
            life = self.sum_v / max(1.0, self.samples)
            rw = self.recency_mass / (self.recency_mass + 8.0)
            blend = life * (1.0 - rw) + self.recent * rw
            disagreement = abs(self.recent - life)
            var = max(0.0, (self.sum_sq / max(1.0, self.samples)) - life * life)
            sample_conf = self.samples / (self.samples + 10.0)
            recency_conf = min(1.0, self.recency_mass / 14.0)
            consistency = max(0.0, 1.0 - min(1.0, disagreement * 2.0))
            conf = _clamp(0.45 * sample_conf + 0.30 * recency_conf + 0.25 * consistency, 0.0, 1.0)
            conf = conf * _clamp(self.reliability, 0.20, 1.0)
            vol = _clamp(0.50 * min(1.0, np.sqrt(var)) + 0.30 * self.disp_ema + 0.20 * disagreement, 0.0, 1.0)
            return life, self.recent, blend, self.recency_mass, disagreement, vol, conf

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.evaluator = _base_eval
        self._running_pnl = 0
        self._hands_completed = 0
        self._opp_model = _OppDiscardModel()
        self._last_community = []
        self._last_opp_discards = []
        self._last_my_cards = []
        self._opp_folded = False
        self._last_was_bet = False
        self._last_street = 0
        self._hand_idx = 0
        self._opp_raised_prev_street = False
        self._last_opp_aggr_street = 0
        self._last_preflop_raise = False
        self._last_after_shock = False
        self._we_folded = False
        self._event_store = []
        self._debug_snapshots = []
        self._line_state = {
            "we_checked_this_node": False,
            "we_bet_this_node": False,
            "villain_checked_this_node": False,
            "villain_had_initiative_prev_street": False,
            "initiative_carried_to_current_street": False,
            "initiative_owner_entering_street": "none",
            "last_aggressor_previous_street": "none",
            "is_continuation_of_prior_initiative": False,
        }

        self._binary_nodes = {}
        for node in self._FACE_BET_NODES:
            base_rel = 1.0 if "flop" in node else (0.85 if "turn" in node else 0.70)
            for a in ("fold", "call", "raise"):
                self._binary_nodes[f"{node}_{a}"] = self.BinaryNodeEstimator(
                    prior=0.34 if a != "raise" else 0.18,
                    reliability=base_rel
                )
        for node in self._BET_INIT_NODES:
            for a in ("bet",):
                self._binary_nodes[f"{node}_{a}"] = self.BinaryNodeEstimator(
                    prior=0.30,
                    reliability=0.80 if "river" not in node else 0.65
                )
        for node in self._STRUCT_BINARY_NODES:
            rare_rel = 0.65 if ("river" in node or "check_raise" in node) else 0.85
            self._binary_nodes[node] = self.BinaryNodeEstimator(
                prior=0.14 if ("check_raise" in node or "stab_" in node) else 0.30,
                reliability=rare_rel
            )

        self._continuous_nodes = {
            "bet_size_frac_flop": self.ContinuousNodeEstimator(prior=0.58, reliability=1.0),
            "bet_size_frac_turn": self.ContinuousNodeEstimator(prior=0.62, reliability=0.90),
            "bet_size_frac_river": self.ContinuousNodeEstimator(prior=0.72, reliability=0.78),
            "raise_size_frac_flop": self.ContinuousNodeEstimator(prior=0.66, reliability=0.95),
            "raise_size_frac_turn": self.ContinuousNodeEstimator(prior=0.72, reliability=0.88),
            "raise_size_frac_river": self.ContinuousNodeEstimator(prior=0.80, reliability=0.75),
            "river_bet_strength_small": self.ContinuousNodeEstimator(prior=0.55, reliability=0.55),
            "river_bet_strength_medium": self.ContinuousNodeEstimator(prior=0.58, reliability=0.52),
            "river_bet_strength_large": self.ContinuousNodeEstimator(prior=0.62, reliability=0.48),
            "river_raise_strength": self.ContinuousNodeEstimator(prior=0.66, reliability=0.42),
            "turn_barrel_strength": self.ContinuousNodeEstimator(prior=0.60, reliability=0.52),
            "check_raise_strength_flop": self.ContinuousNodeEstimator(prior=0.63, reliability=0.55),
            "check_raise_strength_turn": self.ContinuousNodeEstimator(prior=0.65, reliability=0.50),
            "opp_preflop_raise_rate": self.ContinuousNodeEstimator(prior=0.30, reliability=0.95),
        }

        self._exploit_state = {
            "bluff_freq_adj": 0.0,
            "value_bet_size_adj": 0.0,
            "semi_bluff_freq_adj": 0.0,
            "bluff_catch_adj": 0.0,
            "hero_fold_adj": 0.0,
            "probe_freq_adj": 0.0,
            "delayed_barrel_adj": 0.0,
            "trap_freq_adj": 0.0,
            "preflop_pressure_adj": 0.0,
            "preflop_defense_adj": 0.0,
        }
        self._shift_signals = {k: 0.0 for k in self._SHIFT_KEYS}
        self._regime_shift_score = 0.0
        self._last_profile = {}
        self._last_applied_adjustments = {}
        self._last_shadow_compare = {}
        self._decision_idx = 0
        self._last_opp_semantic_action = ""
        self._last_decision_snapshot = {}
        self._in_hand = False

    # ── Opponent profiling helpers ───────────────────────────────────────────

    def _safe_rate(self, key):
        if key == "fold_to_bet":
            p = self._last_profile.get("fold_medium_turn", 0.35)
            c = self._last_profile.get("conf_fold_medium_turn", 0.0)
            return p * c + 0.35 * (1.0 - c)
        if key == "call_down":
            p = self._last_profile.get("call_down_vs_bet_river", 0.40)
            c = self._last_profile.get("conf_call_down_vs_bet_river", 0.0)
            return p * c + 0.40 * (1.0 - c)
        return 0.35

    @staticmethod
    def _street_name(street):
        if street == 1:
            return "flop"
        if street == 2:
            return "turn"
        return "river"

    @staticmethod
    def _street_name_safe(street):
        if street == 0:
            return "preflop"
        if street == DISCARD:
            return "discard"
        if street == 1:
            return "flop"
        if street == 2:
            return "turn"
        if street == 3:
            return "river"
        return "unknown"

    @staticmethod
    def _action_name(action_id):
        m = {
            FOLD: "fold",
            CHECK: "check",
            CALL: "call",
            RAISE: "raise",
            DISCARD: "discard",
        }
        return m.get(int(action_id), "unknown")

    @staticmethod
    def _cards_to_str(cards):
        out = []
        for c in cards:
            try:
                out.append(PokerEnv.int_card_to_str(int(c)))
            except Exception:
                out.append(str(c))
        return out

    @staticmethod
    def _valid_actions_map(valid):
        return {
            "fold": bool(valid[FOLD]),
            "check": bool(valid[CHECK]),
            "call": bool(valid[CALL]),
            "raise": bool(valid[RAISE]),
            "discard": bool(valid[DISCARD]),
        }

    @staticmethod
    def _match_id():
        return os.getenv("MATCH_ID", "unknown")

    def _log_event(self, event_type, payload):
        if not self._LOG_EVENTS:
            return
        try:
            event = {
                "event_type": event_type,
                "match_id": self._match_id(),
                "hand_idx": int(self._hand_idx),
                "decision_idx": int(self._decision_idx),
                "street": int(payload.get("street", self._last_street)),
                "street_name": payload.get("street_name", self._street_name_safe(int(payload.get("street", self._last_street)))),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            event.update(payload)
            self.logger.info(json.dumps(event, default=str, separators=(",", ":")))
        except Exception:
            return

    def _should_log_node_update(self, key):
        mode = self._LOG_NODE_UPDATES
        if mode == "off":
            return False
        if mode == "all":
            return True
        core_prefixes = ("face_bet_", "raise_vs_bet_", "check_raise_", "stab_after_check_", "barrel_", "call_down_")
        return key.startswith(core_prefixes)

    @staticmethod
    def _size_bucket(to_call, pot):
        p = max(1.0, float(pot))
        ratio = float(to_call) / p
        if ratio < 0.40:
            return "small", ratio
        if ratio < 0.80:
            return "medium", ratio
        return "large", ratio

    @staticmethod
    def _bet_size_bucket(bet_amount, pot):
        p = max(1.0, float(pot))
        ratio = float(max(0.0, bet_amount)) / p
        if ratio < 0.40:
            return "small", ratio
        if ratio < 0.80:
            return "medium", ratio
        return "large", ratio

    @staticmethod
    def _classify_opp_action(action, line_we_bet, line_we_checked):
        if action in ("fold", "call", "check"):
            return action
        if action in ("raise", "bet"):
            if line_we_bet:
                return "raise_vs_bet"
            return "initiative_bet"
        return "unknown"

    def _record_event(self, street, node, size_bucket, our_prior_action, opp_action, pot, facing_frac=0.0, opp_amount=0, initiative=False):
        self._event_store.append({
            "hand_idx": self._hand_idx,
            "street": street,
            "node_id": node,
            "size_bucket": size_bucket,
            "pot_size": pot,
            "facing_size_frac": facing_frac,
            "our_prior_line": our_prior_action,
            "opp_action": opp_action,
            "opp_amount": opp_amount,
            "initiative_flag": bool(initiative),
            "showdown_later": False,
            "terminal_strength_bucket": None,
            "after_large_bankroll_swing": self._last_after_shock,
            "after_revealed_bluff": False,
            "after_revealed_value": False,
        })
        if len(self._event_store) > self._EVENT_BUFFER_LIMIT:
            self._event_store = self._event_store[-(self._EVENT_BUFFER_LIMIT - 120):]

    def _node_binary(self, key, cond):
        est = self._binary_nodes.get(key)
        if est is not None:
            pre = est.metrics()
            obs = bool(cond)
            est.observe(obs)
            if self._should_log_node_update(key):
                post = est.metrics()
                self._log_event("node_update", {
                    "street": self._last_street,
                    "node_updated": key,
                    "node_kind": "binary",
                    "observed_outcome": obs,
                    "pre_life_n": pre[0],
                    "post_life_n": post[0],
                    "pre_recent_n": pre[1],
                    "post_recent_n": post[1],
                    "pre_blend": pre[2],
                    "post_blend": post[2],
                    "pre_vol": pre[5],
                    "post_vol": post[5],
                    "pre_conf": pre[6],
                    "post_conf": post[6],
                })

    def _node_continuous(self, key, value, street):
        est = self._continuous_nodes.get(key)
        if est is None:
            return
        pre = est.metrics()
        x = float(value)
        est.observe(x)
        if self._should_log_node_update(key):
            post = est.metrics()
            self._log_event("node_update", {
                "street": street,
                "node_updated": key,
                "node_kind": "continuous",
                "observed_outcome": x,
                "pre_life_n": pre[0],
                "post_life_n": post[0],
                "pre_recent_n": pre[1],
                "post_recent_n": post[1],
                "pre_blend": pre[2],
                "post_blend": post[2],
                "pre_vol": pre[5],
                "post_vol": post[5],
                "pre_conf": pre[6],
                "post_conf": post[6],
            })

    def _update_regime_signal(self):
        def metric_bin(k):
            if k not in self._binary_nodes:
                return 0.0
            _, _, _, _, d, _, c = self._binary_nodes[k].metrics()
            return d * c

        def metric_cont(k):
            if k not in self._continuous_nodes:
                return 0.0
            _, _, _, _, d, _, c = self._continuous_nodes[k].metrics()
            return d * c

        fold_shift = np.mean([
            metric_bin("face_bet_flop_medium_fold"),
            metric_bin("face_bet_turn_medium_fold"),
            metric_bin("face_bet_river_medium_fold"),
        ])
        aggr_shift = np.mean([
            metric_bin("raise_vs_bet_flop"),
            metric_bin("raise_vs_bet_turn"),
            metric_bin("raise_vs_bet_river"),
            metric_bin("check_raise_turn"),
        ])
        sizing_shift = np.mean([
            metric_cont("bet_size_frac_flop"),
            metric_cont("bet_size_frac_turn"),
            metric_cont("raise_size_frac_turn"),
            metric_cont("raise_size_frac_river"),
        ])
        honesty_shift = np.mean([
            metric_cont("river_bet_strength_large"),
            metric_cont("river_raise_strength"),
            metric_cont("turn_barrel_strength"),
        ])
        self._shift_signals["shift_fold_pressure"] = 0.90 * self._shift_signals["shift_fold_pressure"] + 0.10 * float(fold_shift)
        self._shift_signals["shift_aggression"] = 0.90 * self._shift_signals["shift_aggression"] + 0.10 * float(aggr_shift)
        self._shift_signals["shift_sizing"] = 0.90 * self._shift_signals["shift_sizing"] + 0.10 * float(sizing_shift)
        self._shift_signals["shift_showdown_honesty"] = 0.90 * self._shift_signals["shift_showdown_honesty"] + 0.10 * float(honesty_shift)
        self._regime_shift_score = _clamp(float(np.mean(list(self._shift_signals.values()))), 0.0, 1.0)

    def _node_dist(self, node_prefix):
        out = {}
        for a in ("fold", "call", "raise"):
            k = f"{node_prefix}_{a}"
            _, _, blend, _, _, vol, conf = self._binary_nodes[k].metrics()
            out[a] = blend
            out[f"conf_{a}"] = conf
            out[f"vol_{a}"] = vol
        return out

    def _process_opponent_action(self, observation, opp_action, last_was_bet, last_street):
        if not opp_action:
            return
        if opp_action == "fold":
            self._opp_folded = True
        current_street = observation.get("street", 0)
        pot_obs = observation.get("pot_size", observation.get("opp_bet", 0) + observation.get("my_bet", 0))
        my_bet_obs = observation.get("my_bet", 0)
        opp_bet_obs = observation.get("opp_bet", 0)
        # gym_env exposes cumulative committed chips; use the post-action bet gap
        # as the incremental pressure amount for this node.
        to_call = max(0, opp_bet_obs - my_bet_obs)
        line_we_checked = bool(self._line_state.get("we_checked_this_node", False))
        line_we_bet = bool(self._line_state.get("we_bet_this_node", False))
        opp_sem = self._classify_opp_action(opp_action, line_we_bet, line_we_checked)
        self._last_opp_semantic_action = opp_sem
        inc_raise = to_call

        if last_was_bet and last_street >= 1:
            sname = self._street_name(last_street)
            bkt, ratio = self._size_bucket(to_call, pot_obs)
            base = f"face_bet_{sname}_{bkt}"
            self._node_binary(f"{base}_fold", opp_action == "fold")
            self._node_binary(f"{base}_call", opp_action == "call")
            self._node_binary(f"{base}_raise", opp_sem == "raise_vs_bet")
            self._record_event(last_street, base, bkt, "facing_bet", opp_action, pot_obs, ratio, inc_raise, initiative=False)
            rvb_key = f"raise_vs_bet_{sname}"
            self._node_binary(rvb_key, opp_sem == "raise_vs_bet")
            self._record_event(last_street, rvb_key, bkt, "vs_bet", opp_action, pot_obs, ratio, inc_raise, initiative=False)
            if last_street == 2:
                self._node_binary("call_down_turn", opp_action == "call")
            elif last_street == 3:
                self._node_binary("call_down_river", opp_action == "call")

        # True opportunity-based initiative-bet rates:
        # when we check and villain can act, record one opportunity.
        if last_street >= 1 and line_we_checked:
            sname = self._street_name(last_street)
            if opp_sem == "initiative_bet":
                bkt, ratio = self._bet_size_bucket(inc_raise, pot_obs)
            else:
                bkt, ratio = "small", 0.0
            for b in ("small", "medium", "large"):
                init_key = f"bet_initiative_{sname}_{b}"
                self._node_binary(f"{init_key}_bet", opp_sem == "initiative_bet" and b == bkt)
            if opp_sem == "initiative_bet":
                init_key = f"bet_initiative_{sname}_{bkt}"
                self._record_event(last_street, init_key, bkt, "opp_initiative", opp_action, pot_obs, ratio, inc_raise, initiative=True)

        if last_street >= 1:
            sname = self._street_name(last_street)
            # check_raise_* tracks true check-raise: villain checked earlier in
            # this node, then raised after we bet.
            cr_key = f"check_raise_{sname}"
            stab_key = f"stab_after_check_{sname}"
            if line_we_bet:
                is_true_check_raise = bool(self._line_state.get("villain_checked_this_node", False)) and opp_sem == "raise_vs_bet"
                self._node_binary(cr_key, is_true_check_raise)
                if is_true_check_raise:
                    self._record_event(last_street, cr_key, "na", "villain_checked_then_raised", opp_action, pot_obs, 0.0, inc_raise, initiative=True)
            if line_we_checked:
                self._node_binary(stab_key, opp_sem == "initiative_bet")
                if opp_sem == "initiative_bet":
                    self._record_event(last_street, stab_key, "na", "check_to_opp", opp_action, pot_obs, 0.0, inc_raise, initiative=True)

        if last_street == 2:
            cond = self._line_state.get("is_continuation_of_prior_initiative", False) and opp_sem == "initiative_bet"
            self._node_binary("barrel_turn_after_flop_aggr", cond)
            if self._line_state.get("is_continuation_of_prior_initiative", False):
                self._record_event(last_street, "barrel_turn_after_flop_aggr", "na", "after_flop_aggr", opp_action, pot_obs, 0.0, inc_raise, initiative=True)
        elif last_street == 3:
            cond = self._line_state.get("is_continuation_of_prior_initiative", False) and opp_sem == "initiative_bet"
            self._node_binary("barrel_river_after_turn_aggr", cond)
            if self._line_state.get("is_continuation_of_prior_initiative", False):
                self._record_event(last_street, "barrel_river_after_turn_aggr", "na", "after_turn_aggr", opp_action, pot_obs, 0.0, inc_raise, initiative=True)

        if opp_sem in ("initiative_bet", "raise_vs_bet"):
            street_name = self._street_name(last_street if last_street >= 1 else current_street)
            if opp_sem == "initiative_bet":
                if street_name in ("flop", "turn", "river"):
                    self._node_continuous(f"bet_size_frac_{street_name}", _clamp(inc_raise / max(1.0, pot_obs), 0.0, 2.5), last_street)
            else:
                if street_name in ("flop", "turn", "river"):
                    self._node_continuous(f"raise_size_frac_{street_name}", _clamp(inc_raise / max(1.0, pot_obs), 0.0, 2.5), last_street)

        if last_street == 0 and opp_action in ("raise", "call", "check", "fold"):
            self._node_continuous("opp_preflop_raise_rate", 1.0 if opp_action == "raise" else 0.0, last_street)

        if opp_sem in ("initiative_bet", "raise_vs_bet") and last_street >= 1:
            self._last_opp_aggr_street = last_street
        self._opp_raised_prev_street = (opp_sem in ("initiative_bet", "raise_vs_bet"))
        if opp_sem == "check":
            self._line_state["villain_checked_this_node"] = True
        self._line_state["last_aggressor_previous_street"] = "villain" if self._opp_raised_prev_street else "none"
        self._update_regime_signal()

    def _build_opponent_profile(self):
        profile = {}
        for sname in ("flop", "turn", "river"):
            for b in ("small", "medium", "large"):
                d = self._node_dist(f"face_bet_{sname}_{b}")
                for a in ("fold", "call", "raise"):
                    profile[f"{a}_{b}_{sname}"] = d[a]
                    profile[f"conf_{a}_{b}_{sname}"] = d[f"conf_{a}"]
                    profile[f"vol_{a}_{b}_{sname}"] = d[f"vol_{a}"]
                ik = f"bet_initiative_{sname}_{b}_bet"
                _, _, ib, _, _, iv, ic = self._binary_nodes[ik].metrics()
                profile[f"bet_initiative_{b}_{sname}"] = ib
                profile[f"conf_bet_initiative_{b}_{sname}"] = ic
                profile[f"vol_bet_initiative_{b}_{sname}"] = iv

        for k in (
            "raise_vs_bet_flop", "raise_vs_bet_turn", "raise_vs_bet_river",
            "check_raise_flop", "check_raise_turn", "check_raise_river",
            "stab_after_check_flop", "stab_after_check_turn", "stab_after_check_river",
            "barrel_turn_after_flop_aggr", "barrel_river_after_turn_aggr",
            "call_down_turn", "call_down_river",
        ):
            _, _, b, _, _, v, c = self._binary_nodes[k].metrics()
            out_key = k.replace("_after_flop_aggr", "").replace("_after_turn_aggr", "")
            profile[out_key] = b
            profile[f"conf_{out_key}"] = c
            profile[f"vol_{out_key}"] = v
        profile["call_down_vs_bet_turn"] = profile.get("call_down_turn", 0.0)
        profile["call_down_vs_bet_river"] = profile.get("call_down_river", 0.0)
        profile["conf_call_down_vs_bet_turn"] = profile.get("conf_call_down_turn", 0.0)
        profile["conf_call_down_vs_bet_river"] = profile.get("conf_call_down_river", 0.0)

        for ck in (
            "bet_size_frac_flop", "bet_size_frac_turn", "bet_size_frac_river",
            "raise_size_frac_flop", "raise_size_frac_turn", "raise_size_frac_river",
            "river_bet_strength_small", "river_bet_strength_medium", "river_bet_strength_large",
            "river_raise_strength", "turn_barrel_strength",
            "check_raise_strength_flop", "check_raise_strength_turn",
        ):
            _, _, b, _, _, v, c = self._continuous_nodes[ck].metrics()
            alias = ck.replace("bet_size_frac_", "avg_bet_size_").replace("raise_size_frac_", "avg_raise_size_")
            profile[alias] = b
            profile[f"conf_{alias}"] = c
            profile[f"vol_{alias}"] = v

        profile["style_volatility"] = _clamp(
            0.30 * profile.get("vol_raise_vs_bet_turn", 0.0)
            + 0.20 * profile.get("vol_check_raise_turn", 0.0)
            + 0.20 * profile.get("vol_avg_raise_size_river", 0.0)
            + 0.30 * self._regime_shift_score, 0.0, 1.0)
        profile["global_regime_shift"] = self._regime_shift_score
        for sk in self._SHIFT_KEYS:
            profile[sk] = self._shift_signals[sk]

        # Helper aggregates (explicitly derived only)
        profile["fold_medium_bet_overall"] = float(np.mean([
            profile.get("fold_medium_flop", 0.35),
            profile.get("fold_medium_turn", 0.35),
            profile.get("fold_medium_river", 0.35),
        ]))
        profile["raise_rate_overall"] = float(np.mean([
            profile.get("raise_medium_flop", 0.18),
            profile.get("raise_medium_turn", 0.18),
            profile.get("raise_medium_river", 0.18),
        ]))
        self._last_profile = profile
        return profile

    def _compute_exploit_adjustments(self, profile, live_stage=1):
        raw = {
            "bluff_freq_adj": (
                0.45 * (profile.get("fold_medium_flop", 0.35) - 0.35)
                + 0.40 * (profile.get("fold_medium_turn", 0.35) - 0.35)
                + 0.25 * (profile.get("fold_large_river", 0.35) - 0.32)
                - 0.45 * (profile.get("call_down_vs_bet_river", 0.40) - 0.40)
            ),
            "value_bet_size_adj": (
                0.55 * (profile.get("call_down_vs_bet_turn", 0.42) - 0.42)
                + 0.50 * (profile.get("call_down_vs_bet_river", 0.40) - 0.40)
                - 0.40 * (profile.get("fold_large_turn", 0.34) - 0.34)
                - 0.35 * (profile.get("fold_large_river", 0.34) - 0.34)
            ),
            "semi_bluff_freq_adj": (
                0.40 * (profile.get("fold_medium_flop", 0.35) - 0.35)
                + 0.40 * (profile.get("fold_medium_turn", 0.35) - 0.35)
                - 0.35 * (profile.get("raise_vs_bet_turn", 0.20) - 0.20)
                - 0.30 * (profile.get("barrel_river", 0.18) - 0.18)
            ),
            "bluff_catch_adj": (
                0.60 * (0.55 - profile.get("river_raise_strength", 0.55))
                + 0.40 * (0.58 - profile.get("river_bet_strength_large", 0.58))
                + 0.20 * (profile.get("barrel_river", 0.18) - 0.18)
            ),
            "hero_fold_adj": (
                0.62 * (profile.get("river_raise_strength", 0.60) - 0.60)
                + 0.38 * (profile.get("river_bet_strength_large", 0.60) - 0.60)
                + 0.25 * (profile.get("raise_vs_bet_river", 0.22) - 0.22)
            ),
            "probe_freq_adj": 0.40 * (profile.get("stab_after_check_flop", 0.20) - 0.20),
            "delayed_barrel_adj": 0.35 * (profile.get("barrel_turn", 0.22) - 0.22),
            "trap_freq_adj": 0.30 * (profile.get("check_raise_turn", 0.14) - 0.14),
            "preflop_pressure_adj": (
                0.30 * (profile.get("fold_medium_flop", 0.35) - 0.35)
                - 0.30 * (profile.get("check_raise_flop", 0.12) - 0.12)
            ),
            "preflop_defense_adj": (
                0.45 * (profile.get("raise_vs_bet_flop", 0.20) - 0.20)
                + 0.25 * (profile.get("call_down_vs_bet_turn", 0.40) - 0.40)
            ),
        }

        stability_factor = _clamp(1.0 - 0.50 * profile.get("style_volatility", 0.0) - 0.50 * profile.get("global_regime_shift", 0.0), 0.15, 1.0)
        out = {}
        for k, x in raw.items():
            cap = self._COEFF_CAPS[k]
            if k in ("bluff_catch_adj", "hero_fold_adj"):
                conf = float(np.mean([
                    profile.get("conf_river_raise_strength", 0.0),
                    profile.get("conf_river_bet_strength_large", 0.0),
                ]))
            elif k in ("bluff_freq_adj", "semi_bluff_freq_adj"):
                conf = float(np.mean([
                    profile.get("conf_fold_medium_flop", 0.0),
                    profile.get("conf_fold_medium_turn", 0.0),
                    profile.get("conf_fold_large_river", 0.0),
                ]))
            else:
                conf = _clamp(0.35 + 0.40 * profile.get("conf_call_down_vs_bet_turn", 0.0) + 0.25 * profile.get("conf_raise_vs_bet_turn", 0.0), 0.0, 1.0)
            confidence_factor = _clamp(conf, 0.0, 1.0)
            effective = _clamp(x * confidence_factor * stability_factor, -cap, cap)
            if profile.get("global_regime_shift", 0.0) > 0.35:
                effective *= 0.75
            if k in ("bluff_catch_adj", "hero_fold_adj"):
                effective *= 0.80
            stale = 0.95
            self._exploit_state[k] = stale * (0.90 * self._exploit_state[k] + 0.10 * effective)
            out[k] = self._exploit_state[k]

        # Stage gate: keep preflop exploit disabled until final stage
        if live_stage < 4:
            out["preflop_pressure_adj"] = 0.0
            out["preflop_defense_adj"] = 0.0
            self._exploit_state["preflop_pressure_adj"] = 0.0
            self._exploit_state["preflop_defense_adj"] = 0.0

        self._last_applied_adjustments = dict(out)
        return out

    # ── Observe + showdown learning ──────────────────────────────────────────

    def observe(self, observation, reward, terminated, truncated, info):
        raw = observation.get("opp_last_action", "")
        opp_action = _normalize_action(raw)
        if opp_action == "fold":
            self._opp_folded = True

        if terminated:
            pot_final = int(observation.get("pot_size", observation.get("my_bet", 0) + observation.get("opp_bet", 0)))
            my_cards_now = [c for c in observation.get("my_cards", []) if c != -1]
            board_now = [c for c in observation.get("community_cards", []) if c != -1]
            opp_disc = [c for c in observation.get("opp_discarded_cards", []) if c != -1]
            self._running_pnl += int(reward)
            self._hands_completed += 1
            self._last_after_shock = abs(int(reward)) >= 20

            self._log_event("bankroll_progress", {
                "street": observation.get("street", self._last_street),
                "running_pnl": int(self._running_pnl),
                "reward_delta": int(reward),
                "hand_number": int(info.get("hand_number", self._hand_idx)),
                "went_to_showdown": bool(info.get("player_0_cards") and info.get("player_1_cards")),
                "hero_folded": bool(self._we_folded),
                "villain_folded": bool(self._opp_folded),
                "pot_size_final": pot_final,
                "my_cards": my_cards_now,
                "my_cards_str": self._cards_to_str(my_cards_now),
                "community": board_now,
                "community_str": self._cards_to_str(board_now),
                "opp_discards": opp_disc,
                "opp_discards_str": self._cards_to_str(opp_disc),
            })

            self._learn_from_showdown(observation, info)

            self._opp_folded = False
            self._we_folded = False
            self._last_was_bet = False
            self._last_street = 0
            self._opp_raised_prev_street = False
            self._line_state["we_checked_this_node"] = False
            self._line_state["we_bet_this_node"] = False
            self._line_state["villain_checked_this_node"] = False
            self._line_state["initiative_owner_entering_street"] = "none"
            self._line_state["last_aggressor_previous_street"] = "none"
            self._line_state["is_continuation_of_prior_initiative"] = False
            self._hand_idx += 1
            self._in_hand = False

    @staticmethod
    def _showdown_bucket_value(hand_cat, board_len):
        if hand_cat == "trips_plus":
            return 1.00 if board_len >= 5 else 0.75
        if hand_cat == "two_pair":
            return 0.75
        if hand_cat == "one_pair":
            return 0.50
        if hand_cat == "nothing":
            return 0.00
        return 0.25

    def _learn_from_showdown(self, observation, info):
        p0_cards = info.get("player_0_cards")
        p1_cards = info.get("player_1_cards")
        if not p0_cards or not p1_cards:
            return
        if len(self._last_opp_discards) < 3 or len(self._last_community) < 3:
            return
        try:
            p0_ints = [_str_to_int(c) for c in p0_cards]
            p1_ints = [_str_to_int(c) for c in p1_cards]
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
        if len(self._last_community) >= 5:
            river_board = self._last_community[:5]
            cat = _hand_rank_category(opp_kept[:2], river_board)
            strength = self._showdown_bucket_value(cat, len(river_board))
            revealed_bluff = strength <= 0.25
            my_cat = _hand_rank_category(self._last_my_cards[:2], river_board) if len(self._last_my_cards) >= 2 else "unknown"
            try:
                my_rank = int(_lut_eval_7(self._last_my_cards[:2] + river_board)) if len(self._last_my_cards) >= 2 else -1
                opp_rank = int(_lut_eval_7(opp_kept[:2] + river_board))
            except Exception:
                my_rank = -1
                opp_rank = -1
            winner = "tie"
            if my_rank >= 0 and opp_rank >= 0:
                if my_rank < opp_rank:
                    winner = "hero"
                elif opp_rank < my_rank:
                    winner = "villain"
            self._log_event("showdown_result", {
                "street": 3,
                "went_to_showdown": True,
                "my_final_cards": self._last_my_cards[:2],
                "my_final_cards_str": self._cards_to_str(self._last_my_cards[:2]),
                "opp_final_cards": opp_kept[:2],
                "opp_final_cards_str": self._cards_to_str(opp_kept[:2]),
                "board": river_board,
                "board_str": self._cards_to_str(river_board),
                "my_hand_cat": my_cat,
                "opp_hand_cat": cat,
                "my_rank": my_rank,
                "opp_rank": opp_rank,
                "winner": winner,
                "pot_size_final": int(observation.get("pot_size", observation.get("my_bet", 0) + observation.get("opp_bet", 0))),
                "hero_equity_before_showdown": self._last_decision_snapshot.get("equity"),
                "hero_decision_before_showdown": self._last_decision_snapshot.get("action_name"),
            })
            self._log_event("showdown_interpretation", {
                "street": 3,
                "opp_strength_bucket": float(strength),
                "revealed_bluff": bool(revealed_bluff),
                "revealed_value": bool(strength >= 0.75),
                "hero_equity_before_showdown": self._last_decision_snapshot.get("equity"),
                "prior_action_context": self._last_decision_snapshot,
            })
            for i in range(len(self._event_store) - 1, -1, -1):
                ev = self._event_store[i]
                if ev["hand_idx"] != self._hand_idx:
                    continue
                ev["showdown_later"] = True
                ev["terminal_strength_bucket"] = strength
                ev["after_revealed_bluff"] = revealed_bluff
                ev["after_revealed_value"] = strength >= 0.75
                node = ev.get("node_id", "")
                s_bucket = ev.get("size_bucket", "medium")
                if node.startswith("bet_initiative_river_") and ev.get("opp_action") in ("bet", "raise"):
                    self._node_continuous(f"river_bet_strength_{s_bucket}", strength, 3)
                elif node == "raise_vs_bet_river" and ev.get("opp_action") == "raise":
                    self._node_continuous("river_raise_strength", strength, 3)
                elif node == "barrel_turn_after_flop_aggr" and ev.get("opp_action") in ("bet", "raise"):
                    self._node_continuous("turn_barrel_strength", strength, 2)
                elif node == "check_raise_flop" and ev.get("opp_action") == "raise":
                    self._node_continuous("check_raise_strength_flop", strength, 1)
                elif node == "check_raise_turn" and ev.get("opp_action") == "raise":
                    self._node_continuous("check_raise_strength_turn", strength, 2)

    def __name__(self):
        return "OMICRON_V1"

    # ── MC Equity (random range — for discard screening + preflop) ───────────

    def _compute_equity(self, my_cards, community, opp_discards, my_discards, num_sims=300):
        dead = set(my_cards)
        for c in community:
            if c != -1:
                dead.add(c)
        for c in opp_discards:
            if c != -1:
                dead.add(c)
        for c in my_discards:
            if c != -1:
                dead.add(c)

        remaining = [i for i in range(DECK_SIZE) if i not in dead]
        board_needed = 5 - len(community)
        sample_size = 2 + board_needed

        if sample_size > len(remaining):
            return 0.5

        community_l = list(community)
        my_keep = list(my_cards)

        wins = 0
        total = 0
        for _ in range(num_sims):
            sample = random.sample(remaining, sample_size)
            opp_cards = sample[:2]
            board_5 = community_l + sample[2:]

            my_rank = _lut_eval_7(my_keep + board_5)
            opp_rank = _lut_eval_7(opp_cards + board_5)
            if my_rank < opp_rank:
                wins += 1
            elif my_rank == opp_rank:
                wins += 0.5
            total += 1

        return wins / total if total > 0 else 0.5

    # ── Range-weighted MC Equity (for postflop decisions) ────────────────────

    def _compute_equity_ranged(self, my2, community, dead, opp_discards,
                               passive_signal, aggr_signal, num_sims=300):
        remaining = [i for i in range(DECK_SIZE) if i not in dead]
        board_needed = 5 - len(community)
        sample_size = 2 + board_needed
        if sample_size > len(remaining):
            return 0.5

        have_discards = len(opp_discards) >= 3
        flop = community[:3] if len(community) >= 3 else []

        flop_cache = None
        if have_discards and flop:
            fr = [_RANK[c] for c in flop]
            fs = [_SUIT[c] for c in flop]
            bsc = [0, 0, 0]
            for s in fs:
                bsc[s] += 1
            flop_cache = (fr, fs, bsc)

        reject_nothing = aggr_signal >= 2.0
        reject_one_pair = aggr_signal >= 3.2
        max_retries = 3 if reject_nothing else 0

        community_l = list(community)
        my_keep = list(my2)

        opp_weight_map = {}
        if have_discards and flop_cache:
            for pair in combinations(remaining, 2):
                w = self._opp_keep_weight(list(pair), community, opp_discards, flop_cache)
                if w >= 0.01:
                    opp_weight_map[pair] = w

        wins = 0.0
        total_weight = 0.0

        for _ in range(num_sims):
            sample = random.sample(remaining, sample_size)
            opp = sample[:2]
            runout = sample[2:]

            if max_retries > 0 and len(community) >= 3:
                for _retry in range(max_retries):
                    cat = _hand_rank_category(list(opp), community)
                    if cat == "nothing" or (reject_one_pair and cat == "one_pair"):
                        sample = random.sample(remaining, sample_size)
                        opp = sample[:2]
                        runout = sample[2:]
                    else:
                        break

            w = 1.0
            if have_discards and flop_cache:
                key = (opp[0], opp[1]) if opp[0] < opp[1] else (opp[1], opp[0])
                w = opp_weight_map.get(key, 0.0)
                if w < 0.01:
                    continue
                if passive_signal > 2.0:
                    cat = _hand_rank_category(list(opp), community)
                    if cat in ("nothing",):
                        w *= 0.75

            board_5 = community_l + runout
            my_rank = _lut_eval_7(my_keep + board_5)
            opp_rank = _lut_eval_7(opp + board_5)
            if my_rank < opp_rank:
                wins += w
            elif my_rank == opp_rank:
                wins += 0.5 * w
            total_weight += w

        return wins / total_weight if total_weight > 0 else 0.5

    # ── Preflop equity ───────────────────────────────────────────────────────

    def _preflop_equity(self, my5, dead):
        scores = []
        for i, j in combinations(range(len(my5)), 2):
            keep = [my5[i], my5[j]]
            toss = [my5[k] for k in range(len(my5)) if k not in (i, j)]
            eq = self._compute_equity(keep, [], [], toss, num_sims=80)
            scores.append(eq)
        scores.sort(reverse=True)
        top = scores[:3]
        return sum(top) / len(top) if top else 0.45

    # ── Exact discard methods (subgame solver) ───────────────────────────────

    def _exact_discard_equity(self, my_keep, community, dead_cards):
        return _exact_discard_equity_lut(my_keep, community, dead_cards)

    def _opp_keep_weight(self, opp_hand, community, opp_discards, _flop_cache=None):
        return _opp_keep_weight_lut(opp_hand, community, opp_discards,
                                    self._opp_model.weights, _flop_cache)

    def _exact_discard_equity_weighted(self, my_keep, community, dead_cards, opp_discards):
        return _exact_discard_equity_weighted_lut(
            my_keep, community, dead_cards, opp_discards,
            self._opp_model.weights)

    # ── Postflop helper systems ──────────────────────────────────────────────

    def _street_adjust(self, equity, street, has_draw, flush_outs, straight_outs,
                       hand_cat="unknown", my_cards=None, community=None,
                       opp_discards=None):
        outs = max(flush_outs, straight_outs)
        if street == 1 and has_draw and outs > 0:
            equity += min(0.10, outs * 0.025)
        elif street == 3 and has_draw and hand_cat in ("nothing", "one_pair"):
            equity -= 0.10

        if my_cards and community:
            equity += _board_monotone_penalty(my_cards, community)
            equity += _board_connected_penalty(my_cards, community)
        if community and opp_discards:
            equity += _opp_flush_inference(community, opp_discards)

        return _clamp(equity, 0.0, 0.98)

    @staticmethod
    def _cat_to_strength(hand_cat, has_draw, my_cards=None, community=None):
        if hand_cat == "trips_plus":
            return "monster"
        if hand_cat == "two_pair":
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
                          exploit_adj, has_draw=None, flush_outs_v=None, straight_outs_v=None,
                          rand_ctx=None):
        if street not in (1, 2):
            return False, None
        if not (valid[RAISE] and max_raise >= min_raise):
            return False, None

        if has_draw is not None:
            if not has_draw:
                return False, None
            f_outs = flush_outs_v if flush_outs_v is not None else 0
            s_outs = straight_outs_v if straight_outs_v is not None else 0
        else:
            sc, f_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
            _, s_outs, _ = _count_straight_outs(my_cards, community, opp_discards, my_discards)
            if not ((sc >= 4 and f_outs >= 2) or (s_outs >= 3)):
                return False, None

        outs = max(f_outs, s_outs)
        if outs < 2:
            return False, None
        if to_call > pot_size * 0.40:
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
            miss_one = max(0, remaining - outs) / remaining
            miss_two = max(0, remaining - outs - 1) / max(1, remaining - 1)
            draw_eq = 1.0 - miss_one * miss_two

        sname = self._street_name(street)
        fold_prob = self._last_profile.get(f"fold_medium_{sname}", self._safe_rate("fold_to_bet"))
        sb_adj = self._coeff(exploit_adj, "semi_bluff_freq_adj", street)
        fold_prob = _clamp(fold_prob + 0.55 * sb_adj, 0.10, 0.85)
        pot = max(pot_size, 1)

        if fold_prob > 0.48:
            sizing_frac = rand_ctx["semi_bluff_high_frac"] if rand_ctx else random.uniform(0.65, 0.80)
        elif fold_prob < 0.25:
            sizing_frac = rand_ctx["semi_bluff_low_frac"] if rand_ctx else random.uniform(0.45, 0.60)
        else:
            sizing_frac = rand_ctx["semi_bluff_mid_frac"] if rand_ctx else random.uniform(0.55, 0.70)

        bet_size = pot * sizing_frac
        ev = (fold_prob * pot
              + (1 - fold_prob) * (draw_eq * (pot + 2 * bet_size) - bet_size))

        if ev > 0:
            amt = _clamp(int(bet_size), min_raise, max_raise)
            return True, (RAISE, amt, 0, 0)
        return False, None

    def _dynamic_sizing(self, base_amount, strength, street, is_semi_bluff, exploit_adj):
        mult = 1.0
        bluff_freq = self._coeff(exploit_adj, "bluff_freq_adj", street)
        semi_bluff = self._coeff(exploit_adj, "semi_bluff_freq_adj", street)
        value_size = self._coeff(exploit_adj, "value_bet_size_adj", street)
        if is_semi_bluff or strength in ("draw", "weak"):
            mult += bluff_freq * 0.35
            mult += semi_bluff * 0.35
        else:
            mult += value_size * (0.55 if street >= 2 else 0.40)
        mult = _clamp(mult, 0.82, 1.22)
        return max(1, int(base_amount * mult))

    def _coeff(self, exploit_adj, coeff_key, street):
        street_name = "preflop" if street == 0 else self._street_name(street)
        if street_name not in self._COEFF_RELEVANCE.get(coeff_key, frozenset()):
            return 0.0
        if self._SHADOW_ONLY:
            return 0.0
        if self._LIVE_STAGE <= 0:
            return 0.0
        if self._LIVE_STAGE == 1 and coeff_key not in ("value_bet_size_adj",):
            return 0.0
        if self._LIVE_STAGE == 2 and coeff_key in ("probe_freq_adj", "delayed_barrel_adj", "trap_freq_adj", "preflop_pressure_adj", "preflop_defense_adj"):
            return 0.0
        if self._LIVE_STAGE == 3 and coeff_key in ("preflop_pressure_adj", "preflop_defense_adj"):
            return 0.0
        return exploit_adj.get(coeff_key, 0.0)

    @staticmethod
    def _action_category(action_tuple):
        if action_tuple[0] == FOLD:
            return "fold"
        if action_tuple[0] == CALL:
            return "call"
        if action_tuple[0] == RAISE:
            return "aggressive"
        return "check"

    # ── Main act() ───────────────────────────────────────────────────────────

    def act(self, observation, reward, terminated, truncated, info):
        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards = [c for c in observation["my_discarded_cards"] if c != -1]
        valid = observation["valid_actions"]
        street = observation["street"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        prev_street = self._last_street

        self._last_my_cards = list(my_cards)
        if opp_discards:
            self._last_opp_discards = list(opp_discards)
        if community:
            self._last_community = list(community)
        pot_size = observation.get("pot_size", my_bet + opp_bet)
        is_new_hand = (street == 0 and len(community) == 0 and my_bet <= 2 and opp_bet <= 2)
        if is_new_hand and not self._in_hand:
            self._decision_idx = 0
            self._in_hand = True
            self._we_folded = False

        in_early_phase = self._hands_completed < EARLY_PHASE_HANDS

        time_left = observation.get("time_left", 1000.0)
        hands_left = max(1, TOTAL_HANDS - self._hands_completed)
        budget_per_hand = time_left / hands_left
        if budget_per_hand < 0.15:
            sim_mode = "emergency"
        elif budget_per_hand < 0.50:
            sim_mode = "conservative"
        else:
            sim_mode = "full"

        raw = observation.get("opp_last_action", "")
        opp_action = _normalize_action(raw)
        if street != prev_street:
            self._line_state["villain_checked_this_node"] = False
        if opp_action:
            if street != prev_street:
                self._line_state["initiative_owner_entering_street"] = self._line_state.get("last_aggressor_previous_street", "none")
                self._line_state["is_continuation_of_prior_initiative"] = (
                    self._line_state.get("initiative_owner_entering_street") == "villain"
                )
            self._process_opponent_action(observation, opp_action,
                                          self._last_was_bet, self._last_street)
        profile = self._build_opponent_profile()
        exploit_adj = self._compute_exploit_adjustments(profile, live_stage=self._LIVE_STAGE)
        forced_lock_action = None

        # ── Bleed-out lock ───────────────────────────────────────────────────
        if not valid[DISCARD]:
            hands_remaining = max(0, TOTAL_HANDS - self._hands_completed)
            sb_left = (hands_remaining + 1) // 2
            bb_left = hands_remaining // 2
            max_bleed = sb_left * 1 + bb_left * 2

            if self._running_pnl > max_bleed:
                if valid[FOLD]:
                    forced_lock_action = (FOLD, 0, 0, 0)
                elif valid[CHECK]:
                    forced_lock_action = (CHECK, 0, 0, 0)

        # ── Discard phase ────────────────────────────────────────────────────
        if valid[DISCARD]:
            self._last_community = list(community)
            self._last_opp_discards = list(opp_discards)

            dead_base = set(my_cards) | set(c for c in community if c != -1)
            have_opp_discards = len(opp_discards) >= 3

            all_keeps = []
            for i, j in combinations(range(len(my_cards)), 2):
                keep = [my_cards[i], my_cards[j]]
                toss = [my_cards[k] for k in range(len(my_cards)) if k not in (i, j)]
                all_keeps.append((i, j, keep, toss))

            if have_opp_discards:
                if sim_mode == "emergency":
                    opp_disc_set = set(opp_discards)
                    community_l = list(community)
                    board_needed = 5 - len(community)
                    ss = 2 + board_needed
                    best_eq = -1.0
                    best_ij = (0, 1)
                    for i, j, keep, toss in all_keeps:
                        per_dead = dead_base | set(toss) | opp_disc_set
                        per_remaining = [c for c in range(DECK_SIZE) if c not in per_dead]
                        if ss > len(per_remaining):
                            eq = 0.5
                        else:
                            w = 0
                            t = 0
                            for _ in range(100):
                                samp = random.sample(per_remaining, ss)
                                board_5 = community_l + samp[2:]
                                mr = _lut_eval_7(list(keep) + board_5)
                                orr = _lut_eval_7(samp[:2] + board_5)
                                if mr < orr:
                                    w += 1
                                elif mr == orr:
                                    w += 0.5
                                t += 1
                            eq = w / t if t else 0.5
                        if eq > best_eq:
                            best_eq = eq
                            best_ij = (i, j)
                    if self._LOG_DISCARD_CANDIDATES:
                        for rank_i, (i, j, keep, toss) in enumerate(all_keeps, start=1):
                            self._log_event("discard_candidate", {
                                "street": street,
                                "street_name": "discard",
                                "phase": "discard",
                                "candidate_keep": keep,
                                "candidate_keep_str": self._cards_to_str(keep),
                                "candidate_toss": toss,
                                "candidate_toss_str": self._cards_to_str(toss),
                                "screen_eq": None,
                                "exact_eq": None,
                                "weighted_eq": None,
                                "rank_among_candidates": rank_i,
                                "selected": bool((i, j) == best_ij),
                            })
                else:
                    weights = list(self._opp_model.weights)
                    jobs_w = []
                    for i, j, keep, toss in all_keeps:
                        dead = dead_base | set(toss) | set(opp_discards)
                        jobs_w.append(
                            (keep, community, dead, opp_discards, weights)
                        )
                    results_w = _run_discard_equities_parallel(
                        _exact_discard_equity_weighted_lut, jobs_w
                    )
                    best_eq = -1.0
                    best_ij = (0, 1)
                    weighted_map = {}
                    for (i, j, keep, toss), eq in zip(all_keeps, results_w):
                        eqf = float(eq)
                        weighted_map[(i, j)] = eqf
                        if eqf > best_eq:
                            best_eq = eqf
                            best_ij = (i, j)
                    if self._LOG_DISCARD_CANDIDATES:
                        ranked_weighted = sorted(
                            [(i, j, keep, toss, weighted_map.get((i, j))) for i, j, keep, toss in all_keeps],
                            key=lambda x: x[4] if x[4] is not None else -1.0,
                            reverse=True
                        )
                        for rank_i, (i, j, keep, toss, weq) in enumerate(ranked_weighted, start=1):
                            self._log_event("discard_candidate", {
                                "street": street,
                                "street_name": "discard",
                                "phase": "discard",
                                "candidate_keep": keep,
                                "candidate_keep_str": self._cards_to_str(keep),
                                "candidate_toss": toss,
                                "candidate_toss_str": self._cards_to_str(toss),
                                "screen_eq": None,
                                "exact_eq": None,
                                "weighted_eq": weq,
                                "rank_among_candidates": rank_i,
                                "selected": bool((i, j) == best_ij),
                            })
            else:
                opp_disc_set = set(opp_discards)
                community_l = list(community)
                board_needed = 5 - len(community)
                screen_sample_size = 2 + board_needed

                if sim_mode == "emergency":
                    mc_sims = 100
                elif sim_mode == "conservative":
                    mc_sims = 300
                else:
                    mc_sims = 500

                candidates = []
                for i, j, keep, toss in all_keeps:
                    per_dead = dead_base | set(toss) | opp_disc_set
                    per_remaining = [c for c in range(DECK_SIZE) if c not in per_dead]
                    if screen_sample_size > len(per_remaining):
                        candidates.append((i, j, 0.5, keep, toss))
                        continue

                    w = 0
                    t = 0
                    for _ in range(mc_sims):
                        samp = random.sample(per_remaining, screen_sample_size)
                        board_5 = community_l + samp[2:]
                        mr = _lut_eval_7(list(keep) + board_5)
                        orr = _lut_eval_7(samp[:2] + board_5)
                        if mr < orr:
                            w += 1
                        elif mr == orr:
                            w += 0.5
                        t += 1
                    candidates.append((i, j, w / t if t else 0.5, keep, toss))

                candidates.sort(key=lambda c: c[2], reverse=True)

                if sim_mode == "emergency":
                    best_ij = (candidates[0][0], candidates[0][1])
                    if self._LOG_DISCARD_CANDIDATES:
                        for rank_i, (i, j, s_eq, keep, toss) in enumerate(candidates, start=1):
                            self._log_event("discard_candidate", {
                                "street": street,
                                "street_name": "discard",
                                "phase": "discard",
                                "candidate_keep": keep,
                                "candidate_keep_str": self._cards_to_str(keep),
                                "candidate_toss": toss,
                                "candidate_toss_str": self._cards_to_str(toss),
                                "screen_eq": float(s_eq),
                                "exact_eq": None,
                                "weighted_eq": float(s_eq),
                                "rank_among_candidates": rank_i,
                                "selected": bool((i, j) == best_ij),
                            })
                else:
                    top_n = 3 if sim_mode == "conservative" else 5
                    jobs_u = []
                    ij_top = []
                    for i, j, _, keep, toss in candidates[:top_n]:
                        dead = dead_base | set(toss) | opp_disc_set
                        ij_top.append((i, j))
                        jobs_u.append((keep, community, dead))
                    results_u = _run_discard_equities_parallel(
                        _exact_discard_equity_lut, jobs_u
                    )
                    best_eq = -1.0
                    best_ij = (0, 1)
                    exact_map = {}
                    for (i, j), eq in zip(ij_top, results_u):
                        eqf = float(eq)
                        exact_map[(i, j)] = eqf
                        if eqf > best_eq:
                            best_eq = eqf
                            best_ij = (i, j)
                    if self._LOG_DISCARD_CANDIDATES:
                        ranked = sorted(
                            [(i, j, keep, toss, exact_map.get((i, j))) for i, j, _, keep, toss in candidates[:top_n]],
                            key=lambda x: x[4] if x[4] is not None else -1.0,
                            reverse=True
                        )
                        for rank_i, (i, j, keep, toss, xeq) in enumerate(ranked, start=1):
                            self._log_event("discard_candidate", {
                                "street": street,
                                "street_name": "discard",
                                "phase": "discard",
                                "candidate_keep": keep,
                                "candidate_keep_str": self._cards_to_str(keep),
                                "candidate_toss": toss,
                                "candidate_toss_str": self._cards_to_str(toss),
                                "screen_eq": None,
                                "exact_eq": xeq,
                                "weighted_eq": xeq,
                                "rank_among_candidates": rank_i,
                                "selected": bool((i, j) == best_ij),
                            })

            k1, k2 = best_ij
            keep_cards = [my_cards[k1], my_cards[k2]]
            toss_cards = [my_cards[k] for k in range(len(my_cards)) if k not in best_ij]
            solver_mode = "exact_weighted" if have_opp_discards and sim_mode != "emergency" else ("mc_only" if sim_mode == "emergency" else "exact_unweighted")
            self._log_event("discard_choice", {
                "street": street,
                "street_name": "discard",
                "phase": "discard",
                "sim_mode": sim_mode,
                "have_opp_discards": bool(have_opp_discards),
                "community": community,
                "community_str": self._cards_to_str(community),
                "my_5_cards": my_cards,
                "my_5_cards_str": self._cards_to_str(my_cards),
                "chosen_keep_idx": [int(k1), int(k2)],
                "chosen_keep_cards": keep_cards,
                "chosen_keep_cards_str": self._cards_to_str(keep_cards),
                "chosen_toss_cards": toss_cards,
                "chosen_toss_cards_str": self._cards_to_str(toss_cards),
                "chosen_eq": float(best_eq) if 'best_eq' in locals() and best_eq >= 0 else None,
                "solver_mode": solver_mode,
            })
            return (DISCARD, 0, k1, k2)

        # ── Pre-flop (street 0) ──────────────────────────────────────────────
        result = None
        baseline_result = None
        baseline_action_kind = "same"
        decision_tags = []
        preflop_reason = "none"

        if street == 0:
            premium = _has_any_premium(my_cards)
            premium_pair = _has_premium_pair(my_cards)
            to_call = max(0, opp_bet - my_bet)
            pressure = self._coeff(exploit_adj, "preflop_pressure_adj", 0)
            early_eq_gate = EARLY_PREFLOP_MIN_EQUITY - 0.05 * pressure
            normal_eq_gate = NORMAL_PREFLOP_MIN_EQUITY - 0.05 * pressure

            if in_early_phase:
                preflop_eq = self._preflop_equity(my_cards, set()) if len(my_cards) == 5 else 0.45
                if premium:
                    if premium_pair and opp_bet >= PREFLOP_COMMIT_THRESHOLD and valid[RAISE]:
                        result = (RAISE, max_raise, 0, 0)
                    elif to_call > 0 and valid[RAISE] and random.random() < 0.70:
                        amt = _clamp(int(to_call * 2.5 * random.uniform(0.9, 1.1)), min_raise, max_raise)
                        result = (RAISE, amt, 0, 0)
                    else:
                        noise = random.uniform(0.85, 1.15)
                        open_mult = _clamp(1.0 + 0.35 * pressure, 0.80, 1.25)
                        open_size = _clamp(int(max(10, STANDARD_OPEN) * EARLY_OPEN_MULTIPLIER * open_mult * noise), min_raise, max_raise)
                        if valid[RAISE]:
                            result = (RAISE, open_size, 0, 0)
                        elif valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        else:
                            result = (CHECK, 0, 0, 0)
                elif preflop_eq >= early_eq_gate:
                    if valid[RAISE] and random.random() < 0.65:
                        amt = _clamp(int(9 * (1.0 + 0.25 * pressure) * random.uniform(0.9, 1.1)), min_raise, max_raise)
                        result = (RAISE, amt, 0, 0)
                    elif valid[CALL]:
                        result = (CALL, 0, 0, 0)
                    elif valid[CHECK]:
                        result = (CHECK, 0, 0, 0)
                if result is None:
                    if valid[CHECK]:
                        result = (CHECK, 0, 0, 0)
                    else:
                        result = (FOLD, 0, 0, 0)

            else:
                if premium:
                    if premium_pair and opp_bet >= PREFLOP_COMMIT_THRESHOLD and valid[RAISE]:
                        result = (RAISE, max_raise, 0, 0)
                    elif premium_pair and random.random() < SLOW_PLAY_CHANCE:
                        if valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        else:
                            result = (CHECK, 0, 0, 0)
                    else:
                        noise = random.uniform(0.85, 1.15)
                        open_mult = _clamp(1.0 + 0.35 * pressure, 0.80, 1.25)
                        open_size = _clamp(int(max(10, STANDARD_OPEN) * open_mult * noise), min_raise, max_raise)
                        if valid[RAISE]:
                            result = (RAISE, open_size, 0, 0)
                        elif valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        else:
                            result = (CHECK, 0, 0, 0)
                else:
                    preflop_eq = self._preflop_equity(my_cards, set()) if len(my_cards) == 5 else 0.45
                    if preflop_eq >= normal_eq_gate:
                        open_prob = _clamp(0.40 + 0.35 * pressure, 0.20, 0.70)
                        if to_call <= 0 and valid[RAISE] and random.random() < open_prob:
                            amt = _clamp(int(8 * (1.0 + 0.25 * pressure) * random.uniform(0.9, 1.1)), min_raise, max_raise)
                            result = (RAISE, amt, 0, 0)
                        elif valid[CALL] and to_call <= 4:
                            result = (CALL, 0, 0, 0)
                        elif valid[CHECK]:
                            result = (CHECK, 0, 0, 0)
                    if result is None:
                        if valid[CHECK]:
                            result = (CHECK, 0, 0, 0)
                        else:
                            result = (FOLD, 0, 0, 0)
            if result[0] == RAISE and premium_pair and opp_bet >= PREFLOP_COMMIT_THRESHOLD:
                preflop_reason = "premium_pair_commit"
            elif result[0] == RAISE and premium:
                preflop_reason = "premium_open"
            elif result[0] == RAISE:
                preflop_reason = "eq_gate_open"
            elif result[0] == CALL and premium:
                preflop_reason = "premium_call"
            elif result[0] == CALL:
                preflop_reason = "eq_gate_call"
            elif result[0] == CHECK:
                preflop_reason = "default_check"
            else:
                preflop_reason = "default_fold"

        # ── Post-flop (streets 1-3) ──────────────────────────────────────────
        else:
            if len(my_cards) > 2:
                my_cards = my_cards[:2]

            dead = set(my_cards) | set(community) | set(opp_discards) | set(my_discards)

            baseline_passive_signal = 1.0
            baseline_aggr_signal = 1.0
            passive_sticky_signal = _clamp(
                1.0 + 1.3 * profile.get("call_down_turn", 0.40) + 0.9 * profile.get("call_down_river", 0.38), 0.0, 3.0
            )
            aggr_value_signal = _clamp(
                1.0 + 1.2 * profile.get("raise_vs_bet_turn", 0.20) + 1.0 * profile.get("raise_vs_bet_river", 0.20), 0.0, 3.0
            )
            use_profile_signal = (not self._SHADOW_ONLY) and self._LIVE_STAGE >= 3
            signal_passive = passive_sticky_signal if use_profile_signal else baseline_passive_signal
            signal_aggr = aggr_value_signal if use_profile_signal else baseline_aggr_signal
            eq_sims = 100 if sim_mode == "emergency" else (200 if sim_mode == "conservative" else 300)
            equity = self._compute_equity_ranged(
                my_cards, community, dead, opp_discards, signal_passive, signal_aggr, num_sims=eq_sims)

            hand_cat = _hand_rank_category(my_cards, community)
            # OPT7: compute outs once, derive has_draw without re-calling
            suit_count, flush_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
            run_count, straight_outs, _ = _count_straight_outs(my_cards, community, opp_discards, my_discards)
            has_draw = (suit_count >= 4 and flush_outs >= 2) or (run_count >= 4 and straight_outs >= 3)
            strength = self._cat_to_strength(hand_cat, has_draw, my_cards, community)
            equity_before_adjust = float(equity)
            outs = max(flush_outs, straight_outs)
            draw_adj = 0.0
            if street == 1 and has_draw and outs > 0:
                draw_adj = min(0.10, outs * 0.025)
            elif street == 3 and has_draw and hand_cat in ("nothing", "one_pair"):
                draw_adj = -0.10
            board_monotone_penalty = _board_monotone_penalty(my_cards, community)
            board_connected_penalty = _board_connected_penalty(my_cards, community)
            opp_flush_inference_penalty = _opp_flush_inference(community, opp_discards)
            street_adjust_total = draw_adj + board_monotone_penalty + board_connected_penalty + opp_flush_inference_penalty
            equity = _clamp(equity_before_adjust + street_adjust_total, 0.0, 0.98)

            to_call = max(0, opp_bet - my_bet)
            pot_ref = max(pot_size, 1)
            rand_ctx = {
                "monster_flop_frac": random.uniform(0.55, 0.72),
                "monster_turn_frac": random.uniform(0.70, 0.90),
                "strong_flop_frac": random.uniform(0.55, 0.72),
                "strong_turn_frac": random.uniform(0.65, 0.80),
                "strong_river_frac": random.uniform(0.75, 0.90),
                "good_frac": random.uniform(0.30, 0.50),
                "semi_bluff_high_frac": random.uniform(0.65, 0.80),
                "semi_bluff_mid_frac": random.uniform(0.55, 0.70),
                "semi_bluff_low_frac": random.uniform(0.45, 0.60),
            }
            pot_odds = to_call / (pot_ref + to_call) if to_call > 0 else 0.0
            bluff_catch = self._coeff(exploit_adj, "bluff_catch_adj", street)
            hero_fold = self._coeff(exploit_adj, "hero_fold_adj", street)
            value_sizing = self._coeff(exploit_adj, "value_bet_size_adj", street)
            bluff_freq = self._coeff(exploit_adj, "bluff_freq_adj", street)
            monster_gate = _clamp(MONSTER_THRESHOLD - 0.03 * value_sizing, 0.78, 0.86)
            strong_gate = _clamp(STRONG_THRESHOLD - 0.03 * value_sizing, 0.58, 0.72)
            good_gate = _clamp(GOOD_THRESHOLD - 0.03 * bluff_freq, 0.40, 0.54)
            call_gate = _clamp(pot_odds + 0.10 * hero_fold - 0.10 * bluff_catch, 0.0, 0.95)
            baseline_call_gate = pot_odds
            baseline_good = GOOD_THRESHOLD
            zero_adj = {k: 0.0 for k in self._exploit_state}

            baseline_result = None
            if to_call <= 0:
                b_fire, b_sb = self._semi_bluff_check(
                    my_cards, community, opp_discards, my_discards,
                    pot_size, to_call, street, valid, min_raise, max_raise, zero_adj,
                    has_draw=has_draw, flush_outs_v=flush_outs, straight_outs_v=straight_outs,
                    rand_ctx=rand_ctx)
                if b_fire:
                    baseline_result = b_sb
            if baseline_result is None:
                if equity > MONSTER_THRESHOLD:
                    bfrac = rand_ctx["monster_flop_frac"] if street == 1 else (rand_ctx["monster_turn_frac"] if street == 2 else 1.0)
                    bamt = _clamp(int(pot_ref * bfrac), min_raise, max_raise)
                    b_raise = _clamp(self._dynamic_sizing(bamt, strength, street, False, zero_adj), min_raise, max_raise)
                    baseline_result = (RAISE, max_raise, 0, 0) if (street == 3 and valid[RAISE]) else ((RAISE, b_raise, 0, 0) if valid[RAISE] else ((CALL, 0, 0, 0) if valid[CALL] else (CHECK, 0, 0, 0)))
                elif equity > STRONG_THRESHOLD:
                    bfrac = rand_ctx["strong_flop_frac"] if street == 1 else (rand_ctx["strong_turn_frac"] if street == 2 else rand_ctx["strong_river_frac"])
                    bamt = _clamp(int(pot_ref * bfrac), min_raise, max_raise)
                    b_raise = _clamp(self._dynamic_sizing(bamt, strength, street, False, zero_adj), min_raise, max_raise)
                    baseline_result = (RAISE, b_raise, 0, 0) if valid[RAISE] else ((CALL, 0, 0, 0) if valid[CALL] else (CHECK, 0, 0, 0))
                elif equity > baseline_good:
                    bfrac = rand_ctx["good_frac"]
                    bamt = _clamp(int(pot_ref * bfrac), min_raise, max_raise)
                    b_raise = _clamp(self._dynamic_sizing(bamt, strength, street, False, zero_adj), min_raise, max_raise)
                    if to_call <= 0 and valid[RAISE]:
                        baseline_result = (RAISE, b_raise, 0, 0)
                    elif to_call > 0 and equity >= baseline_call_gate and valid[CALL]:
                        baseline_result = (CALL, 0, 0, 0)
                    else:
                        baseline_result = (CHECK, 0, 0, 0) if valid[CHECK] else (FOLD, 0, 0, 0)
                elif equity >= baseline_call_gate and to_call > 0 and to_call <= pot_ref * 0.35:
                    baseline_result = (CALL, 0, 0, 0) if valid[CALL] else ((CHECK, 0, 0, 0) if valid[CHECK] else (FOLD, 0, 0, 0))
                else:
                    baseline_result = (CHECK, 0, 0, 0) if valid[CHECK] else (FOLD, 0, 0, 0)

            if to_call <= 0:
                fire, sb_action = self._semi_bluff_check(
                    my_cards, community, opp_discards, my_discards,
                    pot_size, to_call, street, valid, min_raise, max_raise, exploit_adj,
                    has_draw=has_draw, flush_outs_v=flush_outs, straight_outs_v=straight_outs,
                    rand_ctx=rand_ctx)
                if fire:
                    result = sb_action

            if result is None:
                if equity > monster_gate:
                    if street == 1:
                        bet_frac = rand_ctx["monster_flop_frac"]
                    elif street == 2:
                        bet_frac = rand_ctx["monster_turn_frac"]
                    else:
                        bet_frac = 1.0
                    base_amt = _clamp(int(pot_ref * bet_frac), min_raise, max_raise)
                    raise_amt = self._dynamic_sizing(base_amt, strength, street, False, exploit_adj)
                    raise_amt = _clamp(raise_amt, min_raise, max_raise)
                    if street == 3 and valid[RAISE]:
                        result = (RAISE, max_raise, 0, 0)
                    elif valid[RAISE]:
                        result = (RAISE, raise_amt, 0, 0)
                    elif valid[CALL]:
                        result = (CALL, 0, 0, 0)
                    else:
                        result = (CHECK, 0, 0, 0)

                elif equity > strong_gate:
                    if street == 1:
                        bet_frac = rand_ctx["strong_flop_frac"]
                    elif street == 2:
                        bet_frac = rand_ctx["strong_turn_frac"]
                    else:
                        bet_frac = rand_ctx["strong_river_frac"]
                    base_amt = _clamp(int(pot_ref * bet_frac), min_raise, max_raise)
                    raise_amt = self._dynamic_sizing(base_amt, strength, street, False, exploit_adj)
                    raise_amt = _clamp(raise_amt, min_raise, max_raise)
                    if valid[RAISE]:
                        result = (RAISE, raise_amt, 0, 0)
                    elif valid[CALL]:
                        result = (CALL, 0, 0, 0)
                    else:
                        result = (CHECK, 0, 0, 0)

                elif equity > good_gate:
                    bet_frac = rand_ctx["good_frac"]
                    base_amt = _clamp(int(pot_ref * bet_frac), min_raise, max_raise)
                    raise_amt = self._dynamic_sizing(base_amt, strength, street, False, exploit_adj)
                    raise_amt = _clamp(raise_amt, min_raise, max_raise)
                    if to_call <= 0 and valid[RAISE]:
                        result = (RAISE, raise_amt, 0, 0)
                    elif to_call > 0 and equity >= call_gate and valid[CALL]:
                        result = (CALL, 0, 0, 0)
                    elif valid[CHECK]:
                        result = (CHECK, 0, 0, 0)
                    else:
                        result = (FOLD, 0, 0, 0)

                elif equity >= call_gate and to_call > 0 and to_call <= pot_ref * 0.35:
                    if valid[CALL]:
                        result = (CALL, 0, 0, 0)
                    elif valid[CHECK]:
                        result = (CHECK, 0, 0, 0)
                    else:
                        result = (FOLD, 0, 0, 0)
                else:
                    if valid[CHECK] and to_call <= 0:
                        result = (CHECK, 0, 0, 0)
                    else:
                        result = (FOLD, 0, 0, 0)

            baseline_action_kind = self._action_category(baseline_result)

            # Decision guards (postflop only)
            if result[0] == FOLD and to_call <= 0 and valid[CHECK]:
                result = (CHECK, 0, 0, 0)

            if result[0] == FOLD and my_bet > 0 and pot_ref > 0 and my_bet >= pot_ref * 0.40:
                if valid[CALL]:
                    result = (CALL, 0, 0, 0)

            if (result[0] == FOLD and len(my_cards) == 2
                    and _is_premium_pair(my_cards[0], my_cards[1])
                    and to_call > 0 and to_call <= pot_ref * 0.20
                    and valid[CALL]):
                result = (CALL, 0, 0, 0)

        if forced_lock_action is not None:
            result = forced_lock_action
            decision_tags.append("forced_lock")

        if street == 0 and self._LOG_DECISIONS:
            self._log_event("preflop_decision", {
                "street": street,
                "premium": bool(_has_any_premium(my_cards)),
                "premium_pair": bool(_has_premium_pair(my_cards)),
                "preflop_eq": float(locals().get("preflop_eq", 0.45)),
                "early_phase": bool(in_early_phase),
                "early_eq_gate": float(locals().get("early_eq_gate", EARLY_PREFLOP_MIN_EQUITY)),
                "normal_eq_gate": float(locals().get("normal_eq_gate", NORMAL_PREFLOP_MIN_EQUITY)),
                "pressure_adj": float(locals().get("pressure", 0.0)),
                "open_prob": float(locals().get("open_prob", 0.0)) if "open_prob" in locals() else None,
                "to_call": int(locals().get("to_call", 0)),
                "decision_reason": preflop_reason,
                "final_action_name": self._action_name(result[0]),
                "final_action_amount": int(result[1]),
            })

        if street > 0 and self._LOG_DECISIONS:
            self._log_event("postflop_line_context", {
                "street": street,
                "we_checked_this_node": bool(self._line_state.get("we_checked_this_node", False)),
                "we_bet_this_node": bool(self._line_state.get("we_bet_this_node", False)),
                "villain_checked_this_node": bool(self._line_state.get("villain_checked_this_node", False)),
                "initiative_owner_entering_street": self._line_state.get("initiative_owner_entering_street", "none"),
                "is_continuation_of_prior_initiative": bool(self._line_state.get("is_continuation_of_prior_initiative", False)),
                "last_aggressor_previous_street": self._line_state.get("last_aggressor_previous_street", "none"),
                "opp_last_action": opp_action,
                "opp_semantic_action": self._last_opp_semantic_action,
            })

        if not valid[DISCARD] and (street == 3 or self._hands_completed % 20 == 0):
            final_kind = "fold" if result[0] == FOLD else ("call" if result[0] == CALL else ("aggressive" if result[0] == RAISE else "check"))
            if street == 0 and baseline_action_kind == "same":
                baseline_action_kind = final_kind
            baseline_tuple = baseline_result if street > 0 else result
            category_changed = (final_kind != baseline_action_kind)
            same_type = baseline_tuple[0] == result[0]
            amount_changed = int(baseline_tuple[1]) != int(result[1])
            size_changed_only = same_type and amount_changed and not category_changed
            self._last_shadow_compare = {
                "baseline_action_category": baseline_action_kind,
                "final_action_category": final_kind,
                "baseline_action_type": int(baseline_tuple[0]),
                "baseline_action_amount": int(baseline_tuple[1]),
                "final_action_type": int(result[0]),
                "final_action_amount": int(result[1]),
                "category_changed": category_changed,
                "size_changed_only": size_changed_only,
                "drivers": {
                    "value_bet_size_adj": self._last_applied_adjustments.get("value_bet_size_adj", 0.0),
                    "bluff_freq_adj": self._last_applied_adjustments.get("bluff_freq_adj", 0.0),
                    "hero_fold_adj": self._last_applied_adjustments.get("hero_fold_adj", 0.0),
                    "scaffolding_probe_freq_adj": self._last_applied_adjustments.get("probe_freq_adj", 0.0),
                },
            }
            if self._DEBUG_VERBOSE:
                print(
                    "[OMICRON] profile="
                    f"fMturn={profile.get('fold_medium_turn', 0.0):.2f}/c{profile.get('conf_fold_medium_turn', 0.0):.2f} "
                    f"cdr={profile.get('call_down_vs_bet_river', 0.0):.2f}/c{profile.get('conf_call_down_vs_bet_river', 0.0):.2f} "
                    f"rvbR={profile.get('raise_vs_bet_river', 0.0):.2f} "
                    f"vol={profile.get('style_volatility', 0.0):.2f} "
                    f"reg={self._regime_shift_score:.2f} "
                    f"live_stage={self._LIVE_STAGE} shadow={int(self._SHADOW_ONLY)} "
                    f"adj={self._last_applied_adjustments} "
                    f"baseline={baseline_action_kind} final={final_kind} "
                    f"b_act={baseline_tuple[0]}:{baseline_tuple[1]} f_act={result[0]}:{result[1]} "
                    f"category_changed={int(category_changed)} size_only={int(size_changed_only)}",
                    flush=True
                )
            if self._DEBUG_VERBOSE:
                self._debug_snapshots.append({
                    "hand": self._hand_idx,
                    "street": street,
                    "profile": dict(profile),
                    "adjustments": dict(self._last_applied_adjustments),
                    "action": result,
                })
                if len(self._debug_snapshots) > 120:
                    self._debug_snapshots = self._debug_snapshots[-80:]

        if self._LOG_DECISIONS and not valid[DISCARD]:
            baseline_tuple = baseline_result if street > 0 and baseline_result is not None else result
            category_changed = self._action_category(baseline_tuple) != self._action_category(result)
            size_changed_only = (baseline_tuple[0] == result[0]) and (int(baseline_tuple[1]) != int(result[1])) and (not category_changed)
            if street > 0:
                if result[0] == RAISE and equity > monster_gate:
                    decision_tags.append("monster_value")
                elif result[0] == RAISE and equity > strong_gate:
                    decision_tags.append("strong_value")
                elif result[0] == CALL and equity >= call_gate:
                    decision_tags.append("bluff_catch")
                elif result[0] == RAISE and to_call <= 0:
                    decision_tags.append("thin_value")
                elif result[0] == CHECK:
                    decision_tags.append("pot_control")
            else:
                if result[0] == RAISE and _has_any_premium(my_cards):
                    decision_tags.append("premium_preflop")
            self._log_event("decision_common", {
                "street": street,
                "phase": "preflop" if street == 0 else "postflop",
                "valid_actions": self._valid_actions_map(valid),
                "my_bet": int(my_bet),
                "opp_bet": int(opp_bet),
                "pot_size": int(pot_size),
                "to_call": int(max(0, opp_bet - my_bet)),
                "min_raise": int(min_raise),
                "max_raise": int(max_raise),
                "blind_position": int(observation.get("blind_position", 0)),
                "time_left": float(observation.get("time_left", 0.0)),
                "sim_mode": sim_mode,
                "my_cards": my_cards,
                "community": community,
                "opp_discards": opp_discards,
                "my_discards": my_discards,
                "my_cards_str": self._cards_to_str(my_cards),
                "community_str": self._cards_to_str(community),
                "opp_discards_str": self._cards_to_str(opp_discards),
                "my_discards_str": self._cards_to_str(my_discards),
                "equity": float(locals().get("equity")) if "equity" in locals() else None,
                "hand_cat": locals().get("hand_cat"),
                "strength": locals().get("strength"),
                "has_draw": bool(locals().get("has_draw")) if "has_draw" in locals() else None,
                "flush_outs": int(locals().get("flush_outs")) if "flush_outs" in locals() else None,
                "straight_outs": int(locals().get("straight_outs")) if "straight_outs" in locals() else None,
                "suit_count": int(locals().get("suit_count")) if "suit_count" in locals() else None,
                "run_count": int(locals().get("run_count")) if "run_count" in locals() else None,
                "pot_odds": float(locals().get("pot_odds")) if "pot_odds" in locals() else None,
                "monster_gate": float(locals().get("monster_gate")) if "monster_gate" in locals() else None,
                "strong_gate": float(locals().get("strong_gate")) if "strong_gate" in locals() else None,
                "good_gate": float(locals().get("good_gate")) if "good_gate" in locals() else None,
                "call_gate": float(locals().get("call_gate")) if "call_gate" in locals() else None,
                "baseline_call_gate": float(locals().get("baseline_call_gate")) if "baseline_call_gate" in locals() else None,
                "board_monotone_penalty": float(locals().get("board_monotone_penalty", 0.0)),
                "board_connected_penalty": float(locals().get("board_connected_penalty", 0.0)),
                "opp_flush_inference_penalty": float(locals().get("opp_flush_inference_penalty", 0.0)),
                "street_adjust_total": float(locals().get("street_adjust_total", 0.0)),
                "equity_before_adjust": float(locals().get("equity_before_adjust")) if "equity_before_adjust" in locals() else None,
                "equity_after_adjust": float(locals().get("equity")) if "equity" in locals() else None,
                "profile_fold_medium_flop": profile.get("fold_medium_flop"),
                "profile_fold_medium_turn": profile.get("fold_medium_turn"),
                "profile_fold_medium_river": profile.get("fold_medium_river"),
                "profile_call_down_turn": profile.get("call_down_vs_bet_turn"),
                "profile_call_down_river": profile.get("call_down_vs_bet_river"),
                "profile_raise_vs_bet_flop": profile.get("raise_vs_bet_flop"),
                "profile_raise_vs_bet_turn": profile.get("raise_vs_bet_turn"),
                "profile_raise_vs_bet_river": profile.get("raise_vs_bet_river"),
                "profile_check_raise_flop": profile.get("check_raise_flop"),
                "profile_check_raise_turn": profile.get("check_raise_turn"),
                "profile_bet_initiative_flop_small": profile.get("bet_initiative_small_flop"),
                "profile_bet_initiative_flop_medium": profile.get("bet_initiative_medium_flop"),
                "profile_bet_initiative_flop_large": profile.get("bet_initiative_large_flop"),
                "style_volatility": profile.get("style_volatility"),
                "global_regime_shift": profile.get("global_regime_shift"),
                "conf_fold_medium_turn": profile.get("conf_fold_medium_turn"),
                "conf_call_down_vs_bet_river": profile.get("conf_call_down_vs_bet_river"),
                "conf_raise_vs_bet_turn": profile.get("conf_raise_vs_bet_turn"),
                "conf_raise_vs_bet_river": profile.get("conf_raise_vs_bet_river"),
                "exploit_adj": dict(self._last_applied_adjustments),
                "baseline_action_type": int(baseline_tuple[0]),
                "baseline_action_name": self._action_name(baseline_tuple[0]),
                "baseline_action_amount": int(baseline_tuple[1]),
                "final_action_type": int(result[0]),
                "final_action_name": self._action_name(result[0]),
                "final_action_amount": int(result[1]),
                "category_changed": bool(category_changed),
                "size_changed_only": bool(size_changed_only),
                "decision_tags": list(dict.fromkeys(decision_tags)),
            })
            self._last_decision_snapshot = {
                "street": street,
                "equity": float(locals().get("equity")) if "equity" in locals() else None,
                "action_name": self._action_name(result[0]),
                "action_amount": int(result[1]),
            }
            self._decision_idx += 1

        # ── Track our action ─────────────────────────────────────────────────
        self._we_folded = (result[0] == FOLD)
        self._last_was_bet = result[0] == RAISE
        self._last_street = street
        self._line_state["we_checked_this_node"] = (result[0] == CHECK)
        self._line_state["we_bet_this_node"] = (result[0] == RAISE)

        return result


# Ensure module is in sys.modules for ProcessPoolExecutor pickle compatibility
# (needed when loaded dynamically via importlib)
import sys as _sys
if __name__ not in _sys.modules:
    from types import ModuleType as _MT
    _self_mod = _MT(__name__)
    _self_mod.__dict__.update(globals())
    _sys.modules[__name__] = _self_mod