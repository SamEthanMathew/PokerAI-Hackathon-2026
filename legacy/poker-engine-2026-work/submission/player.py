import json
import os
import random
import sys
import time
import math
import struct
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

EARLY_PHASE_HANDS = _PROFILE.get("early_phase_hands", 7)
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

# Preflop equity LUT (C(27,5) = 80,730 entries; both players keep best 2 from 5, optimal opponent)
_PY_TO_CPP = [(i % 9) * 3 + (i // 9) for i in range(DECK_SIZE)]
_PREFLOP_LUT: dict = {}
_preflop_lut_path = os.path.join(os.path.dirname(__file__), "preflop_equities_50k.json")
if os.path.isfile(_preflop_lut_path):
    try:
        with open(_preflop_lut_path, encoding="utf-8") as _f:
            _PREFLOP_LUT = json.load(_f)
    except (json.JSONDecodeError, OSError):
        pass

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


# ══════════════════════════════════════════════════════════════════════════════
#  BB Discard Table Loader
# ══════════════════════════════════════════════════════════════════════════════

def _py_to_cpp(py_card):
    rank = py_card % 9
    suit = py_card // 9
    return rank * 3 + suit

def _cpp_to_py(cpp_card):
    rank = cpp_card // 3
    suit = cpp_card % 3
    return suit * 9 + rank

def _canonicalize_cpp(cards_cpp):
    """Canonicalize 8 cards in C++ encoding space (must match C++ exactly)."""
    hole_sorted = sorted(cards_cpp[:5])
    flop_sorted = sorted(cards_cpp[5:8])
    ordered = hole_sorted + flop_sorted

    suit_map = [-1, -1, -1]
    next_suit = 0
    for c in ordered:
        suit = c % 3
        if suit_map[suit] == -1:
            suit_map[suit] = next_suit
            next_suit += 1

    canon = []
    for c in ordered:
        rank = c // 3
        suit = c % 3
        canon.append(rank * 3 + suit_map[suit])

    hole = sorted(canon[:5])
    flop = sorted(canon[5:8])
    return hole + flop

def _canonicalize_py(hole_py, flop_py):
    """Canonicalize (5-hole, 3-flop) from Python encoding. Returns tuple of 8 Python-encoded cards."""
    cards_cpp = [_py_to_cpp(c) for c in hole_py] + [_py_to_cpp(c) for c in flop_py]
    canon_cpp = _canonicalize_cpp(cards_cpp)
    return tuple(_cpp_to_py(c) for c in canon_cpp)


class BBTable:
    """Fast BB discard table using sorted numpy arrays + binary search.

    New format: 8 key bytes + 10 uint8 raw equities per entry.
    Equity byte 0-255 maps linearly to [0.0, 1.0].
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            n_entries = struct.unpack("<Q", f.read(8))[0]
            data = f.read(n_entries * 18)
        raw = np.frombuffer(data, dtype=np.uint8).reshape(n_entries, 18)
        self._keys = np.zeros(n_entries, dtype=np.uint64)
        for i in range(8):
            self._keys += raw[:, i].astype(np.uint64) << np.uint64(i * 8)
        self._equities = raw[:, 8:].copy()  # uint8[n, 10] raw equities
        order = np.argsort(self._keys)
        self._keys = self._keys[order]
        self._equities = self._equities[order]

    def lookup(self, canon_key_tuple):
        """Look up 8-card canonical key. Returns uint8[10] equities or None."""
        packed = np.uint64(0)
        for i, c in enumerate(canon_key_tuple):
            packed += np.uint64(c) << np.uint64(i * 8)
        idx = np.searchsorted(self._keys, packed)
        if idx < len(self._keys) and self._keys[idx] == packed:
            return self._equities[idx]
        return None

def _load_bb_table():
    """Load BB discard table. Returns BBTable or None."""
    path = os.path.join(os.path.dirname(__file__), "data", "bb_discard_table.bin")
    if not os.path.exists(path):
        # Try local data path if submission67 data is missing
        path = os.path.join(os.path.dirname(__file__), "..", "data", "bb_discard_table.bin")
        if not os.path.exists(path):
            # Try original location just in case
            path = os.path.join(os.path.dirname(__file__), "..", "submission67", "data", "bb_discard_table.bin")
            if not os.path.exists(path):
                return None
    try:
        return BBTable(path)
    except Exception as e:
        print(f"Error loading bb table: {e}")
        return None

# ── Action constants ─────────────────────────────────────────────────────────

FOLD = PokerEnv.ActionType.FOLD.value
RAISE = PokerEnv.ActionType.RAISE.value
CHECK = PokerEnv.ActionType.CHECK.value
CALL = PokerEnv.ActionType.CALL.value
DISCARD = PokerEnv.ActionType.DISCARD.value

_KEEP2 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

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


TOP_5_PERCENT_PAIRS = {
    frozenset([RANK_A, RANK_A]),
    frozenset([RANK_9, RANK_9]),
    frozenset([RANK_8, RANK_8]),
}

TOP_5_PERCENT_SUITED = {
    frozenset([RANK_A, RANK_9]),
    frozenset([RANK_A, RANK_8]),
}


def _is_top_5_percent_hand(c1, c2):
    r1, r2 = _RANK[c1], _RANK[c2]
    s1, s2 = _SUIT[c1], _SUIT[c2]
    ranks = frozenset([r1, r2])
    if r1 == r2 and ranks in TOP_5_PERCENT_PAIRS:
        return True
    if s1 == s2 and ranks in TOP_5_PERCENT_SUITED:
        return True
    return False


def _has_top_5_percent(cards):
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _is_top_5_percent_hand(cards[i], cards[j]):
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


def _is_top_pair_or_better(my_cards, community):
    if len(my_cards) < 2 or len(community) < 3:
        return False
    hand_cat = _hand_rank_category(my_cards, community)
    if hand_cat in ("trips_plus", "two_pair"):
        return True
    if hand_cat == "one_pair":
        my_ranks = [_RANK[c] for c in my_cards[:2]]
        comm_ranks = [_RANK[c] for c in community]
        max_comm = max(comm_ranks)
        if my_ranks[0] == my_ranks[1]:
            return my_ranks[0] >= max_comm
        else:
            return (my_ranks[0] == max_comm and my_ranks[0] in comm_ranks) or \
                   (my_ranks[1] == max_comm and my_ranks[1] in comm_ranks)
    return False


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


def _flush_nuttedness(my_cards, community):
    """Returns a multiplier [0.05, 1.0] based on how high the flush draw is."""
    comm_suits = [_SUIT[c] for c in community]
    comm_ranks = [_RANK[c] for c in community]
    
    best_nut = 0.05
    for suit_idx in range(3):
        my_ranks_in_suit = [_RANK[c] for c in my_cards if _SUIT[c] == suit_idx]
        if not my_ranks_in_suit: continue
        
        max_mine = max(my_ranks_in_suit)
        on_board = set(comm_ranks[i] for i in range(len(community)) if comm_suits[i] == suit_idx)
        
        higher = sum(1 for r in range(max_mine + 1, NUM_RANKS) 
                     if r not in on_board and r not in my_ranks_in_suit)
                     
        if higher == 0: best_nut = max(best_nut, 1.0)
        elif higher == 1: best_nut = max(best_nut, 0.60)
        elif higher == 2: best_nut = max(best_nut, 0.25)
        
    return best_nut
    
def _compute_draw_bonus(my_cards, community):
    """Used for discard scoring. Returns 0.0 to 0.30."""
    nut_factor = _flush_nuttedness(my_cards, community)
    sc, f_outs, _ = _count_flush_outs(my_cards, community, [], [])
    _, s_outs, _ = _count_straight_outs(my_cards, community, [], [])
    
    outs = max(f_outs, s_outs)
    if outs < 2: return 0.0
    return min(0.30, (outs * 0.025) * nut_factor)

def _discard_quick_score(keep_cards, board):
    # Base rank score (using _lut_eval_7 to get rank, invert it so higher is better)
    # The pad [-1] ensures we have 7 cards for the LUT
    pad_len = max(0, 5 - len(board))
    eval_hand = list(keep_cards) + list(board) + [-1] * pad_len
    rank = _lut_eval_7(eval_hand) 
    score = (8000 - rank) / 100.0  # Normalize (0 to ~80)
    
    # Add draw bonus (Using our new Draw Nuttedness gradient)
    score += _compute_draw_bonus(keep_cards, board) * 20.0
    return score

_RANK_DIST = (0.65, 0.15, 0.08, 0.05, 0.03, 0.015, 0.01, 0.005, 0.005, 0.005)

def _opp_keep_weight_lut(opp_hand, community, opp_discards, model_weights=None, _flop_cache=None, bb_table=None):
    original_five = list(opp_hand) + list(opp_discards)
    if len(original_five) != 5 or len(community) < 3: 
        return 1.0
    
    flop = community[:3]
    keep_scores = []
    
    if bb_table is not None:
        try:
            canon_key = _canonicalize_py(original_five, flop)
            eq_uint8 = bb_table.lookup(canon_key)
            if eq_uint8 is not None:
                # The table equities correspond to KEEP2 pairs on the CANONICALIZED hole cards
                canon_hole = list(canon_key[:5])
                cards_cpp = [_py_to_cpp(c) for c in original_five] + [_py_to_cpp(c) for c in flop]
                ordered_cpp = sorted(cards_cpp[:5]) + sorted(cards_cpp[5:8])
                suit_map = [-1, -1, -1]
                next_suit = 0
                for c in ordered_cpp:
                    s = c % 3
                    if suit_map[s] == -1:
                        suit_map[s] = next_suit
                        next_suit += 1
                        
                canon_to_orig = {}
                for orig_idx, card_py in enumerate(original_five):
                    card_cpp = _py_to_cpp(card_py)
                    canon_cpp = (card_cpp // 3) * 3 + suit_map[card_cpp % 3]
                    canon_py = _cpp_to_py(canon_cpp)
                    canon_to_orig[canon_py] = orig_idx

                KEEP2 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
                
                for idx, eq_val in enumerate(eq_uint8):
                    canon_i, canon_j = KEEP2[idx]
                    target_a = canon_hole[canon_i]
                    target_b = canon_hole[canon_j]
                    if target_a in canon_to_orig and target_b in canon_to_orig:
                        orig_i = canon_to_orig[target_a]
                        orig_j = canon_to_orig[target_b]
                        i, j = min(orig_i, orig_j), max(orig_i, orig_j)
                        float_eq = float(eq_val) / 255.0
                        keep_scores.append((frozenset([original_five[i], original_five[j]]), float_eq))
        except Exception as e:
            # Fall back below
            pass

    if not keep_scores:
        for kc in combinations(original_five, 2):
            score = _discard_quick_score(kc, community)
            keep_scores.append((frozenset(kc), score))
    
    keep_scores.sort(key=lambda x: x[1], reverse=True)
    target = frozenset(opp_hand)
    
    for rank_pos, (keep_set, _) in enumerate(keep_scores):
        if keep_set == target:
            if rank_pos < len(_RANK_DIST):
                return max(0.001, _RANK_DIST[rank_pos])
            return 0.001
            
    return 0.001


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
                                       opp_discards, model_weights, bb_table=None):
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
    sum_weights = 0.0
    for opp in combinations(remaining, 2):
        w = _opp_keep_weight_lut(list(opp), community, opp_discards,
                                 model_weights, flop_cache, bb_table=bb_table)
        opp_weights[opp] = w
        sum_weights += w
        
    if sum_weights > 0:
        for opp in opp_weights:
            opp_weights[opp] /= sum_weights
    else:
        # Fallback if sum_weights is 0 (should not happen with weight floor)
        for opp in opp_weights:
            opp_weights[opp] = 1.0 / len(opp_weights)

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


def _compute_discard_equity_mc_optimal(my_keep, community, dead_cards, num_sims=600):
    """MC equity where opponent draws 5 cards and keeps their best 2-card combination.

    This correctly models the discard game: both players start with 5 cards and
    keep only their best 2. Using random 2-card opponent sampling (the old approach)
    systematically overestimates our equity by ~40pp because it ignores that the
    opponent also selects optimally from 5 cards.
    """
    remaining = [c for c in range(DECK_SIZE) if c not in dead_cards]
    board_needed = 5 - len(community)
    sample_size = 5 + board_needed
    if sample_size > len(remaining):
        return 0.5
    community_l = list(community)
    my_keep_l = list(my_keep)
    wins = 0.0
    total = 0
    _opp_indices = list(combinations(range(5), 2))  # pre-compute C(5,2)=10 pairs
    for _ in range(num_sims):
        sample = random.sample(remaining, sample_size)
        opp5 = sample[:5]
        board_5 = community_l + sample[5:]
        # opponent picks their best (lowest rank = strongest) 2-card keep
        best_opp_rank = None
        for oi, oj in _opp_indices:
            r = _lut_eval_7([opp5[oi], opp5[oj]] + board_5)
            if best_opp_rank is None or r < best_opp_rank:
                best_opp_rank = r
        my_rank = _lut_eval_7(my_keep_l + board_5)
        if my_rank < best_opp_rank:
            wins += 1.0
        elif my_rank == best_opp_rank:
            wins += 0.5
        total += 1
    return wins / total if total > 0 else 0.5


# ── PlayerAgent ──────────────────────────────────────────────────────────────


class PlayerAgent(Agent):

    _LIVE_STAGE = int(os.getenv("OMICRON_LIVE_STAGE", "1"))
    _SHADOW_ONLY = os.getenv("OMICRON_SHADOW_ONLY", "0") == "1"
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
        
        # ── Load precomputed data ────────────────────────────────────────
        self._bb_table = _load_bb_table()
        self._full_deck = set(range(DECK_SIZE))
        
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
        
        # Phoenix Protocol State
        self._pnl_deltas = []
        self._current_gear = "SNIPER"
        self._hands_in_current_gear = 0
        self._gear_entry_pnl = 0
        
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
        self._last_opp_semantic_action = ""
        self._in_hand = False
        self._last_opp_kept_at_showdown: list = []
        self._preflop_reason: str = ""
        self._pflog_eq = None
        self._pflog_gate = None
        self._pflog_open_prob = None
        self._last_urgency: float = 0.0
        self._last_lead_ratio: float = 0.0
        self._last_position: str = "BB"

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
        if street == 0:
            return "preflop"
        if street == 1:
            return "flop"
        if street == 2:
            return "turn"
        return "river"

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
            obs = bool(cond)
            est.observe(obs)

    def _node_continuous(self, key, value, street):
        est = self._continuous_nodes.get(key)
        if est is None:
            return
        x = float(value)
        est.observe(x)

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
            hand_num = self._hands_completed
            delta = int(reward)
            self._running_pnl += delta
            self._pnl_deltas.append(delta)
            
            self._hands_completed += 1
            self._last_after_shock = abs(delta) >= 20

            self._learn_from_showdown(observation, info)

            self._log_event("hand_result",
                hand=hand_num,
                pnl=int(reward),
                running_pnl=self._running_pnl,
                outcome="win" if reward > 0 else ("loss" if reward < 0 else "tie"),
                showdown=bool(info.get("player_0_cards") and info.get("player_1_cards")),
                our_kept_cards=self._cards_to_str(self._last_my_cards[:2]),
                opp_kept_cards=self._last_opp_kept_at_showdown,
                community=self._cards_to_str(self._last_community),
                we_folded=self._we_folded,
                opp_folded=self._opp_folded,
                large_swing=bool(self._last_after_shock),
                urgency=round(self._last_urgency, 3),
                lead_ratio=round(self._last_lead_ratio, 3),
            )

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
            if self._hands_completed % 25 == 0 and self._hands_completed > 0:
                p = self._last_profile
                self._log_event("opp_profile_snapshot",
                    hand=self._hands_completed,
                    fold_rates={k: round(float(p.get(k, 0)), 4) for k in [
                        "fold_small_flop", "fold_medium_flop", "fold_large_flop",
                        "fold_small_turn", "fold_medium_turn", "fold_large_turn",
                        "fold_small_river", "fold_medium_river", "fold_large_river",
                    ]},
                    call_down={
                        "turn": round(float(p.get("call_down_turn", 0)), 4),
                        "river": round(float(p.get("call_down_river", 0)), 4),
                    },
                    barrel_rates={
                        "turn": round(float(p.get("barrel_turn_after_flop_aggr", 0)), 4),
                        "river": round(float(p.get("barrel_river_after_turn_aggr", 0)), 4),
                    },
                    raise_vs_bet={k: round(float(p.get(k, 0)), 4) for k in [
                        "raise_vs_bet_flop", "raise_vs_bet_turn", "raise_vs_bet_river"
                    ]},
                    avg_bet_sizes={k: round(float(p.get(k, 0)), 4) for k in [
                        "avg_bet_size_flop", "avg_bet_size_turn", "avg_bet_size_river"
                    ]},
                    exploit_state={k: round(float(v), 4) for k, v in self._exploit_state.items()},
                    regime_shift=round(float(self._regime_shift_score), 4),
                    shift_signals={k: round(float(v), 4) for k, v in self._shift_signals.items()},
                    style_volatility=round(float(p.get("style_volatility", 0)), 4),
                    opp_model_weights=[round(float(w), 3) for w in self._opp_model.weights],
                )

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
        self._last_opp_kept_at_showdown = self._cards_to_str(opp_kept[:2])
        if len(self._last_community) >= 5:
            river_board = self._last_community[:5]
            cat = _hand_rank_category(opp_kept[:2], river_board)
            strength = self._showdown_bucket_value(cat, len(river_board))
            revealed_bluff = strength <= 0.25
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

    @classmethod
    def run(cls, stream: bool = False, port: int = 8000, host: str = "0.0.0.0", player_id: str = None):
        """Override base run() to survive port-in-use errors between sequential
        validation test runs.

        Root cause of failures: on Linux, ProcessPoolExecutor workers are
        spawned via fork and inherit the socket fd. When the parent process
        (uvicorn) shuts down and closes its copy, the port stays bound until
        all worker processes also exit. Workers become orphans after the parent
        exits, keeping the port bound for 30+ seconds between validation tests.

        Fix:
        - timeout_graceful_shutdown=5: uvicorn force-closes HTTP connections
          after 5s of SIGTERM so server.run() returns promptly.
        - finally block shuts down _POOL with wait=True: waits for all worker
          processes to fully exit, releasing their inherited socket fd copies.
        - Pre-bind socket with SO_REUSEADDR + 30-second retry loop as safety net.
        """
        import socket as _socket
        import time as _time
        import uvicorn as _uvicorn

        if player_id is not None:
            os.environ["PLAYER_ID"] = player_id
        bot = cls(stream)
        bot.logger.info(f"Starting agent server on {host}:{port}")

        # Retry until the port is available (previous test may still be dying).
        # With timeout_graceful_shutdown=5 in place, previous process releases
        # the port within ~5s — so attempts 1-5 may fail, attempt 6 succeeds.
        # 30 retries provides a generous 30-second safety window.
        sock = None
        for attempt in range(30):
            try:
                s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                sock = s
                break
            except OSError as exc:
                s.close()
                if attempt < 29:
                    bot.logger.info(
                        f"Port {port} busy (attempt {attempt + 1}/30): {exc} — retrying in 1s"
                    )
                    _time.sleep(1)
                else:
                    bot.logger.error(f"Port {port} still in use after 30 attempts; giving up")
                    raise

        # Tell the worker initializer (_discard_pool_init) which fd to close.
        # Workers forked from this process inherit all open fds including the
        # server socket; the initializer closes it immediately so workers never
        # keep the port bound after this process exits.
        global _SERVER_SOCK_FD
        _SERVER_SOCK_FD = sock.fileno()

        config = _uvicorn.Config(
            bot.app,
            log_level="info",
            access_log=False,
            timeout_graceful_shutdown=5,
            timeout_keep_alive=1,
        )
        server = _uvicorn.Server(config)
        try:
            server.run(sockets=[sock])
        finally:
            # Clear fd sentinel so stale fd numbers are never closed by future workers.
            _SERVER_SOCK_FD = -1
            # Signal workers to exit (no wait — they already released the socket
            # in their initializer, so the port is freed when we close sock below).
            global _POOL
            _p = _POOL
            _POOL = None
            if _p is not None:
                try:
                    _p.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            try:
                sock.close()
            except Exception:
                pass

    # ── MC Equity (random range — for discard screening + preflop) ───────────

    def _compute_equity(self, my_cards, community, opp_discards, my_discards,
                        num_sims=300, opp_optimal=False):
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
        # opp_optimal: draw 5 cards for opponent so they can pick their best 2
        sample_size = (5 if opp_optimal else 2) + board_needed

        if sample_size > len(remaining):
            return 0.5

        community_l = list(community)
        my_keep = list(my_cards)
        _opp_idx = list(combinations(range(5), 2)) if opp_optimal else None

        wins = 0.0
        total = 0
        for _ in range(num_sims):
            sample = random.sample(remaining, sample_size)
            if opp_optimal:
                opp5 = sample[:5]
                board_5 = community_l + sample[5:]
                best_opp_rank = None
                for oi, oj in _opp_idx:
                    r = _lut_eval_7([opp5[oi], opp5[oj]] + board_5)
                    if best_opp_rank is None or r < best_opp_rank:
                        best_opp_rank = r
                opp_rank = best_opp_rank
            else:
                board_5 = community_l + sample[2:]
                opp_rank = _lut_eval_7(sample[:2] + board_5)

            my_rank = _lut_eval_7(my_keep + board_5)
            if my_rank < opp_rank:
                wins += 1
            elif my_rank == opp_rank:
                wins += 0.5
            total += 1

        return wins / total if total > 0 else 0.5

    def _compute_equity_optimal(self, my_cards, community, opp_discards, my_discards,
                                num_sims=300):
        """Equity simulation with optimal-opponent model (draws 5, keeps best 2)."""
        return self._compute_equity(my_cards, community, opp_discards, my_discards,
                                    num_sims=num_sims, opp_optimal=True)

    # ── Range-weighted Exact Enumeration Equity (for postflop decisions) ────────────

    def _compute_equity_ranged(self, my2, community, dead, opp_discards,
                               passive_signal, aggr_signal):
        remaining = [i for i in range(DECK_SIZE) if i not in dead]
        board_needed = 5 - len(community)
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
        reject_two_pair = aggr_signal >= 4.2
        
        community_l = list(community)
        my_keep = list(my2)

        opp_combos = []
        for pair in combinations(remaining, 2):
            w = 1.0
            if have_discards and flop_cache:
                w = self._opp_keep_weight(list(pair), community, opp_discards, flop_cache)
            
            # Apply passive/aggr signal adjustments to the weight
            if w >= 0.001:
                cat = _hand_rank_category(list(pair), community)
                if reject_nothing and cat == "nothing":
                    w *= 0.1
                elif reject_one_pair and cat == "one_pair":
                    w *= 0.1
                elif reject_two_pair and cat in ("one_pair", "two_pair"):
                    w *= 0.1
                elif passive_signal > 2.0 and cat == "nothing":
                    w *= 0.75
                
                opp_combos.append((pair, w))

        if not opp_combos:
            return 0.5
            
        # Sort by weight descending for early breaking
        opp_combos.sort(key=lambda x: x[1], reverse=True)

        wins = 0.0
        total_weight = 0.0
        start_time = time.time()

        for opp, weight in opp_combos:
            if time.time() - start_time > 0.040:  # 40ms safety valve
                break
                
            opp_l = list(opp)
            runout_pool = [c for c in remaining if c not in opp_l]
            
            for runout in combinations(runout_pool, board_needed):
                board_5 = community_l + list(runout)
                my_rank = _lut_eval_7(my_keep + board_5)
                opp_rank = _lut_eval_7(opp_l + board_5)
                
                if my_rank < opp_rank:
                    wins += weight
                elif my_rank == opp_rank:
                    wins += 0.5 * weight
                total_weight += weight

        return wins / total_weight if total_weight > 0 else 0.5

    # ── Preflop equity ───────────────────────────────────────────────────────

    def _preflop_equity(self, my5):
        key = str(sorted(_PY_TO_CPP[c] for c in my5))
        return _PREFLOP_LUT.get(key, 0.45)

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
        
        if street == 1 and has_draw and (outs > 0):
            # Calculate base bonus
            base_bonus = min(0.10, outs * 0.025)
            # Scale by nuttedness (custom gradient)
            nut_factor = _flush_nuttedness(my_cards, community) if my_cards else 0.5
            
            # If it is a straight draw, we give a default 0.6 nut factor for now
            if straight_outs > flush_outs:
                nut_factor = max(nut_factor, 0.6) 
                
            # Multiply our flat bonus by the nuttedness gradient
            equity += (base_bonus * nut_factor)
            
        elif street == 3 and has_draw and (hand_cat in ('nothing', 'one_pair')):
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

    def _log_event(self, event_type: str, **fields) -> None:
        def _coerce(v):
            if isinstance(v, dict):
                return {kk: _coerce(vv) for kk, vv in v.items()}
            if isinstance(v, (list, tuple)):
                return [_coerce(x) for x in v]
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.floating):
                return float(round(float(v), 5))
            if isinstance(v, np.bool_):
                return bool(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, float):
                return round(v, 5)
            return v
        payload = {"event": event_type, "ts": round(time.time(), 3)}
        payload.update({k: _coerce(v) for k, v in fields.items()})
        self.logger.info(json.dumps(payload))

    # ── Main act() ───────────────────────────────────────────────────────────

    def _calculate_pnl_slope(self):
        """
        Calculates the velocity of ruin using EMA of deltas instead of linear regression.
        Formula: EMA_new = (Current_Hand_Result * alpha) + (EMA_old * (1 - alpha))
        """
        alpha = 0.15
        ema = 0.0
        for delta in self._pnl_deltas:
            # Cap deltas at +/- 15 to ignore massive variance spikes ("coolers")
            capped_delta = max(-15, min(15, delta))
            if ema == 0.0:
                ema = capped_delta
            else:
                ema = (capped_delta * alpha) + (ema * (1 - alpha))
        return ema

    def _determine_gear(self, hands_left, ema_slope, total_hands):
        if self._running_pnl >= 0:
            return "STANDARD"

        # Dynamic Slope Cap: Strict early to ignore coolers, relaxed late to catch death spirals
        if hands_left > 70:
            cap = 1.5
        elif hands_left < 30:
            cap = 5.0
        else:
            # Linear interpolation between 70 and 30 hands left
            cap = 1.5 + (5.0 - 1.5) * ((70 - hands_left) / 40.0)
            
        capped_slope = max(-cap, min(cap, ema_slope))

        # Time-weight the projection: Don't panic early in the match
        # We estimate total hands dynamically. If winning, we evaluate when we might hit the 1.5 loss threshold.
        total_hands_est = self._hands_completed + hands_left
        time_weight = (hands_left / max(1, total_hands_est)) ** 1.5
        
        if self._running_pnl > 0 and ema_slope < 0:
            # If we are winning but bleeding, find intercept of hard bleed
            projected_final = self._running_pnl + (capped_slope * hands_left * time_weight)
        else:
            projected_final = self._running_pnl + (capped_slope * hands_left * time_weight)
        
        # Evaluate Target Gear
        target_gear = "SNIPER"
        # Severity Threshold for Two-Minute Warning
        if hands_left < 30 and self._running_pnl < -20:
            target_gear = "CHAOS"
        elif projected_final < -100 or self._running_pnl < -100:
            target_gear = "CHAOS"
        elif projected_final < -30 or self._running_pnl < -30:
            target_gear = "PRESSURE"
            
        # Hysteresis: Require a 15-BB recovery buffer before down-shifting
        if target_gear == "PRESSURE" and getattr(self, '_current_gear', 'STANDARD') == "CHAOS":
            if self._running_pnl < getattr(self, '_gear_entry_pnl', 0) + 15:
                return "CHAOS" # Stay in CHAOS until we recover 15 BBs
                
        return target_gear

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
        self._preflop_reason = ""
        self._pflog_eq = None
        self._pflog_gate = None
        self._pflog_open_prob = None
        time_left = observation.get("time_left", 1000.0)
        hands_left = max(1, TOTAL_HANDS - self._hands_completed)
        budget_per_hand = time_left / hands_left
        in_early_phase = self._hands_completed < EARLY_PHASE_HANDS

        # ── Urgency / lead-protection signals ────────────────────────────────
        # urgency: how close the opponent is to locking the match (0=neutral, 1=locked)
        if self._running_pnl < 0 and hands_left > 0:
            urgency = _clamp(-self._running_pnl / (hands_left * 1.5), 0.0, 1.2)
        else:
            urgency = 0.0
        in_comeback_mode = urgency > 0.65
        in_critical_mode = urgency > 0.85
        # lead_ratio: how close WE are to locking the match (>1.0 = already locked)
        if self._running_pnl > 0 and hands_left > 0:
            lead_ratio = self._running_pnl / (hands_left * 1.5)
        else:
            lead_ratio = 0.0
        protecting_lead = lead_ratio > 0.55
        self._last_urgency = float(urgency)
        self._last_lead_ratio = float(lead_ratio)
        self._last_position = "SB" if observation.get("blind_position", 0) == 0 else "BB"

        # ── Dynamic Bleed-Out ────────────────────────────────────────────────
        in_dynamic_bleedout = False
        surplus_for_hand = 0.0
        alpha = 0.0
        if hands_left > 0:
            ratio = 0.33 + (6.0 / math.sqrt(hands_left))
            threshold_lead = ratio * hands_left
            if self._running_pnl > threshold_lead:
                in_dynamic_bleedout = True
                surplus_for_hand = self._running_pnl - threshold_lead
                max_burn = 1.5 * max(1, hands_left)
                alpha = min(1.0, max(0.0, surplus_for_hand / max_burn))

        if is_new_hand and not self._in_hand:
            self._in_hand = True
            self._we_folded = False
            self._last_opp_kept_at_showdown = []
            self._hands_in_current_gear += 1
            
            # ── Asymmetric Gear Shifter Logic (Phoenix Protocol) ─────────────
            if hands_left > 0:
                slope = self._calculate_pnl_slope()
                target_gear = self._determine_gear(hands_left, slope, TOTAL_HANDS) 
                
                gear_priority = {"STANDARD": -1, "SNIPER": 0, "PRESSURE": 1, "CHAOS": 2}
                
                # Shift logic
                is_up_shift = gear_priority.get(target_gear, -1) > gear_priority.get(self._current_gear, -1)
                can_down_shift = (self._hands_in_current_gear >= 5 and self._running_pnl > self._gear_entry_pnl)
                
                if is_up_shift or can_down_shift:
                    if self._current_gear != target_gear:
                        self._log_event("gear_transition", 
                            hand=self._hands_completed,
                            old_gear=self._current_gear,
                            new_gear=target_gear,
                            reason="emergency_upshift" if is_up_shift else "delayed_downshift",
                            slope=round(float(slope), 3)
                        )
                    self._current_gear = target_gear
                    # CRITICAL for Hysteresis & cooldowns
                    self._gear_entry_pnl = self._running_pnl
                    self._hands_in_current_gear = 0
            
            self._log_event("hand_start",
                hand=self._hands_completed,
                position="SB" if observation.get("blind_position", 0) == 0 else "BB",
                hole_cards=self._cards_to_str(my_cards),
                running_pnl=self._running_pnl,
                time_left=round(float(time_left), 2),
                hands_remaining=hands_left,
                budget_per_hand=round(float(budget_per_hand), 3),
                early_phase=in_early_phase,
                urgency=round(float(urgency), 3),
                lead_ratio=round(float(lead_ratio), 3),
                comeback_mode=in_comeback_mode,
                protecting_lead=protecting_lead,
                current_gear=self._current_gear,
                pnl_slope=round(float(slope if hands_left > 0 else 0.0), 3),
                projected_pnl=round(float(projected_final if hands_left > 0 else self._running_pnl), 3)
            )
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
                if forced_lock_action is not None:
                    self._log_event("bleedout_lock",
                        hand=self._hands_completed,
                        running_pnl=self._running_pnl,
                        max_bleed=int(max_bleed),
                        hands_remaining=int(hands_remaining),
                        street=street,
                        action_forced=PokerEnv.ActionType(forced_lock_action[0]).name,
                    )

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

            combo_equities: dict = {}  # (i,j) -> float equity for logging

            if have_opp_discards:
                weights = list(self._opp_model.weights)
                best_eq = -1.0
                best_ij = (0, 1)
                for i, j, keep, toss in all_keeps:
                    dead = dead_base | set(toss) | set(opp_discards)
                    eqf = float(_exact_discard_equity_weighted_lut(
                        keep, community, dead, opp_discards, weights, bb_table=getattr(self, "_bb_table", None)
                    ))
                    combo_equities[(i, j)] = eqf
                    if eqf > best_eq:
                        best_eq = eqf
                        best_ij = (i, j)
            else:
                # No opp discards known (we are BB). Use O(1) table lookup or exact fallback.
                bb_table_success = False
                best_eq = 0.0
                best_ij = (0, 1)

                if getattr(self, "_bb_table", None):
                    try:
                        canon_key = _canonicalize_py(my_cards, community[:3])
                        eq_uint8 = self._bb_table.lookup(canon_key)
                        if eq_uint8 is not None:
                            bb_table_success = True
                            
                            # The table equities are stored in a fixed order corresponding to keeping
                            # the i-th and j-th card from the CANONICALIZED hole cards.
                            # We must map the argmax back to our original Python hole card indices.
                            # (Matching submission67 logic)
                            canon_hole = list(canon_key[:5])
                            cards_cpp = [_py_to_cpp(c) for c in my_cards] + [_py_to_cpp(c) for c in community[:3]]
                            ordered_cpp = sorted(cards_cpp[:5]) + sorted(cards_cpp[5:8])
                            suit_map = [-1, -1, -1]
                            next_suit = 0
                            for c in ordered_cpp:
                                s = c % 3
                                if suit_map[s] == -1:
                                    suit_map[s] = next_suit
                                    next_suit += 1
                                    
                            canon_to_orig = {}
                            for orig_idx, card_py in enumerate(my_cards):
                                card_cpp = _py_to_cpp(card_py)
                                canon_cpp = (card_cpp // 3) * 3 + suit_map[card_cpp % 3]
                                canon_py = _cpp_to_py(canon_cpp)
                                canon_to_orig[canon_py] = orig_idx

                            # The fixed C(5,2)=10 keep pairs used during table generation
                            for idx, eq_val in enumerate(eq_uint8):
                                canon_i, canon_j = _KEEP2[idx]
                                target_a = canon_hole[canon_i]
                                target_b = canon_hole[canon_j]
                                if target_a in canon_to_orig and target_b in canon_to_orig:
                                    orig_i = canon_to_orig[target_a]
                                    orig_j = canon_to_orig[target_b]
                                    i, j = min(orig_i, orig_j), max(orig_i, orig_j)
                                    # eq_val is uint8 [0..255] mapping to [0.0..1.0]
                                    float_eq = float(eq_val) / 255.0
                                    combo_equities[(i, j)] = float_eq
                                    if float_eq > best_eq:
                                        best_eq = float_eq
                                        best_ij = (i, j)
                                        
                            # CROSS-CHECK (first 100 hands)
                            if self._hands_completed < 100:
                                mc_best_eq = -1.0
                                mc_best_ij = (0, 1)
                                for i, j, keep, toss in all_keeps:
                                    dead = dead_base | set(toss)
                                    eq = float(_compute_discard_equity_mc_optimal(keep, community, dead, num_sims=300))
                                    if eq > mc_best_eq:
                                        mc_best_eq = eq
                                        mc_best_ij = (i, j)
                                if mc_best_ij != best_ij and abs(best_eq - mc_best_eq) > 0.05:
                                    self.logger.warning(f"BBTable mismatch! Table chose {best_ij} eq={best_eq:.3f}, MC chose {mc_best_ij} eq={mc_best_eq:.3f}")
                    except Exception as e:
                        self.logger.error(f"BBTable lookup failed: {e}")
                        bb_table_success = False
                
                if not bb_table_success:
                    # Robust Fallback: greedy evaluation
                    from treys import Evaluator
                    evaluator = Evaluator()
                    best_rank = 999999
                    for i, j, keep, toss in all_keeps:
                        rank = _lut_eval_7(list(keep) + list(community[:3]))
                        combo_equities[(i, j)] = 1.0 / float(max(1, rank)) # Arbitrary score for logging
                        if rank < best_rank:
                            best_rank = rank
                            best_ij = (i, j)

            k1, k2 = best_ij
            sorted_combos = sorted(combo_equities.items(), key=lambda x: x[1], reverse=True)
            equity_margin = (sorted_combos[0][1] - sorted_combos[1][1]) if len(sorted_combos) >= 2 else 0.0

            self._log_event("discard_decision",
                hand=self._hands_completed,
                mode=sim_mode,
                budget_per_hand=round(float(budget_per_hand), 3),
                flop_cards=self._cards_to_str(community[:3]),
                all_hole_cards=self._cards_to_str(my_cards),
                opp_discards_known=bool(opp_discards),
                opp_discards=self._cards_to_str(opp_discards),
                keep_combos=[
                    {
                        "keep": self._cards_to_str([my_cards[i], my_cards[j]]),
                        "discard": self._cards_to_str([my_cards[x] for x in range(len(my_cards)) if x not in (i, j)]),
                        "equity": round(float(eq), 4),
                        "rank": rank + 1,
                    }
                    for rank, ((i, j), eq) in enumerate(sorted_combos)
                ],
                chosen_keep=self._cards_to_str([my_cards[k1], my_cards[k2]]),
                chosen_equity=round(float(best_eq), 4),
                equity_margin=round(float(equity_margin), 4),
                opp_model_weights=[round(float(w), 3) for w in self._opp_model.weights],
            )
            return (DISCARD, 0, k1, k2)

        # ── Pre-flop (street 0) ──────────────────────────────────────────────
        result = None
        baseline_result = None
        baseline_action_kind = "same"

        if street == 0:
            premium = _has_any_premium(my_cards)
            premium_pair = _has_premium_pair(my_cards)
            to_call = max(0, opp_bet - my_bet)

            # Shover Detection and Tracking
            if opp_bet >= 50:
                self._shove_ema = (0.85 * getattr(self, '_shove_ema', 0.0)) + (0.15 * 1.0)
            elif self._in_hand:
                self._shove_ema = (0.85 * getattr(self, '_shove_ema', 0.0)) + (0.15 * 0.0)

            is_shover = getattr(self, '_shove_ema', 0.0) > 0.12 and self._hands_completed >= 10

            pressure = self._coeff(exploit_adj, "preflop_pressure_adj", 0)
            early_eq_gate = EARLY_PREFLOP_MIN_EQUITY - 0.05 * pressure
            normal_eq_gate = NORMAL_PREFLOP_MIN_EQUITY - 0.05 * pressure
            
            # ── Gear-Based Pre-Flop Logic (Phoenix Protocol) ─────────────
            if self._current_gear == "SNIPER":
                early_eq_gate = _clamp(early_eq_gate + 0.15, 0.50, 0.80)
                normal_eq_gate = _clamp(normal_eq_gate + 0.15, 0.48, 0.78)
            elif self._current_gear == "STANDARD":
                # STANDARD uses the calculated baseline gates with no modifiers
                pass
            elif self._current_gear == "PRESSURE":
                early_eq_gate = _clamp(early_eq_gate - 0.15, 0.22, 0.48)
                normal_eq_gate = _clamp(normal_eq_gate - 0.15, 0.20, 0.45)
            elif self._current_gear == "CHAOS":
                # CHAOS overrides equity gates to be very loose
                early_eq_gate = 0.25
                normal_eq_gate = 0.25
                
            opp_vpip = profile.get("opp_preflop_raise_rate", 0.5)
            alpha_penalty = 0.25 * alpha
            bully_discount = 0.5 * max(0.0, opp_vpip - 0.5)
            net_penalty = max(0.0, alpha_penalty - bully_discount)
            
            # Skip net penalty application if we are in CHAOS or SNIPER so gears act purely
            if self._current_gear not in ("CHAOS", "SNIPER"):
                early_eq_gate = _clamp(early_eq_gate + net_penalty, 0.22, 0.80)
                normal_eq_gate = _clamp(normal_eq_gate + net_penalty, 0.20, 0.78)

            if is_shover and to_call >= 40:
                early_eq_gate -= 0.18 # Relax calling requirements to trap the shover
                normal_eq_gate -= 0.18

            preflop_eq = self._preflop_equity(my_cards) if len(my_cards) == 5 else 0.45
            self._pflog_eq = preflop_eq

            result = None
            if self._current_gear == "CHAOS" and valid[RAISE]:
                has_ace = any(_RANK[c] == 8 for c in my_cards)
                if opp_vpip < 0.45:
                    chaos_shove = has_ace or preflop_eq >= 0.46
                    if chaos_shove:
                        result = (RAISE, max_raise, 0, 0)
                        self._preflop_reason = "chaos_shove_bluff"
                else:
                    chaos_shove = premium_pair or (has_ace and preflop_eq >= 0.48)
                    if chaos_shove:
                        result = (RAISE, max_raise, 0, 0)
                        self._preflop_reason = "chaos_shove_value"

            if in_early_phase and result is None:
                self._pflog_gate = early_eq_gate
                if premium:
                    if premium_pair and opp_bet >= PREFLOP_COMMIT_THRESHOLD and valid[RAISE]:
                        result = (RAISE, max_raise, 0, 0)
                        self._preflop_reason = "premium_pair_allin"
                    elif to_call > 0 and valid[RAISE] and random.random() < 0.70:
                        amt = _clamp(int(to_call * 2.5 * random.uniform(0.9, 1.1)), min_raise, max_raise)
                        result = (RAISE, amt, 0, 0)
                        self._preflop_reason = "premium_3bet_reraise"
                    else:
                        noise = random.uniform(0.85, 1.15)
                        open_mult = _clamp(1.0 + 0.35 * pressure, 0.80, 1.25)
                        open_size = _clamp(int(max(10, STANDARD_OPEN) * EARLY_OPEN_MULTIPLIER * open_mult * noise), min_raise, max_raise)
                        if valid[RAISE]:
                            result = (RAISE, open_size, 0, 0)
                            self._preflop_reason = "premium_early_open"
                        elif valid[CALL]:
                            result = (CALL, 0, 0, 0)
                            self._preflop_reason = "premium_early_call"
                        else:
                            result = (CHECK, 0, 0, 0)
                            self._preflop_reason = "premium_early_check"
                elif preflop_eq >= early_eq_gate:
                    if valid[RAISE] and random.random() < 0.65:
                        amt = _clamp(int(9 * (1.0 + 0.25 * pressure) * random.uniform(0.9, 1.1)), min_raise, max_raise)
                        result = (RAISE, amt, 0, 0)
                        self._preflop_reason = "equity_raise"
                    elif valid[CALL]:
                        result = (CALL, 0, 0, 0)
                        self._preflop_reason = "equity_call"
                    elif valid[CHECK]:
                        result = (CHECK, 0, 0, 0)
                        self._preflop_reason = "equity_check"
                if result is None:
                    if valid[CHECK]:
                        result = (CHECK, 0, 0, 0)
                        self._preflop_reason = "check_no_equity"
                    else:
                        result = (FOLD, 0, 0, 0)
                        self._preflop_reason = "fold_no_equity"

            elif result is None:
                if premium:
                    if premium_pair and opp_bet >= PREFLOP_COMMIT_THRESHOLD and valid[RAISE]:
                        result = (RAISE, max_raise, 0, 0)
                        self._preflop_reason = "premium_pair_allin"
                    elif premium_pair and random.random() < SLOW_PLAY_CHANCE:
                        if valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        else:
                            result = (CHECK, 0, 0, 0)
                        self._preflop_reason = "premium_slowplay"
                    else:
                        noise = random.uniform(0.85, 1.15)
                        open_mult = _clamp(1.0 + 0.35 * pressure, 0.80, 1.25)
                        open_size = _clamp(int(max(10, STANDARD_OPEN) * open_mult * noise), min_raise, max_raise)
                        if valid[RAISE]:
                            result = (RAISE, open_size, 0, 0)
                            self._preflop_reason = "premium_standard_open"
                        elif valid[CALL]:
                            result = (CALL, 0, 0, 0)
                            self._preflop_reason = "premium_call"
                        else:
                            result = (CHECK, 0, 0, 0)
                            self._preflop_reason = "premium_check"
                else:
                    self._pflog_gate = normal_eq_gate
                    if preflop_eq >= normal_eq_gate:
                        open_prob = _clamp(0.40 + 0.35 * pressure + (0.50 * urgency if in_comeback_mode else 0.0), 0.20, 0.98)
                        self._pflog_open_prob = open_prob
                        if to_call <= 0 and valid[RAISE] and random.random() < open_prob:
                            amt = _clamp(int(8 * (1.0 + 0.25 * pressure) * random.uniform(0.9, 1.1)), min_raise, max_raise)
                            result = (RAISE, amt, 0, 0)
                            self._preflop_reason = "equity_raise"
                        elif valid[CALL] and to_call <= (10 if in_critical_mode else 6 if in_comeback_mode else 4):
                            result = (CALL, 0, 0, 0)
                            self._preflop_reason = "equity_call"
                        elif valid[CHECK]:
                            result = (CHECK, 0, 0, 0)
                            self._preflop_reason = "equity_check"
                    if result is None:
                        if valid[CHECK]:
                            result = (CHECK, 0, 0, 0)
                            self._preflop_reason = "check_no_equity"
                        else:
                            result = (FOLD, 0, 0, 0)
                            self._preflop_reason = "fold_no_equity"

            # Tighten Re-Raises based on Bet Depth
            if result and result[0] == RAISE:
                num_raises = to_call // max(2, min_raise)  # Rough proxy for raise depth (3-bet, 4-bet)
                
                # Dynamically adjust threshold based on opponent's raise rate instead of a strict 0.88
                opp_raise_rate = profile.get('raise_rate_overall', 0.18)
                dynamic_raise_threshold = 0.92 - (opp_raise_rate * 0.20) # Loosen against manic raisers, tighten against nits

                if num_raises >= 2:
                    # We are facing a 3-bet or 4-bet
                    if not (premium_pair or preflop_eq > dynamic_raise_threshold):
                        # Downgrade RAISE to CALL or FOLD
                        if valid[CALL] and preflop_eq > normal_eq_gate:
                            result = (CALL, 0, 0, 0)
                            self._preflop_reason += "_tightened_to_call"
                        else:
                            result = (FOLD, 0, 0, 0)
                            self._preflop_reason += "_tightened_to_fold"

        # ── Post-flop (streets 1-3) ──────────────────────────────────────────
        else:
            if len(my_cards) > 2:
                my_cards = my_cards[:2]

            # ── In _process_opponent_action or start of postflop (Street 1) ──
            if street == 1 and not self._line_state.get('discard_trap_calculated', False):
                self._line_state['discard_trap_calculated'] = True
                self._line_state['opp_discarded_pair'] = False
                if len(opp_discards) >= 2:
                    self._line_state['opp_discarded_pair'] = any(
                        _RANK[opp_discards[i]] == _RANK[opp_discards[j]] 
                        for i in range(len(opp_discards)) 
                        for j in range(i+1, len(opp_discards))
                    )

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
            
            # Add current action aggression
            if opp_bet > my_bet:
                to_call = opp_bet - my_bet
                pot_before_call = pot_size - to_call
                bet_fraction = to_call / max(1.0, float(pot_before_call))
                
                action_aggr = 0.0
                if bet_fraction >= 0.75:
                    action_aggr = 2.5
                elif bet_fraction >= 0.4:
                    action_aggr = 1.5
                else:
                    action_aggr = 0.5
                    
                # Extra respect for big river bets
                if street == 3 and bet_fraction > 0.6:
                    action_aggr += 1.0
                    
                signal_aggr = max(signal_aggr, 1.0 + action_aggr)

            equity = self._compute_equity_ranged(
                my_cards, community, dead, opp_discards, signal_passive, signal_aggr)

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
            raw_texture_adj = board_monotone_penalty + board_connected_penalty + opp_flush_inference_penalty
            if in_comeback_mode:
                # Dampen board texture penalties — in comeback mode we need variance,
                # not overly cautious folds on decent raw equity
                texture_dampen_factor = _clamp(1.0 - urgency, 0.0, 1.0)
                street_adjust_total = draw_adj + texture_dampen_factor * raw_texture_adj
            else:
                texture_dampen_factor = 1.0
                street_adjust_total = draw_adj + raw_texture_adj
                
            equity = _clamp(equity_before_adjust + street_adjust_total, 0.0, 0.98)
            
            # ── Gear-Based Post-Flop Adjustments (Phoenix Protocol) ──────
            if self._current_gear == "CHAOS":
                chaos_boost = 0.10
                if self._line_state.get('opp_discarded_pair', False):
                    # "Discard Trap" Inversion: Proceed with more confidence!
                    # They are mathematically incapable of having a set with the discarded rank.
                    chaos_boost += 0.15
                equity = _clamp(equity + chaos_boost, 0.0, 0.98)
            elif self._current_gear == "SNIPER":
                equity = _clamp(equity - 0.05, 0.0, 0.98) # slightly tighter

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
            # Lead protection: shrink bet sizing to keep pots small and preserve our buffer
            if protecting_lead:
                lead_size_scale = _clamp(1.0 - 0.30 * lead_ratio, 0.60, 1.0)
                rand_ctx = {k: v * lead_size_scale for k, v in rand_ctx.items()}
            pot_odds = to_call / (pot_ref + to_call) if to_call > 0 else 0.0
            bluff_catch = self._coeff(exploit_adj, "bluff_catch_adj", street)
            hero_fold = self._coeff(exploit_adj, "hero_fold_adj", street)
            value_sizing = self._coeff(exploit_adj, "value_bet_size_adj", street)
            bluff_freq = self._coeff(exploit_adj, "bluff_freq_adj", street)
            monster_gate = _clamp(MONSTER_THRESHOLD - 0.03 * value_sizing, 0.78, 0.86)
            strong_gate = _clamp(STRONG_THRESHOLD - 0.03 * value_sizing, 0.58, 0.72)
            good_gate = _clamp(GOOD_THRESHOLD - 0.03 * bluff_freq, 0.40, 0.54)
            urgency_call_adj = (-0.08 * urgency if in_comeback_mode else
                                0.05 if protecting_lead else 0.0)
            call_gate = _clamp(pot_odds + 0.10 * hero_fold - 0.10 * bluff_catch + urgency_call_adj, 0.0, 0.95)

            # Apply MDF Gate for Large Bets
            pot_before = max(2 * my_bet, 1)
            if to_call > 1.2 * pot_before:
                mdf = pot_before / float(pot_before + to_call)
                # Calculate the equity threshold required to defend the top MDF% of hands.
                # We use a conservative proxy: 1.0 - (mdf * 0.8)
                mdf_equity_threshold = max(0.40, 1.0 - (mdf * 0.8))
                
                # If opponent rarely bluffs (Nit), we relax the MDF enforcement
                # We calculate a combined aggregate proxy using turn/river/flop strengths
                profile = self._build_opponent_profile()
                opp_bluff_rate = 1.0 - ((
                    profile.get('river_bet_strength_large', 0.5) + 
                    profile.get('river_raise_strength', 0.5) +
                    profile.get('turn_barrel_strength', 0.5)
                ) / 3.0)
                if opp_bluff_rate < 0.25:
                    mdf_equity_threshold += 0.15  # Don't blindly call nits
                    
                call_gate = min(call_gate, mdf_equity_threshold)

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

            fire = False
            if to_call <= 0:
                fire, sb_action = self._semi_bluff_check(
                    my_cards, community, opp_discards, my_discards,
                    pot_size, to_call, street, valid, min_raise, max_raise, exploit_adj,
                    has_draw=has_draw, flush_outs_v=flush_outs, straight_outs_v=straight_outs,
                    rand_ctx=rand_ctx)
                if fire:
                    result = sb_action

            if result is None:
                if self._current_gear == "CHAOS" and equity > strong_gate and valid[RAISE]:
                    result = (RAISE, max_raise, 0, 0)
                elif equity > monster_gate:
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

        # ── Dynamic Bleed-Out Execution & Risk Cap ───────────────────────────
        if in_dynamic_bleedout and not valid[DISCARD]:
            if street >= 1:
                has_top_pair_plus = _is_top_pair_or_better(my_cards, community)
                strong_draw = (flush_outs >= 4 or straight_outs >= 4)
                if not has_top_pair_plus and not strong_draw:
                    if valid[FOLD]:
                        result = (FOLD, 0, 0, 0)
                    elif valid[CHECK]:
                        result = (CHECK, 0, 0, 0)
                
                # No bluffs: downgrade RAISE to CALL unless trips_plus
                if result[0] == RAISE and hand_cat != "trips_plus":
                    if valid[CALL] and opp_bet > my_bet:
                        result = (CALL, 0, 0, 0)
                    elif valid[CHECK]:
                        result = (CHECK, 0, 0, 0)
                    else:
                        result = (FOLD, 0, 0, 0)
            
            # Global Risk Cap: Check if the action commits more than the surplus
            cost = 0
            if result[0] == RAISE:
                cost = max(0, result[1] - my_bet)
            elif result[0] == CALL:
                cost = max(0, opp_bet - my_bet)
            
            opp_honesty = (profile.get("river_bet_strength_large", 0.6) + profile.get("river_raise_strength", 0.6)) / 2.0
            opp_bluff_rate = max(0.0, 1.0 - opp_honesty)
            risk_factor = 0.2 + (0.8 * opp_bluff_rate)
            adjusted_risk_cap = surplus_for_hand * risk_factor

            river_jam_override_active = False
            if street == 3 and equity > strong_gate and _is_top_pair_or_better(my_cards, community):
                river_jam_override_active = True
                
            if cost > adjusted_risk_cap and not river_jam_override_active:
                if valid[FOLD]:
                    result = (FOLD, 0, 0, 0)
                elif valid[CHECK]:
                    result = (CHECK, 0, 0, 0)

        if forced_lock_action is not None:
            result = forced_lock_action

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

        # ── Log decisions ─────────────────────────────────────────────────────
        if not valid[DISCARD]:
            if street == 0:
                self._log_event("preflop_decision",
                    hand=self._hands_completed,
                    position=self._last_position,
                    hole_cards=self._cards_to_str(my_cards),
                    is_early_phase=in_early_phase,
                    is_premium=bool(_has_any_premium(my_cards)),
                    is_premium_pair=bool(_has_premium_pair(my_cards)),
                    equity=self._pflog_eq,
                    eq_gate=self._pflog_gate,
                    open_prob=round(float(self._pflog_open_prob), 3) if self._pflog_open_prob is not None else None,
                    to_call=int(max(0, opp_bet - my_bet)),
                    pot=int(pot_size),
                    action=PokerEnv.ActionType(result[0]).name,
                    amount=int(result[1]),
                    reason=self._preflop_reason,
                    urgency=round(float(urgency), 3),
                    comeback_mode=in_comeback_mode,
                    protecting_lead=protecting_lead,
                    pressure_adj=float(self._coeff(exploit_adj, "preflop_pressure_adj", 0)),
                    preflop_defense_adj=float(self._coeff(exploit_adj, "preflop_defense_adj", 0)),
                    phoenix_state={
                        "current_gear": self._current_gear,
                        "hands_in_gear": int(self._hands_in_current_gear),
                        "projected_pnl": float(locals().get("projected_final", 0.0)),
                        "slope": float(locals().get("slope", 0.0))
                    },
                    bleedout_state={
                        "alpha": float(alpha),
                        "surplus": float(surplus_for_hand),
                        "opp_vpip": float(profile.get("opp_preflop_raise_rate", 0.5)),
                        "alpha_penalty": float(0.25 * alpha) if in_dynamic_bleedout else 0.0,
                        "bully_discount": float(0.5 * max(0.0, profile.get("opp_preflop_raise_rate", 0.5) - 0.5)) if in_dynamic_bleedout else 0.0
                    } if in_dynamic_bleedout else None,
                )
            elif street >= 1:
                active_exploits = {k: round(float(v), 4) for k, v in exploit_adj.items() if abs(v) > 0.001}
                self._log_event("postflop_decision",
                    hand=self._hands_completed,
                    street=street,
                    street_name=self._street_name(street),
                    position=self._last_position,
                    my_cards=self._cards_to_str(my_cards[:2]),
                    community=self._cards_to_str(community),
                    opp_last_action=opp_action if opp_action else "none",
                    raw_equity=round(float(equity_before_adjust), 4),
                    adj_equity=round(float(equity), 4),
                    hand_cat=hand_cat,
                    strength=strength,
                    has_draw=bool(has_draw),
                    flush_outs=int(flush_outs),
                    straight_outs=int(straight_outs),
                    texture_adj=round(float(street_adjust_total), 4),
                    texture_dampen_factor=round(float(texture_dampen_factor), 3),
                    texture_breakdown={
                        "draw": round(float(draw_adj), 4),
                        "monotone": round(float(board_monotone_penalty), 4),
                        "connected": round(float(board_connected_penalty), 4),
                        "flush_inference": round(float(opp_flush_inference_penalty), 4),
                    },
                    chaos_discard_trap_active=bool(locals().get("opp_discarded_pair", False)) if self._current_gear == "CHAOS" else False,
                    to_call=int(to_call),
                    pot=int(pot_ref),
                    pot_odds=round(float(pot_odds), 4),
                    thresholds={
                        "monster": round(float(monster_gate), 4),
                        "strong": round(float(strong_gate), 4),
                        "good": round(float(good_gate), 4),
                        "call": round(float(call_gate), 4),
                    },
                    exploit_adj_active=active_exploits,
                    baseline_action=self._action_category(baseline_result) if baseline_result else "none",
                    final_action=PokerEnv.ActionType(result[0]).name,
                    final_amount=int(result[1]),
                    baseline_changed=bool(baseline_result is not None and baseline_result[0] != result[0]),
                    semi_bluff_fired=bool(fire),
                    bleedout_lock=forced_lock_action is not None,
                    urgency=round(float(urgency), 3),
                    comeback_mode=in_comeback_mode,
                    protecting_lead=protecting_lead,
                    phoenix_state={
                        "current_gear": self._current_gear,
                        "hands_in_gear": int(self._hands_in_current_gear),
                        "projected_pnl": float(locals().get("projected_final", 0.0)),
                        "slope": float(locals().get("slope", 0.0))
                    },
                    bleedout_state={
                        "alpha": float(alpha),
                        "opp_bluff_rate": float(opp_bluff_rate) if 'opp_bluff_rate' in locals() else 0.0,
                        "risk_factor": float(risk_factor) if 'risk_factor' in locals() else 1.0,
                        "raw_surplus": float(surplus_for_hand),
                        "adjusted_risk_cap": float(adjusted_risk_cap) if 'adjusted_risk_cap' in locals() else 0.0,
                        "action_cost": float(cost) if 'cost' in locals() else 0.0,
                        "river_jam_override_active": bool(river_jam_override_active) if 'river_jam_override_active' in locals() else False
                    } if in_dynamic_bleedout else None,
                )

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