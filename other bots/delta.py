# DELTA V1: Frankenstein bot — merges OMICRoN V1 (LUT + parallel exact solver),
#           Genesis V2 (equity-aware strength, river logic, tiered draws, preflop
#           fixes), Meta V5 (safety caps, mid-hand override, smooth ramp-in), plus
#           a new adaptive opponent learning system with EMA stats, Bayesian
#           archetype posteriors, per-street profiling, regime detection, and
#           bet-sizing tell tracking.

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
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
PREFLOP_COMMIT_THRESHOLD = 20
TOTAL_HANDS = _PROFILE.get("total_hands", 1000)

EARLY_PHASE_HANDS = _PROFILE.get("early_phase_hands", 50)
EARLY_PREFLOP_MIN_EQUITY = _PROFILE.get("early_preflop_min_equity", 0.48)
NORMAL_PREFLOP_MIN_EQUITY = 0.45
EARLY_OPEN_MULTIPLIER = 1.20

RIVER_CALL_PREMIUM = 0.08

RANKS = "23456789A"

int_to_card = PokerEnv.int_to_card

# ── Card caches for fast evaluation ──────────────────────────────────────────

_INT_TO_TREYS = [PokerEnv.int_to_card(i) for i in range(DECK_SIZE)]
_INT_TO_TREYS_ALT = []
for _tc in _INT_TO_TREYS:
    _s = Card.int_to_str(_tc)
    _INT_TO_TREYS_ALT.append(Card.new(_s.replace("A", "T")))

_base_eval = Evaluator()

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


def _has_pair_of_rank(cards, rank):
    return sum(1 for c in cards if _RANK[c] == rank) >= 2


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
    if not raw:
        return ""
    s = raw.strip().lower()
    if s == "none":
        return ""
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


def _has_live_draw(my_cards, community, opp_discards, my_discards):
    suit_count, flush_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
    if suit_count >= 4 and flush_outs >= 2:
        return True
    run_count, straight_outs, _ = _count_straight_outs(my_cards, community, opp_discards, my_discards)
    if run_count >= 4 and straight_outs >= 3:
        return True
    return False


# ── String/int conversion ────────────────────────────────────────────────────

_RANKS_STR = "23456789A"
_SUITS_STR = "dhs"


def _str_to_int(card_str):
    rank_ch = card_str[0]
    suit_ch = card_str[1]
    return _RANKS_STR.index(rank_ch) + _SUITS_STR.index(suit_ch) * NUM_RANKS


# ── Adaptive Opponent Discard Model ──────────────────────────────────────────


class _OppDiscardModel:

    _FEATURE_NAMES = [
        "pair_with_board", "pocket_pair", "flush_draw",
        "suited", "connected", "high_card", "board_suit_match",
    ]
    _NUM_FEATURES = len(_FEATURE_NAMES)
    _DEFAULTS = [10.0, 8.0, 6.0, 3.0, 2.0, 4.0, 3.0]
    _ALPHA_BASE = 0.15
    _WEIGHT_FLOOR = 0.5

    def __init__(self):
        self.weights = list(self._DEFAULTS)
        self._alpha = self._ALPHA_BASE
        self._correct = 0
        self._total = 0

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
        self._total += 1
        if chosen_set == predicted_set:
            self._correct += 1
            self._adapt_learning_rate()
            return
        self._adapt_learning_rate()
        chosen_feats = self._extract_features(list(opp_kept), flop)
        predicted_feats = self._extract_features(predicted_best, flop)
        for i in range(self._NUM_FEATURES):
            diff = chosen_feats[i] - predicted_feats[i]
            self.weights[i] += self._alpha * diff
            self.weights[i] = max(self._WEIGHT_FLOOR, self.weights[i])

    def _adapt_learning_rate(self):
        if self._total < 5:
            return
        accuracy = self._correct / self._total
        if accuracy < 0.30:
            self._alpha = min(0.30, self._alpha * 1.1)
        elif accuracy > 0.60:
            self._alpha = max(0.08, self._alpha * 0.95)


# ── Adaptive Opponent Statistics (EMA-based) ─────────────────────────────────


class _AdaptiveStats:
    PRIORS = {
        "fold_to_bet":       0.35,
        "fold_to_raise":     0.35,
        "check_raise":       0.05,
        "call_down":         0.40,
        "opp_aggression":    0.25,
        "opp_avg_bet_frac":  0.50,
        "opp_preflop_raise": 0.30,
    }
    BASE_FAST_ALPHA = 0.12
    BASE_SLOW_ALPHA = 0.03
    MIN_OBS_FOR_RATE = 3

    def __init__(self):
        self._fast = {k: v for k, v in self.PRIORS.items()}
        self._slow = {k: v for k, v in self.PRIORS.items()}
        self._counts = {k: 0 for k in self.PRIORS}
        self.fast_alpha = self.BASE_FAST_ALPHA
        self.slow_alpha = self.BASE_SLOW_ALPHA

    def update(self, stat_name, value):
        if stat_name not in self._fast:
            self._fast[stat_name] = self.PRIORS.get(stat_name, 0.35)
            self._slow[stat_name] = self.PRIORS.get(stat_name, 0.35)
            self._counts[stat_name] = 0
        self._fast[stat_name] = (1 - self.fast_alpha) * self._fast[stat_name] + self.fast_alpha * value
        self._slow[stat_name] = (1 - self.slow_alpha) * self._slow[stat_name] + self.slow_alpha * value
        self._counts[stat_name] += 1

    def rate(self, stat_name):
        if self._counts.get(stat_name, 0) < self.MIN_OBS_FOR_RATE:
            return self.PRIORS.get(stat_name, 0.35)
        return self._fast[stat_name]

    def slow_rate(self, stat_name):
        if self._counts.get(stat_name, 0) < self.MIN_OBS_FOR_RATE:
            return self.PRIORS.get(stat_name, 0.35)
        return self._slow[stat_name]

    def count(self, stat_name):
        return self._counts.get(stat_name, 0)

    def total_obs(self):
        return sum(self._counts.values())

    def detect_regime_change(self):
        key_stats = ["fold_to_bet", "opp_aggression", "check_raise", "call_down"]
        divergences = []
        for stat in key_stats:
            if self._counts.get(stat, 0) >= 10:
                div = abs(self._fast[stat] - self._slow[stat])
                divergences.append(div)
        if not divergences:
            return False
        avg_div = sum(divergences) / len(divergences)
        if avg_div > 0.12:
            self.fast_alpha = min(0.25, self.fast_alpha * 1.5)
            return True
        else:
            self.fast_alpha = max(self.BASE_FAST_ALPHA, self.fast_alpha * 0.95)
            return False

    def update_bet_frac(self, stat_name, frac_value):
        if stat_name not in self._fast:
            self._fast[stat_name] = self.PRIORS.get(stat_name, 0.50)
            self._slow[stat_name] = self.PRIORS.get(stat_name, 0.50)
            self._counts[stat_name] = 0
        self._fast[stat_name] = (1 - self.fast_alpha) * self._fast[stat_name] + self.fast_alpha * frac_value
        self._slow[stat_name] = (1 - self.slow_alpha) * self._slow[stat_name] + self.slow_alpha * frac_value
        self._counts[stat_name] += 1


# ── Bayesian Archetype Posterior ─────────────────────────────────────────────


class _BayesianArchetype:
    ARCHETYPES = ("overfolder", "station", "maniac", "balanced")
    LIKELIHOODS = {
        "overfolder": {"fold": 0.55, "call": 0.25, "raise": 0.20},
        "station":    {"fold": 0.15, "call": 0.60, "raise": 0.25},
        "maniac":     {"fold": 0.10, "call": 0.30, "raise": 0.60},
        "balanced":   {"fold": 0.33, "call": 0.34, "raise": 0.33},
    }

    def __init__(self):
        self.probs = {a: 0.25 for a in self.ARCHETYPES}

    def update(self, observed_action):
        if observed_action not in ("fold", "call", "raise"):
            return
        total = 0.0
        for arch in self.ARCHETYPES:
            likelihood = self.LIKELIHOODS[arch].get(observed_action, 0.33)
            self.probs[arch] *= likelihood
            total += self.probs[arch]
        if total > 0:
            for arch in self.ARCHETYPES:
                self.probs[arch] /= total

    def dominant(self):
        return max(self.probs, key=self.probs.get)

    def prob(self, archetype):
        return self.probs.get(archetype, 0.0)

    def reset_uniform(self):
        self.probs = {a: 0.25 for a in self.ARCHETYPES}


# ── Bet-Sizing Tell Tracker ─────────────────────────────────────────────────


class _SizingTellTracker:
    MAX_HISTORY = 50

    def __init__(self):
        self._small = []
        self._medium = []
        self._large = []

    def record(self, bet_frac, hand_strength):
        if bet_frac < 0.40:
            bucket = self._small
        elif bet_frac < 0.75:
            bucket = self._medium
        else:
            bucket = self._large
        bucket.append(hand_strength)
        if len(bucket) > self.MAX_HISTORY:
            bucket.pop(0)

    def tell_adjustment(self, current_bet_frac):
        if current_bet_frac < 0.40:
            bucket = self._small
        elif current_bet_frac < 0.75:
            bucket = self._medium
        else:
            bucket = self._large
        if len(bucket) < 8:
            return 0.0
        avg_strength = sum(bucket) / len(bucket)
        if avg_strength > 0.70:
            return -0.06
        if avg_strength < 0.35:
            return 0.06
        return 0.0


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
try:
    import multiprocessing as _mp
    _mp_ctx = _mp.get_context("fork")
except (ValueError, AttributeError):
    _mp_ctx = None

_POOL = None

def _get_pool():
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=_NUM_CPUS, mp_context=_mp_ctx)
    return _POOL

import atexit, signal as _signal

def _shutdown_pool(*_args):
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False)
        _POOL = None

atexit.register(_shutdown_pool)
try:
    _signal.signal(_signal.SIGTERM, lambda s, f: (_shutdown_pool(), os._exit(0)))
except (OSError, ValueError):
    pass


# ── PlayerAgent ──────────────────────────────────────────────────────────────


class PlayerAgent(Agent):

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

        self._global_stats = _AdaptiveStats()
        self._street_stats = {s: _AdaptiveStats() for s in range(4)}
        self._bayesian = _BayesianArchetype()
        self._sizing_tells = _SizingTellTracker()

        self._opp_hand_aggr = 0.0
        self._opp_archetype = "default"
        self._opp_folded = False
        self._last_was_bet = False
        self._last_was_check = False
        self._last_street = 0
        self._hand_override = None
        self._position = "BB"
        self._cached_preflop_eq = None

        self._opp_last_bet_fracs = []

    # ── Opponent profiling helpers ───────────────────────────────────────────

    def _safe_rate(self, key, street=None):
        if street is not None and street in self._street_stats:
            ss = self._street_stats[street]
            if ss.count(key) >= 5:
                return ss.rate(key)
        return self._global_stats.rate(key)

    def _total_obs(self):
        return self._global_stats.total_obs()

    def _select_mode(self):
        total = self._total_obs()
        if total < 5:
            self._opp_archetype = "default"
            return

        ramp = min(1.0, (self._hands_completed - 5) / 7.0)
        if ramp < 0:
            ramp = 0.0

        self._global_stats.detect_regime_change()

        fold_to_bet = self._global_stats.rate("fold_to_bet")
        cr_rate = self._global_stats.rate("check_raise")
        call_down = self._global_stats.rate("call_down")
        opp_aggro = self._global_stats.rate("opp_aggression")
        pf_raise = self._global_stats.rate("opp_preflop_raise")

        bay_dom = self._bayesian.dominant()

        if opp_aggro > 0.45:
            self._opp_archetype = "maniac"
        elif fold_to_bet > 0.48:
            self._opp_archetype = "overfolder"
        elif fold_to_bet < 0.30 or call_down > 0.55:
            self._opp_archetype = "station"
        elif cr_rate > 0.10:
            self._opp_archetype = "maniac"
        elif opp_aggro < 0.15 and self._global_stats.count("opp_aggression") >= 8:
            self._opp_archetype = "overfolder"
        elif pf_raise > 0.50:
            self._opp_archetype = "default"
        elif pf_raise < 0.12:
            self._opp_archetype = "station"
        else:
            if self._bayesian.prob(bay_dom) > 0.40 and bay_dom != "balanced":
                self._opp_archetype = bay_dom
            else:
                self._opp_archetype = "default"

    def _process_opponent_action(self, observation, opp_action, last_was_bet, last_street):
        if not opp_action:
            return
        if opp_action == "fold":
            self._opp_folded = True

        current_street = observation.get("street", 0)
        canon = opp_action if opp_action in ("fold", "call", "raise") else "call"
        self._bayesian.update(canon)

        if last_was_bet:
            is_fold = 1.0 if opp_action == "fold" else 0.0
            self._global_stats.update("fold_to_bet", is_fold)
            self._global_stats.update("fold_to_raise", is_fold)
            self._street_stats[last_street].update("fold_to_bet", is_fold)
            self._street_stats[last_street].update("fold_to_raise", is_fold)
            if opp_action in ("call", "check", "raise"):
                if opp_action == "call" and last_street >= 1:
                    self._global_stats.update("call_down", 1.0)
                    self._street_stats[last_street].update("call_down", 1.0)
                elif last_street >= 1:
                    self._global_stats.update("call_down", 0.0)
                    self._street_stats[last_street].update("call_down", 0.0)

        last_was_check = self._last_was_check
        if last_was_check and opp_action == "raise":
            self._global_stats.update("check_raise", 1.0)
            self._street_stats[last_street].update("check_raise", 1.0)
            if self._hand_override is None:
                self._hand_override = "value_mode"
        elif last_was_check and opp_action in ("check", "call", "fold"):
            self._global_stats.update("check_raise", 0.0)
            self._street_stats[last_street].update("check_raise", 0.0)

        if last_was_bet and opp_action == "raise":
            if self._hand_override is None:
                self._hand_override = "value_mode"

        if opp_action in ("raise", "call", "check"):
            is_raise = 1.0 if opp_action == "raise" else 0.0
            self._global_stats.update("opp_aggression", is_raise)
            self._street_stats[current_street].update("opp_aggression", is_raise)
            if opp_action == "raise":
                opp_bet_obs = observation.get("opp_bet", 0)
                my_bet_obs = observation.get("my_bet", 0)
                pot_obs = observation.get("pot_size", opp_bet_obs + my_bet_obs)
                raise_size = max(0, opp_bet_obs - my_bet_obs)
                if pot_obs > 0 and raise_size > 0:
                    frac = raise_size / pot_obs
                    self._global_stats.update_bet_frac("opp_avg_bet_frac", frac)
                    self._opp_last_bet_fracs.append(frac)
                    if len(self._opp_last_bet_fracs) > 30:
                        self._opp_last_bet_fracs.pop(0)

        if last_street == 0 and opp_action in ("raise", "call", "check", "fold"):
            is_pf_raise = 1.0 if opp_action == "raise" else 0.0
            self._global_stats.update("opp_preflop_raise", is_pf_raise)

        if current_street > last_street:
            self._opp_hand_aggr *= 0.7

        if opp_action == "raise":
            self._opp_hand_aggr += 0.7
        elif opp_action == "call" and last_was_bet:
            self._opp_hand_aggr += 0.2

    # ── Observe + showdown learning ──────────────────────────────────────────

    def observe(self, observation, reward, terminated, truncated, info):
        raw = observation.get("opp_last_action", "")
        opp_action = _normalize_action(raw)
        if opp_action == "fold":
            self._opp_folded = True

        if terminated:
            self._running_pnl += int(reward)
            self._hands_completed += 1

            if self._opp_folded and self._last_was_bet:
                self._global_stats.update("fold_to_bet", 1.0)
                self._global_stats.update("fold_to_raise", 1.0)

            self._learn_from_showdown(observation, info)

            self._opp_folded = False
            self._last_was_bet = False
            self._last_was_check = False
            self._opp_hand_aggr = 0.0
            self._last_street = 0
            self._hand_override = None
            self._cached_preflop_eq = None

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

        if len(self._last_community) >= 5 and len(opp_kept) == 2:
            opp_rank = _lut_eval_7(list(opp_kept) + self._last_community[:5])
            opp_strength = 1.0 - (opp_rank / 7462.0)
            if self._opp_last_bet_fracs:
                avg_frac = sum(self._opp_last_bet_fracs) / len(self._opp_last_bet_fracs)
                self._sizing_tells.record(avg_frac, opp_strength)

        self._opp_last_bet_fracs = []

    def __name__(self):
        return "DELTA_V1"

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
                               opp_signal, num_sims=300):
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

        reject_nothing = opp_signal >= 2.0
        reject_one_pair = opp_signal >= 3.5
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

            board_5 = community_l + runout
            my_rank = _lut_eval_7(my_keep + board_5)
            opp_rank = _lut_eval_7(opp + board_5)
            if my_rank < opp_rank:
                wins += w
            elif my_rank == opp_rank:
                wins += 0.5 * w
            total_weight += w

        return wins / total_weight if total_weight > 0 else 0.5

    # ── Preflop equity (best-keep + cache) ───────────────────────────────────

    def _preflop_equity(self, my5):
        if self._cached_preflop_eq is not None:
            return self._cached_preflop_eq
        best_eq = 0.0
        dead = set(my5)
        for i, j in combinations(range(len(my5)), 2):
            keep = [my5[i], my5[j]]
            toss = [my5[k] for k in range(len(my5)) if k not in (i, j)]
            eq = self._compute_equity(keep, [], [], toss, num_sims=80)
            best_eq = max(best_eq, eq)
        self._cached_preflop_eq = best_eq
        return best_eq

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
            equity += min(0.14, outs * 0.035)
        elif street == 2 and has_draw and outs > 0:
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
    def _cat_to_strength(hand_cat, has_draw, equity=None):
        if equity is not None:
            if equity > MONSTER_THRESHOLD:
                return "monster"
            if equity > STRONG_THRESHOLD:
                return "strong"
            if equity > GOOD_THRESHOLD:
                return "strong" if hand_cat == "trips_plus" else "good"
            if has_draw:
                return "draw"
            return "weak"
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
                          has_draw=None, flush_outs_v=None, straight_outs_v=None):
        if self._hand_override is not None:
            return False, None
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

        fold_prob = self._safe_rate("fold_to_bet", street=street)
        pot = max(pot_size, 1)

        if fold_prob > 0.48:
            sizing_frac = random.uniform(0.65, 0.80)
        elif fold_prob < 0.25:
            sizing_frac = random.uniform(0.45, 0.60)
        else:
            sizing_frac = random.uniform(0.55, 0.70)

        bet_size = pot * sizing_frac
        ev = (fold_prob * pot
              + (1 - fold_prob) * (draw_eq * (pot + 2 * bet_size) - bet_size))

        if ev > 0:
            amt = _clamp(int(bet_size), min_raise, max_raise)
            return True, (RAISE, amt, 0, 0)
        return False, None

    def _dynamic_sizing(self, base_amount, strength, street, is_semi_bluff):
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
        noise = random.uniform(0.90, 1.10)
        return max(1, int(base_amount * mult * noise))

    # ── River-specific decision logic (from Genesis V2) ──────────────────────

    def _act_river(self, my_cards, community, opp_discards, my_discards,
                   valid, pot_size, my_bet, opp_bet, min_raise, max_raise,
                   equity, hand_cat, strength, has_draw):
        to_call = max(0, opp_bet - my_bet)
        pot_ref = max(pot_size, 1)
        pot_odds = to_call / (pot_ref + to_call) if to_call > 0 else 0.0
        arch = self._opp_archetype
        fold_rate = self._safe_rate("fold_to_bet", street=3)

        opp_bet_frac = to_call / pot_ref if to_call > 0 else 0.0
        tell_adj = self._sizing_tells.tell_adjustment(opp_bet_frac) if to_call > 0 else 0.0
        adj_equity = _clamp(equity + tell_adj, 0.0, 0.98)

        result = None

        if to_call <= 0:
            if adj_equity > MONSTER_THRESHOLD:
                if arch == "station":
                    frac = random.uniform(0.85, 1.00)
                elif arch == "overfolder":
                    frac = random.uniform(0.60, 0.75)
                else:
                    frac = random.uniform(0.70, 0.88)
                amt = _clamp(int(pot_ref * frac), min_raise, max_raise)
                if valid[RAISE]:
                    result = (RAISE, _clamp(self._dynamic_sizing(amt, strength, 3, False), min_raise, max_raise), 0, 0)
                elif valid[CALL]:
                    result = (CALL, 0, 0, 0)
                else:
                    result = (CHECK, 0, 0, 0)

            elif adj_equity > STRONG_THRESHOLD:
                if arch == "station":
                    frac = random.uniform(0.65, 0.80)
                elif arch == "overfolder":
                    frac = random.uniform(0.45, 0.60)
                else:
                    frac = random.uniform(0.55, 0.70)
                amt = _clamp(int(pot_ref * frac), min_raise, max_raise)
                if valid[RAISE]:
                    result = (RAISE, _clamp(self._dynamic_sizing(amt, strength, 3, False), min_raise, max_raise), 0, 0)
                else:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

            elif adj_equity > GOOD_THRESHOLD:
                if arch == "overfolder" and valid[RAISE]:
                    frac = random.uniform(0.35, 0.50)
                    amt = _clamp(int(pot_ref * frac), min_raise, max_raise)
                    result = (RAISE, _clamp(self._dynamic_sizing(amt, strength, 3, False), min_raise, max_raise), 0, 0)
                else:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

            else:
                if arch == "overfolder" and fold_rate > 0.55 and valid[RAISE]:
                    if random.random() < 0.25:
                        amt = _clamp(int(pot_ref * random.uniform(0.55, 0.70)), min_raise, max_raise)
                        result = (RAISE, amt, 0, 0)
                if result is None:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

        else:
            if adj_equity > MONSTER_THRESHOLD:
                if valid[RAISE]:
                    amt = _clamp(int(pot_ref * 1.0), min_raise, max_raise)
                    result = (RAISE, amt, 0, 0)
                elif valid[CALL]:
                    result = (CALL, 0, 0, 0)
                else:
                    result = (CHECK, 0, 0, 0)

            elif adj_equity > STRONG_THRESHOLD:
                if valid[CALL]:
                    result = (CALL, 0, 0, 0)
                else:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

            elif adj_equity > GOOD_THRESHOLD:
                if adj_equity >= pot_odds + RIVER_CALL_PREMIUM and to_call <= pot_ref * 0.45:
                    if valid[CALL]:
                        result = (CALL, 0, 0, 0)
                if result is None:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

            else:
                if adj_equity >= pot_odds + RIVER_CALL_PREMIUM + 0.05 and to_call <= pot_ref * 0.20:
                    if valid[CALL]:
                        result = (CALL, 0, 0, 0)
                if result is None:
                    result = (CHECK if valid[CHECK] else FOLD, 0, 0, 0)

        if result[0] == FOLD and to_call <= 0 and valid[CHECK]:
            result = (CHECK, 0, 0, 0)
        if result[0] == FOLD and my_bet > 0 and to_call > 0 and to_call <= pot_ref * 0.15:
            if valid[CALL]:
                result = (CALL, 0, 0, 0)

        return result

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

        self._last_my_cards = list(my_cards)
        if opp_discards:
            self._last_opp_discards = list(opp_discards)
        if community:
            self._last_community = list(community)
        pot_size = observation.get("pot_size", my_bet + opp_bet)

        in_early_phase = self._hands_completed < EARLY_PHASE_HANDS

        blind_pos = observation.get("blind_position", 1)
        self._position = "SB" if blind_pos == 0 else "BB"

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
        if opp_action:
            self._process_opponent_action(observation, opp_action,
                                          self._last_was_bet, self._last_street)

        if street == 0 and not valid[DISCARD]:
            self._select_mode()

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
                else:
                    weights = list(self._opp_model.weights)
                    futures = []
                    for i, j, keep, toss in all_keeps:
                        dead = dead_base | set(toss) | set(opp_discards)
                        fut = _get_pool().submit(_exact_discard_equity_weighted_lut,
                                           keep, community, dead, opp_discards,
                                           weights)
                        futures.append((i, j, fut))
                    best_eq = -1.0
                    best_ij = (0, 1)
                    for i, j, fut in futures:
                        eq = fut.result()
                        if eq > best_eq:
                            best_eq = eq
                            best_ij = (i, j)
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
                else:
                    top_n = 3 if sim_mode == "conservative" else 5
                    futures = []
                    for i, j, _, keep, toss in candidates[:top_n]:
                        dead = dead_base | set(toss) | opp_disc_set
                        fut = _get_pool().submit(_exact_discard_equity_lut,
                                           keep, community, dead)
                        futures.append((i, j, fut))
                    best_eq = -1.0
                    best_ij = (0, 1)
                    for i, j, fut in futures:
                        eq = fut.result()
                        if eq > best_eq:
                            best_eq = eq
                            best_ij = (i, j)

            return (DISCARD, 0, best_ij[0], best_ij[1])

        # ── Pre-flop (street 0) ──────────────────────────────────────────────
        result = None

        if street == 0:
            premium = _has_any_premium(my_cards)
            premium_pair = _has_premium_pair(my_cards)
            to_call = max(0, opp_bet - my_bet)

            if in_early_phase:
                preflop_eq = self._preflop_equity(my_cards) if len(my_cards) == 5 else 0.45
                if premium:
                    if premium_pair and opp_bet >= PREFLOP_COMMIT_THRESHOLD and valid[RAISE]:
                        if self._opp_archetype == "maniac" and not _has_pair_of_rank(my_cards, RANK_A):
                            amt = _clamp(int(to_call * 2.5 * random.uniform(0.9, 1.1)),
                                         min_raise, max(min_raise, max_raise // 3))
                            result = (RAISE, amt, 0, 0)
                        else:
                            result = (RAISE, max_raise, 0, 0)
                    elif to_call > 0 and valid[RAISE] and random.random() < 0.70:
                        amt = _clamp(int(to_call * 2.5 * random.uniform(0.9, 1.1)), min_raise, max_raise)
                        result = (RAISE, amt, 0, 0)
                    else:
                        noise = random.uniform(0.85, 1.15)
                        open_size = _clamp(int(max(10, STANDARD_OPEN) * EARLY_OPEN_MULTIPLIER * noise), min_raise, max_raise)
                        if valid[RAISE]:
                            result = (RAISE, open_size, 0, 0)
                        elif valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        else:
                            result = (CHECK, 0, 0, 0)
                elif preflop_eq >= EARLY_PREFLOP_MIN_EQUITY:
                    if valid[RAISE] and random.random() < 0.65:
                        amt = _clamp(int(9 * random.uniform(0.9, 1.1)), min_raise, max_raise)
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
                        if self._opp_archetype == "maniac" and not _has_pair_of_rank(my_cards, RANK_A):
                            amt = _clamp(int(to_call * 2.5 * random.uniform(0.9, 1.1)),
                                         min_raise, max(min_raise, max_raise // 3))
                            result = (RAISE, amt, 0, 0)
                        else:
                            result = (RAISE, max_raise, 0, 0)
                    elif premium_pair and random.random() < SLOW_PLAY_CHANCE:
                        if valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        else:
                            result = (CHECK, 0, 0, 0)
                    else:
                        noise = random.uniform(0.85, 1.15)
                        open_size = _clamp(int(max(10, STANDARD_OPEN) * noise), min_raise, max_raise)
                        if valid[RAISE]:
                            result = (RAISE, open_size, 0, 0)
                        elif valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        else:
                            result = (CHECK, 0, 0, 0)
                else:
                    preflop_eq = self._preflop_equity(my_cards) if len(my_cards) == 5 else 0.45

                    if self._position == "BB" and to_call <= 4 and preflop_eq >= 0.40 and valid[CALL]:
                        result = (CALL, 0, 0, 0)
                    elif preflop_eq >= NORMAL_PREFLOP_MIN_EQUITY:
                        if to_call <= 0 and valid[RAISE] and random.random() < 0.25:
                            amt = _clamp(int(8 * random.uniform(0.9, 1.1)), min_raise, max_raise)
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

        # ── Post-flop (streets 1-3) ──────────────────────────────────────────
        else:
            if len(my_cards) > 2:
                my_cards = my_cards[:2]

            dead = set(my_cards) | set(community) | set(opp_discards) | set(my_discards)

            opp_signal = 0.0 if self._opp_archetype == "maniac" else self._opp_hand_aggr
            eq_sims = 100 if sim_mode == "emergency" else (200 if sim_mode == "conservative" else 300)
            equity = self._compute_equity_ranged(
                my_cards, community, dead, opp_discards, opp_signal, num_sims=eq_sims)

            hand_cat = _hand_rank_category(my_cards, community)
            suit_count, flush_outs, _ = _count_flush_outs(my_cards, community, opp_discards, my_discards)
            run_count, straight_outs, _ = _count_straight_outs(my_cards, community, opp_discards, my_discards)
            has_draw = (suit_count >= 4 and flush_outs >= 2) or (run_count >= 4 and straight_outs >= 3)

            equity = self._street_adjust(equity, street, has_draw, flush_outs, straight_outs,
                                         hand_cat, my_cards, community, opp_discards)

            strength = self._cat_to_strength(hand_cat, has_draw, equity=equity)

            to_call = max(0, opp_bet - my_bet)
            pot_ref = max(pot_size, 1)
            pot_odds = to_call / (pot_ref + to_call) if to_call > 0 else 0.0

            if street == 3:
                result = self._act_river(
                    my_cards, community, opp_discards, my_discards,
                    valid, pot_size, my_bet, opp_bet, min_raise, max_raise,
                    equity, hand_cat, strength, has_draw,
                )
            else:
                if to_call <= 0:
                    fire, sb_action = self._semi_bluff_check(
                        my_cards, community, opp_discards, my_discards,
                        pot_size, to_call, street, valid, min_raise, max_raise,
                        has_draw=has_draw, flush_outs_v=flush_outs, straight_outs_v=straight_outs)
                    if fire:
                        result = sb_action

                if result is None:
                    if self._hand_override is not None:
                        if equity > MONSTER_THRESHOLD and valid[RAISE]:
                            bet_frac = random.uniform(0.55, 0.72) if street == 1 else random.uniform(0.70, 0.90)
                            base_amt = _clamp(int(pot_ref * bet_frac), min_raise, max_raise)
                            raise_amt = self._dynamic_sizing(base_amt, strength, street, False)
                            result = (RAISE, _clamp(raise_amt, min_raise, max_raise), 0, 0)
                        elif equity > STRONG_THRESHOLD and valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        elif equity >= pot_odds and to_call > 0 and valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        elif valid[CHECK]:
                            result = (CHECK, 0, 0, 0)
                        else:
                            result = (FOLD, 0, 0, 0)

                if result is None:
                    if equity > MONSTER_THRESHOLD:
                        if street == 1:
                            bet_frac = random.uniform(0.55, 0.72)
                        else:
                            bet_frac = random.uniform(0.70, 0.90)
                        base_amt = _clamp(int(pot_ref * bet_frac), min_raise, max_raise)
                        raise_amt = self._dynamic_sizing(base_amt, strength, street, False)
                        raise_amt = _clamp(raise_amt, min_raise, max_raise)
                        if valid[RAISE]:
                            result = (RAISE, raise_amt, 0, 0)
                        elif valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        else:
                            result = (CHECK, 0, 0, 0)

                    elif equity > STRONG_THRESHOLD:
                        if street == 1:
                            bet_frac = random.uniform(0.55, 0.72)
                        else:
                            bet_frac = random.uniform(0.65, 0.80)
                        base_amt = _clamp(int(pot_ref * bet_frac), min_raise, max_raise)
                        raise_amt = self._dynamic_sizing(base_amt, strength, street, False)
                        raise_amt = _clamp(raise_amt, min_raise, max_raise)
                        if valid[RAISE]:
                            result = (RAISE, raise_amt, 0, 0)
                        elif valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        else:
                            result = (CHECK, 0, 0, 0)

                    elif equity > GOOD_THRESHOLD:
                        bet_frac = random.uniform(0.30, 0.50)
                        base_amt = _clamp(int(pot_ref * bet_frac), min_raise, max_raise)
                        raise_amt = self._dynamic_sizing(base_amt, strength, street, False)
                        raise_amt = _clamp(raise_amt, min_raise, max_raise)
                        if to_call <= 0 and valid[RAISE]:
                            result = (RAISE, raise_amt, 0, 0)
                        elif to_call > 0 and equity >= pot_odds and valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        elif valid[CHECK]:
                            result = (CHECK, 0, 0, 0)
                        else:
                            result = (FOLD, 0, 0, 0)

                    elif equity >= pot_odds and to_call > 0 and to_call <= pot_ref * 0.35:
                        if valid[CALL]:
                            result = (CALL, 0, 0, 0)
                        elif valid[CHECK]:
                            result = (CHECK, 0, 0, 0)
                        else:
                            result = (FOLD, 0, 0, 0)

                    else:
                        if valid[CHECK]:
                            result = (CHECK, 0, 0, 0)
                        else:
                            result = (FOLD, 0, 0, 0)

            # ── Hand-category safety caps (from Meta V5) ─────────────────────
            if len(my_cards) == 2 and len(community) >= 3:
                cap_draw = _has_live_draw(my_cards, community, opp_discards, my_discards)

                if hand_cat == "one_pair":
                    if cap_draw:
                        if result[0] == CALL and to_call > pot_ref * 0.70:
                            if valid[FOLD]:
                                result = (FOLD, 0, 0, 0)
                    else:
                        if result[0] == RAISE:
                            if to_call > 0 and valid[CALL]:
                                result = (CALL, 0, 0, 0)
                            elif valid[CHECK]:
                                result = (CHECK, 0, 0, 0)
                        if result[0] == CALL and to_call > pot_ref * 0.25:
                            if valid[FOLD]:
                                result = (FOLD, 0, 0, 0)
                elif hand_cat == "nothing":
                    if cap_draw:
                        if result[0] == CALL and to_call > pot_ref * 0.50:
                            if valid[FOLD]:
                                result = (FOLD, 0, 0, 0)
                    else:
                        if to_call > 0:
                            if valid[FOLD]:
                                result = (FOLD, 0, 0, 0)
                        elif valid[CHECK]:
                            result = (CHECK, 0, 0, 0)

            # ── Decision guards (postflop only) ──────────────────────────────
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

        # ── Track our action ─────────────────────────────────────────────────
        self._last_was_bet = result[0] == RAISE
        self._last_was_check = result[0] == CHECK
        self._last_street = street

        return result


# Ensure module is in sys.modules for ProcessPoolExecutor pickle compatibility
import sys as _sys
if __name__ not in _sys.modules:
    from types import ModuleType as _MT
    _self_mod = _MT(__name__)
    _self_mod.__dict__.update(globals())
    _sys.modules[__name__] = _self_mod
