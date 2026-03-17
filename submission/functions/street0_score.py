"""
Street 0 Scoring System
=======================
Preflop hand-quality scoring for the 27-card poker variant.

Input:  hand5  – list of 5 ints in [0, 27)
        opponent_profile – OpponentProfile dict/object (or None)
Output: float in [0, 1] (higher = better), plus optional ScoreBreakdown
"""

from __future__ import annotations

import math
import os
import pickle
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

# Parallel Street 0: chunk size for flop batching (reduce IPC)
_FLOP_CHUNK_SIZE = 15
_MIN_FLOPS_FOR_PARALLEL = 20

# ---------------------------------------------------------------------------
# Import the engine's hand evaluator
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from treys import Card, Evaluator

from gym_env import PokerEnv, WrappedEval

# =========================================================================
# SECTION 1 – Constants & Card Utilities
# =========================================================================

DECK_SIZE = 27
NUM_RANKS = 9
NUM_SUITS = 3
RANKS_STR = "23456789A"
SUITS_STR = "dhs"

# --- Top-level base-score weights (spec §6) ---
W_FUTURE = 0.58
W_OPTIONALITY = 0.20
W_DISCARD = 0.14
W_REVEAL = 0.08

# --- Keep-score Q weights (spec §7) ---
A_G = 0.62
A_R = 0.20
A_L = 0.12
A_I = 0.06

# --- G sub-weights (spec §8) ---
LAMBDA_STREET1 = 0.15
LAMBDA_STREET2 = 0.25
LAMBDA_STREET3 = 0.60

# --- Within-class refinement (spec §8.2) ---
ETA = 0.30

# --- R sub-weights (spec §9) ---
B_STRAIGHT = 0.28
B_FLUSH = 0.24
B_FULLHOUSE = 0.20
B_MULTIPATH = 0.28

# --- L sub-weights (spec §10) ---
C_RANK = 0.27
C_SUIT = 0.24
C_CONN = 0.24
C_PORT = 0.25

# --- I sub-weights (spec §11) ---
D_KEEP = 0.45
D_DEAD = 0.20
D_PATH = 0.35

# --- Optionality aggregation weights (spec §12.2) ---
OPT_W1 = 0.55
OPT_W2 = 0.30
OPT_W3 = 0.15

# --- Street utility baselines (spec §8.1) ---
# Index 0 = Straight Flush … 7 = High Card
HAND_CLASS_NAMES = [
    "Straight Flush", "Full House", "Flush", "Three of a Kind",
    "Straight", "Two Pair", "One Pair", "High Card",
]

STREET_UTILITY: Dict[int, List[float]] = {
    1: [0.99879, 0.98562, 0.95938, 0.79013, 0.89469, 0.65109, 0.41503, 0.13193],
    2: [0.99570, 0.96062, 0.88380, 0.62941, 0.75954, 0.45831, 0.20758, 0.03802],
    3: [0.99063, 0.92160, 0.77999, 0.45266, 0.59436, 0.27398, 0.07250, 0.00584],
}

# --- treys LookupTable class boundaries ---
_CLASS_BOUNDS: List[Tuple[int, int]] = [
    (1, 10),       # 0 Straight Flush
    (167, 322),    # 1 Full House
    (323, 1599),   # 2 Flush
    (1610, 2467),  # 3 Three of a Kind
    (1600, 1609),  # 4 Straight
    (2468, 3325),  # 5 Two Pair
    (3326, 6185),  # 6 One Pair
    (6186, 7462),  # 7 High Card
]

# All six straight windows in this variant: A2345 23456 34567 45678 56789 6789A
# Represented as sets of rank indices (0=2, 1=3, …, 7=9, 8=A)
STRAIGHT_WINDOWS: List[set] = [
    {8, 0, 1, 2, 3},  # A2345
    {0, 1, 2, 3, 4},  # 23456
    {1, 2, 3, 4, 5},  # 34567
    {2, 3, 4, 5, 6},  # 45678
    {3, 4, 5, 6, 7},  # 56789
    {4, 5, 6, 7, 8},  # 6789A
]


def rank(card: int) -> int:
    """Rank index 0-8 (2=0, 3=1, …, 9=7, A=8)."""
    return card % NUM_RANKS


def suit(card: int) -> int:
    """Suit index 0-2 (d=0, h=1, s=2)."""
    return card // NUM_RANKS


def card_str(card: int) -> str:
    return RANKS_STR[rank(card)] + SUITS_STR[suit(card)]


def hand_rank_counts(cards: List[int]) -> Counter:
    return Counter(rank(c) for c in cards)


def hand_suit_counts(cards: List[int]) -> Counter:
    return Counter(suit(c) for c in cards)


# Shared evaluator singleton
_evaluator: Optional[WrappedEval] = None


def _get_evaluator() -> WrappedEval:
    global _evaluator
    if _evaluator is None:
        _evaluator = WrappedEval()
    return _evaluator


def _int_to_treys(card_int: int) -> int:
    return Card.new(PokerEnv.int_card_to_str(card_int))


# =========================================================================
# SECTION 2 – Hand Structure Features
# =========================================================================

def longest_run(rank_set: set) -> int:
    """Longest consecutive run among rank indices, treating Ace as both low (below 2) and high (above 9)."""
    if not rank_set:
        return 0
    expanded = set(rank_set)
    if 8 in expanded:
        expanded.add(-1)  # Ace-low sentinel
        expanded.add(9)   # Ace-high sentinel
    sorted_r = sorted(expanded)
    best = cur = 1
    for i in range(1, len(sorted_r)):
        if sorted_r[i] == sorted_r[i - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def count_straight_windows(rank_set: set) -> int:
    """How many of the 6 straight windows are fully covered."""
    return sum(1 for w in STRAIGHT_WINDOWS if w.issubset(rank_set))


def near_straight_windows(rank_set: set) -> int:
    """Windows needing exactly 1 more card."""
    return sum(1 for w in STRAIGHT_WINDOWS if len(w - rank_set) == 1)


def ace_flex_features(rank_set: set) -> dict:
    has_ace = 8 in rank_set
    low_connectors = rank_set & {0, 1, 2, 3}
    high_connectors = rank_set & {5, 6, 7}
    return {
        "has_ace": has_ace,
        "ace_low_support": len(low_connectors) if has_ace else 0,
        "ace_high_support": len(high_connectors) if has_ace else 0,
    }


def suit_concentration_entropy(suit_counts: Counter) -> float:
    """Shannon entropy of the suit distribution; lower = more concentrated."""
    total = sum(suit_counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for cnt in suit_counts.values():
        if cnt > 0:
            p = cnt / total
            ent -= p * math.log2(p)
    return ent


def hand_structure_features(hand5: List[int]) -> dict:
    """Compute all structural features for a 5-card starting hand."""
    rc = hand_rank_counts(hand5)
    sc = hand_suit_counts(hand5)
    rank_set = set(rc.keys())
    mults = sorted(rc.values(), reverse=True)

    max_suit = max(sc.values()) if sc else 0
    dominant_suit = max(sc, key=sc.get) if sc else 0
    highest_in_dominant = max(
        (rank(c) for c in hand5 if suit(c) == dominant_suit), default=0
    )

    ace_f = ace_flex_features(rank_set)

    return {
        "rank_counts": rc,
        "suit_counts": sc,
        "sorted_mults": mults,
        "has_trips": 3 in rc.values(),
        "has_two_pair": mults.count(2) >= 2 if len(mults) >= 2 else False,
        "has_one_pair": 2 in rc.values(),
        "all_distinct": max(rc.values()) == 1,
        "highest_rank": max(rank_set),
        "num_aces": rc.get(8, 0),
        "max_suit_count": max_suit,
        "num_suits_present": len(sc),
        "has_5_suited": max_suit == 5,
        "has_4_suited": max_suit == 4,
        "has_3_suited": max_suit >= 3,
        "dominant_suit": dominant_suit,
        "highest_in_dominant_suit": highest_in_dominant,
        "suit_entropy": suit_concentration_entropy(sc),
        "longest_run": longest_run(rank_set),
        "straight_windows_covered": count_straight_windows(rank_set),
        "near_straight_windows": near_straight_windows(rank_set),
        **ace_f,
    }


# =========================================================================
# SECTION 3 – Enumeration Helpers
# =========================================================================

def all_keeps(hand5: List[int]) -> List[Tuple[Tuple[int, int], List[int], List[int]]]:
    """
    Return all 10 keeps.
    Each entry: (keep_indices, keep_cards, discard_cards).
    """
    result = []
    for i, j in combinations(range(5), 2):
        keep = [hand5[i], hand5[j]]
        discard = [hand5[k] for k in range(5) if k != i and k != j]
        result.append(((i, j), keep, discard))
    return result


def remaining_cards(hand5: List[int]) -> List[int]:
    """Cards not in our hand (unknown 22 cards)."""
    hand_set = set(hand5)
    return [c for c in range(DECK_SIZE) if c not in hand_set]


def sample_or_enumerate_flops(
    remaining: List[int], n_samples: Optional[int] = None
) -> List[Tuple[int, ...]]:
    if n_samples is None:
        return list(combinations(remaining, 3))
    pool = list(combinations(remaining, 3))
    if n_samples >= len(pool):
        return pool
    return random.sample(pool, n_samples)


def sample_or_enumerate_turn_river(
    remaining: List[int], n_samples: Optional[int] = None
) -> List[Tuple[int, ...]]:
    if n_samples is None:
        return list(combinations(remaining, 2))
    pool = list(combinations(remaining, 2))
    if n_samples >= len(pool):
        return pool
    return random.sample(pool, n_samples)


# =========================================================================
# SECTION 4 – Hand Evaluation
# =========================================================================

def evaluate_hand(hole: List[int], board: List[int]) -> int:
    """Return treys rank score (lower = better) using WrappedEval."""
    ev = _get_evaluator()
    h = [_int_to_treys(c) for c in hole]
    b = [_int_to_treys(c) for c in board]
    return ev.evaluate(h, b)


def hand_class_from_score(score: int) -> int:
    """Map treys score → class index 0-7."""
    for idx, (lo, hi) in enumerate(_CLASS_BOUNDS):
        if lo <= score <= hi:
            return idx
    return 7  # fallback High Card


def within_class_percentile(score: int, cls: int) -> float:
    """Return b(x) in [0,1]; 1 = best in class."""
    lo, hi = _CLASS_BOUNDS[cls]
    span = hi - lo
    if span == 0:
        return 1.0
    return max(0.0, min(1.0, (hi - score) / span))


def refined_utility(street: int, score: int) -> float:
    """U_s(x) = u_s(class) + eta * gap_s(class) * b(x)."""
    cls = hand_class_from_score(score)
    u = STREET_UTILITY[street][cls]
    b = within_class_percentile(score, cls)
    gap = _class_gap(street, cls)
    return u + ETA * gap * b


def _class_gap(street: int, cls: int) -> float:
    """Gap to the next stronger class (0 for Straight Flush)."""
    if cls == 0:
        return 0.0
    utils = STREET_UTILITY[street]
    return utils[cls - 1] - utils[cls]


# =========================================================================
# SECTION 5 – G: Future Made-Hand Strength
# =========================================================================

def future_strength_G(
    keep: List[int],
    flop: Tuple[int, ...],
    remaining_after_flop: List[int],
    n_tr_samples: Optional[int] = None,
) -> float:
    """
    G(H,K,F) = λ1*U1(x1) + λ2*E_T[U2] + λ3*E_{T,R}[U3]
    """
    board3 = list(flop)

    # Street 1: keep(2) + flop(3) = 5 cards
    score1 = evaluate_hand(keep, board3)
    u1 = refined_utility(1, score1)

    # Enumerate or sample turn-river pairs
    tr_pairs = sample_or_enumerate_turn_river(remaining_after_flop, n_tr_samples)
    if not tr_pairs:
        return u1

    u2_sum = 0.0
    u3_sum = 0.0
    u2_count = 0
    u3_count = 0

    # Collect unique turns for street-2 scoring
    turn_scores: Dict[int, int] = {}
    for tr in tr_pairs:
        t = tr[0]
        if t not in turn_scores:
            board4 = board3 + [t]
            turn_scores[t] = evaluate_hand(keep, board4)

    # Street 2 expectations (average over unique turns)
    for t, sc in turn_scores.items():
        u2_sum += refined_utility(2, sc)
        u2_count += 1

    # Street 3 expectations (average over all turn-river pairs)
    for tr in tr_pairs:
        board5 = board3 + list(tr)
        sc3 = evaluate_hand(keep, board5)
        u3_sum += refined_utility(3, sc3)
        u3_count += 1

    e_u2 = u2_sum / u2_count if u2_count else u1
    e_u3 = u3_sum / u3_count if u3_count else e_u2

    return LAMBDA_STREET1 * u1 + LAMBDA_STREET2 * e_u2 + LAMBDA_STREET3 * e_u3


# =========================================================================
# SECTION 6 – R: Retained Structural Richness
# =========================================================================

def _live_straight_info(keep: List[int], flop: Tuple[int, ...], remaining: List[int]) -> float:
    """Straight richness r_straight in [0,1]."""
    combined_ranks = set(rank(c) for c in keep) | set(rank(c) for c in flop)
    remaining_ranks = set(rank(c) for c in remaining)

    total_score = 0.0
    for i, w in enumerate(STRAIGHT_WINDOWS):
        have = w & combined_ranks
        need = w - combined_ranks
        if len(need) == 0:
            total_score += 1.0
        elif len(need) == 1:
            live = need & remaining_ranks
            if live:
                weight = 0.7
                if 8 in w:
                    weight += 0.1
                total_score += weight
        elif len(need) == 2:
            live = need & remaining_ranks
            if len(live) == 2:
                total_score += 0.25
    return min(1.0, total_score / 3.0)


def _live_flush_info(keep: List[int], flop: Tuple[int, ...], remaining: List[int]) -> float:
    """Flush richness r_flush in [0,1]."""
    combined = list(keep) + list(flop)
    sc = hand_suit_counts(combined)
    remaining_suits = hand_suit_counts(remaining)

    best = 0.0
    for s in range(NUM_SUITS):
        have = sc.get(s, 0)
        live = remaining_suits.get(s, 0)
        if have >= 5:
            best = max(best, 1.0)
        elif have == 4 and live >= 1:
            highest = max((rank(c) for c in combined if suit(c) == s), default=0)
            best = max(best, 0.65 + 0.10 * (highest / 8.0))
        elif have == 3 and live >= 2:
            best = max(best, 0.25 + 0.05 * (live / 6.0))
    return min(1.0, best)


def _live_fullhouse_info(keep: List[int], flop: Tuple[int, ...], remaining: List[int]) -> float:
    """Full-house richness r_fullhouse in [0,1]."""
    combined = list(keep) + list(flop)
    rc = hand_rank_counts(combined)
    remaining_rc = hand_rank_counts(remaining)

    score = 0.0
    for r, cnt in rc.items():
        live = remaining_rc.get(r, 0)
        if cnt >= 3:
            score += 0.6
        elif cnt == 2:
            if live >= 1:
                score += 0.35
            else:
                score += 0.10
        elif cnt == 1 and live >= 2:
            score += 0.08
    return min(1.0, score / 1.5)


def _multipath_richness(keep: List[int], flop: Tuple[int, ...], remaining: List[int]) -> float:
    """Count distinct strong-category branches still alive."""
    combined = list(keep) + list(flop)
    rc = hand_rank_counts(combined)
    sc = hand_suit_counts(combined)
    combined_ranks = set(rc.keys())
    remaining_ranks = set(rank(c) for c in remaining)
    remaining_suit_counts = hand_suit_counts(remaining)

    paths = 0.0

    # Pair/trips path
    has_pair_potential = any(
        (cnt >= 2 or (cnt == 1 and remaining_ranks & {r}))
        for r, cnt in rc.items()
    )
    if has_pair_potential:
        paths += 1.0

    # Flush path
    flush_alive = any(
        sc.get(s, 0) + remaining_suit_counts.get(s, 0) >= 5 and sc.get(s, 0) >= 2
        for s in range(NUM_SUITS)
    )
    if flush_alive:
        paths += 1.0

    # Straight path
    straight_alive = any(
        len(w - combined_ranks) <= 2 and (w - combined_ranks).issubset(remaining_ranks)
        for w in STRAIGHT_WINDOWS
    )
    if straight_alive:
        paths += 1.0

    # Full house path
    fh_alive = sum(1 for cnt in rc.values() if cnt >= 2) >= 1 and any(
        remaining_ranks & {r} for r, cnt in rc.items()
    )
    if fh_alive:
        paths += 0.5

    return min(1.0, paths / 3.5)


def retained_richness_R(keep: List[int], flop: Tuple[int, ...], remaining: List[int]) -> float:
    r_s = _live_straight_info(keep, flop, remaining)
    r_f = _live_flush_info(keep, flop, remaining)
    r_h = _live_fullhouse_info(keep, flop, remaining)
    r_m = _multipath_richness(keep, flop, remaining)
    return B_STRAIGHT * r_s + B_FLUSH * r_f + B_FULLHOUSE * r_h + B_MULTIPATH * r_m


# =========================================================================
# SECTION 7 – L: Discard Opportunity Loss
# =========================================================================

def _rank_loss(keep: List[int], discard: List[int]) -> float:
    """How much rank multiplicity / strength is destroyed."""
    keep_rc = hand_rank_counts(keep)
    disc_rc = hand_rank_counts(discard)

    loss = 0.0
    for r, cnt in disc_rc.items():
        if r in keep_rc:
            continue
        rank_value = r / 8.0
        if cnt >= 2:
            loss += 0.4 + 0.15 * rank_value
        else:
            loss += 0.1 + 0.1 * rank_value
    return min(1.0, loss)


def _suit_loss(keep: List[int], discard: List[int], hand5: List[int]) -> float:
    """How much suit support is destroyed."""
    hand_sc = hand_suit_counts(hand5)
    keep_sc = hand_suit_counts(keep)
    disc_sc = hand_suit_counts(discard)

    dominant = max(hand_sc, key=hand_sc.get)
    dominant_count = hand_sc[dominant]
    kept_dominant = keep_sc.get(dominant, 0)
    lost_dominant = disc_sc.get(dominant, 0)

    loss = 0.0
    if dominant_count >= 4 and kept_dominant <= 1:
        loss += 0.5
    elif dominant_count >= 3 and kept_dominant == 0:
        loss += 0.35

    ace_suited_lost = any(rank(c) == 8 and suit(c) == dominant for c in discard)
    if ace_suited_lost and dominant_count >= 3:
        loss += 0.2

    high_suited_lost = sum(1 for c in discard if suit(c) == dominant and rank(c) >= 6)
    loss += 0.08 * high_suited_lost

    return min(1.0, loss)


def _connectivity_loss(keep: List[int], discard: List[int], hand5: List[int]) -> float:
    """How much straight connectivity is destroyed."""
    hand_ranks = set(rank(c) for c in hand5)
    keep_ranks = set(rank(c) for c in keep)
    disc_ranks = set(rank(c) for c in discard) - keep_ranks

    windows_before = sum(
        1 for w in STRAIGHT_WINDOWS if len(w - hand_ranks) <= 2
    )
    windows_after = sum(
        1 for w in STRAIGHT_WINDOWS if len(w - keep_ranks) <= 2
    )
    window_loss = max(0, windows_before - windows_after)

    bridge_lost = 0
    for r in disc_ranks:
        adjacent_count = sum(1 for dr in [-1, 1] if (r + dr) in keep_ranks)
        if adjacent_count >= 1:
            bridge_lost += 1

    loss = 0.15 * window_loss + 0.2 * bridge_lost

    if 8 in disc_ranks:
        if {0, 1} & keep_ranks or {6, 7} & keep_ranks:
            loss += 0.15

    return min(1.0, loss)


def _portfolio_loss(
    current_gr: float, all_gr_scores: List[float]
) -> float:
    """Normalized drop from best raw G+R keep score."""
    if len(all_gr_scores) <= 1:
        return 0.0
    best = max(all_gr_scores)
    worst = min(all_gr_scores)
    span = best - worst
    if span < 1e-9:
        return 0.0
    return max(0.0, (best - current_gr) / span)


def discard_loss_L(
    keep: List[int],
    discard: List[int],
    hand5: List[int],
    current_gr: float,
    all_gr_scores: List[float],
) -> float:
    l_r = _rank_loss(keep, discard)
    l_s = _suit_loss(keep, discard, hand5)
    l_c = _connectivity_loss(keep, discard, hand5)
    l_p = _portfolio_loss(current_gr, all_gr_scores)
    return C_RANK * l_r + C_SUIT * l_s + C_CONN * l_c + C_PORT * l_p


# =========================================================================
# SECTION 8 – I: Reveal-Information Cost
# =========================================================================

def _keep_inference_leakage(
    discard: List[int],
    flop: Tuple[int, ...],
    all_keeps_list: List[Tuple[Tuple[int, int], List[int], List[int]]],
) -> float:
    """
    Entropy-based keep-inference leakage.
    i_keep = 1 - H(P(K'|D,F)) / log2(10)
    """
    flop_ranks = set(rank(c) for c in flop)
    flop_suits = hand_suit_counts(list(flop))
    disc_ranks = Counter(rank(c) for c in discard)
    disc_suits = Counter(suit(c) for c in discard)

    scores = []
    for _, kc, dc in all_keeps_list:
        if sorted(dc) == sorted(discard):
            scores.append(1.0)
            continue
        plausibility = 1.0
        k_ranks = set(rank(c) for c in kc)
        k_suits = hand_suit_counts(kc)

        # Pair keep → plausible
        if len(k_ranks) == 1:
            plausibility *= 1.2

        # Flush draw: keep matches dominant flop suit
        dominant_flop_suit = max(flop_suits, key=flop_suits.get) if flop_suits else -1
        keep_dominant_count = k_suits.get(dominant_flop_suit, 0)
        if keep_dominant_count == 2:
            plausibility *= 1.3
        elif keep_dominant_count == 0 and flop_suits.get(dominant_flop_suit, 0) >= 2:
            plausibility *= 0.6

        # Connectivity: keep near flop ranks
        connectivity_bonus = sum(1 for r in k_ranks if any(abs(r - fr) <= 1 for fr in flop_ranks))
        plausibility *= (1.0 + 0.1 * connectivity_bonus)

        scores.append(max(0.01, plausibility))

    total = sum(scores)
    probs = [s / total for s in scores]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(scores))
    if max_entropy < 1e-9:
        return 0.0
    return max(0.0, 1.0 - entropy / max_entropy)


def _dead_card_signal(discard: List[int], flop: Tuple[int, ...]) -> float:
    """How much useful negative info the discard conveys."""
    disc_ranks = set(rank(c) for c in discard)
    disc_suits = Counter(suit(c) for c in discard)
    flop_ranks = set(rank(c) for c in flop)

    signal = 0.0

    # Suit branches killed: if all 3 discards share a suit, that suit is exposed as abandoned
    for s, cnt in disc_suits.items():
        if cnt >= 3:
            signal += 0.35
        elif cnt == 2:
            signal += 0.10

    # Rank duplication killed
    disc_rc = Counter(rank(c) for c in discard)
    for r, cnt in disc_rc.items():
        if cnt >= 2:
            signal += 0.15

    # Straight windows killed
    for w in STRAIGHT_WINDOWS:
        killed = disc_ranks & w
        if len(killed) >= 2 and not (w - disc_ranks - flop_ranks):
            signal += 0.10

    return min(1.0, signal)


def _path_reveal(keep: List[int], discard: List[int], flop: Tuple[int, ...]) -> float:
    """How obviously the discard reveals our strategy."""
    keep_ranks = set(rank(c) for c in keep)
    keep_suits = hand_suit_counts(keep)
    disc_ranks = set(rank(c) for c in discard)
    disc_suits = Counter(suit(c) for c in discard)
    flop_suits = hand_suit_counts(list(flop))
    flop_ranks = set(rank(c) for c in flop)

    reveal = 0.0

    # Obvious pair keep: both keep cards same rank
    if len(keep_ranks) == 1:
        reveal += 0.40

    # Obvious flush keep: both keep cards share suit with dominant flop suit
    for s in range(NUM_SUITS):
        if keep_suits.get(s, 0) == 2 and flop_suits.get(s, 0) >= 2:
            reveal += 0.35
            break

    # Obvious straight keep: keep cards are consecutive and near flop
    if len(keep_ranks) == 2:
        kr = sorted(keep_ranks)
        diff = kr[1] - kr[0]
        if diff <= 2:
            overlap = sum(1 for r in keep_ranks if any(abs(r - fr) <= 1 for fr in flop_ranks))
            if overlap >= 1:
                reveal += 0.15

    # If discard is very homogeneous (all same suit or sequential), strategy is transparent
    if max(disc_suits.values()) == 3:
        reveal += 0.10

    return min(1.0, reveal)


def reveal_cost_I(
    keep: List[int],
    discard: List[int],
    flop: Tuple[int, ...],
    all_keeps_list: List[Tuple[Tuple[int, int], List[int], List[int]]],
) -> float:
    i_k = _keep_inference_leakage(discard, flop, all_keeps_list)
    i_d = _dead_card_signal(discard, flop)
    i_p = _path_reveal(keep, discard, flop)
    return D_KEEP * i_k + D_DEAD * i_d + D_PATH * i_p


# =========================================================================
# SECTION 9 – Aggregation & Base Score
# =========================================================================

def keep_score_Q(
    g: float, r: float, l: float, i: float
) -> float:
    return A_G * g + A_R * r - A_L * l - A_I * i


def _compute_flop_chunk(
    args: Tuple[List[Tuple[int, ...]], List[int], Optional[int]],
) -> Tuple[float, float, float, float, Dict[Tuple[int, int], int]]:
    """
    Worker for parallel Street 0: process a chunk of flops.
    args = (flops_chunk, hand5, n_tr_samples). Returns aggregated contributions
    for this chunk so the main process can sum and merge best_keep_counts.
    Top-level and picklable for ProcessPoolExecutor.
    """
    flops_chunk, hand5, n_tr_samples = args
    keeps = all_keeps(hand5)
    rem = remaining_cards(hand5)
    v_future_sum = 0.0
    v_opt_sum = 0.0
    c_disc_sum = 0.0
    c_rev_sum = 0.0
    best_keep_counts: Counter = Counter()

    for flop in flops_chunk:
        flop_set = set(flop)
        rem_after_flop = [c for c in rem if c not in flop_set]

        gr_values: List[Tuple[float, float]] = []
        for _, kc, dc in keeps:
            g = future_strength_G(kc, flop, rem_after_flop, n_tr_samples)
            r = retained_richness_R(kc, flop, rem_after_flop)
            gr_values.append((g, r))

        all_gr_scores = [g + r for g, r in gr_values]

        q_values: List[float] = []
        for idx, ((ki, kc, dc), (g, r)) in enumerate(zip(keeps, gr_values)):
            current_gr = all_gr_scores[idx]
            l = discard_loss_L(kc, dc, hand5, current_gr, all_gr_scores)
            i = reveal_cost_I(kc, dc, flop, keeps)
            q = keep_score_Q(g, r, l, i)
            q_values.append(q)

        best_idx = max(range(len(q_values)), key=lambda x: q_values[x])
        best_keep_counts[keeps[best_idx][0]] += 1

        v_future_sum += max(q_values)

        sorted_q = sorted(q_values, reverse=True)
        v_opt = OPT_W1 * sorted_q[0]
        if len(sorted_q) > 1:
            v_opt += OPT_W2 * sorted_q[1]
        if len(sorted_q) > 2:
            v_opt += OPT_W3 * sorted_q[2]
        v_opt_sum += v_opt

        best_kc = keeps[best_idx][1]
        best_dc = keeps[best_idx][2]
        best_gr = all_gr_scores[best_idx]
        c_disc_sum += discard_loss_L(best_kc, best_dc, hand5, best_gr, all_gr_scores)
        c_rev_sum += reveal_cost_I(best_kc, best_dc, flop, keeps)

    return (
        v_future_sum,
        v_opt_sum,
        c_disc_sum,
        c_rev_sum,
        dict(best_keep_counts),
    )


def _get_n_workers() -> int:
    """Number of workers for parallel Street 0 (1 = sequential). Capped 1-4."""
    try:
        n = int(os.environ.get("POKER_N_WORKERS", "0"))
    except (ValueError, TypeError):
        n = 0
    if n <= 0:
        n = min(2, os.cpu_count() or 2)
    return max(1, min(n, 4))


# Reused pool (created on first parallel use, reused for process lifetime)
_executor: Optional[ProcessPoolExecutor] = None


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=_get_n_workers())
    return _executor


def compute_base_street0_score(
    hand5: List[int],
    n_flop_samples: Optional[int] = 150,
    n_tr_samples: Optional[int] = 50,
) -> Tuple[float, "ScoreBreakdown"]:
    """
    Compute the full base Street 0 score for a 5-card starting hand.
    Returns (raw_score, breakdown).
    Uses parallel flop chunks when len(flops) >= _MIN_FLOPS_FOR_PARALLEL and
    POKER_N_WORKERS > 1 (env POKER_N_WORKERS, default 2).
    """
    keeps = all_keeps(hand5)
    rem = remaining_cards(hand5)
    flops = sample_or_enumerate_flops(rem, n_flop_samples)

    if not flops:
        return 0.0, ScoreBreakdown()

    n_flops = len(flops)
    n_workers = _get_n_workers()
    use_parallel = n_flops >= _MIN_FLOPS_FOR_PARALLEL and n_workers > 1

    if use_parallel:
        # Chunk flops (10–15 per task to limit IPC)
        chunks: List[List[Tuple[int, ...]]] = []
        for i in range(0, n_flops, _FLOP_CHUNK_SIZE):
            chunks.append(flops[i : i + _FLOP_CHUNK_SIZE])
        args_list: List[Tuple[List[Tuple[int, ...]], List[int], Optional[int]]] = [
            (chunk, hand5, n_tr_samples) for chunk in chunks
        ]
        executor = _get_executor()
        chunk_results = list(executor.map(_compute_flop_chunk, args_list))
        v_future_sum = sum(r[0] for r in chunk_results)
        v_opt_sum = sum(r[1] for r in chunk_results)
        c_disc_sum = sum(r[2] for r in chunk_results)
        c_rev_sum = sum(r[3] for r in chunk_results)
        best_keep_counts = Counter()
        for r in chunk_results:
            best_keep_counts.update(r[4])
    else:
        v_future_sum = 0.0
        v_opt_sum = 0.0
        c_disc_sum = 0.0
        c_rev_sum = 0.0
        best_keep_counts = Counter()
        for flop in flops:
            flop_set = set(flop)
            rem_after_flop = [c for c in rem if c not in flop_set]

            gr_values: List[Tuple[float, float]] = []
            for _, kc, dc in keeps:
                g = future_strength_G(kc, flop, rem_after_flop, n_tr_samples)
                r = retained_richness_R(kc, flop, rem_after_flop)
                gr_values.append((g, r))

            all_gr_scores = [g + r for g, r in gr_values]

            q_values: List[float] = []
            for idx, ((ki, kc, dc), (g, r)) in enumerate(zip(keeps, gr_values)):
                current_gr = all_gr_scores[idx]
                l = discard_loss_L(kc, dc, hand5, current_gr, all_gr_scores)
                i = reveal_cost_I(kc, dc, flop, keeps)
                q = keep_score_Q(g, r, l, i)
                q_values.append(q)

            best_idx = max(range(len(q_values)), key=lambda x: q_values[x])
            best_keep_counts[keeps[best_idx][0]] += 1

            v_future_sum += max(q_values)

            sorted_q = sorted(q_values, reverse=True)
            v_opt = OPT_W1 * sorted_q[0]
            if len(sorted_q) > 1:
                v_opt += OPT_W2 * sorted_q[1]
            if len(sorted_q) > 2:
                v_opt += OPT_W3 * sorted_q[2]
            v_opt_sum += v_opt

            best_kc = keeps[best_idx][1]
            best_dc = keeps[best_idx][2]
            best_gr = all_gr_scores[best_idx]
            c_disc_sum += discard_loss_L(best_kc, best_dc, hand5, best_gr, all_gr_scores)
            c_rev_sum += reveal_cost_I(best_kc, best_dc, flop, keeps)

    n = n_flops
    v_future = v_future_sum / n
    v_opt = v_opt_sum / n
    c_disc = c_disc_sum / n
    c_rev = c_rev_sum / n

    raw = W_FUTURE * v_future + W_OPTIONALITY * v_opt - W_DISCARD * c_disc - W_REVEAL * c_rev

    breakdown = ScoreBreakdown(
        v_future=v_future,
        v_optionality=v_opt,
        c_discard=c_disc,
        c_reveal=c_rev,
        s_base=raw,
        best_keep_counts=dict(best_keep_counts),
    )
    return raw, breakdown


# =========================================================================
# SECTION 10 – Opponent Model
# =========================================================================

@dataclass
class OpponentProfile:
    """Stores raw opponent stats and supports smoothed estimation."""

    # Preflop stats
    vpip_opportunities: int = 0
    vpip_successes: int = 0
    pfr_opportunities: int = 0
    pfr_successes: int = 0

    # Aggression
    raise_count: int = 0
    call_count: int = 0

    # Non-river betting
    non_river_bet_opportunities: int = 0
    non_river_bet_successes: int = 0

    # Fold to bet
    fold_non_river_opportunities: int = 0
    fold_non_river_successes: int = 0
    fold_river_opportunities: int = 0
    fold_river_successes: int = 0

    # Showdown reach
    showdown_opportunities: int = 0
    showdown_reached: int = 0

    # Preflop response by sizing (fold/call/raise for limp, small, medium, large)
    preflop_limp_faced: int = 0
    preflop_limp_fold: int = 0
    preflop_limp_call: int = 0
    preflop_limp_raise: int = 0

    preflop_small_open_faced: int = 0
    preflop_small_open_fold: int = 0
    preflop_small_open_call: int = 0
    preflop_small_open_raise: int = 0

    preflop_medium_open_faced: int = 0
    preflop_medium_open_fold: int = 0
    preflop_medium_open_call: int = 0
    preflop_medium_open_raise: int = 0

    preflop_large_open_faced: int = 0
    preflop_large_open_fold: int = 0
    preflop_large_open_call: int = 0
    preflop_large_open_raise: int = 0

    # Postflop continuation
    cbet_opportunities: int = 0
    cbet_successes: int = 0
    fold_to_flop_bet_opportunities: int = 0
    fold_to_flop_bet_successes: int = 0
    fold_to_turn_bet_opportunities: int = 0
    fold_to_turn_bet_successes: int = 0

    # Board texture reactions (fold counts / opportunities)
    paired_board_faced: int = 0
    paired_board_fold: int = 0
    suited_board_faced: int = 0
    suited_board_fold: int = 0
    connected_board_faced: int = 0
    connected_board_fold: int = 0
    disconnected_board_faced: int = 0
    disconnected_board_fold: int = 0

    # Discard-reaction (villain response to our discard class)
    react_pair_keep_faced: int = 0
    react_pair_keep_fold: int = 0
    react_pair_keep_raise: int = 0
    react_flush_keep_faced: int = 0
    react_flush_keep_fold: int = 0
    react_flush_keep_raise: int = 0
    react_straight_keep_faced: int = 0
    react_straight_keep_fold: int = 0
    react_straight_keep_raise: int = 0
    react_ambiguous_faced: int = 0
    react_ambiguous_fold: int = 0
    react_ambiguous_raise: int = 0

    # Recency ring buffers (last N outcomes: 1=success, 0=fail)
    recent_vpip_100: List[int] = field(default_factory=list)
    recent_pfr_100: List[int] = field(default_factory=list)
    recent_fold_nr_100: List[int] = field(default_factory=list)
    recent_fold_r_100: List[int] = field(default_factory=list)

    # Total hands
    total_hands: int = 0


def smoothed_rate(successes: int, n: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    """Beta-posterior mean: (successes + alpha) / (n + alpha + beta)."""
    return (successes + alpha) / (n + alpha + beta)


def _recent_rate(buf: List[int], window: int = 100) -> float:
    if not buf:
        return 0.5
    recent = buf[-window:]
    return sum(recent) / len(recent)


def get_vpip(p: OpponentProfile) -> float:
    return smoothed_rate(p.vpip_successes, p.vpip_opportunities)


def get_pfr(p: OpponentProfile) -> float:
    return smoothed_rate(p.pfr_successes, p.pfr_opportunities)


def get_af(p: OpponentProfile) -> float:
    if p.call_count == 0:
        return min(5.0, p.raise_count) if p.raise_count > 0 else 1.0
    return min(5.0, p.raise_count / p.call_count)


def get_non_river_bet_pct(p: OpponentProfile) -> float:
    return smoothed_rate(p.non_river_bet_successes, p.non_river_bet_opportunities)


def get_fold_non_river(p: OpponentProfile) -> float:
    return smoothed_rate(p.fold_non_river_successes, p.fold_non_river_opportunities)


def get_fold_river(p: OpponentProfile) -> float:
    return smoothed_rate(p.fold_river_successes, p.fold_river_opportunities)


def get_showdown_reach(p: OpponentProfile) -> float:
    return smoothed_rate(p.showdown_reached, p.showdown_opportunities)


# --- Profile indices (spec §18) ---

def pressure_index(p: OpponentProfile) -> float:
    """0-1; higher = more pressure they apply."""
    pfr = get_pfr(p)
    af = get_af(p) / 5.0
    nr_bet = get_non_river_bet_pct(p)
    return min(1.0, 0.35 * pfr + 0.35 * af + 0.30 * nr_bet)


def foldability_index(p: OpponentProfile) -> float:
    """0-1; higher = they fold more (exploitable with bluffs)."""
    fnr = get_fold_non_river(p)
    fr = get_fold_river(p)
    return min(1.0, 0.55 * fnr + 0.45 * fr)


def stickiness_index(p: OpponentProfile) -> float:
    """0-1; higher = they call down more (station-like)."""
    vpip = get_vpip(p)
    sd = get_showdown_reach(p)
    low_fold = 1.0 - foldability_index(p)
    return min(1.0, 0.30 * vpip + 0.35 * sd + 0.35 * low_fold)


def transparency_exploitation_index(p: OpponentProfile) -> float:
    """0-1; higher = they exploit our transparent discards well."""
    pair_raise = smoothed_rate(p.react_pair_keep_raise, p.react_pair_keep_faced, 0.5, 1.0)
    flush_raise = smoothed_rate(p.react_flush_keep_raise, p.react_flush_keep_faced, 0.5, 1.0)
    ambig_fold = smoothed_rate(p.react_ambiguous_fold, p.react_ambiguous_faced, 0.5, 1.0)
    return min(1.0, 0.35 * pair_raise + 0.35 * flush_raise + 0.30 * (1.0 - ambig_fold))


def discard_weakness_index(p: OpponentProfile) -> float:
    """0-1; higher = they make bad discard decisions (we can exploit texture)."""
    # Proxy: if they fold a lot on boards that should favor strong discarders,
    # they probably have weak discard strategy
    paired_fold = smoothed_rate(p.paired_board_fold, p.paired_board_faced, 0.5, 1.0)
    suited_fold = smoothed_rate(p.suited_board_fold, p.suited_board_faced, 0.5, 1.0)
    return min(1.0, 0.50 * paired_fold + 0.50 * suited_fold)


def volatility_index(p: OpponentProfile) -> float:
    """0-1; higher = they play aggressively / wildly."""
    pfr = get_pfr(p)
    af = get_af(p) / 5.0
    return min(1.0, 0.50 * pfr + 0.50 * af)


# --- Per-feature confidence (spec §20) ---

def _feature_confidence(n: int, k: float = 30.0) -> float:
    return n / (n + k)


def opponent_confidence(p: OpponentProfile) -> float:
    """Overall model confidence λ_θ ∈ [0, 0.80]."""
    confs = [
        _feature_confidence(p.vpip_opportunities),
        _feature_confidence(p.pfr_opportunities),
        _feature_confidence(p.raise_count + p.call_count),
        _feature_confidence(p.fold_non_river_opportunities),
        _feature_confidence(p.fold_river_opportunities),
    ]
    avg = sum(confs) / len(confs) if confs else 0.0
    return min(0.80, avg)


# --- Opponent-adjusted score components (spec §19) ---

def opponent_adjusted_future_value(
    v_future: float, p: OpponentProfile, hand_features: dict
) -> float:
    pi = pressure_index(p)
    fi = foldability_index(p)
    si = stickiness_index(p)
    vi = volatility_index(p)

    adj = v_future

    # High pressure → discount fragile speculative hands
    fragility = 1.0 - hand_features.get("longest_run", 0) / 5.0
    adj -= 0.08 * pi * fragility

    # Overfolder → boost flexible hands
    flexibility = hand_features.get("near_straight_windows", 0) / 6.0
    adj += 0.06 * fi * flexibility

    # Sticky opponent → boost robust value hands
    robustness = 0.0
    if hand_features.get("has_trips"):
        robustness = 0.9
    elif hand_features.get("has_two_pair"):
        robustness = 0.6
    elif hand_features.get("has_one_pair"):
        robustness = 0.3
    adj += 0.05 * si * robustness

    # High volatility → discount fragile hands more
    adj -= 0.04 * vi * fragility

    return max(0.0, adj)


def opponent_adjusted_optionality(
    v_opt: float, p: OpponentProfile, hand_features: dict
) -> float:
    tei = transparency_exploitation_index(p)
    fi = foldability_index(p)

    adj = v_opt

    # If villain misreads ambiguity, optionality is more valuable
    adj += 0.04 * (1.0 - tei)

    # Against folders, multi-path hands gain more
    multi = hand_features.get("num_suits_present", 1) / 3.0
    adj += 0.03 * fi * multi

    return max(0.0, adj)


def opponent_adjusted_discard_cost(
    c_disc: float, p: OpponentProfile
) -> float:
    pi = pressure_index(p)
    adj = c_disc
    # High-pressure villain punishes capped ranges harder
    adj *= (1.0 + 0.15 * pi)
    return adj


def opponent_adjusted_reveal_cost(
    c_reveal: float, p: OpponentProfile
) -> float:
    tei = transparency_exploitation_index(p)
    adj = c_reveal
    # Villain who exploits reveals well → increase cost; ignores → decrease
    adj *= (0.5 + tei)
    return adj


def opponent_adjusted_score(
    breakdown: "ScoreBreakdown",
    p: OpponentProfile,
    hand_features: dict,
) -> float:
    v_f = opponent_adjusted_future_value(breakdown.v_future, p, hand_features)
    v_o = opponent_adjusted_optionality(breakdown.v_optionality, p, hand_features)
    c_d = opponent_adjusted_discard_cost(breakdown.c_discard, p)
    c_r = opponent_adjusted_reveal_cost(breakdown.c_reveal, p)
    return W_FUTURE * v_f + W_OPTIONALITY * v_o - W_DISCARD * c_d - W_REVEAL * c_r


# =========================================================================
# SECTION 11 – Final Score, Classification & Caching
# =========================================================================

@dataclass
class ScoreBreakdown:
    v_future: float = 0.0
    v_optionality: float = 0.0
    c_discard: float = 0.0
    c_reveal: float = 0.0
    s_base: float = 0.0
    s_opp: float = 0.0
    s_final: float = 0.0
    confidence: float = 0.0
    best_keep_counts: dict = field(default_factory=dict)


# --- Global normalization bounds (estimated, can be recomputed offline) ---
_GLOBAL_MIN = -0.05
_GLOBAL_MAX = 0.65


def normalize_score(raw: float, lo: float = _GLOBAL_MIN, hi: float = _GLOBAL_MAX) -> float:
    if hi - lo < 1e-9:
        return 0.5
    return max(0.0, min(1.0, (raw - lo) / (hi - lo)))


def final_street0_score(
    hand5: List[int],
    opponent_profile: Optional[OpponentProfile] = None,
    n_flop_samples: Optional[int] = 150,
    n_tr_samples: Optional[int] = 50,
) -> Tuple[float, ScoreBreakdown]:
    """
    Main entry point.

    Args:
        hand5: list of 5 card ints in [0, 27)
        opponent_profile: OpponentProfile or None for base-only scoring
        n_flop_samples: None for exact enumeration, int for sampled
        n_tr_samples: None for exact enumeration, int for sampled

    Returns:
        (score_0_to_1, breakdown)
    """
    raw_base, breakdown = compute_base_street0_score(hand5, n_flop_samples, n_tr_samples)
    s_base = normalize_score(raw_base)
    breakdown.s_base = s_base

    if opponent_profile is None:
        breakdown.s_final = s_base
        return s_base, breakdown

    features = hand_structure_features(hand5)
    s_opp_raw = opponent_adjusted_score(breakdown, opponent_profile, features)
    s_opp = normalize_score(s_opp_raw)
    breakdown.s_opp = s_opp

    lam = opponent_confidence(opponent_profile)
    breakdown.confidence = lam

    s_final = (1.0 - lam) * s_base + lam * s_opp
    breakdown.s_final = s_final
    return s_final, breakdown


# --- Discard-class labels (spec §26) ---

def classify_discard(
    keep: List[int], discard: List[int], community: List[int]
) -> str:
    keep_ranks = set(rank(c) for c in keep)
    keep_suits = hand_suit_counts(keep)
    disc_ranks = set(rank(c) for c in discard)
    disc_suits = hand_suit_counts(discard)
    comm_suits = hand_suit_counts(community)
    comm_ranks = set(rank(c) for c in community)

    # Obvious pair keep
    if len(keep_ranks) == 1:
        return "obvious_pair_keep"

    # Obvious flush keep
    for s in range(NUM_SUITS):
        if keep_suits.get(s, 0) == 2 and comm_suits.get(s, 0) >= 2:
            return "obvious_flush_keep"

    # Obvious straight keep
    if len(keep_ranks) == 2:
        kr = sorted(keep_ranks)
        if kr[1] - kr[0] <= 2:
            overlap = sum(1 for r in keep_ranks if any(abs(r - cr) <= 2 for cr in comm_ranks))
            if overlap >= 1:
                return "obvious_straight_keep"

    # Suit abandon
    if max(disc_suits.values(), default=0) == 3:
        return "suit_abandon"

    # Rank duplication abandon
    disc_rc = Counter(rank(c) for c in discard)
    if any(cnt >= 2 for cnt in disc_rc.values()):
        return "rank_duplication_abandon"

    # Connectivity abandon
    sorted_disc = sorted(disc_ranks)
    if len(sorted_disc) >= 2 and sorted_disc[-1] - sorted_disc[0] <= 3:
        return "connectivity_abandon"

    # Weak-looking keep: neither card is high
    if all(rank(c) < 5 for c in keep):
        return "weak_looking_keep"

    return "ambiguous_keep"


# --- Board-texture labels (spec §27) ---

def classify_board_texture(community: List[int]) -> str:
    if not community:
        return "no_board"

    rc = hand_rank_counts(community)
    sc = hand_suit_counts(community)
    rank_set = set(rc.keys())
    max_suit = max(sc.values()) if sc else 0

    is_paired = any(cnt >= 2 for cnt in rc.values())
    is_mono = max_suit >= 3 and len(community) <= 3
    is_two_suited = max_suit == 2 and len(sc) <= 2

    is_connected = False
    sorted_r = sorted(rank_set)
    if len(sorted_r) >= 3:
        for i in range(len(sorted_r) - 2):
            if sorted_r[i + 2] - sorted_r[i] <= 4:
                is_connected = True
                break

    if len(community) <= 3:
        if is_mono:
            return "monotone"
        if is_paired:
            return "paired_flop"
        if is_connected and is_two_suited:
            return "connected_two_suited"
        if is_connected:
            return "connected_flop"
        if is_two_suited:
            return "two_suited_flop"
        return "rainbow_disconnected"

    # Turn / river labels
    if len(community) == 4:
        sc4 = hand_suit_counts(community)
        if max(sc4.values()) >= 4:
            return "flush_completing_turn"
        if is_paired and any(cnt >= 2 for cnt in hand_rank_counts(community[-1:]).values()):
            return "pair_making_turn"
        return "turn_" + ("connected" if is_connected else "disconnected")

    if len(community) == 5:
        sc5 = hand_suit_counts(community)
        if max(sc5.values()) >= 5:
            return "flush_completed_river"
        sorted_all = sorted(set(rank(c) for c in community))
        if len(sorted_all) >= 5:
            for i in range(len(sorted_all) - 4):
                if sorted_all[i + 4] - sorted_all[i] == 4:
                    return "straight_completed_river"
        return "river_standard"

    return "unknown_texture"


# --- Score cache ---

_score_cache: Dict[Tuple[int, ...], Tuple[float, ScoreBreakdown]] = {}


def get_cached_base_score(
    hand5: List[int],
    n_flop_samples: Optional[int] = 150,
    n_tr_samples: Optional[int] = 50,
) -> Tuple[float, ScoreBreakdown]:
    key = tuple(sorted(hand5))
    if key in _score_cache:
        return _score_cache[key]
    raw, bd = compute_base_street0_score(hand5, n_flop_samples, n_tr_samples)
    s = normalize_score(raw)
    bd.s_base = s
    bd.s_final = s
    _score_cache[key] = (s, bd)
    return s, bd


def save_cache(path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(_score_cache, f)


def load_cache(path: str) -> None:
    global _score_cache
    if os.path.exists(path):
        with open(path, "rb") as f:
            _score_cache = pickle.load(f)


def recompute_global_min_max(
    sample_size: int = 2000,
    n_flop_samples: int = 80,
    n_tr_samples: int = 30,
) -> Tuple[float, float]:
    """Sample random hands and estimate global score bounds."""
    all_cards = list(range(DECK_SIZE))
    lo, hi = float("inf"), float("-inf")
    for _ in range(sample_size):
        hand = random.sample(all_cards, 5)
        raw, _ = compute_base_street0_score(hand, n_flop_samples, n_tr_samples)
        lo = min(lo, raw)
        hi = max(hi, raw)
    return lo, hi
