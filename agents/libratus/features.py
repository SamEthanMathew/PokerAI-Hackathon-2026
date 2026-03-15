"""
Feature extraction for the decision-tree policy.

Computes betting, hole-card, board texture, opponent discard,
and opponent behavior features.
"""

from typing import List, Tuple, Dict
import numpy as np

RANKS = "23456789A"
SUITS = "dhs"
NUM_RANKS = 9
NUM_SUITS = 3

STRAIGHT_WINDOWS = [
    {8, 0, 1, 2, 3},  # A2345
    {0, 1, 2, 3, 4},  # 23456
    {1, 2, 3, 4, 5},  # 34567
    {2, 3, 4, 5, 6},  # 45678
    {3, 4, 5, 6, 7},  # 56789
    {4, 5, 6, 7, 8},  # 6789A
]

RANK_CENTRALITY = [0] * NUM_RANKS
for _w in STRAIGHT_WINDOWS:
    for _r in _w:
        RANK_CENTRALITY[_r] += 1


def card_rank(card: int) -> int:
    return card % NUM_RANKS


def card_suit(card: int) -> int:
    return card // NUM_RANKS


def is_connector(r1: int, r2: int) -> bool:
    return abs(r1 - r2) == 1 or {r1, r2} == {8, 0}


def is_gap1(r1: int, r2: int) -> bool:
    return abs(r1 - r2) == 2 or {r1, r2} == {8, 1}


# ---- Betting features ----

def betting_features(obs: dict) -> dict:
    continue_cost = max(0, obs["opp_bet"] - obs["my_bet"])
    pot = obs["pot_size"]
    pot_odds = continue_cost / (continue_cost + pot) if continue_cost > 0 else 0.0
    max_raise = obs.get("max_raise", 0)
    spr_proxy = max_raise / max(pot, 1)
    street = obs["street"]
    blind_pos = obs.get("blind_position", 0)
    in_position = 1 if (street >= 1 and blind_pos == 0) else 0
    return {
        "continue_cost": continue_cost,
        "pot": pot,
        "pot_odds": pot_odds,
        "spr_proxy": spr_proxy,
        "in_position": in_position,
        "street": street,
        "min_raise": obs.get("min_raise", 2),
        "max_raise": max_raise,
    }


# ---- Hole card features (pre-discard, 5 cards) ----

def hole5_features(cards5: List[int]) -> dict:
    ranks = [card_rank(c) for c in cards5]
    suits = [card_suit(c) for c in cards5]
    rank_counts = [0] * NUM_RANKS
    suit_counts = [0] * NUM_SUITS
    for r in ranks:
        rank_counts[r] += 1
    for s in suits:
        suit_counts[s] += 1
    has_trips = int(max(rank_counts) >= 3)
    top2 = sorted(ranks, reverse=True)[:2]
    return {
        "five_has_trips": has_trips,
        "rank_counts_5": rank_counts,
        "suit_counts_5": suit_counts,
        "top2_ranks_5": top2,
    }


# ---- Hole card features (post-discard, 2 cards) ----

def hole2_features(c0: int, c1: int) -> dict:
    r0, r1 = card_rank(c0), card_rank(c1)
    s0, s1 = card_suit(c0), card_suit(c1)
    rank_hi = max(r0, r1)
    rank_lo = min(r0, r1)
    is_pair = int(r0 == r1)
    is_suited = int(s0 == s1)
    if is_connector(r0, r1):
        gap_class = 1
    elif is_gap1(r0, r1):
        gap_class = 2
    elif abs(r0 - r1) <= 3 or {r0, r1} in [{8, 2}]:
        gap_class = 3
    else:
        gap_class = 0
    coverage = RANK_CENTRALITY[r0] + RANK_CENTRALITY[r1]
    return {
        "hole_is_pair": is_pair,
        "hole_is_suited": is_suited,
        "gap_class": gap_class,
        "rank_hi": rank_hi,
        "rank_lo": rank_lo,
        "hole_straight_coverage": coverage,
    }


# ---- Board texture features ----

def board_features(board_cards: List[int]) -> dict:
    if not board_cards:
        return {
            "board_max_suit": 0, "board_is_monotone": 0,
            "board_pair_flag": 0, "board_trips_flag": 0,
            "board_straight_density": 0.0, "board_four_to_straight": 0,
        }
    ranks = [card_rank(c) for c in board_cards]
    suits = [card_suit(c) for c in board_cards]
    rank_counts = [0] * NUM_RANKS
    suit_counts = [0] * NUM_SUITS
    for r in ranks:
        rank_counts[r] += 1
    for s in suits:
        suit_counts[s] += 1
    max_suit = max(suit_counts)
    is_mono = int(max_suit >= 3 and len(board_cards) == 3)
    pair_flag = int(max(rank_counts) >= 2)
    trips_flag = int(max(rank_counts) >= 3)
    rank_set = set(ranks)
    best_window = 0
    four_to_straight = 0
    for w in STRAIGHT_WINDOWS:
        overlap = len(rank_set & w)
        if overlap > best_window:
            best_window = overlap
        if overlap >= 4:
            four_to_straight = 1
    density = best_window / max(len(rank_set), 1)
    return {
        "board_max_suit": max_suit,
        "board_is_monotone": is_mono,
        "board_pair_flag": pair_flag,
        "board_trips_flag": trips_flag,
        "board_straight_density": density,
        "board_four_to_straight": four_to_straight,
    }


# ---- Opponent discard features ----

def opp_discard_features(opp_discards: List[int]) -> dict:
    visible = [c for c in opp_discards if c >= 0]
    if len(visible) < 3:
        return {"opp_discards_visible": 0, "opp_discard_highness": 0, "opp_discard_drawish": 0}
    ranks = [card_rank(c) for c in visible]
    suits = [card_suit(c) for c in visible]
    highness = max(ranks)
    suit_counts = [0] * NUM_SUITS
    for s in suits:
        suit_counts[s] += 1
    drawish = 0
    for i in range(3):
        for j in range(i + 1, 3):
            if suits[i] == suits[j] and is_connector(ranks[i], ranks[j]):
                drawish = 1
    return {"opp_discards_visible": 1, "opp_discard_highness": highness, "opp_discard_drawish": drawish}


# ---- Discard scoring features (per keep candidate) ----

DISCARD_FEATURE_DIM = 8


def discard_candidate_features(
    cards5: List[int],
    flop3: List[int],
    keep_i: int,
    keep_j: int,
    rank7_table=None,
) -> List[float]:
    """Feature vector for keeping cards at indices (keep_i, keep_j)."""
    c0, c1 = cards5[keep_i], cards5[keep_j]
    r0, r1 = card_rank(c0), card_rank(c1)
    s0, s1 = card_suit(c0), card_suit(c1)

    board_ranks = [card_rank(c) for c in flop3]
    board_suits = [card_suit(c) for c in flop3]

    is_pair = int(r0 == r1)
    is_suited = int(s0 == s1)

    all_suits = [s0, s1] + board_suits
    suit_counts = [0] * NUM_SUITS
    for s in all_suits:
        suit_counts[s] += 1
    flush_strength = max(suit_counts)

    keep_ranks = {r0, r1}
    all_ranks = keep_ranks | set(board_ranks)
    straight_draw = 0
    for w in STRAIGHT_WINDOWS:
        if len(keep_ranks & w) >= 1 and len(all_ranks & w) >= 4:
            straight_draw += 1

    high_card = max(r0, r1)

    connectivity = 0
    for br in board_ranks:
        if is_connector(r0, br) or is_connector(r1, br):
            connectivity += 1
        if r0 == br or r1 == br:
            connectivity += 2

    coverage = RANK_CENTRALITY[r0] + RANK_CENTRALITY[r1]

    return [
        float(is_pair),
        float(r0) / 8.0 if is_pair else 0.0,
        float(is_suited),
        float(flush_strength) / 5.0,
        float(straight_draw) / 6.0,
        float(high_card) / 8.0,
        float(connectivity) / 6.0,
        float(coverage) / 12.0,
    ]


# ---- Opponent behavior tracker ----

class OpponentStats:
    """Track opponent tendencies with Beta-distribution posteriors."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._fold_to_raise = [1.0, 1.0]
        self._call_down = [1.0, 1.0]
        self._raise_freq = [1.0, 1.0]

    @property
    def fold_to_raise_est(self) -> float:
        a, b = self._fold_to_raise
        return a / (a + b)

    @property
    def call_down_est(self) -> float:
        a, b = self._call_down
        return a / (a + b)

    @property
    def raise_freq_est(self) -> float:
        a, b = self._raise_freq
        return a / (a + b)

    def update(self, opp_action: str, was_facing_raise: bool):
        action = (opp_action or "").strip().upper()
        if was_facing_raise:
            if action == "FOLD":
                self._fold_to_raise[0] += 1
            else:
                self._fold_to_raise[1] += 1
            if action == "CALL":
                self._call_down[0] += 1
            else:
                self._call_down[1] += 1
        if action == "RAISE":
            self._raise_freq[0] += 1
        else:
            self._raise_freq[1] += 1

    def to_features(self) -> dict:
        return {
            "opp_fold_to_raise_est": self.fold_to_raise_est,
            "opp_call_down_est": self.call_down_est,
            "opp_raise_freq_est": self.raise_freq_est,
        }
