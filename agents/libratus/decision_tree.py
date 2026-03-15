"""
Parameterized decision tree for discard and betting.

All thresholds and weights live in a 'theta' dict loaded from JSON,
making the entire policy tunable via black-box optimization (CMA-ES).
"""

import json
import os
from typing import List, Tuple, Optional
import numpy as np

from .features import (
    DISCARD_FEATURE_DIM,
    discard_candidate_features,
    hole5_features,
    board_features,
    betting_features,
    card_rank,
    card_suit,
    STRAIGHT_WINDOWS,
)

FOLD = 0
RAISE = 1
CHECK = 2
CALL = 3
DISCARD = 4


def load_theta(path: Optional[str] = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "theta_default.json")
    with open(path, "r") as f:
        return json.load(f)


def theta_to_vector(theta: dict) -> np.ndarray:
    """Flatten theta dict to a 1-D float array for CMA-ES."""
    parts = list(theta["w_discard"])
    for key in [
        "call_margin",
        "value_raise_thresh_0", "value_raise_thresh_1",
        "value_raise_thresh_2", "value_raise_thresh_3",
        "value_bet_thresh_0", "value_bet_thresh_1",
        "value_bet_thresh_2", "value_bet_thresh_3",
        "bluff_equity_ceiling", "bluff_draw_min",
        "raise_size_small_frac", "raise_size_medium_frac", "raise_size_large_frac",
        "tau_emergency", "tau_exact_discard",
    ]:
        parts.append(theta[key])
    parts.append(float(theta["K_refine"]))
    return np.array(parts, dtype=np.float64)


def vector_to_theta(vec: np.ndarray) -> dict:
    """Reconstruct theta dict from flat vector."""
    d = DISCARD_FEATURE_DIM
    return {
        "w_discard": list(vec[:d]),
        "call_margin": float(vec[d]),
        "value_raise_thresh_0": float(vec[d + 1]),
        "value_raise_thresh_1": float(vec[d + 2]),
        "value_raise_thresh_2": float(vec[d + 3]),
        "value_raise_thresh_3": float(vec[d + 4]),
        "value_bet_thresh_0": float(vec[d + 5]),
        "value_bet_thresh_1": float(vec[d + 6]),
        "value_bet_thresh_2": float(vec[d + 7]),
        "value_bet_thresh_3": float(vec[d + 8]),
        "bluff_equity_ceiling": float(vec[d + 9]),
        "bluff_draw_min": float(vec[d + 10]),
        "raise_size_small_frac": float(vec[d + 11]),
        "raise_size_medium_frac": float(vec[d + 12]),
        "raise_size_large_frac": float(vec[d + 13]),
        "tau_emergency": float(vec[d + 14]),
        "tau_exact_discard": float(vec[d + 15]),
        "K_refine": max(1, int(round(vec[d + 16]))),
    }


# ---- Heuristic discard (O(1) emergency fallback) ----

def heuristic_keep_pair(cards5: List[int], flop3: List[int]) -> Tuple[int, int]:
    """Ultra-fast fallback: keep the pair with highest combined rank."""
    best = (0, 1)
    best_score = -1
    for i in range(5):
        for j in range(i + 1, 5):
            r0, r1 = card_rank(cards5[i]), card_rank(cards5[j])
            score = r0 + r1
            if r0 == r1:
                score += 20
            if card_suit(cards5[i]) == card_suit(cards5[j]):
                score += 3
            if score > best_score:
                best_score = score
                best = (i, j)
    return best


# ---- Discard tree ----

def choose_discard(
    obs: dict,
    theta: dict,
    exact_equity_fn=None,
    rank7_table=None,
) -> Tuple[int, int, int, int]:
    """Decision tree for discard. Returns (DISCARD, 0, keep_i, keep_j)."""
    cards5 = [c for c in obs["my_cards"] if c != -1]
    flop3 = [c for c in obs["community_cards"] if c != -1]
    time_left = obs.get("time_left", 1e9)
    opp_discards = [c for c in obs.get("opp_discarded_cards", [-1, -1, -1]) if c >= 0]

    if len(cards5) != 5 or len(flop3) < 3:
        return (DISCARD, 0, 0, 1)

    if time_left < theta.get("tau_emergency", 5.0):
        ki, kj = heuristic_keep_pair(cards5, flop3)
        return (DISCARD, 0, ki, kj)

    h5 = hole5_features(cards5)
    if h5["five_has_trips"]:
        trip_rank = -1
        for r_idx in range(9):
            if h5["rank_counts_5"][r_idx] >= 3:
                trip_rank = r_idx
                break
        if trip_rank >= 0:
            trip_indices = [i for i in range(5) if card_rank(cards5[i]) == trip_rank]
            best = (trip_indices[0], trip_indices[1])
            best_score = -1
            for a in range(len(trip_indices)):
                for b in range(a + 1, len(trip_indices)):
                    i, j = trip_indices[a], trip_indices[b]
                    phi = discard_candidate_features(cards5, flop3, i, j, rank7_table)
                    score = sum(theta["w_discard"][k] * phi[k] for k in range(len(phi)))
                    if score > best_score:
                        best_score = score
                        best = (i, j)
            return (DISCARD, 0, best[0], best[1])

    w = theta["w_discard"]
    scored = []
    for i in range(5):
        for j in range(i + 1, 5):
            phi = discard_candidate_features(cards5, flop3, i, j, rank7_table)
            score = sum(w[k] * phi[k] for k in range(min(len(w), len(phi))))
            scored.append((score, i, j))
    scored.sort(key=lambda x: -x[0])

    K = theta.get("K_refine", 3)
    tau_exact = theta.get("tau_exact_discard", 30.0)

    if exact_equity_fn is not None and time_left >= tau_exact:
        top_k = scored[:K]
        best_eq = -1.0
        best_keep = (scored[0][1], scored[0][2])
        for _, i, j in top_k:
            opp_disc = opp_discards if opp_discards else None
            results = exact_equity_fn(cards5, flop3, opp_disc)
            for (ki, kj), eq in results:
                if ki == i and kj == j:
                    if eq > best_eq:
                        best_eq = eq
                        best_keep = (i, j)
                    break
        return (DISCARD, 0, best_keep[0], best_keep[1])

    return (DISCARD, 0, scored[0][1], scored[0][2])


# ---- Raise sizing ----

def choose_raise_amount(obs: dict, theta: dict, equity: float, mode: str = "value") -> int:
    pot = obs["pot_size"]
    min_r = obs["min_raise"]
    max_r = obs["max_raise"]

    if mode == "bluff_small":
        frac = theta.get("raise_size_small_frac", 0.4)
    elif equity >= 0.85:
        frac = theta.get("raise_size_large_frac", 1.5)
    elif equity >= 0.72:
        frac = theta.get("raise_size_medium_frac", 0.75)
    else:
        frac = theta.get("raise_size_small_frac", 0.4)

    amount = int(pot * frac)
    amount = max(amount, min_r)
    amount = min(amount, max_r)
    return amount


# ---- Betting tree ----

def choose_bet_action(
    obs: dict,
    theta: dict,
    equity: float,
    hand_features: dict,
) -> Tuple[int, int, int, int]:
    """Decision tree for betting. Returns (action_type, raise_amount, 0, 0)."""
    va = obs["valid_actions"]
    bf = betting_features(obs)
    street = bf["street"]
    pot_odds = bf["pot_odds"]
    continue_cost = bf["continue_cost"]
    call_margin = theta.get("call_margin", 0.05)

    if continue_cost > 0:
        vrt_key = f"value_raise_thresh_{street}"
        vrt = theta.get(vrt_key, 0.75)

        if va[FOLD] and equity < pot_odds + call_margin:
            return (FOLD, 0, 0, 0)

        if va[RAISE] and equity >= vrt:
            amt = choose_raise_amount(obs, theta, equity, "value")
            return (RAISE, amt, 0, 0)

        if va[CALL] and equity >= pot_odds + call_margin:
            return (CALL, 0, 0, 0)

        if va[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    else:
        vbt_key = f"value_bet_thresh_{street}"
        vbt = theta.get(vbt_key, 0.65)

        if not va[RAISE]:
            if va[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        if equity >= vbt:
            amt = choose_raise_amount(obs, theta, equity, "value")
            return (RAISE, amt, 0, 0)

        bluff_ceil = theta.get("bluff_equity_ceiling", 0.35)
        draw_min = theta.get("bluff_draw_min", 0.3)
        has_draw = (
            hand_features.get("board_straight_density", 0) >= draw_min
            or hand_features.get("flush_strength", 0) >= 4
        )
        if has_draw and equity < bluff_ceil and equity > 0.15:
            amt = choose_raise_amount(obs, theta, equity, "bluff_small")
            return (RAISE, amt, 0, 0)

        if va[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)
