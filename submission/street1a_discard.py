"""
Street 1A – Discard Engine
==========================
Given a 5-card hand and a 3-card flop, score every keep-2 / discard-3 option
and return the best (or softmax-mixed) keep choice plus metadata.

Components (per keep K):
    C(K) – current flop hand quality
    F(K) – future turn/river expected value
    D(K) – dead-card penalty (cards we lose)
    R(K) – reveal cost (how much discard exposes our strategy)
    A(K) – ambiguity value (multiple plausible lines)
    I(K) – information exploitation (SB only – opponent discard is known)
    B(K) – betting usability (how well the keep plays in flop betting)

S1A = w_c*C + w_f*F - w_d*D - w_r*R + w_a*A + w_i*I + w_b*B
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from submission.street0_score import (
    DECK_SIZE,
    NUM_RANKS,
    NUM_SUITS,
    STRAIGHT_WINDOWS,
    STREET_UTILITY,
    all_keeps,
    evaluate_hand,
    hand_class_from_score,
    hand_rank_counts,
    hand_suit_counts,
    rank,
    refined_utility,
    remaining_cards,
    suit,
    within_class_percentile,
)

# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class Street1AContext:
    hand5: List[int]
    flop3: List[int]
    position: int          # raw gym blind_position: 0=SB, 1=BB
    opp_discard3: Optional[List[int]]  # list of 3 ints when SB; None when BB
    pot_size: int
    opponent_profile: object  # OpponentProfile from street0_score


@dataclass
class KeepEvalRecord:
    keep_indices: Tuple[int, int]
    keep_cards: List[int]
    discard_cards: List[int]
    C: float = 0.0
    F: float = 0.0
    D: float = 0.0
    R: float = 0.0
    A: float = 0.0
    I_val: float = 0.0
    B: float = 0.0
    S1A: float = 0.0
    current_hand_class: str = ""


DISCARD_CLASS_LABELS = (
    "flush_transparent", "straight_transparent", "pair_transparent",
    "weak_transparent", "ambiguous", "mixed_strength", "capped",
)

# Position-dependent weights: index by gym blind_position
# BB (position=1): low I (no opp discard info at discard time)
W_BB = {"c": 0.16, "f": 0.30, "d": 0.16, "r": 0.14, "a": 0.10, "i": 0.02, "b": 0.12}
# SB (position=0): high I (sees opp discard)
W_SB = {"c": 0.15, "f": 0.28, "d": 0.16, "r": 0.12, "a": 0.08, "i": 0.11, "b": 0.10}

# Street 1 future-value sub-weights (turn vs river relative importance)
_W_TURN = 0.35
_W_RIVER = 0.65

# Sampling limits for future value
_N_TR_SAMPLES = 40

HAND_CLASS_NAMES = [
    "Straight Flush", "Full House", "Flush", "Three of a Kind",
    "Straight", "Two Pair", "One Pair", "High Card",
]


def _weights_for_position(position: int) -> Dict[str, float]:
    return W_BB if position == 1 else W_SB


# =========================================================================
# C(K) – Current Flop Hand Quality
# =========================================================================

def score_current_flop_hand(keep2: List[int], flop3: List[int]) -> float:
    """Best 5-card hand from keep2+flop3, refined utility at street 1."""
    score = evaluate_hand(keep2, flop3)
    return refined_utility(1, score)


# =========================================================================
# F(K) – Future Turn/River Expected Value
# =========================================================================

def score_future_turn_river(
    keep2: List[int],
    flop3: List[int],
    dead_cards: List[int],
    opp_dead_cards: Optional[List[int]] = None,
) -> float:
    """E_T[U2]*0.35 + E_{T,R}[U3]*0.65 over sampled turn/river cards."""
    all_dead = set(keep2) | set(flop3) | set(dead_cards)
    if opp_dead_cards:
        all_dead |= set(c for c in opp_dead_cards if c != -1)

    deck = [c for c in range(DECK_SIZE) if c not in all_dead]
    if len(deck) < 2:
        return score_current_flop_hand(keep2, flop3)

    board3 = list(flop3)
    tr_pairs = list(combinations(deck, 2))
    if len(tr_pairs) > _N_TR_SAMPLES:
        tr_pairs = random.sample(tr_pairs, _N_TR_SAMPLES)

    if not tr_pairs:
        return score_current_flop_hand(keep2, flop3)

    turn_scores: Dict[int, float] = {}
    u3_sum = 0.0

    for t, r_card in tr_pairs:
        if t not in turn_scores:
            board4 = board3 + [t]
            sc2 = evaluate_hand(keep2, board4)
            turn_scores[t] = refined_utility(2, sc2)

        board5 = board3 + [t, r_card]
        sc3 = evaluate_hand(keep2, board5)
        u3_sum += refined_utility(3, sc3)

    e_u2 = sum(turn_scores.values()) / len(turn_scores) if turn_scores else 0.0
    e_u3 = u3_sum / len(tr_pairs) if tr_pairs else e_u2

    return _W_TURN * e_u2 + _W_RIVER * e_u3


# =========================================================================
# D(K) – Dead-Card Penalty
# =========================================================================

def score_dead_card_penalty(
    keep2: List[int],
    discard3: List[int],
    flop3: List[int],
    opp_discard3: Optional[List[int]],
    position: int,
) -> float:
    """Penalty for losing useful cards. Higher when SB (opp cards known)."""
    keep_ranks = set(rank(c) for c in keep2)
    keep_suits = hand_suit_counts(keep2)
    disc_ranks = Counter(rank(c) for c in discard3)
    disc_suits = hand_suit_counts(discard3)
    flop_ranks = set(rank(c) for c in flop3)
    flop_suits = hand_suit_counts(flop3)

    loss = 0.0

    # Rank loss: discarding cards that pair with flop or keep
    for r, cnt in disc_ranks.items():
        if r in keep_ranks:
            loss += 0.15 * cnt
        if r in flop_ranks:
            loss += 0.10 * cnt
        if cnt >= 2:
            loss += 0.08

    # Suit loss: abandoning flush potential
    combined_suits = hand_suit_counts(list(keep2) + list(flop3))
    for s in range(NUM_SUITS):
        discarded_in_suit = disc_suits.get(s, 0)
        combined_in_suit = combined_suits.get(s, 0)
        if combined_in_suit >= 3 and discarded_in_suit >= 1:
            loss += 0.12 * discarded_in_suit
        elif combined_in_suit >= 4 and discarded_in_suit >= 1:
            loss += 0.20

    # Connectivity loss: discarding cards near flop ranks
    for c in discard3:
        r = rank(c)
        adj_to_flop = sum(1 for fr in flop_ranks if abs(r - fr) <= 1)
        adj_to_keep = sum(1 for kr in keep_ranks if abs(r - kr) <= 1)
        loss += 0.04 * adj_to_flop + 0.03 * adj_to_keep

    # SB bonus: if opp_discard known, factor in alternative-path loss
    if position == 0 and opp_discard3:
        opp_dead_ranks = set(rank(c) for c in opp_discard3 if c != -1)
        for r, cnt in disc_ranks.items():
            if r in opp_dead_ranks:
                loss -= 0.05

    return max(0.0, min(1.0, loss))


# =========================================================================
# R(K) – Reveal Cost
# =========================================================================

def score_reveal_cost(keep2: List[int], discard3: List[int], flop3: List[int]) -> float:
    """How much the discard reveals our keep/class."""
    keep_ranks = set(rank(c) for c in keep2)
    keep_suits = hand_suit_counts(keep2)
    disc_suits = hand_suit_counts(discard3)
    flop_suits = hand_suit_counts(flop3)
    flop_ranks = set(rank(c) for c in flop3)

    reveal = 0.0

    # Obvious pair keep
    if len(keep_ranks) == 1:
        reveal += 0.40

    # Obvious flush keep: both keep cards match dominant flop suit
    for s in range(NUM_SUITS):
        if keep_suits.get(s, 0) == 2 and flop_suits.get(s, 0) >= 2:
            reveal += 0.35
            break

    # Obvious straight keep: keep cards consecutive and near flop
    if len(keep_ranks) == 2:
        kr = sorted(keep_ranks)
        if kr[1] - kr[0] <= 2:
            overlap = sum(1 for r in keep_ranks if any(abs(r - fr) <= 1 for fr in flop_ranks))
            if overlap >= 1:
                reveal += 0.15

    # Homogeneous discard: all same suit
    if max(disc_suits.values(), default=0) == 3:
        reveal += 0.10

    return min(1.0, reveal)


# =========================================================================
# A(K) – Ambiguity Value
# =========================================================================

def score_ambiguity_value(keep2: List[int], discard3: List[int], flop3: List[int]) -> float:
    """Multiple plausible lines for our keep."""
    keep_ranks = set(rank(c) for c in keep2)
    keep_suits = hand_suit_counts(keep2)
    flop_ranks = set(rank(c) for c in flop3)
    flop_suits = hand_suit_counts(flop3)
    combined_ranks = keep_ranks | flop_ranks

    paths = 0.0

    # Pair/trips potential
    if len(keep_ranks) == 1 or any(rank(k) in flop_ranks for k in keep2):
        paths += 1.0

    # Flush potential
    for s in range(NUM_SUITS):
        total_suited = keep_suits.get(s, 0) + flop_suits.get(s, 0)
        if total_suited >= 3:
            paths += 1.0
            break

    # Straight potential
    for w in STRAIGHT_WINDOWS:
        if len(w - combined_ranks) <= 2:
            paths += 0.5
            break

    # High cards give more ambiguity
    if any(rank(c) >= 7 for c in keep2):
        paths += 0.3

    # Diverse suits in keep + flop = harder to read
    if len(keep_suits) == 2 and keep_suits.get(max(keep_suits, key=keep_suits.get), 0) == 1:
        paths += 0.2

    return min(1.0, paths / 3.0)


# =========================================================================
# I(K) – Information Exploitation (SB only)
# =========================================================================

def score_info_exploitation(
    keep2: List[int],
    flop3: List[int],
    opp_discard3: Optional[List[int]],
    position: int,
) -> float:
    """Returns 0.0 when BB (position==1). For SB: path elimination, keep inference, dead-out benefit."""
    if position == 1 or not opp_discard3:
        return 0.0

    opp_disc = [c for c in opp_discard3 if c != -1]
    if not opp_disc:
        return 0.0

    opp_disc_ranks = set(rank(c) for c in opp_disc)
    opp_disc_suits = hand_suit_counts(opp_disc)
    flop_ranks = set(rank(c) for c in flop3)
    keep_ranks = set(rank(c) for c in keep2)

    score = 0.0

    # Path elimination: opponent abandoned certain draws
    for w in STRAIGHT_WINDOWS:
        killed_by_opp = w & opp_disc_ranks
        if len(killed_by_opp) >= 2:
            if keep_ranks & w:
                score += 0.15
            break

    # Suit abandonment
    for s in range(NUM_SUITS):
        if opp_disc_suits.get(s, 0) >= 2:
            keep_in_suit = sum(1 for c in keep2 if suit(c) == s)
            flop_in_suit = sum(1 for c in flop3 if suit(c) == s)
            if keep_in_suit + flop_in_suit >= 3:
                score += 0.20
            break

    # Dead-card benefit: opponent's discards block their potential paths
    opp_pair_discarded = any(cnt >= 2 for cnt in Counter(rank(c) for c in opp_disc).values())
    if opp_pair_discarded:
        score += 0.10

    # Keep inference: knowing what they threw away tells us about their keep
    opp_max_suit = max(opp_disc_suits.values()) if opp_disc_suits else 0
    if opp_max_suit == 3:
        score += 0.15

    return min(1.0, score)


# =========================================================================
# B(K) – Betting Usability
# =========================================================================

def score_betting_usability(
    keep2: List[int],
    flop3: List[int],
    opponent_profile: object,
    position: int,
) -> float:
    """How well this keep plays in flop betting (robustness to raises, value/pressure potential)."""
    c_score = score_current_flop_hand(keep2, flop3)
    keep_ranks = set(rank(c) for c in keep2)
    flop_ranks = set(rank(c) for c in flop3)
    combined = keep_ranks | flop_ranks

    usability = 0.0

    # Strong current hand -> good for value betting
    if c_score >= 0.70:
        usability += 0.40
    elif c_score >= 0.50:
        usability += 0.25
    elif c_score >= 0.30:
        usability += 0.10

    # Robustness: paired board + pair in keep -> less vulnerable to bluffs
    flop_rc = hand_rank_counts(flop3)
    if any(cnt >= 2 for cnt in flop_rc.values()) and len(keep_ranks) == 1:
        usability += 0.15

    # Pressure potential: draws that can represent strength
    for w in STRAIGHT_WINDOWS:
        if len(w - combined) == 1:
            usability += 0.10
            break

    for s in range(NUM_SUITS):
        combined_suits = hand_suit_counts(list(keep2) + list(flop3))
        if combined_suits.get(s, 0) >= 4:
            usability += 0.15
            break

    # Position bonus: acting last (SB in post-flop) gives more control
    if position == 0:
        usability += 0.05

    return min(1.0, usability)


# =========================================================================
# S1A Aggregation
# =========================================================================

def street1a_keep_score(
    keep2: List[int],
    discard3: List[int],
    flop3: List[int],
    ctx: Street1AContext,
    weights: Dict[str, float],
) -> Tuple[float, KeepEvalRecord]:
    """Compute S1A for a single keep option. Returns (score, record)."""
    opp_disc = ctx.opp_discard3

    C = score_current_flop_hand(keep2, flop3)
    F = score_future_turn_river(keep2, flop3, discard3, opp_disc)
    D = score_dead_card_penalty(keep2, discard3, flop3, opp_disc, ctx.position)
    R = score_reveal_cost(keep2, discard3, flop3)
    A = score_ambiguity_value(keep2, discard3, flop3)
    I = score_info_exploitation(keep2, flop3, opp_disc, ctx.position)
    B = score_betting_usability(keep2, flop3, ctx.opponent_profile, ctx.position)

    s1a = (
        weights["c"] * C
        + weights["f"] * F
        - weights["d"] * D
        - weights["r"] * R
        + weights["a"] * A
        + weights["i"] * I
        + weights["b"] * B
    )

    cls = hand_class_from_score(evaluate_hand(keep2, flop3))
    cls_name = HAND_CLASS_NAMES[cls] if cls < len(HAND_CLASS_NAMES) else "Unknown"

    record = KeepEvalRecord(
        keep_indices=(0, 0),  # placeholder; filled by caller
        keep_cards=list(keep2),
        discard_cards=list(discard3),
        C=C, F=F, D=D, R=R, A=A, I_val=I, B=B,
        S1A=s1a,
        current_hand_class=cls_name,
    )
    return s1a, record


# =========================================================================
# Main: choose_keep_with_controlled_mixing
# =========================================================================

def choose_keep_with_controlled_mixing(
    hand5: List[int],
    flop3: List[int],
    ctx: Street1AContext,
) -> Tuple[int, int, List[int], List[int], str, List[KeepEvalRecord]]:
    """
    Evaluate all 10 keeps, rank, pick best (or softmax-mix when close).

    Returns:
        (keep_idx0, keep_idx1, chosen_keep, chosen_discard, discard_class, all_records)
    where keep_idx0 and keep_idx1 are indices 0-4 into observation["my_cards"].
    """
    keeps = all_keeps(hand5)
    weights = _weights_for_position(ctx.position)

    records: List[KeepEvalRecord] = []
    for (ki, kc, dc) in keeps:
        s1a, rec = street1a_keep_score(kc, dc, flop3, ctx, weights)
        rec.keep_indices = ki
        records.append(rec)

    # Sort by S1A descending
    sorted_records = sorted(records, key=lambda r: r.S1A, reverse=True)

    # Controlled mixing: if top two are very close, use softmax
    top = sorted_records[0]
    second = sorted_records[1] if len(sorted_records) > 1 else None

    if second and abs(top.S1A - second.S1A) < 0.03:
        # Softmax between top candidates
        candidates = [r for r in sorted_records if r.S1A >= top.S1A - 0.05]
        if len(candidates) < 2:
            candidates = sorted_records[:2]

        temp = 0.05
        max_s = max(r.S1A for r in candidates)
        exp_scores = [math.exp((r.S1A - max_s) / temp) for r in candidates]
        total_exp = sum(exp_scores)
        probs = [e / total_exp for e in exp_scores]

        r_val = random.random()
        cumulative = 0.0
        chosen = candidates[0]
        for prob, cand in zip(probs, candidates):
            cumulative += prob
            if r_val <= cumulative:
                chosen = cand
                break
    else:
        chosen = top

    chosen_keep = chosen.keep_cards
    chosen_discard = chosen.discard_cards
    ki0, ki1 = chosen.keep_indices

    discard_class = classify_discard(chosen_keep, chosen_discard, flop3)

    return ki0, ki1, chosen_keep, chosen_discard, discard_class, records


# =========================================================================
# Discard Classification
# =========================================================================

def classify_discard(keep2: List[int], discard3: List[int], flop3: List[int]) -> str:
    """Classify our discard for logging and opponent recon attribution."""
    keep_ranks = set(rank(c) for c in keep2)
    keep_suits = hand_suit_counts(keep2)
    disc_ranks = Counter(rank(c) for c in discard3)
    disc_suits = hand_suit_counts(discard3)
    flop_suits = hand_suit_counts(flop3)
    flop_ranks = set(rank(c) for c in flop3)

    # Flush transparent: both keep cards suited with flop
    for s in range(NUM_SUITS):
        if keep_suits.get(s, 0) == 2 and flop_suits.get(s, 0) >= 2:
            return "flush_transparent"

    # Pair transparent: keep is a pair
    if len(keep_ranks) == 1:
        return "pair_transparent"

    # Straight transparent: keep cards consecutive and near flop
    if len(keep_ranks) == 2:
        kr = sorted(keep_ranks)
        if kr[1] - kr[0] <= 2:
            overlap = sum(1 for r in keep_ranks if any(abs(r - fr) <= 1 for fr in flop_ranks))
            if overlap >= 1:
                return "straight_transparent"

    # Weak transparent: both keep cards are low
    if all(rank(c) < 4 for c in keep2) and all(rank(c) < 4 for c in discard3):
        return "weak_transparent"

    # Capped: discarding all high cards
    if all(rank(c) >= 6 for c in discard3):
        return "capped"

    return "ambiguous"


# =========================================================================
# Flop texture classifier (shared with genesis / recon)
# =========================================================================

def classify_flop_texture(flop3: List[int]) -> str:
    """Quick texture label for a 3-card flop."""
    rc = hand_rank_counts(flop3)
    sc = hand_suit_counts(flop3)
    rank_set = set(rc.keys())
    max_suit = max(sc.values()) if sc else 0

    if max_suit == 3:
        return "monotone"
    if any(cnt >= 2 for cnt in rc.values()):
        return "paired_flop"

    sorted_r = sorted(rank_set)
    is_connected = len(sorted_r) >= 3 and sorted_r[-1] - sorted_r[0] <= 4

    if is_connected and max_suit == 2:
        return "connected_two_suited"
    if is_connected:
        return "connected_flop"
    if max_suit == 2:
        return "two_suited_flop"
    return "rainbow_disconnected"
