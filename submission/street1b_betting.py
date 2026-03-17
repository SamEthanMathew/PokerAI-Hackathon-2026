"""
Street 1B – Flop Betting Engine
================================
After both players discard, compute a flop betting action.

This module is designed to be called **multiple times per hand** on the same
street (multi-round betting: bet -> raise -> re-raise -> ...).

Score components:
    H   – current post-discard hand strength
    F'  – future strategic value with all dead cards known
    E   – dead-card equity-position advantage
    T   – transparency penalty (how readable our line is)
    P   – pressure suitability (can we push folds?)
    X   – opponent exploit adjustment

S1B = v_h*H + v_f*F' + v_e*E - v_t*T + v_p*P + v_x*X
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from submission.street0_score import (
    DECK_SIZE,
    NUM_RANKS,
    NUM_SUITS,
    STRAIGHT_WINDOWS,
    evaluate_hand,
    hand_class_from_score,
    hand_rank_counts,
    hand_suit_counts,
    rank,
    refined_utility,
    suit,
)

# Action type constants (match gym_env.PokerEnv.ActionType)
FOLD = 0
RAISE = 1
CHECK = 2
CALL = 3

# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class Street1BContext:
    our_keep2: List[int]
    our_discard3: List[int]
    opp_discard3: List[int]
    flop3: List[int]
    pot_size: int
    amount_to_call: int
    valid_actions: List[int]
    min_raise: int
    max_raise: int
    opponent_profile: object      # OpponentProfile from street0_score
    our_discard_class: str
    opp_discard_class: str
    flop_texture: str
    opp_last_action: str          # what opponent just did (or "DISCARD"/"None")
    position: int                 # raw gym blind_position: 0=SB, 1=BB


@dataclass
class Street1BScoreBreakdown:
    H: float = 0.0
    F_prime: float = 0.0
    E: float = 0.0
    T: float = 0.0
    P: float = 0.0
    X: float = 0.0
    S1B: float = 0.0


# S1B component weights
V_H = 0.24
V_F = 0.24
V_E = 0.14
V_T = 0.12
V_P = 0.14
V_X = 0.12

# Sampling limit for future value
_N_TR_SAMPLES_1B = 30
_W_TURN = 0.35
_W_RIVER = 0.65


# =========================================================================
# H – Post-Discard Current Hand Strength
# =========================================================================

def score_post_discard_current_hand(keep2: List[int], flop3: List[int]) -> float:
    score = evaluate_hand(keep2, flop3)
    return refined_utility(1, score)


# =========================================================================
# F' – Post-Discard Future Value (with all dead cards known)
# =========================================================================

def score_post_discard_future_value(
    keep2: List[int],
    flop3: List[int],
    our_discard: List[int],
    opp_discard: List[int],
) -> float:
    from itertools import combinations

    all_dead = set(keep2) | set(flop3)
    all_dead |= set(c for c in our_discard if c != -1)
    all_dead |= set(c for c in opp_discard if c != -1)

    deck = [c for c in range(DECK_SIZE) if c not in all_dead]
    if len(deck) < 2:
        return score_post_discard_current_hand(keep2, flop3)

    board3 = list(flop3)
    tr_pairs = list(combinations(deck, 2))
    if len(tr_pairs) > _N_TR_SAMPLES_1B:
        tr_pairs = random.sample(tr_pairs, _N_TR_SAMPLES_1B)

    if not tr_pairs:
        return score_post_discard_current_hand(keep2, flop3)

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
# E – Dead-Card Equity-Position Advantage
# =========================================================================

def score_dead_card_advantage(
    keep2: List[int],
    flop3: List[int],
    our_discard: List[int],
    opp_discard: List[int],
) -> float:
    """Do dead cards help us more than them? Positive = advantage."""
    our_disc_ranks = set(rank(c) for c in our_discard if c != -1)
    opp_disc_ranks = set(rank(c) for c in opp_discard if c != -1)
    opp_disc_suits = hand_suit_counts([c for c in opp_discard if c != -1])
    flop_ranks = set(rank(c) for c in flop3)
    keep_ranks = set(rank(c) for c in keep2)

    advantage = 0.0

    # Opponent killed their own straight draws
    for w in STRAIGHT_WINDOWS:
        opp_killed = w & opp_disc_ranks
        our_covered = w & (keep_ranks | flop_ranks)
        if len(opp_killed) >= 2 and len(our_covered) >= 3:
            advantage += 0.15

    # Opponent killed their own flush draws
    for s in range(NUM_SUITS):
        if opp_disc_suits.get(s, 0) >= 2:
            our_suited = sum(1 for c in list(keep2) + list(flop3) if suit(c) == s)
            if our_suited >= 3:
                advantage += 0.20

    # Our discards don't overlap with our draws (already handled by choosing discard)
    # But if our discards block opponent's paths, that's good
    for w in STRAIGHT_WINDOWS:
        our_blocking = w & our_disc_ranks
        opp_needs = w - (opp_disc_ranks | flop_ranks)
        if len(our_blocking) >= 1 and len(opp_needs) <= 1:
            advantage += 0.05

    return max(0.0, min(1.0, advantage))


# =========================================================================
# T – Transparency Penalty
# =========================================================================

def score_transparency_penalty(
    keep2: List[int],
    our_discard: List[int],
    flop3: List[int],
) -> float:
    """How readable our line is post-discard."""
    keep_ranks = set(rank(c) for c in keep2)
    keep_suits = hand_suit_counts(keep2)
    disc_suits = hand_suit_counts([c for c in our_discard if c != -1])
    flop_suits = hand_suit_counts(flop3)

    penalty = 0.0

    # Obvious pair
    if len(keep_ranks) == 1:
        penalty += 0.35

    # Obvious flush
    for s in range(NUM_SUITS):
        if keep_suits.get(s, 0) == 2 and flop_suits.get(s, 0) >= 2:
            penalty += 0.30
            break

    # Obvious straight: keep cards are consecutive
    if len(keep_ranks) == 2:
        kr = sorted(keep_ranks)
        if kr[1] - kr[0] <= 1:
            penalty += 0.15

    # All discard same suit = obvious suit abandon
    if max(disc_suits.values(), default=0) == 3:
        penalty += 0.10

    return min(1.0, penalty)


# =========================================================================
# P – Pressure Suitability
# =========================================================================

def score_pressure_suitability(
    keep2: List[int],
    flop3: List[int],
    opponent_profile: object,
    our_discard_class: str,
) -> float:
    """Can we push folds? Based on fold tendencies and our apparent strength."""
    from submission.opponent_recon import (
        get_flop_fold_vs_our_discard_class,
        get_flop_fold_by_texture,
    )
    from submission.street1a_discard import classify_flop_texture

    h_score = score_post_discard_current_hand(keep2, flop3)
    pressure = 0.0

    # Base: medium hands benefit most from pressure
    if 0.25 <= h_score <= 0.65:
        pressure += 0.30
    elif h_score > 0.65:
        pressure += 0.15

    # Fold tendency: if opponent folds a lot vs our discard class, pressure is useful
    if opponent_profile and hasattr(opponent_profile, '_recon_ref'):
        recon = opponent_profile._recon_ref
        fold_rate = get_flop_fold_vs_our_discard_class(recon, our_discard_class)
        pressure += 0.30 * max(0.0, fold_rate - 0.33)

    # Draw backup: if we have draws, pressure + equity = strong
    combined = list(keep2) + list(flop3)
    combined_suits = hand_suit_counts(combined)
    combined_ranks = set(rank(c) for c in combined)
    has_draw = False

    for s in range(NUM_SUITS):
        if combined_suits.get(s, 0) >= 4:
            has_draw = True
            break
    if not has_draw:
        for w in STRAIGHT_WINDOWS:
            if len(w - combined_ranks) == 1:
                has_draw = True
                break

    if has_draw:
        pressure += 0.20

    return min(1.0, pressure)


# =========================================================================
# X – Opponent Exploit Adjustment
# =========================================================================

def score_exploit_adjustment(
    opponent_profile: object,
    our_discard_class: str,
    opp_discard_class: str,
    texture: str,
) -> float:
    """Overfold/overcall/overbluff adjustments."""
    from submission.opponent_recon import (
        get_flop_fold_vs_our_discard_class,
        get_flop_fold_by_texture,
        get_opponent_flop_aggression_after_discard,
    )

    adj = 0.0

    if not opponent_profile or not hasattr(opponent_profile, '_recon_ref'):
        return 0.0

    recon = opponent_profile._recon_ref

    # Overfolder: opponent folds too much -> we should bluff/pressure more
    fold_vs_us = get_flop_fold_vs_our_discard_class(recon, our_discard_class)
    if fold_vs_us > 0.55:
        adj += 0.20 * (fold_vs_us - 0.55) / 0.45

    # Overcaller: opponent calls too much -> we should value bet more
    if fold_vs_us < 0.30:
        adj += 0.10

    # Opponent overbluffs after certain discards
    if opp_discard_class:
        opp_agg = get_opponent_flop_aggression_after_discard(recon, opp_discard_class)
        if opp_agg > 0.60:
            adj += 0.10

    # Texture-based fold rate
    tex_fold = get_flop_fold_by_texture(recon, texture)
    if tex_fold > 0.55:
        adj += 0.10 * (tex_fold - 0.55) / 0.45

    return max(-0.5, min(0.5, adj))


# =========================================================================
# S1B Aggregate Score
# =========================================================================

def street1b_score(ctx: Street1BContext) -> Tuple[float, Street1BScoreBreakdown]:
    H = score_post_discard_current_hand(ctx.our_keep2, ctx.flop3)
    F_prime = score_post_discard_future_value(
        ctx.our_keep2, ctx.flop3, ctx.our_discard3, ctx.opp_discard3
    )
    E = score_dead_card_advantage(
        ctx.our_keep2, ctx.flop3, ctx.our_discard3, ctx.opp_discard3
    )
    T = score_transparency_penalty(ctx.our_keep2, ctx.our_discard3, ctx.flop3)
    P = score_pressure_suitability(
        ctx.our_keep2, ctx.flop3, ctx.opponent_profile, ctx.our_discard_class
    )
    X = score_exploit_adjustment(
        ctx.opponent_profile, ctx.our_discard_class,
        ctx.opp_discard_class, ctx.flop_texture
    )

    s1b = V_H * H + V_F * F_prime + V_E * E - V_T * T + V_P * P + V_X * X

    bd = Street1BScoreBreakdown(
        H=H, F_prime=F_prime, E=E, T=T, P=P, X=X, S1B=s1b,
    )
    return s1b, bd


# =========================================================================
# Motives
# =========================================================================

def compute_flop_action_motives(s1b: float, ctx: Street1BContext) -> Dict[str, float]:
    """Compute motives based on whether we're opening or facing a bet."""
    is_opening = ctx.amount_to_call <= 0
    opp_al = (ctx.opp_last_action or "").lower()
    facing_raise = "raise" in opp_al

    m_value = max(0.0, s1b - 0.20) / 0.80 if s1b > 0.20 else 0.0
    m_pressure = 0.0
    m_continue = 0.0
    m_defend = 0.0
    m_reraise = 0.0

    if is_opening:
        # Opening: value and pressure dominate
        m_pressure = max(0.0, min(1.0, 0.5 + (0.50 - s1b)))
        if s1b < 0.25:
            m_pressure = max(m_pressure, 0.3)
    else:
        # Facing a bet or raise: continue and defend dominate
        pot_odds = ctx.amount_to_call / (ctx.pot_size + ctx.amount_to_call + 1)
        m_continue = max(0.0, s1b - pot_odds)
        m_defend = max(0.0, 0.60 - pot_odds) if s1b > 0.35 else 0.0

        if facing_raise:
            m_defend *= 0.7
            m_reraise = max(0.0, s1b - 0.60) / 0.40 if s1b > 0.60 else 0.0
        else:
            m_reraise = max(0.0, s1b - 0.55) / 0.45 if s1b > 0.55 else 0.0

    return {
        "m_value": m_value,
        "m_pressure": m_pressure,
        "m_continue": m_continue,
        "m_defend": m_defend,
        "m_reraise": m_reraise,
    }


# =========================================================================
# Action Selection
# =========================================================================

def compute_flop_action(
    ctx: Street1BContext,
    motives: Dict[str, float],
    s1b: float,
) -> Tuple[int, int, float, str]:
    """
    Returns (action_type, raise_amount, confidence, size_bucket).
    action_type uses gym constants: FOLD=0, RAISE=1, CHECK=2, CALL=3.
    """
    va = ctx.valid_actions
    is_opening = ctx.amount_to_call <= 0

    can_check = len(va) > CHECK and va[CHECK] == 1
    can_call = len(va) > CALL and va[CALL] == 1
    can_raise = len(va) > RAISE and va[RAISE] == 1
    can_fold = len(va) > FOLD and va[FOLD] == 1

    if is_opening:
        return _opening_action(ctx, motives, s1b, can_check, can_raise)
    else:
        return _facing_bet_action(ctx, motives, s1b, can_call, can_raise, can_fold, can_check)


def _opening_action(
    ctx: Street1BContext,
    motives: Dict[str, float],
    s1b: float,
    can_check: bool,
    can_raise: bool,
) -> Tuple[int, int, float, str]:
    """We are opening (amount_to_call == 0)."""
    m_value = motives["m_value"]
    m_pressure = motives["m_pressure"]

    # Strong hands: bet for value
    if s1b >= 0.55 and can_raise:
        amount, bucket = _compute_bet_size(ctx, s1b, "value")
        return RAISE, amount, m_value, bucket

    # Medium hands with pressure motive
    if s1b >= 0.30 and m_pressure > 0.35 and can_raise:
        amount, bucket = _compute_bet_size(ctx, s1b, "pressure")
        return RAISE, amount, m_pressure, bucket

    # Bluff candidates: weak but with some draw equity
    if s1b >= 0.15 and m_pressure > 0.45 and can_raise:
        r_val = random.random()
        bluff_freq = min(0.30, m_pressure * 0.4)
        if r_val < bluff_freq:
            amount, bucket = _compute_bet_size(ctx, s1b, "pressure")
            return RAISE, amount, m_pressure * 0.5, bucket

    if can_check:
        return CHECK, 0, 0.0, ""
    return CHECK, 0, 0.0, ""


def _facing_bet_action(
    ctx: Street1BContext,
    motives: Dict[str, float],
    s1b: float,
    can_call: bool,
    can_raise: bool,
    can_fold: bool,
    can_check: bool,
) -> Tuple[int, int, float, str]:
    """We are facing a bet (amount_to_call > 0)."""
    m_continue = motives["m_continue"]
    m_defend = motives["m_defend"]
    m_reraise = motives["m_reraise"]

    pot_odds = ctx.amount_to_call / (ctx.pot_size + ctx.amount_to_call + 1)

    # Re-raise with very strong hands
    if m_reraise > 0.30 and s1b > 0.55 and can_raise:
        amount, bucket = _compute_bet_size(ctx, s1b, "value")
        return RAISE, amount, m_reraise, bucket

    # Call with decent hands
    if s1b > pot_odds + 0.05 and can_call:
        return CALL, 0, m_continue, ""

    # Defend marginal hands at good pot odds
    if m_defend > 0.20 and pot_odds < 0.30 and can_call:
        return CALL, 0, m_defend, ""

    # Small calls
    if ctx.amount_to_call <= 5 and s1b > 0.15 and can_call:
        return CALL, 0, 0.3, ""

    if can_check:
        return CHECK, 0, 0.0, ""
    if can_fold:
        return FOLD, 0, 0.0, ""
    return FOLD, 0, 0.0, ""


# =========================================================================
# Bet Sizing
# =========================================================================

def _compute_bet_size(
    ctx: Street1BContext,
    s1b: float,
    mode: str,
) -> Tuple[int, str]:
    """
    Compute bet amount and size bucket.
    mode: "value" or "pressure"
    """
    pot = max(1, ctx.pot_size)

    if mode == "value":
        # Value: bet proportional to hand strength
        if s1b >= 0.75:
            frac = 0.70
            bucket = "large"
        elif s1b >= 0.55:
            frac = 0.45
            bucket = "medium"
        else:
            frac = 0.30
            bucket = "small"
    else:
        # Pressure: bet enough to induce folds
        if s1b >= 0.50:
            frac = 0.55
            bucket = "medium"
        elif s1b >= 0.30:
            frac = 0.40
            bucket = "small"
        else:
            frac = 0.30
            bucket = "small"

    base_amount = int(pot * frac)

    # Confidence dampening: less confident = smaller bets
    # Jitter: small random noise to avoid being predictable
    jitter = random.uniform(-0.05, 0.05)
    base_amount = int(base_amount * (1.0 + jitter))

    amount = max(ctx.min_raise, min(base_amount, ctx.max_raise))
    return amount, bucket


# =========================================================================
# Action Mixing (gradient: more mixing when confidence is low)
# =========================================================================

def apply_flop_action_mixing(
    action_type: int,
    confidence: float,
    ctx: Street1BContext,
    s1b: float,
) -> int:
    """Optionally randomize the action when confidence is low."""
    if confidence > 0.60:
        return action_type

    can_check = len(ctx.valid_actions) > CHECK and ctx.valid_actions[CHECK] == 1

    # Low confidence + opening = might check instead of betting
    if action_type == RAISE and confidence < 0.30 and can_check:
        if random.random() < 0.30:
            return CHECK

    return action_type


# =========================================================================
# Top-level entry point
# =========================================================================

def get_street1b_action(ctx: Street1BContext) -> Tuple[int, int, str, Street1BScoreBreakdown]:
    """
    Main entry point for Street 1B decisions.

    Returns: (action_type, raise_amount, size_bucket, breakdown)
    """
    s1b, bd = street1b_score(ctx)
    motives = compute_flop_action_motives(s1b, ctx)
    action_type, raise_amount, confidence, size_bucket = compute_flop_action(ctx, motives, s1b)
    action_type = apply_flop_action_mixing(action_type, confidence, ctx, s1b)

    # If mixing changed to CHECK but we were going to RAISE, clear amount
    if action_type != RAISE:
        raise_amount = 0
        size_bucket = ""

    return action_type, raise_amount, size_bucket, bd
