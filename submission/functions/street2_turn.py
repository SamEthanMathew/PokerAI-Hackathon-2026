"""
Street 2 (Turn) Engine
=====================
Single engine for turn: re-evaluation + betting. No discard on turn.
Tighter than Street 1: more deterministic mixing, smaller jitter.

Score: S2 = w_h*H + w_r*Rv + w_e*E - w_t*T + w_p*P + w_l*L + w_x*X
Components: H (hand), Rv (river value), E (dead-card advantage), T (transparency),
P (pressure), L (line consistency), X (opponent exploit).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from submission.functions.street0_score import (
    DECK_SIZE,
    NUM_RANKS,
    NUM_SUITS,
    STRAIGHT_WINDOWS,
    classify_board_texture,
    evaluate_hand,
    hand_class_from_score,
    hand_rank_counts,
    hand_suit_counts,
    rank,
    refined_utility,
    suit,
    within_class_percentile,
)

# Action type constants (match gym_env.PokerEnv.ActionType)
FOLD = 0
RAISE = 1
CHECK = 2
CALL = 3

# S2 weights (plan §2.1)
W_H = 0.26
W_R = 0.20
W_E = 0.14
W_T = 0.12
W_P = 0.12
W_L = 0.08
W_X = 0.08

# X clamp so opponent exploit doesn't dominate
X_CLAMP = 0.15

# Sizing neutrals (plan §2.6)
Z_NEUTRAL_TURN_BET = 0.46
Z_NEUTRAL_TURN_RAISE = 0.58

# Mixing: only when action scores within this
MIXING_THRESHOLD = 0.05
SOFTMAX_TEMP = 0.15


# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class Street2Context:
    our_keep2: List[int]
    our_discard3: List[int]
    opp_discard3: List[int]
    flop3: List[int]
    turn_card: int
    board4: List[int]  # flop3 + [turn_card]
    pot_size: int
    amount_to_call: int
    valid_actions: List[int]
    min_raise: int
    max_raise: int
    opponent_profile: Any
    recon: Any  # OpponentRecon for turn getters
    our_discard_class: str
    opp_discard_class: str
    flop_texture: str
    turn_texture: str
    flop_line: str  # e.g. "we_checked_opp_checked", "we_bet_small_opp_called"
    opp_last_action: str
    position: int  # 0=SB, 1=BB


@dataclass
class Street2ScoreBreakdown:
    H: float = 0.0
    Rv: float = 0.0
    E: float = 0.0
    T: float = 0.0
    P: float = 0.0
    L: float = 0.0
    X: float = 0.0
    S2: float = 0.0


# =========================================================================
# Turn texture classification
# =========================================================================

def classify_turn_texture(board4: List[int]) -> str:
    """Classify 4-card board (flop + turn) for turn texture. Reuse street0_score."""
    if not board4 or len(board4) < 4:
        return "blank_turn"
    return classify_board_texture(board4)


def _turn_texture_bucket_for_t(turn_texture: str) -> str:
    """Map to short bucket: paired_turn, flush_turn, straight_turn, blank_turn."""
    if not turn_texture:
        return "blank_turn"
    t = turn_texture.lower()
    if "pair" in t or "paired" in t:
        return "paired_turn"
    if "flush" in t:
        return "flush_turn"
    if "straight" in t or "connect" in t:
        return "straight_turn"
    return "blank_turn"


# =========================================================================
# H – Current turn hand strength (with vulnerability discount)
# =========================================================================

def _vulnerability_discount(score: int, cls: int, b: float) -> float:
    """Fragile strong hands get a small reduction; nuts/robust get full weight."""
    if cls <= 1:  # Straight Flush, Full House – robust
        return 1.0
    if cls == 2:  # Flush – low flush is fragile
        if b < 0.4:
            return 0.92
        return 1.0
    if cls == 3:  # Trips – brittle on draw-heavy boards
        if b < 0.5:
            return 0.94
        return 1.0
    if cls == 5:  # Two pair – usually robust
        return 1.0
    return 1.0


def score_turn_hand_strength(keep2: List[int], board4: List[int]) -> float:
    """H: evaluate keep2 + board4 at street 2 with optional vulnerability discount."""
    score = evaluate_hand(keep2, board4)
    cls = hand_class_from_score(score)
    u = refined_utility(2, score)
    b = within_class_percentile(score, cls)
    mult = _vulnerability_discount(score, cls, b)
    return u * mult


# =========================================================================
# Rv – River value potential
# =========================================================================

def _remaining_deck(keep2: List[int], our_discard: List[int], opp_discard: List[int], board4: List[int]) -> List[int]:
    used = set(keep2) | set(c for c in board4) | set(c for c in our_discard if c != -1) | set(c for c in opp_discard if c != -1)
    return [c for c in range(DECK_SIZE) if c not in used]


def score_river_value_potential(
    keep2: List[int],
    board4: List[int],
    our_discard: List[int],
    opp_discard: List[int],
    opp_discard_class: str,
) -> float:
    """
    Rv: one river card remains. Weight: reward improvement/safe rivers,
    penalize counterfeits and rivers that complete villain's likely draws.
    """
    deck = _remaining_deck(keep2, our_discard, opp_discard, board4)
    if not deck:
        return 0.0

    current_score = evaluate_hand(keep2, board4)
    u_now = refined_utility(2, current_score)
    cls_now = hand_class_from_score(current_score)

    rv_sum = 0.0
    for r in deck:
        board5 = list(board4) + [r]
        sc3 = evaluate_hand(keep2, board5)
        u_river = refined_utility(3, sc3)
        cls_river = hand_class_from_score(sc3)

        # Improvement: better class or within-class
        if cls_river < cls_now:
            improve = 1.0 + (u_river - u_now)
        elif cls_river == cls_now:
            b_now = within_class_percentile(current_score, cls_now)
            b_river = within_class_percentile(sc3, cls_river)
            improve = 0.5 + 0.5 * (b_river - b_now)
        else:
            # Counterfeit (e.g. board pairs and we have two pair)
            improve = -0.2

        # Simple villain-completion penalty: if turn texture suggests draws, river that completes flush/straight
        # We don't have full villain hand; use opp_discard_class. Penalize flush/straight completing rivers
        # when opp was draw-heavy (flush_transparent, straight_transparent).
        penalty = 0.0
        r_suit = suit(r)
        r_rank = rank(r)
        board_suits = hand_suit_counts(board4)
        board_ranks = set(rank(c) for c in board4)
        if opp_discard_class == "flush_transparent" and board_suits.get(r_suit, 0) >= 2:
            if max(board_suits.values()) >= 3 or board_suits.get(r_suit, 0) >= 3:
                penalty += 0.15
        if opp_discard_class == "straight_transparent":
            for w in STRAIGHT_WINDOWS:
                if r_rank in w and len(w & board_ranks) >= 2:
                    penalty += 0.10
                    break

        rv_sum += max(0.0, improve - penalty)

    n = len(deck)
    return rv_sum / n if n else 0.0


# =========================================================================
# E – Dead-card / equity-position advantage
# =========================================================================

def score_turn_dead_card_advantage(
    keep2: List[int],
    board4: List[int],
    our_discard: List[int],
    opp_discard: List[int],
) -> float:
    """E: do dead cards reduce villain's improvements more than ours?"""
    our_disc_ranks = set(rank(c) for c in our_discard if c != -1)
    opp_disc_ranks = set(rank(c) for c in opp_discard if c != -1)
    opp_disc_suits = hand_suit_counts([c for c in opp_discard if c != -1])
    board_ranks = set(rank(c) for c in board4)
    keep_ranks = set(rank(c) for c in keep2)

    advantage = 0.0

    for w in STRAIGHT_WINDOWS:
        opp_killed = w & opp_disc_ranks
        our_covered = w & (keep_ranks | board_ranks)
        if len(opp_killed) >= 2 and len(our_covered) >= 3:
            advantage += 0.12
    for s in range(NUM_SUITS):
        if opp_disc_suits.get(s, 0) >= 2:
            advantage += 0.08
    # Our discard removed our outs
    for w in STRAIGHT_WINDOWS:
        our_killed = w & our_disc_ranks
        if len(our_killed) >= 2:
            advantage -= 0.06
    return max(-0.2, min(0.25, advantage))


# =========================================================================
# T – Transparency / line-exposure penalty
# =========================================================================

def score_turn_transparency(
    our_discard_class: str,
    flop_line: str,
    turn_texture: str,
) -> float:
    """T: high when our line is very readable (e.g. flush discard + flush turn)."""
    t = 0.0
    dc = (our_discard_class or "").lower()
    tt = (turn_texture or "").lower()
    fl = (flop_line or "").lower()

    if "flush" in dc and ("flush" in tt or "complet" in tt):
        t += 0.35
    if "straight" in dc and ("connect" in tt or "straight" in tt):
        t += 0.28
    if "pair" in dc and "pair" in tt:
        t += 0.20
    if "we_checked" in fl and "we_bet" not in fl:
        if "flush" in dc or "straight" in dc:
            t += 0.15  # check then turn bet with draw discard is readable
    return min(1.0, t)


# =========================================================================
# P – Pressure suitability (recon getters)
# =========================================================================

def score_turn_pressure(
    ctx: Street2Context,
    recon: Any,
) -> float:
    """P: reward villain capped, overfolds turn; penalize transparent bluff, sticky villain."""
    if recon is None:
        return 0.5
    try:
        from submission.functions.opponent_recon import (
            get_turn_fold_vs_size_bucket,
            get_turn_fold_by_turn_texture,
            get_turn_fold_vs_our_discard_class,
        )
    except ImportError:
        return 0.5

    fold_vs_size = get_turn_fold_vs_size_bucket(recon, "medium")
    fold_vs_tex = get_turn_fold_by_turn_texture(recon, ctx.turn_texture)
    fold_vs_dc = get_turn_fold_vs_our_discard_class(recon, ctx.our_discard_class) if ctx.our_discard_class else 0.5

    p = (fold_vs_size + fold_vs_tex + fold_vs_dc) / 3.0
    # VCAP: villain capped (opp_discard_class + flop line)
    if (ctx.opp_discard_class or "").lower() == "capped":
        p += 0.12
    if (ctx.our_discard_class or "").lower() in ("flush_transparent", "straight_transparent"):
        p -= 0.15  # penalize bluffing with transparent discard
    return max(0.0, min(1.0, p))


# =========================================================================
# L – Line consistency value
# =========================================================================

def score_turn_line_consistency(
    our_discard_class: str,
    flop_line: str,
    turn_texture: str,
    considering_bet: bool,
) -> float:
    """L: reward turn action that fits discard + flop line; penalize erratic lines."""
    l_val = 0.5
    dc = (our_discard_class or "").lower()
    fl = (flop_line or "").lower()

    if considering_bet:
        if "we_bet" in fl and ("value" in dc or "pair" in dc or "ambiguous" in dc):
            l_val += 0.2
        if "we_checked" in fl and ("flush" in dc or "straight" in dc):
            l_val -= 0.15  # check-call flop then big turn bluff is erratic
    else:
        if "we_bet" in fl:
            l_val += 0.1  # bet flop then check turn can be consistent
    return max(0.0, min(1.0, l_val))


# =========================================================================
# X – Opponent exploit adjustment (clamp ±0.15)
# =========================================================================

def score_turn_exploit(ctx: Street2Context, recon: Any) -> float:
    """X: tilt S2 up when villain overfolds, down when overraises; clamp ±X_CLAMP."""
    if recon is None:
        return 0.0
    try:
        from submission.functions.opponent_recon import (
            get_turn_fold_vs_size_bucket,
            get_turn_raise_vs_size_bucket,
            get_turn_fold_by_turn_texture,
            get_turn_fold_vs_our_discard_class,
            get_turn_aggression_after_flop_line,
        )
    except ImportError:
        return 0.0

    fold_vs = get_turn_fold_vs_size_bucket(recon, "medium")
    raise_vs = get_turn_raise_vs_size_bucket(recon, "medium")
    fold_tex = get_turn_fold_by_turn_texture(recon, ctx.turn_texture)
    fold_dc = get_turn_fold_vs_our_discard_class(recon, ctx.our_discard_class) if ctx.our_discard_class else 0.5

    # Overfold -> positive X (we can bluff more); overraise -> negative X
    x = (fold_vs - 0.5) * 0.5 + (0.5 - raise_vs) * 0.3 + (fold_tex - 0.5) * 0.1 + (fold_dc - 0.5) * 0.1
    return max(-X_CLAMP, min(X_CLAMP, x))


# =========================================================================
# S2 composite and breakdown
# =========================================================================

def compute_s2_breakdown(ctx: Street2Context) -> Tuple[float, Street2ScoreBreakdown]:
    """Compute H, Rv, E, T, P, L, X and S2."""
    bd = Street2ScoreBreakdown()

    bd.H = score_turn_hand_strength(ctx.our_keep2, ctx.board4)
    bd.Rv = score_river_value_potential(
        ctx.our_keep2, ctx.board4, ctx.our_discard3, ctx.opp_discard3, ctx.opp_discard_class or "",
    )
    bd.E = score_turn_dead_card_advantage(
        ctx.our_keep2, ctx.board4, ctx.our_discard3, ctx.opp_discard3,
    )
    bd.T = score_turn_transparency(ctx.our_discard_class or "", ctx.flop_line or "", ctx.turn_texture or "")
    bd.P = score_turn_pressure(ctx, ctx.recon)
    bd.L = score_turn_line_consistency(
        ctx.our_discard_class or "", ctx.flop_line or "", ctx.turn_texture or "", considering_bet=True,
    )
    bd.X = score_turn_exploit(ctx, ctx.recon)

    bd.S2 = (
        W_H * bd.H + W_R * bd.Rv + W_E * bd.E
        - W_T * bd.T + W_P * bd.P + W_L * bd.L + W_X * bd.X
    )
    bd.S2 = max(0.0, min(1.0, bd.S2))
    return bd.S2, bd


# =========================================================================
# Descriptors
# =========================================================================

def compute_turn_descriptors(s2: float, breakdown: Street2ScoreBreakdown, ctx: Street2Context) -> Dict[str, float]:
    """Descriptors for motives and sizing."""
    H, Rv, E, T, P, L, X = breakdown.H, breakdown.Rv, breakdown.E, breakdown.T, breakdown.P, breakdown.L, breakdown.X

    HV2 = H * (1.0 - 0.3 * (1.0 - min(1.0, H + 0.2)))  # value heaviness
    ROB2 = min(1.0, H * 1.2 + Rv * 0.3)  # robustness vs raise
    DRAW2 = Rv * (1.0 - H)  # river-improvement dependence
    TCAP = T  # how capped our line looks
    VCAP = 0.6 if (ctx.opp_discard_class or "").lower() == "capped" else 0.3
    TPEN = T
    DENY = P * (1.0 - H) * 0.8  # denial value of betting
    SDV = H * (1.0 - Rv * 0.3)  # showdown value
    BLUFF2 = max(0.0, P * (1.0 - T) * (1.0 - SDV) - 0.1)

    return {
        "S2": s2,
        "HV2": HV2,
        "ROB2": ROB2,
        "DRAW2": DRAW2,
        "TCAP": TCAP,
        "VCAP": VCAP,
        "TPEN": TPEN,
        "DENY": DENY,
        "SDV": SDV,
        "BLUFF2": BLUFF2,
    }


# =========================================================================
# Motives (continuous)
# =========================================================================

def compute_turn_motives(
    s2: float,
    breakdown: Street2ScoreBreakdown,
    descriptors: Dict[str, float],
    ctx: Street2Context,
) -> Dict[str, float]:
    """M_continue_2, M_press_2, M_value_2, M_defend_2, M_raise_2."""
    H, Rv, T, P, L, X = breakdown.H, breakdown.Rv, breakdown.T, breakdown.P, breakdown.L, breakdown.X
    ROB2 = descriptors["ROB2"]
    SDV = descriptors["SDV"]
    DRAW2 = descriptors["DRAW2"]
    TPEN = descriptors["TPEN"]
    DENY = descriptors["DENY"]
    VCAP = descriptors["VCAP"]
    BLUFF2 = descriptors["BLUFF2"]
    HV2 = descriptors["HV2"]

    # Indices from recon (simplified if no recon)
    stickiness = 0.5
    fold_turn_idx = 0.5
    if ctx.recon is not None:
        try:
            from submission.functions.opponent_recon import get_turn_fold_vs_size_bucket
            fold_turn_idx = get_turn_fold_vs_size_bucket(ctx.recon, "medium")
            stickiness = 1.0 - fold_turn_idx
        except ImportError:
            pass

    adj = 0.5 + X * 2.0  # exploit tilt
    adj = max(0.3, min(0.7, adj))

    M_continue_2 = s2 * 0.4 + ROB2 * 0.3 + SDV * 0.2 + Rv * 0.1 - TPEN * 0.2
    M_continue_2 = max(0.0, min(1.0, M_continue_2 + (adj - 0.5)))

    M_press_2 = P * 0.3 + DENY * 0.25 + VCAP * 0.2 + (1.0 - SDV) * 0.15 + BLUFF2 * 0.2 - TPEN * 0.2 - stickiness * 0.1
    M_press_2 = max(0.0, min(1.0, M_press_2 + (adj - 0.5) * 0.5))

    M_value_2 = HV2 * 0.35 + H * 0.25 + ROB2 * 0.2 + (1.0 - TPEN) * 0.15 + stickiness * 0.05 - (1.0 - fold_turn_idx) * 0.05
    M_value_2 = max(0.0, min(1.0, M_value_2))

    M_defend_2 = s2 * 0.35 + ROB2 * 0.25 + SDV * 0.2 + Rv * 0.1 - TPEN * 0.2
    M_defend_2 = max(0.0, min(1.0, M_defend_2 - (1.0 - adj) * 0.1))

    M_raise_2 = 0.4 * (M_press_2 + M_value_2) / 2 + DENY * 0.2 + VCAP * 0.15 + ROB2 * 0.15 - TPEN * 0.2 - stickiness * 0.1
    M_raise_2 = max(0.0, min(1.0, M_raise_2))

    return {
        "M_continue_2": M_continue_2,
        "M_press_2": M_press_2,
        "M_value_2": M_value_2,
        "M_defend_2": M_defend_2,
        "M_raise_2": M_raise_2,
    }


# =========================================================================
# Action scoring and selection
# =========================================================================

def _can_check(ctx: Street2Context) -> bool:
    va = ctx.valid_actions
    return len(va) > CHECK and va[CHECK] == 1


def _can_call(ctx: Street2Context) -> bool:
    va = ctx.valid_actions
    return len(va) > CALL and va[CALL] == 1


def _can_raise(ctx: Street2Context) -> bool:
    va = ctx.valid_actions
    return len(va) > RAISE and va[RAISE] == 1


def _can_fold(ctx: Street2Context) -> bool:
    va = ctx.valid_actions
    return len(va) > FOLD and va[FOLD] == 1


def compute_turn_action(
    ctx: Street2Context,
    s2: float,
    motives: Dict[str, float],
    descriptors: Dict[str, float],
) -> Tuple[int, int, float, str]:
    """
    Returns (action_type, raise_amount, confidence, size_bucket).
    Checked to us vs facing bet vs facing raise (after we bet).
    """
    is_opening = ctx.amount_to_call <= 0
    can_check = _can_check(ctx)
    can_call = _can_call(ctx)
    can_raise = _can_raise(ctx)
    can_fold = _can_fold(ctx)

    if is_opening:
        return _turn_opening(ctx, s2, motives, descriptors, can_check, can_raise)
    # Facing a bet (or a raise after we bet)
    return _turn_facing_bet(ctx, s2, motives, descriptors, can_call, can_raise, can_fold, can_check)


def _turn_opening(
    ctx: Street2Context,
    s2: float,
    motives: Dict[str, float],
    descriptors: Dict[str, float],
    can_check: bool,
    can_raise: bool,
) -> Tuple[int, int, float, str]:
    """Checked to us: A_check_2 vs A_bet_2."""
    m_value = motives["M_value_2"]
    m_press = motives["M_press_2"]

    if s2 >= 0.52 and can_raise:
        amount, bucket = _compute_turn_bet_size(ctx, s2, "value", descriptors, motives)
        return RAISE, amount, m_value, bucket
    if s2 >= 0.28 and m_press > 0.32 and can_raise:
        amount, bucket = _compute_turn_bet_size(ctx, s2, "pressure", descriptors, motives)
        return RAISE, amount, m_press, bucket
    if s2 >= 0.12 and m_press > 0.42 and can_raise:
        if random.random() < 0.22:
            amount, bucket = _compute_turn_bet_size(ctx, s2, "pressure", descriptors, motives)
            return RAISE, amount, m_press * 0.5, bucket
    if can_check:
        return CHECK, 0, 0.0, ""
    return CHECK, 0, 0.0, ""


def _turn_facing_bet(
    ctx: Street2Context,
    s2: float,
    motives: Dict[str, float],
    descriptors: Dict[str, float],
    can_call: bool,
    can_raise: bool,
    can_fold: bool,
    can_check: bool,
) -> Tuple[int, int, float, str]:
    """Facing a bet: fold / call / raise. Re-raise only in strong spots."""
    m_continue = motives["M_continue_2"]
    m_defend = motives["M_defend_2"]
    m_raise = motives["M_raise_2"]

    pot_odds = ctx.amount_to_call / (ctx.pot_size + ctx.amount_to_call + 1) if (ctx.pot_size + ctx.amount_to_call) > 0 else 0.5

    if m_raise > 0.28 and s2 > 0.52 and can_raise:
        amount, bucket = _compute_turn_bet_size(ctx, s2, "value", descriptors, motives)
        return RAISE, amount, m_raise, bucket
    if s2 > pot_odds + 0.06 and can_call:
        return CALL, 0, m_continue, ""
    if motives["M_defend_2"] > 0.18 and pot_odds < 0.28 and can_call:
        return CALL, 0, m_defend, ""
    if ctx.amount_to_call <= 5 and s2 > 0.12 and can_call:
        return CALL, 0, 0.25, ""
    if can_check:
        return CHECK, 0, 0.0, ""
    if can_fold:
        return FOLD, 0, 0.0, ""
    return FOLD, 0, 0.0, ""


# =========================================================================
# Sizing: Z_press_2, Z_value_2, blend, confidence damp, jitter (tighter than Street 1)
# =========================================================================

def _compute_turn_bet_size(
    ctx: Street2Context,
    s2: float,
    mode: str,
    descriptors: Dict[str, float],
    motives: Dict[str, float],
) -> Tuple[int, str]:
    """Z_press_2 / Z_value_2 blend; confidence damp; small jitter."""
    pot = max(1, ctx.pot_size)
    m_press = motives["M_press_2"]
    m_value = motives["M_value_2"]
    eps = 1e-6
    beta = m_press / (m_press + m_value + eps)
    beta = max(0.0, min(1.0, beta))

    if mode == "value":
        Z_val = 0.35 + s2 * 0.45
        Z_press = 0.30
    else:
        Z_val = 0.25
        Z_press = 0.30 + descriptors.get("DENY", 0.2) * 0.3 + descriptors.get("VCAP", 0.3) * 0.2

    Z_raw = beta * Z_press + (1.0 - beta) * Z_val
    # Confidence damp toward neutral
    Z_neutral = Z_NEUTRAL_TURN_BET
    Z_raw = Z_raw * 0.85 + Z_neutral * 0.15
    jitter = random.uniform(-0.03, 0.03)
    Z_raw = max(0.0, min(1.0, Z_raw + jitter))

    frac = 0.25 + Z_raw * 0.55
    base_amount = int(pot * frac)
    amount = max(ctx.min_raise, min(base_amount, ctx.max_raise))

    if frac >= 0.65:
        bucket = "large"
    elif frac >= 0.42:
        bucket = "medium"
    else:
        bucket = "small"
    return amount, bucket


# =========================================================================
# Mixing (more deterministic than Street 1)
# =========================================================================

def apply_turn_action_mixing(
    action_type: int,
    confidence: float,
    ctx: Street2Context,
    s2: float,
) -> int:
    """Only mix when action scores are close; low temperature."""
    if confidence > 0.55:
        return action_type
    can_check = _can_check(ctx)
    if action_type == RAISE and confidence < 0.28 and can_check:
        if random.random() < 0.22:
            return CHECK
    return action_type


# =========================================================================
# Top-level API
# =========================================================================

def get_street2_action(ctx: Street2Context) -> Tuple[int, int, str, Street2ScoreBreakdown]:
    """
    Main entry point for Street 2 (turn) decisions.

    Returns: (action_type, raise_amount, size_bucket, breakdown)
    """
    s2, breakdown = compute_s2_breakdown(ctx)
    descriptors = compute_turn_descriptors(s2, breakdown, ctx)
    motives = compute_turn_motives(s2, breakdown, descriptors, ctx)

    action_type, raise_amount, confidence, size_bucket = compute_turn_action(ctx, s2, motives, descriptors)
    action_type = apply_turn_action_mixing(action_type, confidence, ctx, s2)

    if action_type != RAISE:
        raise_amount = 0
        size_bucket = ""

    return action_type, raise_amount, size_bucket, breakdown
