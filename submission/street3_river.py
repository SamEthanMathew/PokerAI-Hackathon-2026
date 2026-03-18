"""
Street 3 (River) Engine
=======================
Final hand strength on full board (keep2 + board5), board-adjusted confidence,
flop/turn line, and river recon for check/bet/call/fold/raise and sizing.

Score: R3 = w_h*H3 + w_c*C3 + w_l*L3 + w_x*X3
Components: H3 (hand strength), C3 (confidence), L3 (line consistency), X3 (exploit).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from submission.street0_score import (
    classify_board_texture,
    evaluate_hand,
    hand_class_from_score,
    hand_rank_counts,
    refined_utility,
    within_class_percentile,
)

# Action type constants (match gym_env.PokerEnv.ActionType)
FOLD = 0
RAISE = 1
CHECK = 2
CALL = 3

# R3 weights (plan §5)
W_H = 0.35
W_C = 0.35
W_L = 0.15
W_X = 0.15

# X clamp so exploit doesn't dominate
X_CLAMP = 0.15

# Mixing (plan §8): more deterministic than turn
MIXING_THRESHOLD = 0.04
SOFTMAX_TEMP = 0.12

# River sample size below which we damp sizing toward medium (plan §7)
RIVER_SAMPLE_DAMP_THRESHOLD = 5

# Jitter (plan §7): smaller than turn
RIVER_JITTER = 0.02


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class Street3Context:
    our_keep2: List[int]
    board5: List[int]
    pot_size: int
    amount_to_call: int
    valid_actions: List[int]
    min_raise: int
    max_raise: int
    position: int  # 0=SB, 1=BB
    recon: Any  # OpponentRecon for river getters
    flop_line: str
    turn_we_bet: bool
    turn_opp_action: str
    opp_last_action: str
    river_texture: str = ""  # classify_board_texture(board5) for this hand
    our_discard_class: str = ""
    opp_discard_class: str = ""
    opp_type: str = ""  # "loose" / "tight" / "balanced" from opponent_recon


@dataclass
class Street3ScoreBreakdown:
    H3: float = 0.0
    C3: float = 0.0
    L3: float = 0.0
    X3: float = 0.0
    R3: float = 0.0


# =========================================================================
# River texture and board helpers
# =========================================================================

def is_board_paired(board5: List[int]) -> bool:
    """True if any rank appears >= 2 on board."""
    if not board5 or len(board5) < 5:
        return False
    from submission.street0_score import rank
    rc = hand_rank_counts(board5)
    return any(cnt >= 2 for cnt in rc.values())


def river_texture_info(board5: List[int]) -> Tuple[str, bool, List[str]]:
    """
    Returns (texture, is_paired, possible_best_hand_types).
    texture from classify_board_texture; possible_best_hand_types list of hand types
    that can be nut/top on this board.
    """
    if not board5 or len(board5) < 5:
        return "unknown_texture", False, []
    texture = classify_board_texture(board5)
    paired = is_board_paired(board5)
    possible: List[str] = []
    t = texture.lower()
    if "flush_completed" in t:
        possible.append("flush")
        possible.append("straight_flush")
    if "straight_completed" in t:
        possible.append("straight")
        possible.append("straight_flush")
    if "river_standard" in t:
        if paired:
            possible.extend(["full_house", "trips", "two_pair"])
        else:
            possible.extend(["trips", "two_pair", "one_pair"])
    if paired:
        possible.extend(["full_house", "trips", "two_pair"])
    return texture, paired, list(dict.fromkeys(possible))


# =========================================================================
# C3 – Board-adjusted confidence (plan §2, §11)
# =========================================================================

def river_confidence(score: int, board5: List[int]) -> float:
    """
    Board-adjusted confidence in [0, 1]. 1 = nut or near-nut for this board.
    Uses river_texture, is_board_paired, hand_class, within_class_percentile.
    """
    if not board5 or len(board5) < 5:
        return 0.5
    cls = hand_class_from_score(score)
    b = within_class_percentile(score, cls)
    texture, paired = river_texture_info(board5)[:2]
    t = texture.lower()

    # Straight Flush (class 0): always very high on any board
    if cls == 0:
        return 0.95 + 0.05 * b

    # flush_completed_river: flush possible; our flush → confidence by percentile; no flush → reduced
    if "flush_completed" in t:
        if cls == 2:  # We have flush
            return 0.25 + 0.75 * b  # nut flush ≈ 1, low flush < 1
        # We don't have flush; they could have flush
        return 0.12

    # straight_completed_river: straight possible; same idea
    if "straight_completed" in t:
        if cls == 4:  # We have straight (class 4 in _CLASS_BOUNDS)
            return 0.25 + 0.75 * b
        return 0.12

    # river_standard: no flush/straight on board → our flush/straight are nut; full house/trips strong
    if "river_standard" in t:
        if cls == 2:  # Flush on dry board = nut flush
            return 0.82 + 0.18 * b
        if cls == 4:  # Straight on dry board = nut straight
            return 0.82 + 0.18 * b
        if cls == 1:  # Full house
            return 0.85 + 0.15 * b
        if cls == 3:  # Trips
            return 0.5 + 0.5 * b if paired else 0.6 + 0.4 * b
        if cls == 5:  # Two pair
            return 0.45 + 0.55 * b if paired else 0.55 + 0.45 * b
        if cls == 6:  # One pair
            return 0.25 + 0.35 * b
        # High card
        return 0.1 + 0.2 * b

    # Fallback (e.g. turn texture passed by mistake)
    return 0.3 + 0.5 * b


# =========================================================================
# H3 – Final hand strength
# =========================================================================

def score_river_hand_strength(keep2: List[int], board5: List[int]) -> float:
    """H3: refined_utility(3, score) for keep2 + board5."""
    if len(keep2) < 2 or len(board5) < 5:
        return 0.0
    score = evaluate_hand(keep2, board5)
    return refined_utility(3, score)


# =========================================================================
# L3 – Line consistency (plan §5)
# =========================================================================

def score_river_line_consistency(
    flop_line: str,
    turn_we_bet: bool,
    turn_opp_action: str,
    considering_bet: bool,
) -> float:
    """
    L3: our river action fits flop/turn line. We bet flop and turn → L3 high for value bet;
    we checked both → L3 high for check; etc.
    """
    l_val = 0.5
    fl = (flop_line or "").lower()
    topp = (turn_opp_action or "").lower()

    if considering_bet:
        if turn_we_bet and "we_bet" in fl:
            l_val += 0.25  # bet flop and turn → value bet on river is consistent
        if not turn_we_bet and "we_checked" in fl:
            l_val -= 0.15  # checked both → donk bet is less consistent
        if "call" in topp or "raise" in topp:
            l_val += 0.1  # they acted → our bet can rep strength
    else:
        if turn_we_bet:
            l_val += 0.1  # bet flop and turn then check river can be consistent (give up)
        if not turn_we_bet and "we_checked" in fl:
            l_val += 0.2  # checked both → check river is very consistent
    return max(0.0, min(1.0, l_val))


# =========================================================================
# X3 – Exploit (river recon getters, plan §5)
# =========================================================================

def score_river_exploit(ctx: Street3Context, recon: Any) -> float:
    """
    X3: tilt R3 up when opponent overfolds to our bets or rarely bets when we check;
    down when sticky or frequently betting when we check. Clamp ±X_CLAMP.
    """
    if recon is None:
        return 0.0
    try:
        from submission.opponent_recon import (
            get_river_fold_vs_size_bucket,
            get_river_raise_vs_size_bucket,
            get_river_fold_by_river_texture,
            get_river_fold_vs_our_discard_class,
            get_river_bet_when_checked_to,
        )
    except ImportError:
        return 0.0

    fold_vs_size = get_river_fold_vs_size_bucket(recon, "medium")
    raise_vs_size = get_river_raise_vs_size_bucket(recon, "medium")
    fold_tex = get_river_fold_by_river_texture(recon, ctx.river_texture or "")
    fold_dc = get_river_fold_vs_our_discard_class(recon, ctx.our_discard_class) if ctx.our_discard_class else 0.5
    bet_when_checked = get_river_bet_when_checked_to(recon)

    # Overfold → positive X; overraise → negative X; they bet when we check a lot → negative X
    x = (fold_vs_size - 0.5) * 0.4 + (0.5 - raise_vs_size) * 0.25
    x += (fold_tex - 0.5) * 0.1 + (fold_dc - 0.5) * 0.1
    x += (0.5 - bet_when_checked) * 0.15  # rarely bet when we check → we can check more
    return max(-X_CLAMP, min(X_CLAMP, x))


# =========================================================================
# R3 composite and breakdown
# =========================================================================

def compute_r3_breakdown(ctx: Street3Context) -> Tuple[float, Street3ScoreBreakdown]:
    """Compute H3, C3, L3, X3 and R3. Requires len(board5)==5 for real computation."""
    bd = Street3ScoreBreakdown()
    if len(ctx.board5) < 5 or len(ctx.our_keep2) < 2:
        bd.R3 = 0.5
        return 0.5, bd

    score = evaluate_hand(ctx.our_keep2, ctx.board5)
    bd.H3 = refined_utility(3, score)
    bd.C3 = river_confidence(score, ctx.board5)
    bd.L3 = score_river_line_consistency(
        ctx.flop_line or "",
        ctx.turn_we_bet,
        ctx.turn_opp_action or "",
        considering_bet=True,
    )
    bd.X3 = score_river_exploit(ctx, ctx.recon)

    bd.R3 = W_H * bd.H3 + W_C * bd.C3 + W_L * bd.L3 + W_X * bd.X3
    bd.R3 = max(0.0, min(1.0, bd.R3))
    return bd.R3, bd


# =========================================================================
# Action helpers (valid_actions)
# =========================================================================

def _can_check(ctx: Street3Context) -> bool:
    va = ctx.valid_actions
    return len(va) > CHECK and va[CHECK] == 1


def _can_call(ctx: Street3Context) -> bool:
    va = ctx.valid_actions
    return len(va) > CALL and va[CALL] == 1


def _can_raise(ctx: Street3Context) -> bool:
    va = ctx.valid_actions
    return len(va) > RAISE and va[RAISE] == 1


def _can_fold(ctx: Street3Context) -> bool:
    va = ctx.valid_actions
    return len(va) > FOLD and va[FOLD] == 1


# =========================================================================
# Action logic (plan §6): checked to us vs facing bet
# =========================================================================

def _river_opening(
    ctx: Street3Context,
    r3: float,
    breakdown: Street3ScoreBreakdown,
    can_check: bool,
    can_raise: bool,
) -> Tuple[int, int, str]:
    """
    Checked to us: value if confidence and H3 above thresholds; bluff sometimes; else check.
    Returns (action_type, raise_amount, size_bucket).
    """
    C3, H3 = breakdown.C3, breakdown.H3
    pot = max(1, ctx.pot_size)

    # Value: confidence and H3 above thresholds
    if C3 >= 0.52 and H3 >= 0.45 and can_raise:
        amount, bucket = _compute_river_bet_size(ctx, C3, H3, "value")
        return RAISE, amount, bucket
    if C3 >= 0.38 and H3 >= 0.35 and can_raise:
        amount, bucket = _compute_river_bet_size(ctx, C3, H3, "value")
        return RAISE, amount, bucket

    # Bluff: low confidence but recon says they fold; line can rep strength
    fold_river = 0.5
    if ctx.recon is not None:
        try:
            from submission.opponent_recon import get_fold_to_river_bet
            fold_river = get_fold_to_river_bet(ctx.recon)
        except ImportError:
            pass
    if C3 < 0.35 and fold_river >= 0.55 and ctx.turn_we_bet and can_raise:
        if random.random() < 0.18:
            amount, bucket = _compute_river_bet_size(ctx, C3, H3, "bluff")
            return RAISE, amount, bucket

    if can_check:
        return CHECK, 0, ""
    return CHECK, 0, ""


def _river_facing_bet(
    ctx: Street3Context,
    r3: float,
    breakdown: Street3ScoreBreakdown,
    can_call: bool,
    can_raise: bool,
    can_fold: bool,
) -> Tuple[int, int, str]:
    """
    Facing a bet: raise only with very high confidence; call with good pot odds or moderate confidence; else fold.
    Returns (action_type, raise_amount, size_bucket).
    """
    C3, H3 = breakdown.C3, breakdown.H3
    pot = max(1, ctx.pot_size)
    to_call = ctx.amount_to_call
    pot_odds = to_call / (pot + to_call + 1) if (pot + to_call) > 0 else 0.5
    opp_loose = (ctx.opp_type or "").strip().lower() == "loose"
    r3_margin = 0.15 if opp_loose else 0.08
    c3_floor_loose = 0.50  # require higher confidence to call vs loose (reduce showdown losses when classified loose)
    c3_baseline = 0.42     # baseline C3 to call (slightly higher to cut marginal showdown calls)
    c3_danger = 0.48       # C3 required on paired/completed boards

    # Fold: huge river raise with marginal strength (reduce showdown losses from calling big raises)
    if can_fold and to_call > 40 and (C3 < 0.50 or H3 < 0.55):
        if r3 < pot_odds + 0.12:
            return FOLD, 0, ""
    if can_fold and pot > 0 and to_call > 0.55 * (pot + to_call) and C3 < 0.55 and r3 < pot_odds + 0.10:
        return FOLD, 0, ""

    # Raise: only with very high confidence (nut/near-nut)
    if C3 >= 0.78 and H3 >= 0.70 and can_raise:
        amount, bucket = _compute_river_bet_size(ctx, C3, H3, "value")
        return RAISE, amount, bucket

    # Call: good pot odds or moderate confidence (stricter when opp is "loose")
    if r3 > pot_odds + r3_margin and can_call:
        if opp_loose and C3 < c3_floor_loose:
            pass  # don't call with marginal confidence vs loose
        else:
            return CALL, 0, ""
    # Board texture: require higher C3 on paired or completed boards (reduce showdown losses)
    texture, paired, _ = river_texture_info(ctx.board5)
    tex_t = (texture or "").lower()
    board_danger = paired or "flush_completed" in tex_t or "straight_completed" in tex_t
    c3_call_thresh = max(c3_floor_loose if opp_loose else 0.0, c3_danger if board_danger else c3_baseline)
    if C3 >= c3_call_thresh and pot_odds < 0.35 and can_call:
        return CALL, 0, ""
    # Small river calls: higher bar to avoid -2 chips and marginal showdown losses
    if to_call <= 5 and (C3 >= 0.32 or r3 > 0.38) and can_call:
        if opp_loose and C3 < c3_floor_loose:
            pass
        else:
            return CALL, 0, ""

    if can_fold:
        return FOLD, 0, ""
    if can_call:
        return CALL, 0, ""
    return FOLD, 0, ""


# =========================================================================
# Sizing (plan §7): value by confidence; bluff one size; jitter ±0.02; confidence damp
# =========================================================================

def _river_sample_size(recon: Any) -> int:
    """Total river data count for damp check."""
    if recon is None:
        return 0
    return (
        getattr(recon, "opp_river_bets_faced", 0)
        + getattr(recon, "river_we_checked_opp_bet_count", 0)
        + getattr(recon, "river_we_checked_opp_check_count", 0)
    )


def _compute_river_bet_size(
    ctx: Street3Context,
    confidence: float,
    h3: float,
    mode: str,
) -> Tuple[int, str]:
    """
    Value: very high confidence → large (0.6–0.75 pot), high → medium (0.45–0.6), medium → small (0.3–0.45).
    Bluff: one size (small or medium). Jitter ±0.02. Confidence damp when river sample < 5.
    """
    pot = max(1, ctx.pot_size)
    if mode == "bluff":
        frac = 0.35 + random.uniform(-RIVER_JITTER, RIVER_JITTER)
        bucket = "small" if random.random() < 0.6 else "medium"
    else:
        if confidence >= 0.75 and h3 >= 0.65:
            frac = 0.67 + random.uniform(-RIVER_JITTER, RIVER_JITTER)
            bucket = "large"
        elif confidence >= 0.55 and h3 >= 0.45:
            frac = 0.52 + random.uniform(-RIVER_JITTER, RIVER_JITTER)
            bucket = "medium"
        else:
            frac = 0.38 + random.uniform(-RIVER_JITTER, RIVER_JITTER)
            bucket = "small"

    # Confidence damp when river sample is small
    sample = _river_sample_size(ctx.recon)
    if sample < RIVER_SAMPLE_DAMP_THRESHOLD:
        frac = frac * 0.6 + 0.50 * 0.4  # pull toward medium (0.5)

    frac = max(0.0, min(1.0, frac))
    base_amount = int(pot * frac)
    amount = max(ctx.min_raise, min(base_amount, ctx.max_raise))

    if frac >= 0.62:
        bucket = "large"
    elif frac >= 0.48:
        bucket = "medium"
    else:
        bucket = "small"
    return amount, bucket


# =========================================================================
# Mixing (plan §8): only when close; low temperature
# =========================================================================

def apply_river_mixing(
    action_type: int,
    ctx: Street3Context,
    r3: float,
    breakdown: Street3ScoreBreakdown,
) -> int:
    """Mix only when bet vs check is close; otherwise deterministic."""
    if breakdown.C3 >= 0.72 or breakdown.C3 <= 0.22:
        return action_type
    can_check = _can_check(ctx)
    if action_type == RAISE and r3 < 0.48 + MIXING_THRESHOLD and can_check:
        if random.random() < 0.15:
            return CHECK
    return action_type


# =========================================================================
# Top-level API
# =========================================================================

def get_street3_action(ctx: Street3Context) -> Tuple[int, int, str, Street3ScoreBreakdown]:
    """
    Main entry point for Street 3 (river) decisions.
    If len(board5) != 5, falls back to check/call/fold based on valid_actions.
    Returns: (action_type, raise_amount, size_bucket, breakdown).
    """
    if len(ctx.board5) < 5 or len(ctx.our_keep2) < 2:
        # Fallback: no full board
        if _can_check(ctx):
            return CHECK, 0, "", Street3ScoreBreakdown(R3=0.5)
        if ctx.amount_to_call > 0 and _can_call(ctx):
            return CALL, 0, "", Street3ScoreBreakdown(R3=0.5)
        if _can_fold(ctx):
            return FOLD, 0, "", Street3ScoreBreakdown(R3=0.5)
        return CHECK, 0, "", Street3ScoreBreakdown(R3=0.5)

    r3, breakdown = compute_r3_breakdown(ctx)
    is_opening = ctx.amount_to_call <= 0
    can_check = _can_check(ctx)
    can_call = _can_call(ctx)
    can_raise = _can_raise(ctx)
    can_fold = _can_fold(ctx)

    if is_opening:
        action_type, raise_amount, size_bucket = _river_opening(ctx, r3, breakdown, can_check, can_raise)
    else:
        action_type, raise_amount, size_bucket = _river_facing_bet(ctx, r3, breakdown, can_call, can_raise, can_fold)

    action_type = apply_river_mixing(action_type, ctx, r3, breakdown)

    if action_type != RAISE:
        raise_amount = 0
        size_bucket = ""

    return action_type, raise_amount, size_bucket, breakdown
