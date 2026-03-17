"""
Street 0 Bet Sizing
===================
Returns preflop action and bet sizing from the Street 0 score and game context.

Use with: score from submission.street0_score.final_street0_score(),
          plus observation and opponent stats from your agent.

Returns: (action_type, raise_amount) for use in act() on street 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

# Action type constants (match gym_env.PokerEnv.ActionType)
FOLD = 0
RAISE = 1
CHECK = 2
CALL = 3
DISCARD = 4


# --- Score buckets (spec §29) ---
SCORE_PREMIUM = 0.85       # 0.85+ premium aggressive
SCORE_STRONG = 0.70        # 0.70–0.85 strong playable/aggressive
SCORE_MEDIUM = 0.55        # 0.55–0.70 medium playable
SCORE_MARGINAL = 0.40      # 0.40–0.55 marginal / context-dependent
# < 0.40 weak / mostly fold or low-investment


@dataclass
class Street0Context:
    """Context for street 0 decision."""
    score: float
    amount_to_call: int
    pot_size: int
    blind_position: int  # 0 = SB, 1 = BB
    valid_actions: List[int]  # [FOLD, RAISE, CHECK, CALL, DISCARD] 0/1
    max_raise: int
    min_raise: int
    # Opponent stats (0.0–1.0); pass None to use defaults
    vpip: Optional[float] = None
    pfr: Optional[float] = None
    fold_to_non_river_bet: Optional[float] = None
    fold_to_river_bet: Optional[float] = None
    non_river_bet_pct: Optional[float] = None
    aggression_factor: Optional[float] = None  # use a cap e.g. 5.0 for inf


def _get_bucket(score: float) -> str:
    if score >= SCORE_PREMIUM:
        return "premium"
    if score >= SCORE_STRONG:
        return "strong"
    if score >= SCORE_MEDIUM:
        return "medium"
    if score >= SCORE_MARGINAL:
        return "marginal"
    return "weak"


def _can_check(ctx: Street0Context) -> bool:
    return ctx.valid_actions[CHECK] == 1


def _can_call(ctx: Street0Context) -> bool:
    return ctx.valid_actions[CALL] == 1


def _can_raise(ctx: Street0Context) -> bool:
    return ctx.valid_actions[RAISE] == 1


def _choose_raise_amount(
    ctx: Street0Context,
    bucket: str,
    is_open: bool,  # we are opening (no call to make) vs facing a bet
) -> int:
    """Return raise size in chips, clamped to [min_raise, max_raise]."""
    vpip = ctx.vpip if ctx.vpip is not None else 0.5
    pfr = ctx.pfr if ctx.pfr is not None else 0.5

    # Base sizes by bucket (small = ~2–6, medium = ~6–15, large = 15+)
    if bucket == "premium":
        small, medium, large = 8, 20, min(50, ctx.max_raise)
    elif bucket == "strong":
        small, medium, large = 5, 12, 25
    elif bucket == "medium":
        small, medium, large = 3, 8, 15
    elif bucket == "marginal":
        small, medium, large = 2, 5, 10
    else:
        small, medium, large = 1, 3, 6

    # Against passive (low VPIP/PFR) we can open larger
    if is_open:
        if vpip < 0.40 and pfr < 0.30:
            amount = large
        elif vpip < 0.55:
            amount = medium
        else:
            amount = small
    else:
        # Facing a bet: raise size depends on our strength
        if bucket in ("premium", "strong"):
            amount = medium if ctx.amount_to_call <= 10 else small
        elif bucket == "medium":
            amount = small
        else:
            amount = max(ctx.min_raise, small)

    return max(ctx.min_raise, min(amount, ctx.max_raise))


def get_street0_action(ctx: Street0Context) -> Tuple[int, int]:
    """
    Compute preflop action and bet sizing from Street 0 score and context.

    Args:
        ctx: Street0Context with score, amounts, valid_actions, and optional opponent stats.

    Returns:
        (action_type, raise_amount) where:
        - action_type is FOLD, RAISE, CHECK, or CALL (DISCARD not used on street 0).
        - raise_amount is the chip amount for RAISE, else -1.

    Uses spec §29 buckets and opponent stats (VPIP, PFR, fold rates) to choose
    check/call/fold/raise and raise sizing.
    """
    action_type, raise_amount = _get_street0_action_impl(ctx)
    return _enforce_valid(action_type, raise_amount, ctx)


def _get_street0_action_impl(ctx: Street0Context) -> Tuple[int, int]:
    """Internal: raw decision without valid_actions enforcement."""
    bucket = _get_bucket(ctx.score)
    amount_to_call = ctx.amount_to_call
    is_open = amount_to_call <= 0
    pfr = ctx.pfr if ctx.pfr is not None else 0.5
    vpip = ctx.vpip if ctx.vpip is not None else 0.5

    # --- 1) If we can check, use score to decide whether to check or open ---
    if _can_check(ctx) and amount_to_call <= 0:
        if bucket in ("weak", "marginal"):
            return CHECK, -1
        if bucket == "medium" and vpip >= 0.5:
            return CHECK, -1
        if _can_raise(ctx) and bucket in ("premium", "strong", "medium"):
            amount = _choose_raise_amount(ctx, bucket, is_open=True)
            return RAISE, amount
        return CHECK, -1

    # --- 2) Facing a bet (amount_to_call > 0) ---
    if amount_to_call > 0:
        # High PFR opponent: don’t fold as easily (call more)
        if pfr > 0.80 and _can_call(ctx):
            if bucket != "weak":
                return CALL, -1
            if amount_to_call <= 10:
                return CALL, -1
        if pfr > 0.50 and _can_call(ctx) and amount_to_call <= 20:
            if bucket in ("premium", "strong", "medium"):
                return CALL, -1
            if bucket == "marginal" and amount_to_call <= 10:
                return CALL, -1

        # Premium: raise or call
        if bucket == "premium":
            if _can_raise(ctx):
                return RAISE, _choose_raise_amount(ctx, bucket, is_open=False)
            if _can_call(ctx):
                return CALL, -1
            return FOLD, -1

        # Strong: call most, raise small bets
        if bucket == "strong":
            if amount_to_call <= 15 and _can_raise(ctx):
                return RAISE, _choose_raise_amount(ctx, bucket, is_open=False)
            if _can_call(ctx):
                return CALL, -1
            return FOLD, -1

        # Medium: call up to ~25, fold larger
        if bucket == "medium":
            if _can_call(ctx) and amount_to_call <= 25:
                return CALL, -1
            if _can_check(ctx):
                return CHECK, -1
            return FOLD, -1

        # Marginal: call only small
        if bucket == "marginal":
            if amount_to_call <= 10 and _can_call(ctx):
                return CALL, -1
            if _can_check(ctx):
                return CHECK, -1
            return FOLD, -1

        # Weak: fold unless call is tiny
        if amount_to_call <= 5 and _can_call(ctx):
            return CALL, -1
        if _can_check(ctx):
            return CHECK, -1
        return FOLD, -1

    # --- 3) We can open (no bet to call, but check was not chosen above) ---
    if _can_raise(ctx) and bucket in ("premium", "strong"):
        return RAISE, _choose_raise_amount(ctx, bucket, is_open=True)
    if _can_check(ctx):
        return CHECK, -1
    if _can_call(ctx):
        return CALL, -1
    return FOLD, -1


def _enforce_valid(action_type: int, raise_amount: int, ctx: Street0Context) -> Tuple[int, int]:
    """If chosen action is invalid, fall back to a valid one."""
    if action_type == RAISE and not _can_raise(ctx):
        if _can_call(ctx):
            return CALL, -1
        if _can_check(ctx):
            return CHECK, -1
        return FOLD, -1
    if action_type == CALL and not _can_call(ctx):
        if _can_check(ctx):
            return CHECK, -1
        return FOLD, -1
    if action_type == CHECK and not _can_check(ctx):
        if _can_call(ctx):
            return CALL, -1
        return FOLD, -1
    return action_type, raise_amount


def get_street0_action_from_obs(
    score: float,
    observation: dict,
    vpip: Optional[float] = None,
    pfr: Optional[float] = None,
    fold_to_non_river_bet: Optional[float] = None,
    fold_to_river_bet: Optional[float] = None,
    non_river_bet_pct: Optional[float] = None,
    aggression_factor: Optional[float] = None,
) -> Tuple[int, int]:
    """
    Convenience: build Street0Context from observation and optional opponent stats,
    then return get_street0_action(ctx).

    observation must have: street, my_bet, opp_bet, valid_actions, blind_position, max_raise, min_raise.
    """
    my_bet = observation.get("my_bet", 0)
    opp_bet = observation.get("opp_bet", 0)
    pot_size = my_bet + opp_bet
    amount_to_call = opp_bet - my_bet
    valid_actions = list(observation.get("valid_actions", [1, 1, 1, 1, 0]))
    # Ensure we don't use DISCARD for street 0
    if len(valid_actions) > 4:
        valid_actions = valid_actions[:5]
    while len(valid_actions) < 5:
        valid_actions.append(0)

    ctx = Street0Context(
        score=score,
        amount_to_call=amount_to_call,
        pot_size=pot_size,
        blind_position=observation.get("blind_position", 0),
        valid_actions=valid_actions,
        max_raise=int(observation.get("max_raise", 100)),
        min_raise=int(observation.get("min_raise", 2)),
        vpip=vpip,
        pfr=pfr,
        fold_to_non_river_bet=fold_to_non_river_bet,
        fold_to_river_bet=fold_to_river_bet,
        non_river_bet_pct=non_river_bet_pct,
        aggression_factor=aggression_factor,
    )
    return get_street0_action(ctx)


def get_street0_action_from_recon(
    score: float,
    observation: dict,
    recon,  # OpponentRecon from opponent_recon
) -> Tuple[int, int]:
    """
    Convenience: use OpponentRecon to fill opponent stats for Street0Context.
    Import and pass submission.opponent_recon.OpponentRecon.
    """
    from submission.opponent_recon import stats_for_street0_context
    stats = stats_for_street0_context(recon)
    return get_street0_action_from_obs(
        score,
        observation,
        vpip=stats["vpip"],
        pfr=stats["pfr"],
        fold_to_non_river_bet=stats["fold_to_non_river_bet"],
        fold_to_river_bet=stats["fold_to_river_bet"],
        non_river_bet_pct=stats["non_river_bet_pct"],
        aggression_factor=stats["aggression_factor"],
    )
