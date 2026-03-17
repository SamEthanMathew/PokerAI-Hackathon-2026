"""
Opponent Recon
==============
Central place for collecting and querying opponent data across the match.
Use throughout the game (all streets). Feed from act() and observe();
consume from street0_bet_sizing, street0_score, and strategy logic.

Data is updated by calling the update_* and record_* functions from your agent.
Getters return smoothed rates for use in decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Default when no data (avoid 0/0)
_DEFAULT_RATE = 0.5
_AF_CAP = 5.0


@dataclass
class OpponentRecon:
    """
    Raw counts and per-hand state for opponent modeling.
    Create one per match; call update_* and record_* from your agent each step.
    """

    # --- Aggression (lifetime) ---
    opp_raise_count: int = 0
    opp_call_count: int = 0

    # --- Fold to our bet (lifetime) ---
    opp_non_river_bets_faced: int = 0
    opp_non_river_folds: int = 0
    opp_river_bets_faced: int = 0
    opp_river_folds: int = 0

    # --- VPIP / PFR (hand-count denominator = total_hands) ---
    total_hands: int = 0
    opp_vpip_count: int = 0
    opp_pfr_count: int = 0

    # --- Non-river betting (flop/turn) ---
    opp_non_river_streets_seen: int = 0
    opp_non_river_bet_count: int = 0

    # --- Per-hand aggression (reset each hand) ---
    preflop_aggressor: bool = False
    flop_aggressor: bool = False
    turn_aggressor: bool = False

    # --- Per-hand state (reset each hand) ---
    last_hand_number: int = -1
    we_bet_this_street: bool = False
    last_street: int = -1
    _counted_flop_seen_this_hand: bool = False
    _counted_turn_seen_this_hand: bool = False
    _counted_flop_bet_this_hand: bool = False
    _counted_turn_bet_this_hand: bool = False
    _counted_vpip_this_hand: bool = False
    _counted_pfr_this_hand: bool = False
    _counted_river_bet_this_hand: bool = False
    _counted_flop_bet_faced_this_hand: bool = False
    _counted_turn_bet_faced_this_hand: bool = False

    # --- Optional: preflop response by our sizing (for future use) ---
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

    # --- Optional: board texture / discard reaction (for street0_score OpponentProfile) ---
    paired_board_faced: int = 0
    paired_board_fold: int = 0
    suited_board_faced: int = 0
    suited_board_fold: int = 0
    react_pair_keep_faced: int = 0
    react_pair_keep_fold: int = 0
    react_pair_keep_raise: int = 0
    react_flush_keep_faced: int = 0
    react_flush_keep_fold: int = 0
    react_flush_keep_raise: int = 0
    react_ambiguous_faced: int = 0
    react_ambiguous_fold: int = 0
    react_ambiguous_raise: int = 0


# ---------------------------------------------------------------------------
# Update API (call from your agent each step / on new hand / on our bet)
# ---------------------------------------------------------------------------

def start_new_hand(recon: OpponentRecon, hand_number: int) -> None:
    """Call at start of each hand (when hand_number changes)."""
    recon.last_hand_number = hand_number
    recon.total_hands = hand_number
    recon.preflop_aggressor = False
    recon.flop_aggressor = False
    recon.turn_aggressor = False
    recon.we_bet_this_street = False
    recon._counted_flop_seen_this_hand = False
    recon._counted_turn_seen_this_hand = False
    recon._counted_flop_bet_this_hand = False
    recon._counted_turn_bet_this_hand = False
    recon._counted_vpip_this_hand = False
    recon._counted_pfr_this_hand = False
    recon._counted_river_bet_this_hand = False
    recon._counted_flop_bet_faced_this_hand = False
    recon._counted_turn_bet_faced_this_hand = False


def start_new_street(recon: OpponentRecon, street: int) -> None:
    """Call when street changes (so we_bet_this_street resets)."""
    recon.last_street = street
    recon.we_bet_this_street = False


def record_our_bet(recon: OpponentRecon, street: int) -> None:
    """Call when we bet or raise (so opponent is counted as facing a bet this street)."""
    recon.we_bet_this_street = True
    if street == 3:
        if not recon._counted_river_bet_this_hand:
            recon._counted_river_bet_this_hand = True
            recon.opp_river_bets_faced += 1
    elif street == 1:
        if not recon._counted_flop_bet_faced_this_hand:
            recon._counted_flop_bet_faced_this_hand = True
            recon.opp_non_river_bets_faced += 1
    elif street == 2:
        if not recon._counted_turn_bet_faced_this_hand:
            recon._counted_turn_bet_faced_this_hand = True
            recon.opp_non_river_bets_faced += 1


def update_opponent_actions(recon: OpponentRecon, opp_last_action: Optional[str]) -> None:
    """Call every step with observation['opp_last_action'] (for raise/call counts and aggression)."""
    if not opp_last_action:
        return
    al = opp_last_action.lower()
    if "raise" in al:
        recon.opp_raise_count += 1
    elif "call" in al:
        recon.opp_call_count += 1


def update_vpip_pfr(recon: OpponentRecon, observation: dict, street: int) -> None:
    """Call every step on street 0 and 1/2 with full observation (VPIP/PFR and non-river bet)."""
    opp_last_action = observation.get("opp_last_action") or ""
    if street == 0 and opp_last_action:
        al = opp_last_action.lower()
        if "raise" in al:
            if not recon._counted_pfr_this_hand:
                recon._counted_pfr_this_hand = True
                recon.opp_vpip_count += 1
                recon.opp_pfr_count += 1
        elif "call" in al:
            my_blind_position = observation.get("blind_position", -1)
            opp_is_bb = my_blind_position == 0
            if not recon._counted_vpip_this_hand:
                if not opp_is_bb or observation.get("my_bet", 0) > observation.get("opp_bet", 0):
                    recon._counted_vpip_this_hand = True
                    recon.opp_vpip_count += 1
    if street in (1, 2):
        if not (recon._counted_flop_seen_this_hand if street == 1 else recon._counted_turn_seen_this_hand):
            if street == 1:
                recon._counted_flop_seen_this_hand = True
            else:
                recon._counted_turn_seen_this_hand = True
            recon.opp_non_river_streets_seen += 1
        if opp_last_action and ("bet" in opp_last_action.lower() or "raise" in opp_last_action.lower()):
            if not (recon._counted_flop_bet_this_hand if street == 1 else recon._counted_turn_bet_this_hand):
                if street == 1:
                    recon._counted_flop_bet_this_hand = True
                else:
                    recon._counted_turn_bet_this_hand = True
                recon.opp_non_river_bet_count += 1


def update_fold_on_terminate(
    recon: OpponentRecon,
    street: int,
    opp_last_action: Optional[str],
    terminated: bool,
) -> None:
    """Call when hand terminates; if opponent folded to our bet, update fold counts."""
    if not terminated or not opp_last_action or "fold" not in opp_last_action.lower():
        return
    if not recon.we_bet_this_street:
        return
    if street == 3:
        recon.opp_river_folds += 1
    elif street in (1, 2):
        recon.opp_non_river_folds += 1


def update_aggression_flags(recon: OpponentRecon, observation: dict, street: int) -> None:
    """Call each step to set preflop_aggressor / flop_aggressor / turn_aggressor."""
    opp_last_action = observation.get("opp_last_action") or ""
    if not opp_last_action or ("raise" not in opp_last_action.lower() and "bet" not in opp_last_action.lower()):
        return
    if street == 0:
        recon.preflop_aggressor = True
    elif street == 1:
        recon.flop_aggressor = True
    elif street == 2:
        recon.turn_aggressor = True


# ---------------------------------------------------------------------------
# Getters (use from street0_bet_sizing, street0_score, and strategy)
# ---------------------------------------------------------------------------

def get_vpip(recon: OpponentRecon) -> float:
    """Opponent VPIP (0â€“1). Default 0.5 when no data."""
    if recon.total_hands == 0:
        return _DEFAULT_RATE
    return recon.opp_vpip_count / recon.total_hands


def get_pfr(recon: OpponentRecon) -> float:
    """Opponent PFR (0â€“1). Default 0.5 when no data."""
    if recon.total_hands == 0:
        return _DEFAULT_RATE
    return recon.opp_pfr_count / recon.total_hands


def get_aggression_factor(recon: OpponentRecon) -> float:
    """Raises / calls. Capped at _AF_CAP. Returns 0 if no calls, 1.0 if no raises."""
    if recon.opp_call_count == 0:
        return _AF_CAP if recon.opp_raise_count > 0 else 1.0
    return min(_AF_CAP, recon.opp_raise_count / recon.opp_call_count)


def get_fold_to_non_river_bet(recon: OpponentRecon) -> float:
    """Fold rate when facing our bet on flop/turn. Default 0.5 when no data."""
    if recon.opp_non_river_bets_faced == 0:
        return _DEFAULT_RATE
    return recon.opp_non_river_folds / recon.opp_non_river_bets_faced


def get_fold_to_river_bet(recon: OpponentRecon) -> float:
    """Fold rate when facing our bet on river. Default 0.5 when no data."""
    if recon.opp_river_bets_faced == 0:
        return _DEFAULT_RATE
    return recon.opp_river_folds / recon.opp_river_bets_faced


def get_non_river_bet_percentage(recon: OpponentRecon) -> float:
    """How often they bet/raise on flop or turn when they see those streets. Default 0.5."""
    if recon.opp_non_river_streets_seen == 0:
        return _DEFAULT_RATE
    return recon.opp_non_river_bet_count / recon.opp_non_river_streets_seen


def get_opponent_type(recon: OpponentRecon) -> str:
    """'tight' if fold_to_non_river > 70%, 'loose' if < 30%, else 'balanced'."""
    f = get_fold_to_non_river_bet(recon)
    if f > 0.70:
        return "tight"
    if f < 0.30:
        return "loose"
    return "balanced"


def get_aggression_street_count(recon: OpponentRecon) -> int:
    """Number of streets (preflop, flop, turn) opponent was aggressor this hand."""
    return sum([recon.preflop_aggressor, recon.flop_aggressor, recon.turn_aggressor])


def stats_for_street0_context(recon: OpponentRecon) -> dict:
    """
    Return a dict of opponent stats for use with street0_bet_sizing.Street0Context.
    Keys: vpip, pfr, fold_to_non_river_bet, fold_to_river_bet, non_river_bet_pct, aggression_factor.
    """
    return {
        "vpip": get_vpip(recon),
        "pfr": get_pfr(recon),
        "fold_to_non_river_bet": get_fold_to_non_river_bet(recon),
        "fold_to_river_bet": get_fold_to_river_bet(recon),
        "non_river_bet_pct": get_non_river_bet_percentage(recon),
        "aggression_factor": get_aggression_factor(recon),
    }


def to_opponent_profile(recon: OpponentRecon):  # -> OpponentProfile
    """
    Build street0_score.OpponentProfile from OpponentRecon so you can pass it to final_street0_score.
    Import OpponentProfile from street0_score inside this function to avoid circular imports.
    """
    from submission.functions.street0_score import OpponentProfile
    return OpponentProfile(
        vpip_opportunities=recon.total_hands,
        vpip_successes=recon.opp_vpip_count,
        pfr_opportunities=recon.total_hands,
        pfr_successes=recon.opp_pfr_count,
        raise_count=recon.opp_raise_count,
        call_count=recon.opp_call_count,
        non_river_bet_opportunities=recon.opp_non_river_streets_seen,
        non_river_bet_successes=recon.opp_non_river_bet_count,
        fold_non_river_opportunities=recon.opp_non_river_bets_faced,
        fold_non_river_successes=recon.opp_non_river_folds,
        fold_river_opportunities=recon.opp_river_bets_faced,
        fold_river_successes=recon.opp_river_folds,
        paired_board_faced=recon.paired_board_faced,
        paired_board_fold=recon.paired_board_fold,
        suited_board_faced=recon.suited_board_faced,
        suited_board_fold=recon.suited_board_fold,
        react_pair_keep_faced=recon.react_pair_keep_faced,
        react_pair_keep_fold=recon.react_pair_keep_fold,
        react_pair_keep_raise=recon.react_pair_keep_raise,
        react_flush_keep_faced=recon.react_flush_keep_faced,
        react_flush_keep_fold=recon.react_flush_keep_fold,
        react_flush_keep_raise=recon.react_flush_keep_raise,
        react_ambiguous_faced=recon.react_ambiguous_faced,
        react_ambiguous_fold=recon.react_ambiguous_fold,
        react_ambiguous_raise=recon.react_ambiguous_raise,
        preflop_limp_faced=recon.preflop_limp_faced,
        preflop_limp_fold=recon.preflop_limp_fold,
        preflop_limp_call=recon.preflop_limp_call,
        preflop_limp_raise=recon.preflop_limp_raise,
        preflop_small_open_faced=recon.preflop_small_open_faced,
        preflop_small_open_fold=recon.preflop_small_open_fold,
        preflop_small_open_call=recon.preflop_small_open_call,
        preflop_small_open_raise=recon.preflop_small_open_raise,
        preflop_medium_open_faced=recon.preflop_medium_open_faced,
        preflop_medium_open_fold=recon.preflop_medium_open_fold,
        preflop_medium_open_call=recon.preflop_medium_open_call,
        preflop_medium_open_raise=recon.preflop_medium_open_raise,
        preflop_large_open_faced=recon.preflop_large_open_faced,
        preflop_large_open_fold=recon.preflop_large_open_fold,
        preflop_large_open_call=recon.preflop_large_open_call,
        preflop_large_open_raise=recon.preflop_large_open_raise,
        total_hands=recon.total_hands,
    )
