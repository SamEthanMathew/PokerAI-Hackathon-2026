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

    # ===================================================================
    # STREET 1 – Per-hand state (reset in start_new_hand)
    # ===================================================================
    our_flop_discard_class: str = ""
    our_flop_size_bucket: str = ""
    flop_texture_this_hand: str = ""
    opp_flop_discard_class: str = ""
    _opp_discard_classified_this_hand: bool = False

    # ===================================================================
    # STREET 1 – Lifetime: reaction to our flop bet by our discard class
    # ===================================================================
    react_flush_transparent_flop_faced: int = 0
    react_flush_transparent_flop_fold: int = 0
    react_flush_transparent_flop_call: int = 0
    react_flush_transparent_flop_raise: int = 0

    react_straight_transparent_flop_faced: int = 0
    react_straight_transparent_flop_fold: int = 0
    react_straight_transparent_flop_call: int = 0
    react_straight_transparent_flop_raise: int = 0

    react_pair_transparent_flop_faced: int = 0
    react_pair_transparent_flop_fold: int = 0
    react_pair_transparent_flop_call: int = 0
    react_pair_transparent_flop_raise: int = 0

    react_weak_transparent_flop_faced: int = 0
    react_weak_transparent_flop_fold: int = 0
    react_weak_transparent_flop_call: int = 0
    react_weak_transparent_flop_raise: int = 0

    react_ambiguous_flop_faced: int = 0
    react_ambiguous_flop_fold: int = 0
    react_ambiguous_flop_call: int = 0
    react_ambiguous_flop_raise: int = 0

    react_capped_flop_faced: int = 0
    react_capped_flop_fold: int = 0
    react_capped_flop_call: int = 0
    react_capped_flop_raise: int = 0

    # ===================================================================
    # STREET 1 – Lifetime: reaction by board texture on flop
    # ===================================================================
    flop_tex_paired_faced: int = 0
    flop_tex_paired_fold: int = 0
    flop_tex_paired_raise: int = 0

    flop_tex_suited_faced: int = 0
    flop_tex_suited_fold: int = 0
    flop_tex_suited_raise: int = 0

    flop_tex_connected_faced: int = 0
    flop_tex_connected_fold: int = 0
    flop_tex_connected_raise: int = 0

    flop_tex_disconnected_faced: int = 0
    flop_tex_disconnected_fold: int = 0
    flop_tex_disconnected_raise: int = 0

    # ===================================================================
    # STREET 1 – Lifetime: reaction to our flop bet by size bucket
    # ===================================================================
    flop_bet_small_faced: int = 0
    flop_bet_small_fold: int = 0
    flop_bet_small_call: int = 0
    flop_bet_small_raise: int = 0

    flop_bet_medium_faced: int = 0
    flop_bet_medium_fold: int = 0
    flop_bet_medium_call: int = 0
    flop_bet_medium_raise: int = 0

    flop_bet_large_faced: int = 0
    flop_bet_large_fold: int = 0
    flop_bet_large_call: int = 0
    flop_bet_large_raise: int = 0

    # ===================================================================
    # STREET 1 – Lifetime: opponent flop aggression by their discard class
    # ===================================================================
    opp_flush_discard_bet_count: int = 0
    opp_flush_discard_check_count: int = 0
    opp_straight_discard_bet_count: int = 0
    opp_straight_discard_check_count: int = 0
    opp_pair_discard_bet_count: int = 0
    opp_pair_discard_check_count: int = 0
    opp_weak_discard_bet_count: int = 0
    opp_weak_discard_check_count: int = 0
    opp_ambiguous_discard_bet_count: int = 0
    opp_ambiguous_discard_check_count: int = 0

    # ===================================================================
    # STREET 2 – Per-hand state (reset in start_new_hand)
    # ===================================================================
    our_turn_size_bucket: str = ""
    turn_texture_this_hand: str = ""
    _flop_opp_action: str = ""  # "check" | "bet" | "call" | "raise" – set during Street 1B

    # ===================================================================
    # STREET 2 – Lifetime: turn response by size bucket
    # ===================================================================
    turn_bet_small_faced: int = 0
    turn_bet_small_fold: int = 0
    turn_bet_small_call: int = 0
    turn_bet_small_raise: int = 0
    turn_bet_medium_faced: int = 0
    turn_bet_medium_fold: int = 0
    turn_bet_medium_call: int = 0
    turn_bet_medium_raise: int = 0
    turn_bet_large_faced: int = 0
    turn_bet_large_fold: int = 0
    turn_bet_large_call: int = 0
    turn_bet_large_raise: int = 0

    # ===================================================================
    # STREET 2 – Lifetime: turn response by our discard class
    # ===================================================================
    turn_react_flush_transparent_faced: int = 0
    turn_react_flush_transparent_fold: int = 0
    turn_react_flush_transparent_call: int = 0
    turn_react_flush_transparent_raise: int = 0
    turn_react_straight_transparent_faced: int = 0
    turn_react_straight_transparent_fold: int = 0
    turn_react_straight_transparent_call: int = 0
    turn_react_straight_transparent_raise: int = 0
    turn_react_pair_transparent_faced: int = 0
    turn_react_pair_transparent_fold: int = 0
    turn_react_pair_transparent_call: int = 0
    turn_react_pair_transparent_raise: int = 0
    turn_react_weak_transparent_faced: int = 0
    turn_react_weak_transparent_fold: int = 0
    turn_react_weak_transparent_call: int = 0
    turn_react_weak_transparent_raise: int = 0
    turn_react_ambiguous_faced: int = 0
    turn_react_ambiguous_fold: int = 0
    turn_react_ambiguous_call: int = 0
    turn_react_ambiguous_raise: int = 0
    turn_react_capped_faced: int = 0
    turn_react_capped_fold: int = 0
    turn_react_capped_call: int = 0
    turn_react_capped_raise: int = 0

    # ===================================================================
    # STREET 2 – Lifetime: turn response by turn texture
    # ===================================================================
    turn_tex_paired_faced: int = 0
    turn_tex_paired_fold: int = 0
    turn_tex_paired_raise: int = 0
    turn_tex_flush_faced: int = 0
    turn_tex_flush_fold: int = 0
    turn_tex_flush_raise: int = 0
    turn_tex_straight_faced: int = 0
    turn_tex_straight_fold: int = 0
    turn_tex_straight_raise: int = 0
    turn_tex_blank_faced: int = 0
    turn_tex_blank_fold: int = 0
    turn_tex_blank_raise: int = 0

    # ===================================================================
    # STREET 2 – Lifetime: turn aggression after flop line
    # ===================================================================
    turn_after_flop_call_fold: int = 0
    turn_after_flop_call_bet: int = 0
    turn_after_flop_bet_called_fold: int = 0
    turn_after_flop_bet_called_bet: int = 0
    turn_after_flop_check_fold: int = 0
    turn_after_flop_check_bet: int = 0

    # ===================================================================
    # STREET 3 – Per-hand state (reset in start_new_hand)
    # ===================================================================
    our_river_size_bucket: str = ""
    river_texture_this_hand: str = ""

    # ===================================================================
    # STREET 3 – Lifetime: opponent response to our river bet by size bucket
    # ===================================================================
    river_bet_small_faced: int = 0
    river_bet_small_fold: int = 0
    river_bet_small_call: int = 0
    river_bet_small_raise: int = 0
    river_bet_medium_faced: int = 0
    river_bet_medium_fold: int = 0
    river_bet_medium_call: int = 0
    river_bet_medium_raise: int = 0
    river_bet_large_faced: int = 0
    river_bet_large_fold: int = 0
    river_bet_large_call: int = 0
    river_bet_large_raise: int = 0

    # ===================================================================
    # STREET 3 – Lifetime: opponent response by our discard class
    # ===================================================================
    river_react_flush_transparent_faced: int = 0
    river_react_flush_transparent_fold: int = 0
    river_react_flush_transparent_call: int = 0
    river_react_flush_transparent_raise: int = 0
    river_react_straight_transparent_faced: int = 0
    river_react_straight_transparent_fold: int = 0
    river_react_straight_transparent_call: int = 0
    river_react_straight_transparent_raise: int = 0
    river_react_pair_transparent_faced: int = 0
    river_react_pair_transparent_fold: int = 0
    river_react_pair_transparent_call: int = 0
    river_react_pair_transparent_raise: int = 0
    river_react_weak_transparent_faced: int = 0
    river_react_weak_transparent_fold: int = 0
    river_react_weak_transparent_call: int = 0
    river_react_weak_transparent_raise: int = 0
    river_react_ambiguous_faced: int = 0
    river_react_ambiguous_fold: int = 0
    river_react_ambiguous_call: int = 0
    river_react_ambiguous_raise: int = 0
    river_react_capped_faced: int = 0
    river_react_capped_fold: int = 0
    river_react_capped_call: int = 0
    river_react_capped_raise: int = 0

    # ===================================================================
    # STREET 3 – Lifetime: opponent response by river texture
    # ===================================================================
    river_tex_flush_river_faced: int = 0
    river_tex_flush_river_fold: int = 0
    river_tex_flush_river_raise: int = 0
    river_tex_straight_river_faced: int = 0
    river_tex_straight_river_fold: int = 0
    river_tex_straight_river_raise: int = 0
    river_tex_standard_river_faced: int = 0
    river_tex_standard_river_fold: int = 0
    river_tex_standard_river_raise: int = 0

    # ===================================================================
    # STREET 3 – Lifetime: opponent aggression when we checked river
    # ===================================================================
    river_we_checked_opp_bet_count: int = 0
    river_we_checked_opp_check_count: int = 0


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
    # Street 1 per-hand state
    recon.our_flop_discard_class = ""
    recon.our_flop_size_bucket = ""
    recon.flop_texture_this_hand = ""
    recon.opp_flop_discard_class = ""
    recon._opp_discard_classified_this_hand = False
    recon.our_turn_size_bucket = ""
    recon.turn_texture_this_hand = ""
    recon._flop_opp_action = ""
    # Street 3 per-hand state
    recon.our_river_size_bucket = ""
    recon.river_texture_this_hand = ""


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
# Street 1 Update API
# ---------------------------------------------------------------------------

_DISCARD_CLASS_LABELS = (
    "flush_transparent", "straight_transparent", "pair_transparent",
    "weak_transparent", "ambiguous", "capped",
)

_TEXTURE_BUCKETS = ("paired", "suited", "connected", "disconnected")
_SIZE_BUCKETS = ("small", "medium", "large")

_BETTING_ACTIONS = {"raise", "call", "fold", "check"}


def _is_betting_action(opp_last_action: Optional[str]) -> bool:
    if not opp_last_action:
        return False
    return opp_last_action.lower() in _BETTING_ACTIONS


def record_our_flop_discard_class(recon: OpponentRecon, our_discard_class: str) -> None:
    recon.our_flop_discard_class = our_discard_class


def record_our_flop_bet(recon: OpponentRecon, size_bucket: str) -> None:
    """Only store on the FIRST bet this hand; re-raises keep original bucket."""
    if not recon.our_flop_size_bucket:
        recon.our_flop_size_bucket = size_bucket


def record_flop_texture(recon: OpponentRecon, texture: str) -> None:
    recon.flop_texture_this_hand = texture


def classify_opponent_flop_discard(
    recon: OpponentRecon,
    opp_discard_cards: list,
    flop: list,
    texture: str,
) -> str:
    """Classify opponent's discard into a label. Runs once per hand (guarded)."""
    if recon._opp_discard_classified_this_hand:
        return recon.opp_flop_discard_class

    recon._opp_discard_classified_this_hand = True

    from submission.street0_score import (
        rank, suit, hand_suit_counts, hand_rank_counts, NUM_SUITS,
    )

    disc = [c for c in opp_discard_cards if c != -1]
    if len(disc) != 3:
        recon.opp_flop_discard_class = "ambiguous"
        return "ambiguous"

    disc_suits = hand_suit_counts(disc)
    disc_ranks = hand_rank_counts(disc)
    flop_suits = hand_suit_counts(flop)
    flop_ranks = set(rank(c) for c in flop)

    max_disc_suit = max(disc_suits.values()) if disc_suits else 0

    # Flush transparent: all 3 discards same suit (abandoned flush draw)
    if max_disc_suit == 3:
        recon.opp_flop_discard_class = "flush_transparent"
        return "flush_transparent"

    # Pair transparent: discarded a pair (2 of same rank)
    if any(cnt >= 2 for cnt in disc_ranks.values()):
        recon.opp_flop_discard_class = "pair_transparent"
        return "pair_transparent"

    # Straight transparent: 3 consecutive or near-consecutive discards
    disc_rank_set = set(disc_ranks.keys())
    sorted_dr = sorted(disc_rank_set)
    if len(sorted_dr) == 3 and sorted_dr[-1] - sorted_dr[0] <= 3:
        recon.opp_flop_discard_class = "straight_transparent"
        return "straight_transparent"

    # Weak transparent: all low cards
    if all(rank(c) < 5 for c in disc):
        recon.opp_flop_discard_class = "weak_transparent"
        return "weak_transparent"

    # Capped: high cards discarded suggest capped range
    if all(rank(c) >= 6 for c in disc):
        recon.opp_flop_discard_class = "capped"
        return "capped"

    recon.opp_flop_discard_class = "ambiguous"
    return "ambiguous"


def _texture_bucket(texture: str) -> str:
    """Map a full texture label to one of our 4 recon buckets."""
    t = texture.lower()
    if "paired" in t:
        return "paired"
    if "suited" in t or "mono" in t:
        return "suited"
    if "connected" in t:
        return "connected"
    return "disconnected"


def update_opponent_flop_response(recon: OpponentRecon, opp_last_action: Optional[str]) -> None:
    """Increment faced/fold/call/raise for current per-hand attribution state.
    Only call when street==1, valid_actions[DISCARD]==0, and opp_last_action is a betting action."""
    if not _is_betting_action(opp_last_action):
        return

    al = opp_last_action.lower()

    # --- By our discard class ---
    dc = recon.our_flop_discard_class
    if dc:
        _inc_discard_class_reaction(recon, dc, al)

    # --- By texture ---
    tex = _texture_bucket(recon.flop_texture_this_hand) if recon.flop_texture_this_hand else ""
    if tex:
        _inc_texture_reaction(recon, tex, al)

    # --- By our size bucket ---
    sb = recon.our_flop_size_bucket
    if sb:
        _inc_size_bucket_reaction(recon, sb, al)


def _inc_discard_class_reaction(recon: OpponentRecon, dc: str, al: str) -> None:
    prefix = f"react_{dc}_flop"
    faced_attr = f"{prefix}_faced"
    if hasattr(recon, faced_attr):
        setattr(recon, faced_attr, getattr(recon, faced_attr) + 1)
        if "fold" in al:
            setattr(recon, f"{prefix}_fold", getattr(recon, f"{prefix}_fold") + 1)
        elif "raise" in al:
            setattr(recon, f"{prefix}_raise", getattr(recon, f"{prefix}_raise") + 1)
        elif "call" in al:
            setattr(recon, f"{prefix}_call", getattr(recon, f"{prefix}_call") + 1)


def _inc_texture_reaction(recon: OpponentRecon, tex: str, al: str) -> None:
    faced_attr = f"flop_tex_{tex}_faced"
    if hasattr(recon, faced_attr):
        setattr(recon, faced_attr, getattr(recon, faced_attr) + 1)
        if "fold" in al:
            setattr(recon, f"flop_tex_{tex}_fold", getattr(recon, f"flop_tex_{tex}_fold") + 1)
        elif "raise" in al:
            setattr(recon, f"flop_tex_{tex}_raise", getattr(recon, f"flop_tex_{tex}_raise") + 1)


def _inc_size_bucket_reaction(recon: OpponentRecon, sb: str, al: str) -> None:
    faced_attr = f"flop_bet_{sb}_faced"
    if hasattr(recon, faced_attr):
        setattr(recon, faced_attr, getattr(recon, faced_attr) + 1)
        if "fold" in al:
            setattr(recon, f"flop_bet_{sb}_fold", getattr(recon, f"flop_bet_{sb}_fold") + 1)
        elif "raise" in al:
            setattr(recon, f"flop_bet_{sb}_raise", getattr(recon, f"flop_bet_{sb}_raise") + 1)
        elif "call" in al:
            setattr(recon, f"flop_bet_{sb}_call", getattr(recon, f"flop_bet_{sb}_call") + 1)


def update_opponent_flop_aggression(recon: OpponentRecon, opp_last_action: Optional[str]) -> None:
    """Track opponent betting vs checking on flop, keyed by their inferred discard class."""
    if not _is_betting_action(opp_last_action):
        return

    al = opp_last_action.lower()
    odc = recon.opp_flop_discard_class
    if not odc:
        return

    is_bet = "raise" in al
    is_check = "check" in al

    mapping = {
        "flush_transparent": ("opp_flush_discard_bet_count", "opp_flush_discard_check_count"),
        "straight_transparent": ("opp_straight_discard_bet_count", "opp_straight_discard_check_count"),
        "pair_transparent": ("opp_pair_discard_bet_count", "opp_pair_discard_check_count"),
        "weak_transparent": ("opp_weak_discard_bet_count", "opp_weak_discard_check_count"),
        "ambiguous": ("opp_ambiguous_discard_bet_count", "opp_ambiguous_discard_check_count"),
        "capped": ("opp_ambiguous_discard_bet_count", "opp_ambiguous_discard_check_count"),
    }

    attrs = mapping.get(odc)
    if not attrs:
        return
    bet_attr, check_attr = attrs
    if is_bet:
        setattr(recon, bet_attr, getattr(recon, bet_attr) + 1)
    elif is_check:
        setattr(recon, check_attr, getattr(recon, check_attr) + 1)


# ---------------------------------------------------------------------------
# Street 2 Update API
# ---------------------------------------------------------------------------

def _turn_texture_bucket(texture: str) -> str:
    """Map turn texture string to recon bucket: paired, flush, straight, blank."""
    if not texture:
        return "blank"
    t = texture.lower()
    if "pair" in t or "paired" in t:
        return "paired"
    if "flush" in t:
        return "flush"
    if "straight" in t or "connect" in t:
        return "straight"
    return "blank"


def _turn_size_bucket_norm(sb: str) -> str:
    """Normalize size bucket to small/medium/large for turn attributes."""
    if sb in _SIZE_BUCKETS:
        return sb
    t = (sb or "").lower()
    if "small" in t:
        return "small"
    if "large" in t:
        return "large"
    return "medium"


def record_our_turn_bet(recon: OpponentRecon, size_bucket: str) -> None:
    """Set our_turn_size_bucket only if currently empty (first turn bet this hand)."""
    if not recon.our_turn_size_bucket and size_bucket:
        recon.our_turn_size_bucket = _turn_size_bucket_norm(size_bucket)


def record_turn_texture(recon: OpponentRecon, texture: str) -> None:
    """Set turn_texture_this_hand for this hand."""
    recon.turn_texture_this_hand = texture or ""


def set_flop_opp_action(recon: OpponentRecon, opp_action: str) -> None:
    """Set opponent's flop action for turn aggression attribution. Call from genesis Street 1B."""
    al = (opp_action or "").lower()
    if "raise" in al:
        recon._flop_opp_action = "raise"
    elif "call" in al:
        recon._flop_opp_action = "call"
    elif "bet" in al:
        recon._flop_opp_action = "bet"
    elif "check" in al:
        recon._flop_opp_action = "check"
    else:
        recon._flop_opp_action = ""


def _inc_turn_size_bucket_reaction(recon: OpponentRecon, sb: str, al: str) -> None:
    sb = _turn_size_bucket_norm(sb)
    faced_attr = f"turn_bet_{sb}_faced"
    if not hasattr(recon, faced_attr):
        return
    setattr(recon, faced_attr, getattr(recon, faced_attr) + 1)
    if "fold" in al:
        setattr(recon, f"turn_bet_{sb}_fold", getattr(recon, f"turn_bet_{sb}_fold") + 1)
    elif "raise" in al:
        setattr(recon, f"turn_bet_{sb}_raise", getattr(recon, f"turn_bet_{sb}_raise") + 1)
    elif "call" in al:
        setattr(recon, f"turn_bet_{sb}_call", getattr(recon, f"turn_bet_{sb}_call") + 1)


def _inc_turn_discard_class_reaction(recon: OpponentRecon, dc: str, al: str) -> None:
    if not dc or dc not in _DISCARD_CLASS_LABELS:
        return
    prefix = f"turn_react_{dc}"
    faced_attr = f"{prefix}_faced"
    if not hasattr(recon, faced_attr):
        return
    setattr(recon, faced_attr, getattr(recon, faced_attr) + 1)
    if "fold" in al:
        setattr(recon, f"{prefix}_fold", getattr(recon, f"{prefix}_fold") + 1)
    elif "raise" in al:
        setattr(recon, f"{prefix}_raise", getattr(recon, f"{prefix}_raise") + 1)
    elif "call" in al:
        setattr(recon, f"{prefix}_call", getattr(recon, f"{prefix}_call") + 1)


def _inc_turn_texture_reaction(recon: OpponentRecon, tex: str, al: str) -> None:
    faced_attr = f"turn_tex_{tex}_faced"
    if not hasattr(recon, faced_attr):
        return
    setattr(recon, faced_attr, getattr(recon, faced_attr) + 1)
    if "fold" in al:
        setattr(recon, f"turn_tex_{tex}_fold", getattr(recon, f"turn_tex_{tex}_fold") + 1)
    elif "raise" in al:
        setattr(recon, f"turn_tex_{tex}_raise", getattr(recon, f"turn_tex_{tex}_raise") + 1)


def _inc_turn_after_flop_line(recon: OpponentRecon, al: str) -> None:
    """Increment turn_after_flop_* based on _flop_opp_action (call/bet/check)."""
    flop = recon._flop_opp_action or ""
    is_fold = "fold" in al
    is_bet = "raise" in al or "bet" in al
    if flop == "call":
        # Villain called our flop bet; now fold vs our turn bet or bet/raise
        if is_fold:
            recon.turn_after_flop_call_fold += 1
        elif is_bet:
            recon.turn_after_flop_call_bet += 1
    elif flop == "check":
        if is_fold:
            recon.turn_after_flop_check_fold += 1
        elif is_bet:
            recon.turn_after_flop_check_bet += 1
    elif flop == "bet":
        # Villain bet flop, we called; now they fold vs our barrel or bet/raise
        if is_fold:
            recon.turn_after_flop_bet_called_fold += 1
        elif is_bet:
            recon.turn_after_flop_bet_called_bet += 1


def update_opponent_turn_response(
    recon: OpponentRecon, opp_last_action: Optional[str], our_discard_class: str = ""
) -> None:
    """
    When street == 2 and opp_last_action is a betting action, increment faced/fold/call/raise
    for current our_turn_size_bucket, our_discard_class, turn_texture_this_hand, and turn
    aggression after flop line. Call from genesis with our_discard_class from _street1_discard_class.
    """
    if not _is_betting_action(opp_last_action):
        return
    al = opp_last_action.lower()

    sb = recon.our_turn_size_bucket
    if sb:
        _inc_turn_size_bucket_reaction(recon, sb, al)

    if our_discard_class:
        _inc_turn_discard_class_reaction(recon, our_discard_class, al)

    tex = _turn_texture_bucket(recon.turn_texture_this_hand)
    if tex:
        _inc_turn_texture_reaction(recon, tex, al)

    _inc_turn_after_flop_line(recon, al)


# ---------------------------------------------------------------------------
# Street 3 (River) Update API
# ---------------------------------------------------------------------------

def _river_texture_bucket(texture: str) -> str:
    """Map classify_board_texture output to recon bucket: flush_river, straight_river, standard_river."""
    if not texture:
        return "standard_river"
    t = texture.lower()
    if "flush_completed" in t or "flush_river" in t:
        return "flush_river"
    if "straight_completed" in t or "straight_river" in t:
        return "straight_river"
    return "standard_river"


def _river_size_bucket_norm(sb: str) -> str:
    """Normalize size bucket to small/medium/large for river attributes."""
    if sb in _SIZE_BUCKETS:
        return sb
    t = (sb or "").lower()
    if "small" in t:
        return "small"
    if "large" in t:
        return "large"
    return "medium"


def record_our_river_bet(recon: OpponentRecon, size_bucket: str) -> None:
    """Set our_river_size_bucket only if currently empty (first river bet this hand)."""
    if not recon.our_river_size_bucket and size_bucket:
        recon.our_river_size_bucket = _river_size_bucket_norm(size_bucket)


def record_river_texture(recon: OpponentRecon, texture: str) -> None:
    """Set river_texture_this_hand for this hand. Call once per hand when building river context."""
    recon.river_texture_this_hand = texture or ""


def _inc_river_size_bucket_reaction(recon: OpponentRecon, sb: str, al: str) -> None:
    sb = _river_size_bucket_norm(sb)
    faced_attr = f"river_bet_{sb}_faced"
    if not hasattr(recon, faced_attr):
        return
    setattr(recon, faced_attr, getattr(recon, faced_attr) + 1)
    if "fold" in al:
        setattr(recon, f"river_bet_{sb}_fold", getattr(recon, f"river_bet_{sb}_fold") + 1)
    elif "raise" in al:
        setattr(recon, f"river_bet_{sb}_raise", getattr(recon, f"river_bet_{sb}_raise") + 1)
    elif "call" in al:
        setattr(recon, f"river_bet_{sb}_call", getattr(recon, f"river_bet_{sb}_call") + 1)


def _inc_river_discard_class_reaction(recon: OpponentRecon, dc: str, al: str) -> None:
    if not dc or dc not in _DISCARD_CLASS_LABELS:
        return
    prefix = f"river_react_{dc}"
    faced_attr = f"{prefix}_faced"
    if not hasattr(recon, faced_attr):
        return
    setattr(recon, faced_attr, getattr(recon, faced_attr) + 1)
    if "fold" in al:
        setattr(recon, f"{prefix}_fold", getattr(recon, f"{prefix}_fold") + 1)
    elif "raise" in al:
        setattr(recon, f"{prefix}_raise", getattr(recon, f"{prefix}_raise") + 1)
    elif "call" in al:
        setattr(recon, f"{prefix}_call", getattr(recon, f"{prefix}_call") + 1)


def _inc_river_texture_reaction(recon: OpponentRecon, tex: str, al: str) -> None:
    faced_attr = f"river_tex_{tex}_faced"
    if not hasattr(recon, faced_attr):
        return
    setattr(recon, faced_attr, getattr(recon, faced_attr) + 1)
    if "fold" in al:
        setattr(recon, f"river_tex_{tex}_fold", getattr(recon, f"river_tex_{tex}_fold") + 1)
    elif "raise" in al:
        setattr(recon, f"river_tex_{tex}_raise", getattr(recon, f"river_tex_{tex}_raise") + 1)


def update_opponent_river_response(
    recon: OpponentRecon, opp_last_action: Optional[str], our_discard_class: str = ""
) -> None:
    """
    Only when we had bet river (our_river_size_bucket non-empty) and opp_last_action is
    a betting action (fold/call/raise), increment faced/fold/call/raise for current
    our_river_size_bucket, our_discard_class, and river_texture_this_hand.
    Call from genesis in act() and observe() when street == 3 and we see their response.
    """
    if not recon.our_river_size_bucket:
        return
    if not _is_betting_action(opp_last_action):
        return
    al = opp_last_action.lower()

    _inc_river_size_bucket_reaction(recon, recon.our_river_size_bucket, al)
    if our_discard_class:
        _inc_river_discard_class_reaction(recon, our_discard_class, al)
    tex = _river_texture_bucket(recon.river_texture_this_hand)
    if tex:
        _inc_river_texture_reaction(recon, tex, al)


def update_opponent_river_aggression(
    recon: OpponentRecon, we_checked: bool, opp_last_action: Optional[str]
) -> None:
    """
    When we had checked river and we see their action: if they bet, increment
    river_we_checked_opp_bet_count; if they check, increment river_we_checked_opp_check_count.
    Call from genesis only when _river_we_checked is True and we see opp_last_action.
    """
    if not we_checked or not opp_last_action:
        return
    al = opp_last_action.lower()
    if "raise" in al or "bet" in al:
        recon.river_we_checked_opp_bet_count += 1
    elif "check" in al:
        recon.river_we_checked_opp_check_count += 1


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


# ---------------------------------------------------------------------------
# Street 1 Getters (smoothed, gradient-ready)
# ---------------------------------------------------------------------------

def _smoothed_rate(successes: int, n: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    return (successes + alpha) / (n + alpha + beta)


def get_flop_fold_vs_our_discard_class(recon: OpponentRecon, our_discard_class: str) -> float:
    faced_attr = f"react_{our_discard_class}_flop_faced"
    fold_attr = f"react_{our_discard_class}_flop_fold"
    if not hasattr(recon, faced_attr):
        return 0.5
    return _smoothed_rate(getattr(recon, fold_attr), getattr(recon, faced_attr))


def get_flop_raise_vs_our_discard_class(recon: OpponentRecon, our_discard_class: str) -> float:
    faced_attr = f"react_{our_discard_class}_flop_faced"
    raise_attr = f"react_{our_discard_class}_flop_raise"
    if not hasattr(recon, faced_attr):
        return 0.5
    return _smoothed_rate(getattr(recon, raise_attr), getattr(recon, faced_attr))


def get_flop_fold_by_texture(recon: OpponentRecon, texture: str) -> float:
    tex = _texture_bucket(texture)
    faced_attr = f"flop_tex_{tex}_faced"
    fold_attr = f"flop_tex_{tex}_fold"
    if not hasattr(recon, faced_attr):
        return 0.5
    return _smoothed_rate(getattr(recon, fold_attr), getattr(recon, faced_attr))


def get_flop_fold_vs_size_bucket(recon: OpponentRecon, size_bucket: str) -> float:
    faced_attr = f"flop_bet_{size_bucket}_faced"
    fold_attr = f"flop_bet_{size_bucket}_fold"
    if not hasattr(recon, faced_attr):
        return 0.5
    return _smoothed_rate(getattr(recon, fold_attr), getattr(recon, faced_attr))


def get_opponent_flop_aggression_after_discard(
    recon: OpponentRecon, their_discard_class: str
) -> float:
    mapping = {
        "flush_transparent": ("opp_flush_discard_bet_count", "opp_flush_discard_check_count"),
        "straight_transparent": ("opp_straight_discard_bet_count", "opp_straight_discard_check_count"),
        "pair_transparent": ("opp_pair_discard_bet_count", "opp_pair_discard_check_count"),
        "weak_transparent": ("opp_weak_discard_bet_count", "opp_weak_discard_check_count"),
        "ambiguous": ("opp_ambiguous_discard_bet_count", "opp_ambiguous_discard_check_count"),
        "capped": ("opp_ambiguous_discard_bet_count", "opp_ambiguous_discard_check_count"),
    }
    attrs = mapping.get(their_discard_class)
    if not attrs:
        return 0.5
    bet_attr, check_attr = attrs
    bets = getattr(recon, bet_attr, 0)
    checks = getattr(recon, check_attr, 0)
    return _smoothed_rate(bets, bets + checks)


# ---------------------------------------------------------------------------
# Street 2 Getters (smoothed)
# ---------------------------------------------------------------------------

def get_turn_fold_vs_size_bucket(recon: OpponentRecon, size_bucket: str) -> float:
    faced_attr = f"turn_bet_{_turn_size_bucket_norm(size_bucket)}_faced"
    fold_attr = f"turn_bet_{_turn_size_bucket_norm(size_bucket)}_fold"
    if not hasattr(recon, faced_attr):
        return _DEFAULT_RATE
    return _smoothed_rate(getattr(recon, fold_attr), getattr(recon, faced_attr))


def get_turn_raise_vs_size_bucket(recon: OpponentRecon, size_bucket: str) -> float:
    sb = _turn_size_bucket_norm(size_bucket)
    faced_attr = f"turn_bet_{sb}_faced"
    raise_attr = f"turn_bet_{sb}_raise"
    if not hasattr(recon, faced_attr):
        return _DEFAULT_RATE
    return _smoothed_rate(getattr(recon, raise_attr), getattr(recon, faced_attr))


def get_turn_fold_vs_our_discard_class(recon: OpponentRecon, our_discard_class: str) -> float:
    faced_attr = f"turn_react_{our_discard_class}_faced"
    fold_attr = f"turn_react_{our_discard_class}_fold"
    if not hasattr(recon, faced_attr):
        return _DEFAULT_RATE
    return _smoothed_rate(getattr(recon, fold_attr), getattr(recon, faced_attr))


def get_turn_fold_by_turn_texture(recon: OpponentRecon, turn_texture: str) -> float:
    tex = _turn_texture_bucket(turn_texture)
    faced_attr = f"turn_tex_{tex}_faced"
    fold_attr = f"turn_tex_{tex}_fold"
    if not hasattr(recon, faced_attr):
        return _DEFAULT_RATE
    return _smoothed_rate(getattr(recon, fold_attr), getattr(recon, faced_attr))


def get_turn_aggression_after_flop_line(
    recon: OpponentRecon, flop_line_key: str
) -> float:
    """
    flop_line_key: e.g. "call_then_bet", "bet_called_then_bet", "check_then_bet".
    Returns rate at which opponent bets/raises on turn after that flop line (smoothed).
    """
    key = (flop_line_key or "").lower()
    if "call_then" in key or "call_then_bet" in key:
        n = recon.turn_after_flop_call_fold + recon.turn_after_flop_call_bet
        return _smoothed_rate(recon.turn_after_flop_call_bet, n) if n else _DEFAULT_RATE
    if "bet_called_then" in key or "bet_called" in key:
        n = recon.turn_after_flop_bet_called_fold + recon.turn_after_flop_bet_called_bet
        return _smoothed_rate(recon.turn_after_flop_bet_called_bet, n) if n else _DEFAULT_RATE
    if "check_then" in key or "check_then_bet" in key:
        n = recon.turn_after_flop_check_fold + recon.turn_after_flop_check_bet
        return _smoothed_rate(recon.turn_after_flop_check_bet, n) if n else _DEFAULT_RATE
    return _DEFAULT_RATE


# ---------------------------------------------------------------------------
# Street 3 (River) Getters (smoothed, alpha=1, beta=1)
# ---------------------------------------------------------------------------

def get_river_fold_vs_size_bucket(recon: OpponentRecon, size_bucket: str) -> float:
    sb = _river_size_bucket_norm(size_bucket)
    faced_attr = f"river_bet_{sb}_faced"
    fold_attr = f"river_bet_{sb}_fold"
    if not hasattr(recon, faced_attr):
        return _DEFAULT_RATE
    return _smoothed_rate(getattr(recon, fold_attr), getattr(recon, faced_attr))


def get_river_raise_vs_size_bucket(recon: OpponentRecon, size_bucket: str) -> float:
    sb = _river_size_bucket_norm(size_bucket)
    faced_attr = f"river_bet_{sb}_faced"
    raise_attr = f"river_bet_{sb}_raise"
    if not hasattr(recon, faced_attr):
        return _DEFAULT_RATE
    return _smoothed_rate(getattr(recon, raise_attr), getattr(recon, faced_attr))


def get_river_fold_vs_our_discard_class(recon: OpponentRecon, our_discard_class: str) -> float:
    faced_attr = f"river_react_{our_discard_class}_faced"
    fold_attr = f"river_react_{our_discard_class}_fold"
    if not hasattr(recon, faced_attr):
        return _DEFAULT_RATE
    return _smoothed_rate(getattr(recon, fold_attr), getattr(recon, faced_attr))


def get_river_fold_by_river_texture(recon: OpponentRecon, river_texture: str) -> float:
    tex = _river_texture_bucket(river_texture)
    faced_attr = f"river_tex_{tex}_faced"
    fold_attr = f"river_tex_{tex}_fold"
    if not hasattr(recon, faced_attr):
        return _DEFAULT_RATE
    return _smoothed_rate(getattr(recon, fold_attr), getattr(recon, faced_attr))


def get_river_bet_when_checked_to(recon: OpponentRecon) -> float:
    """Opponent bet rate when we checked river: opp_bet_count / (opp_bet_count + opp_check_count), smoothed."""
    n = recon.river_we_checked_opp_bet_count + recon.river_we_checked_opp_check_count
    if n == 0:
        return _DEFAULT_RATE
    return _smoothed_rate(recon.river_we_checked_opp_bet_count, n)


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
    from submission.street0_score import OpponentProfile
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
