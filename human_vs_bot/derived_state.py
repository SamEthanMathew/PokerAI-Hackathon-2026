"""
Compute derived_state for logging and RL: hand_category, flush_outs, straight_outs,
cards_known, opp_discard_bucket, board_texture, position, pot_odds, effective_stack.
Uses submission.player helpers for consistency with game rules.
"""
from typing import Any

# Import pure helpers from submission.player (game rules are shared across bots)
try:
    from submission.player import (
        _hand_rank_category,
        _count_flush_outs,
        _count_straight_outs,
        _bucket_opp_discard,
        _bucket_flop_simple,
    )
except ImportError:
    import submission.player as _player_mod
    _hand_rank_category = getattr(_player_mod, "_hand_rank_category", lambda a, b: "unknown")
    _count_flush_outs = getattr(_player_mod, "_count_flush_outs", lambda a, b, c, d: (0, 0, 27))
    _count_straight_outs = getattr(_player_mod, "_count_straight_outs", lambda a, b, c, d: (0, 0, 27))
    _bucket_opp_discard = getattr(_player_mod, "_bucket_opp_discard", lambda x: "unknown")
    _bucket_flop_simple = getattr(_player_mod, "_bucket_flop_simple", lambda x: "medium")


def _to_list(cards: Any) -> list:
    """Normalize cards to list of ints, filtering -1."""
    if hasattr(cards, "__iter__") and not isinstance(cards, dict):
        return [int(c) for c in cards if c != -1]
    return []


def cards_known_for_player(
    my_cards: list,
    community: list,
    opp_discards: list,
    blind_position: int,
) -> tuple[int, list[int]]:
    """
    SB (blind_position 0) sees opp_discards; BB (blind_position 1) does not.
    Returns (count, list of known card indices).
    """
    known = list(_to_list(my_cards)) + _to_list(community)
    if blind_position == 0:  # we are SB -> we see opp discards
        known.extend(_to_list(opp_discards))
    uniq = list(dict.fromkeys(known))
    return len(uniq), uniq


def suit_summary(cards: list, env_int_to_str) -> str:
    """e.g. '5 diamonds seen, 4 live' for one suit."""
    if not cards:
        return "0 cards seen"
    suits = [c // 9 for c in cards if 0 <= c < 27]  # 9 ranks per suit
    from collections import Counter
    sc = Counter(suits)
    suit_names = ["diamonds", "hearts", "spades"]
    parts = [f"{cnt} {suit_names[s]}" for s, cnt in sc.most_common()]
    return ", ".join(parts) + " seen"


def compute_derived_state(
    obs: dict,
    env,
    my_cards_2: list | None = None,
) -> dict[str, Any]:
    """
    Compute derived_state for the acting player.
    obs: observation dict (my_cards, community_cards, opp_discarded_cards, my_discarded_cards, my_bet, opp_bet, etc.)
    env: PokerEnv (for int_card_to_str if needed).
    my_cards_2: if provided, use these 2 cards as "kept" hand (for post-discard); else use obs["my_cards"] first 2 if available.
    """
    my_cards = _to_list(obs.get("my_cards", []))
    community = _to_list(obs.get("community_cards", []))
    opp_discards = _to_list(obs.get("opp_discarded_cards", []))
    my_discards = _to_list(obs.get("my_discarded_cards", []))
    blind_pos = int(obs.get("blind_position", 0))
    my_bet = int(obs.get("my_bet", 0))
    opp_bet = int(obs.get("opp_bet", 0))
    pot_size = int(obs.get("pot_size", my_bet + opp_bet))

    cards_known_count, cards_known_list = cards_known_for_player(
        my_cards, community, opp_discards, blind_pos
    )
    position = "sb" if blind_pos == 0 else "bb"
    to_call = max(0, opp_bet - my_bet)
    pot_odds = to_call / (pot_size + to_call) if (pot_size + to_call) > 0 else 0.0
    max_bet = getattr(env, "MAX_PLAYER_BET", 100)
    stack_0 = max_bet - my_bet if obs.get("acting_agent") == 0 else max_bet - opp_bet
    stack_1 = max_bet - opp_bet if obs.get("acting_agent") == 0 else max_bet - my_bet
    effective_stack = min(stack_0, stack_1)

    hand_category = "nothing"
    flush_outs = 0
    straight_outs = 0
    opp_discard_bucket = _bucket_opp_discard(opp_discards) if opp_discards else "unknown"
    board_texture = _bucket_flop_simple(community) if len(community) >= 3 else "any"

    use_cards = my_cards_2 if my_cards_2 is not None and len(my_cards_2) >= 2 else (my_cards[:2] if len(my_cards) >= 2 else [])
    if len(use_cards) >= 2 and len(community) >= 3:
        hand_category = _hand_rank_category(use_cards, community)
        fc, fl, _ = _count_flush_outs(use_cards, community, opp_discards, my_discards)
        flush_outs = fl
        si, so, _ = _count_straight_outs(use_cards, community, opp_discards, my_discards)
        straight_outs = so

    return {
        "hand_category": hand_category,
        "flush_outs": flush_outs,
        "straight_outs": straight_outs,
        "cards_known": cards_known_count,
        "cards_known_list": cards_known_list,
        "opp_discard_bucket": opp_discard_bucket,
        "board_texture": board_texture,
        "position": position,
        "pot_odds": round(pot_odds, 4),
        "effective_stack": effective_stack,
    }


def analysis_for_discard_options(
    my_cards_5: list,
    community: list,
    opp_discards: list,
    my_discards: list,
    env,
) -> list[dict]:
    """
    For each possible 2-card keep from my_cards_5, return analysis (hand_category, flush_outs, straight_outs, label).
    my_discards can be empty; we use toss = the 3 we're not keeping.
    """
    from itertools import combinations
    results = []
    for i, j in combinations(range(min(5, len(my_cards_5))), 2):
        keep = [my_cards_5[i], my_cards_5[j]]
        toss = [my_cards_5[k] for k in range(len(my_cards_5)) if k not in (i, j)]
        toss_set = set(toss)
        if len(community) < 3:
            results.append({"keep_indices": (i, j), "hand_category": "nothing", "flush_outs": 0, "straight_outs": 0, "label": "pre-flop"})
            continue
        cat = _hand_rank_category(keep, community)
        fc, fl, _ = _count_flush_outs(keep, community, opp_discards, list(toss_set))
        si, so, _ = _count_straight_outs(keep, community, opp_discards, list(toss_set))
        label = f"Keep {i}-{j} -> {cat}, flush_outs={fl}, straight_outs={so}"
        results.append({"keep_indices": (i, j), "hand_category": cat, "flush_outs": fl, "straight_outs": so, "label": label})
    return results
