"""
CLI display and input for human_vs_bot: two prompt modes (discard vs betting),
known card count, analysis block, opponent discard inference, post-hand review.
"""
import time
from typing import Any

# ActionType: FOLD=0, RAISE=1, CHECK=2, CALL=3, DISCARD=4
FOLD, RAISE, CHECK, CALL, DISCARD = 0, 1, 2, 3, 4
ACTION_NAMES = ["FOLD", "RAISE", "CHECK", "CALL", "DISCARD"]

DISCARD_REASON_TAGS = "flush_draw / straight_draw / trips / top_pair / blocker / draw_dead / other"
BETTING_REASON_TAGS = "value / bluff / pot_odds / fold_equity / read_weak / read_strong / other"
OPP_KEEP_OPTIONS = "pair / flush_draw / straight_draw / high_cards / unknown"


def _fmt_cards(env, cards: list) -> list[str]:
    return [env.int_card_to_str(int(c)) for c in cards if c != -1]


def display_discard_prompt(
    env,
    hand_number: int,
    my_cards_5: list,
    board: list,
    opp_discards: list,
    blind_position: int,
    cards_known_count: int,
    cards_known_str: str,
    analysis_lines: list[str],
) -> None:
    """Display discard screen: board, your 5 cards, opp discards (if SB), known count, analysis."""
    pos_str = "SB (discard 2nd)" if blind_position == 0 else "BB (discard 1st)"
    print("\n" + "=" * 50)
    print(f"  Hand #{hand_number} — You are {pos_str}")
    print("=" * 50)
    print(f"Board:  {' '.join(_fmt_cards(env, board))}")
    print(f"Your hand:  {' '.join(f'[{env.int_card_to_str(int(c))}]' for c in my_cards_5)}")
    if opp_discards:
        print(f"Opp discarded:  {' '.join(_fmt_cards(env, opp_discards))}")
    else:
        print("Opp discarded:  (not yet)")
    print(f"\nYou know {cards_known_count}/27 cards. {cards_known_str}")
    if analysis_lines:
        print("\nAnalysis:")
        for line in analysis_lines:
            print("  ", line)
    print("\nKeep which two? (indices 0-4, e.g. 0 1)")


def display_betting_prompt(
    env,
    hand_number: int,
    street_name: str,
    my_cards: list,
    board: list,
    pot: int,
    my_bet: int,
    opp_bet: int,
    valid_actions: list,
    min_raise: int,
    max_raise: int,
    cards_known_count: int,
    cards_known_str: str,
) -> None:
    """Display betting screen: street, cards, pot, bets, valid actions."""
    print("\n" + "=" * 50)
    print(f"  Hand #{hand_number} — {street_name}")
    print("=" * 50)
    print(f"Board:  {' '.join(_fmt_cards(env, board))}")
    print(f"Your hand:  {' '.join(_fmt_cards(env, my_cards))}")
    print(f"Pot: {pot}  |  Your bet: {my_bet}  |  Opp bet: {opp_bet}")
    print(f"You know {cards_known_count}/27 cards. {cards_known_str}")
    valid_names = [ACTION_NAMES[i] for i in range(5) if valid_actions[i]]
    print(f"Valid: {', '.join(valid_names)}")
    if valid_actions[RAISE]:
        print(f"  RAISE between {min_raise} and {max_raise}")
    print("\nAction? (f=fold, c=call, k=check, r=raise [amount], d i j=discard keep i j)")


def parse_action(
    raw: str,
    valid_actions: list,
    min_raise: int,
    max_raise: int,
    is_discard_turn: bool,
) -> tuple[int, int, int, int] | None:
    """
    Parse one line into (action_type, amount, keep1, keep2) or None if invalid.
    For DISCARD, keep1 and keep2 are indices 0-4.
    """
    raw = raw.strip().lower()
    if not raw:
        return None
    parts = raw.split()
    if is_discard_turn and len(parts) >= 2:
        try:
            i, j = int(parts[0]), int(parts[1])
            if 0 <= i <= 4 and 0 <= j <= 4 and i != j and valid_actions[DISCARD]:
                return (DISCARD, 0, i, j)
        except ValueError:
            pass
    if len(parts) >= 1:
        c = parts[0][0]
        if c == "f" and valid_actions[FOLD]:
            return (FOLD, 0, 0, 0)
        if c == "c" and valid_actions[CALL]:
            return (CALL, 0, 0, 0)
        if c == "k" and valid_actions[CHECK]:
            return (CHECK, 0, 0, 0)
        if c == "r" and valid_actions[RAISE]:
            amt = min_raise
            if len(parts) >= 2:
                try:
                    amt = int(parts[1])
                except ValueError:
                    pass
            amt = max(min_raise, min(max_raise, amt))
            return (RAISE, amt, 0, 0)
        if c == "d" and len(parts) >= 3 and valid_actions[DISCARD]:
            try:
                i, j = int(parts[1]), int(parts[2])
                if 0 <= i <= 4 and 0 <= j <= 4 and i != j:
                    return (DISCARD, 0, i, j)
            except ValueError:
                pass
    return None


def prompt_discard_action(
    env,
    hand_number: int,
    my_cards_5: list,
    board: list,
    opp_discards: list,
    blind_position: int,
    cards_known_count: int,
    cards_known_str: str,
    analysis_lines: list[str],
    valid_actions: list,
) -> tuple[tuple[int, int, int, int], float, str, str]:
    """
    Show discard prompt, loop until valid input. Returns (action_tuple, decision_time_sec, human_discard_reason, opp_keep_inference).
    """
    display_discard_prompt(
        env, hand_number, my_cards_5, board, opp_discards,
        blind_position, cards_known_count, cards_known_str, analysis_lines,
    )
    t0 = time.time()
    human_discard_reason = ""
    opp_keep_inference = ""
    while True:
        raw = input("> ").strip()
        action = parse_action(raw, valid_actions, 0, 0, is_discard_turn=True)
        if action is not None:
            decision_time = time.time() - t0
            if opp_discards:
                inf = input("What do you think they kept? (pair/flush_draw/straight_draw/high_cards/unknown) [Enter to skip]: ").strip().lower()
                if inf:
                    opp_keep_inference = inf
            reason = input(f"Why this keep? ({DISCARD_REASON_TAGS}) [Enter to skip]: ").strip().lower()
            if reason:
                human_discard_reason = reason
            return action, decision_time, human_discard_reason, opp_keep_inference
        print("Invalid. Keep two indices 0-4 (e.g. 0 1).")


def prompt_betting_action(
    env,
    hand_number: int,
    street_name: str,
    my_cards: list,
    board: list,
    pot: int,
    my_bet: int,
    opp_bet: int,
    valid_actions: list,
    min_raise: int,
    max_raise: int,
    cards_known_count: int,
    cards_known_str: str,
) -> tuple[tuple[int, int, int, int], float, str, str, str]:
    """
    Show betting prompt, loop until valid input. Returns (action_tuple, decision_time_sec, human_reason, human_confidence, human_read).
    """
    display_betting_prompt(
        env, hand_number, street_name, my_cards, board,
        pot, my_bet, opp_bet, valid_actions, min_raise, max_raise,
        cards_known_count, cards_known_str,
    )
    t0 = time.time()
    human_reason = ""
    human_confidence = ""
    human_read = ""
    while True:
        raw = input("> ").strip()
        action = parse_action(raw, valid_actions, min_raise, max_raise, is_discard_turn=False)
        if action is not None:
            decision_time = time.time() - t0
            reason = input(f"Reason? ({BETTING_REASON_TAGS}) [Enter to skip]: ").strip().lower()
            if reason:
                human_reason = reason
            conf = input("Confidence (1-5)? [Enter to skip]: ").strip()
            if conf and conf in "12345":
                human_confidence = conf
            read = input("Opponent read? (weak/strong/neutral/unknown) [Enter to skip]: ").strip().lower()
            if read:
                human_read = read
            return action, decision_time, human_reason, human_confidence, human_read
        print("Invalid action. Use f/c/k/r [amount] or d i j for discard.")


def display_hand_result(
    env,
    hand_number: int,
    reward_0: int,
    reward_1: int,
    player_0_cards: list,
    player_1_cards: list,
    community_cards: list,
    human_is_0: bool,
) -> None:
    """Show hand over with both hands and board."""
    print("\n" + "-" * 50)
    print(f"Hand #{hand_number} over.")
    print(f"Board: {' '.join(_fmt_cards(env, community_cards))}")
    print(f"Your cards:  {' '.join(_fmt_cards(env, player_0_cards if human_is_0 else player_1_cards))}")
    print(f"Bot cards:   {' '.join(_fmt_cards(env, player_1_cards if human_is_0 else player_0_cards))}")
    r_human = reward_0 if human_is_0 else reward_1
    print(f"Result: {'You' if human_is_0 else 'Bot'} {'win' if r_human > 0 else 'lose'} {abs(r_human)}")
    print("-" * 50)


def prompt_post_hand_review() -> tuple[bool, str, str]:
    """
    Ask: Would you change any decision? (y/N). If yes, which street and what would you do differently?
    Returns (changed, street, note).
    """
    raw = input("Would you change any decision? (y/N): ").strip().lower()
    if raw != "y" and raw != "yes":
        return False, "", ""
    street = input("Which street? (Pre-Flop/Flop/Turn/River): ").strip()
    note = input("What would you do differently? ").strip()
    return True, street, note
