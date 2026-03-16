"""
Entry point for human vs bot: bot discovery, game loop, full logging.
Run from project root: python -m human_vs_bot.run
"""
import importlib
import logging
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from gym_env import PokerEnv
from match import prepare_payload, get_street_name

from human_vs_bot.cli import (
    prompt_discard_action,
    prompt_betting_action,
    display_hand_result,
    prompt_post_hand_review,
    DISCARD,
)
from human_vs_bot.derived_state import (
    compute_derived_state,
    cards_known_for_player,
    analysis_for_discard_options,
)
from human_vs_bot.logger import SessionLogger

logging.basicConfig(level=logging.INFO, format="%(message)s")
HUMAN_PLAYER = 0
BOT_PLAYER = 1
TIME_LIMIT_SECONDS = 600


def discover_bots():
    submission_dir = os.path.join(_root, "submission")
    if not os.path.isdir(submission_dir):
        return []
    results = []
    try:
        from agents.agent import Agent
    except ImportError:
        Agent = object
    for name in sorted(os.listdir(submission_dir)):
        if not name.endswith(".py") or name.startswith("_"):
            continue
        if name == "libratus_tables.py":
            continue
        mod_name = name[:-3]
        try:
            mod = importlib.import_module(f"submission.{mod_name}")
            cls = getattr(mod, "PlayerAgent", None)
            if cls is not None and isinstance(cls, type) and issubclass(cls, Agent):
                results.append((mod_name, cls))
        except Exception:
            continue
    return results


def main():
    bots = discover_bots()
    if not bots:
        print("No bots found in submission/.")
        return
    print("Select a bot to play against:")
    for i, (name, _) in enumerate(bots, 1):
        print(f"  {i}. {name}")
    try:
        idx = int(input("Number: ").strip())
        if idx < 1 or idx > len(bots):
            raise ValueError("Invalid number")
        bot_module_name, BotClass = bots[idx - 1]
    except (ValueError, EOFError) as e:
        print("Invalid choice.", e)
        return

    num_hands_str = input("Number of hands (or 0 for until you quit): ").strip() or "10"
    try:
        num_hands = int(num_hands_str)
    except ValueError:
        num_hands = 10
    if num_hands <= 0:
        num_hands = 9999

    os.environ["MATCH_ID"] = "human_vs_bot"
    os.environ["PLAYER_ID"] = "bot"
    bot = BotClass(stream=False)
    env = PokerEnv(logger=logging.getLogger("PokerEnv"))
    session = SessionLogger()
    session.start_session(bot_name=bot_module_name)

    bankrolls = [0, 0]
    time_used_0 = 0.0
    time_used_1 = 0.0
    hand_number = 0

    while hand_number < num_hands:
        small_blind_player = hand_number % 2
        (obs0, obs1), info = env.reset(options={"small_blind_player": small_blind_player})
        info["hand_number"] = hand_number
        reward0 = reward1 = 0
        terminated = truncated = False
        obs0["time_used"] = time_used_0
        obs0["time_left"] = TIME_LIMIT_SECONDS - time_used_0
        obs1["time_used"] = time_used_1
        obs1["time_left"] = TIME_LIMIT_SECONDS - time_used_1
        obs0["opp_last_action"] = "None"
        obs1["opp_last_action"] = "None"

        last_move_0 = None
        last_move_1 = None
        step_in_hand = 0

        def fmt_cards(cards_list):
            return [env.int_card_to_str(int(c)) for c in cards_list if c != -1]

        while not terminated:
            current_player = obs0["acting_agent"]
            obs_human = obs0 if current_player == HUMAN_PLAYER else obs1
            obs_bot = obs1 if current_player == HUMAN_PLAYER else obs0
            payload = prepare_payload(
                obs_bot, reward1 if current_player == BOT_PLAYER else reward0, terminated, truncated, info
            )
            obs_for_bot = payload

            decision_time_sec = 0.0
            human_reason = human_confidence = human_read = ""
            human_discard_reason = opp_keep_inference = ""

            if current_player == HUMAN_PLAYER:
                my_cards = [c for c in obs_human["my_cards"] if c != -1]
                community = [c for c in obs_human["community_cards"] if c != -1]
                opp_discards = [c for c in obs_human["opp_discarded_cards"] if c != -1]
                blind_pos = int(obs_human.get("blind_position", 0))
                cards_known_count, cards_known_list = cards_known_for_player(
                    my_cards, community, opp_discards, blind_pos
                )
                from collections import Counter
                suits = [c // 9 for c in (my_cards + community + (opp_discards if blind_pos == 0 else [])) if 0 <= c < 27]
                sc = Counter(suits)
                suit_names = ["diamonds", "hearts", "spades"]
                cards_known_str = ", ".join(f"{cnt} {suit_names[s]} seen" for s, cnt in sc.most_common()) or "0 cards seen"

                valid_actions = list(obs_human.get("valid_actions", [1] * 5))
                if len(valid_actions) < 5:
                    valid_actions = valid_actions + [0] * (5 - len(valid_actions))

                is_discard_turn = valid_actions[DISCARD] if len(valid_actions) > DISCARD else False

                if is_discard_turn and len(my_cards) == 5 and len(community) >= 3:
                    analysis_list = analysis_for_discard_options(
                        my_cards, community, opp_discards, [], env
                    )
                    analysis_lines = [a["label"] for a in analysis_list]
                    action_tuple, decision_time_sec, human_discard_reason, opp_keep_inference = prompt_discard_action(
                        env, hand_number, my_cards, community, opp_discards,
                        blind_pos, cards_known_count, cards_known_str, analysis_lines, valid_actions,
                    )
                    time_used_0 += decision_time_sec
                    session.add_time_used(0, decision_time_sec)
                else:
                    street_name = get_street_name(obs_human.get("street", 0))
                    pot = int(obs_human.get("pot_size", 0))
                    my_bet = int(obs_human.get("my_bet", 0))
                    opp_bet = int(obs_human.get("opp_bet", 0))
                    min_raise = int(obs_human.get("min_raise", 2))
                    max_raise = int(obs_human.get("max_raise", 100))
                    action_tuple, decision_time_sec, human_reason, human_confidence, human_read = prompt_betting_action(
                        env, hand_number, street_name, my_cards, community,
                        pot, my_bet, opp_bet, valid_actions, min_raise, max_raise,
                        cards_known_count, cards_known_str,
                    )
                    time_used_0 += decision_time_sec
                    session.add_time_used(0, decision_time_sec)

                action = action_tuple
                last_move_0 = action[0]
                observer_payload = prepare_payload(obs1, reward1, terminated, truncated, info)
                bot.do_bot_observation(
                    observer_payload["observation"],
                    observer_payload["reward"],
                    observer_payload["terminated"],
                    observer_payload["truncated"],
                    observer_payload["info"],
                )
            else:
                import time as _time
                t0 = _time.time()
                action = bot.get_bot_action(
                    obs_for_bot["observation"],
                    obs_for_bot["reward"],
                    obs_for_bot["terminated"],
                    obs_for_bot["truncated"],
                    obs_for_bot["info"],
                )
                elapsed = _time.time() - t0
                time_used_1 += elapsed
                session.add_time_used(1, elapsed)
                if action is None:
                    action = (0, 0, 0, 0)
                if not isinstance(action, (list, tuple)) or len(action) < 4:
                    action = (int(action[0]) if action else 0, int(action[1]) if len(action) > 1 else 0, 0, 0)
                action = (int(action[0]), int(action[1]), int(action[2]), int(action[3]))
                last_move_1 = action[0]

            obs_before_step = obs0 if current_player == 0 else obs1
            derived = compute_derived_state(obs_before_step, env)
            cards_known_count_pre = derived["cards_known"]
            cards_known_list_pre = derived.get("cards_known_list", [])

            (obs0, obs1), (reward0, reward1), terminated, truncated, info = env.step(action)
            info["hand_number"] = hand_number
            step_in_hand += 1

            obs0["time_used"] = time_used_0
            obs1["time_used"] = time_used_1
            obs0["time_left"] = TIME_LIMIT_SECONDS - time_used_0
            obs1["time_left"] = TIME_LIMIT_SECONDS - time_used_1
            obs0["opp_last_action"] = "None" if last_move_0 is None else PokerEnv.ActionType(last_move_0).name
            obs1["opp_last_action"] = "None" if last_move_1 is None else PokerEnv.ActionType(last_move_1).name

            num_board = 0 if obs0["street"] == 0 else obs0["street"] + 2
            obs_for_log = obs_before_step
            cards_known_count = cards_known_count_pre
            cards_known_list = cards_known_list_pre

            session.log_decision(
                hand_number=hand_number,
                street=get_street_name(obs0["street"]),
                active_team=current_player,
                team_0_bankroll=bankrolls[0],
                team_1_bankroll=bankrolls[1],
                team_0_cards=fmt_cards(env.player_cards[0]),
                team_1_cards=fmt_cards(env.player_cards[1]),
                board_cards=fmt_cards(env.community_cards[:num_board]),
                team_0_discarded=fmt_cards(env.discarded_cards[0]),
                team_1_discarded=fmt_cards(env.discarded_cards[1]),
                team_0_bet=obs0["my_bet"] if obs0["acting_agent"] == 0 else obs0["opp_bet"],
                team_1_bet=obs1["my_bet"] if obs1["acting_agent"] == 1 else obs1["opp_bet"],
                action_type=PokerEnv.ActionType(action[0]).name,
                action_amount=action[1],
                action_keep_1=action[2],
                action_keep_2=action[3],
                actor="Human" if current_player == HUMAN_PLAYER else "Bot",
                decision_time_sec=decision_time_sec,
                human_reason=human_reason if current_player == HUMAN_PLAYER else "",
                human_confidence=human_confidence if current_player == HUMAN_PLAYER else "",
                human_read=human_read if current_player == HUMAN_PLAYER else "",
                human_discard_reason=human_discard_reason if current_player == HUMAN_PLAYER else "",
                opp_keep_inference=opp_keep_inference if current_player == HUMAN_PLAYER else "",
                valid_actions=list(obs_for_log.get("valid_actions", [])),
                cards_known_count=cards_known_count,
                cards_known_list=cards_known_list,
                bot_recommendation_keep=f"{action[2]},{action[3]}" if current_player == BOT_PLAYER and action[0] == DISCARD else "",
            )
            obs_native = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in obs_for_log.items()}
            reward_acting = reward0 if current_player == 0 else reward1
            session.log_rl_transition(
                hand_id=hand_number,
                step_in_hand=step_in_hand,
                player=current_player,
                obs=obs_native,
                action=list(action),
                reward=reward_acting,
                done=terminated,
                derived_state=derived,
                is_human=(current_player == HUMAN_PLAYER),
                info=info,
            )

        bankrolls[0] += reward0
        bankrolls[1] += reward1
        hand_number += 1

        if terminated:
            display_hand_result(
                env, hand_number - 1, reward0, reward1,
                list(env.player_cards[0]), list(env.player_cards[1]),
                env.community_cards,
                human_is_0=True,
            )
            changed, rev_street, rev_note = prompt_post_hand_review()
            if changed and rev_note:
                session.record_post_hand_review(hand_number - 1, rev_street, rev_note)

        print(f"Bankrolls — You: {bankrolls[0]}, Bot: {bankrolls[1]}")
        if num_hands < 9999 and hand_number >= num_hands:
            break
        again = input("Next hand? (Enter=yes, n=no): ").strip().lower()
        if again == "n":
            break

    summary_path = session.end_session(hand_number, bankrolls[0], bankrolls[1], bot_module_name)
    print(f"Session over. Summary: {summary_path}")
    print(f"Decisions CSV: {session._csv_path}")
    print(f"RL JSONL: {session._rl_path}")


if __name__ == "__main__":
    main()
