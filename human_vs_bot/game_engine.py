"""
Stateful game runner for the human-vs-bot UI.
Runs one hand at a time; exposes get_state() and submit_action() / next_hand().
"""
import importlib
import logging
import os
import sys
import time as _time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from gym_env import PokerEnv
from match import prepare_payload, get_street_name

from human_vs_bot.derived_state import (
    compute_derived_state,
    cards_known_for_player,
    analysis_for_discard_options,
)
from human_vs_bot.logger import SessionLogger

HUMAN_PLAYER = 0
BOT_PLAYER = 1
TIME_LIMIT_SECONDS = 600
DISCARD = 4


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


def _fmt_cards(env, cards_list):
    return [env.int_card_to_str(int(c)) for c in cards_list if c != -1]


class UIGameRunner:
    """Single-session runner: start(bot_index, num_hands), get_state(), submit_action(), next_hand()."""

    def __init__(self):
        self._phase = "no_game"  # no_game | your_turn | hand_over | game_over
        self._env = None
        self._bot = None
        self._session = None
        self._bot_module_name = None
        self._num_hands = 10
        self._hand_number = 0
        self._bankrolls = [0, 0]
        self._time_used_0 = 0.0
        self._time_used_1 = 0.0
        self._obs0 = None
        self._obs1 = None
        self._terminated = False
        self._truncated = False
        self._info = None
        self._reward0 = 0
        self._reward1 = 0
        self._last_move_0 = None
        self._last_move_1 = None
        self._step_in_hand = 0
        self._bots = discover_bots()

    def get_bots(self):
        return [{"index": i, "name": name} for i, (name, _) in enumerate(self._bots)]

    def start(self, bot_index: int, num_hands: int):
        if bot_index < 0 or bot_index >= len(self._bots):
            raise ValueError("Invalid bot index")
        os.environ["MATCH_ID"] = "human_vs_bot_ui"
        os.environ["PLAYER_ID"] = "bot"
        bot_module_name, BotClass = self._bots[bot_index]
        self._bot_module_name = bot_module_name
        self._bot = BotClass(stream=False)
        self._env = PokerEnv(logger=logging.getLogger("PokerEnv"))
        self._session = SessionLogger()
        self._session.start_session(bot_name=bot_module_name)
        self._num_hands = num_hands if num_hands > 0 else 9999
        self._hand_number = 0
        self._bankrolls = [0, 0]
        self._time_used_0 = 0.0
        self._time_used_1 = 0.0
        self._phase = "your_turn"
        self._start_hand()

    def _start_hand(self):
        small_blind_player = self._hand_number % 2
        (self._obs0, self._obs1), self._info = self._env.reset(
            options={"small_blind_player": small_blind_player}
        )
        self._info["hand_number"] = self._hand_number
        self._reward0 = self._reward1 = 0
        self._terminated = self._truncated = False
        self._obs0["time_used"] = self._time_used_0
        self._obs0["time_left"] = TIME_LIMIT_SECONDS - self._time_used_0
        self._obs1["time_used"] = self._time_used_1
        self._obs1["time_left"] = TIME_LIMIT_SECONDS - self._time_used_1
        self._obs0["opp_last_action"] = "None"
        self._obs1["opp_last_action"] = "None"
        self._last_move_0 = self._last_move_1 = None
        self._step_in_hand = 0
        self._run_bot_until_human_turn()

    def _run_bot_until_human_turn(self):
        env, bot, session = self._env, self._bot, self._session
        while not self._terminated:
            current_player = self._obs0["acting_agent"]
            if current_player == HUMAN_PLAYER:
                break
            obs_bot = self._obs1 if current_player == BOT_PLAYER else self._obs0
            obs_human = self._obs0 if current_player == HUMAN_PLAYER else self._obs1
            payload = prepare_payload(
                obs_bot,
                self._reward1 if current_player == BOT_PLAYER else self._reward0,
                self._terminated,
                self._truncated,
                self._info,
            )
            t0 = _time.time()
            action = bot.get_bot_action(
                payload["observation"],
                payload["reward"],
                payload["terminated"],
                payload["truncated"],
                payload["info"],
            )
            elapsed = _time.time() - t0
            self._time_used_1 += elapsed
            session.add_time_used(1, elapsed)
            if action is None:
                action = (0, 0, 0, 0)
            if not isinstance(action, (list, tuple)) or len(action) < 4:
                action = (
                    int(action[0]) if action else 0,
                    int(action[1]) if len(action) > 1 else 0,
                    0,
                    0,
                )
            action = (int(action[0]), int(action[1]), int(action[2]), int(action[3]))
            if current_player == 0:
                self._last_move_0 = action[0]
            else:
                self._last_move_1 = action[0]

            obs_before = self._obs0 if current_player == 0 else self._obs1
            derived = compute_derived_state(obs_before, env)
            cards_known_count_pre = derived["cards_known"]
            cards_known_list_pre = derived.get("cards_known_list", [])

            (self._obs0, self._obs1), (self._reward0, self._reward1), self._terminated, self._truncated, self._info = env.step(
                action
            )
            self._info["hand_number"] = self._hand_number
            self._step_in_hand += 1

            self._obs0["time_used"] = self._time_used_0
            self._obs1["time_used"] = self._time_used_1
            self._obs0["time_left"] = TIME_LIMIT_SECONDS - self._time_used_0
            self._obs1["time_left"] = TIME_LIMIT_SECONDS - self._time_used_1
            self._obs0["opp_last_action"] = (
                "None" if self._last_move_0 is None else PokerEnv.ActionType(self._last_move_0).name
            )
            self._obs1["opp_last_action"] = (
                "None" if self._last_move_1 is None else PokerEnv.ActionType(self._last_move_1).name
            )

            num_board = 0 if self._obs0["street"] == 0 else self._obs0["street"] + 2
            session.log_decision(
                hand_number=self._hand_number,
                street=get_street_name(self._obs0["street"]),
                active_team=current_player,
                team_0_bankroll=self._bankrolls[0],
                team_1_bankroll=self._bankrolls[1],
                team_0_cards=_fmt_cards(env, env.player_cards[0]),
                team_1_cards=_fmt_cards(env, env.player_cards[1]),
                board_cards=_fmt_cards(env, env.community_cards[:num_board]),
                team_0_discarded=_fmt_cards(env, env.discarded_cards[0]),
                team_1_discarded=_fmt_cards(env, env.discarded_cards[1]),
                team_0_bet=self._obs0["my_bet"] if self._obs0["acting_agent"] == 0 else self._obs0["opp_bet"],
                team_1_bet=self._obs1["my_bet"] if self._obs1["acting_agent"] == 1 else self._obs1["opp_bet"],
                action_type=PokerEnv.ActionType(action[0]).name,
                action_amount=action[1],
                action_keep_1=action[2],
                action_keep_2=action[3],
                actor="Bot",
                decision_time_sec=elapsed,
                bot_recommendation_keep=f"{action[2]},{action[3]}"
                if action[0] == DISCARD else "",
                valid_actions=list(obs_before.get("valid_actions", [])),
                cards_known_count=cards_known_count_pre,
                cards_known_list=cards_known_list_pre,
            )
            obs_native = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in obs_before.items()
            }
            reward_acting = self._reward0 if current_player == 0 else self._reward1
            session.log_rl_transition(
                hand_id=self._hand_number,
                step_in_hand=self._step_in_hand,
                player=current_player,
                obs=obs_native,
                action=list(action),
                reward=reward_acting,
                done=self._terminated,
                derived_state=derived,
                is_human=False,
                info=self._info,
            )

        if self._terminated:
            self._phase = "hand_over"
        else:
            self._phase = "your_turn"

    def get_state(self):
        """Return JSON-serializable state for the UI."""
        if self._phase == "no_game":
            return {
                "phase": "no_game",
                "bots": self.get_bots(),
            }
        if self._phase == "game_over":
            return {
                "phase": "game_over",
                "hand_number": self._hand_number,
                "num_hands": self._num_hands,
                "bankrolls": list(self._bankrolls),
                "bot_name": self._bot_module_name,
            }

        if self._phase == "hand_over":
            env = self._env
            num_board = len([c for c in env.community_cards if c != -1])
            return {
                "phase": "hand_over",
                "hand_number": self._hand_number,
                "num_hands": self._num_hands,
                "bankrolls": list(self._bankrolls),
                "reward0": self._reward0,
                "reward1": self._reward1,
                "your_cards": _fmt_cards(env, env.player_cards[0]),
                "bot_cards": _fmt_cards(env, env.player_cards[1]),
                "board": _fmt_cards(env, env.community_cards[:num_board]),
                "bot_name": self._bot_module_name,
            }

        # your_turn: build human observation payload
        obs = self._obs0  # human is always player 0
        env = self._env
        my_cards = [int(c) for c in obs["my_cards"] if c != -1]
        community = [int(c) for c in obs["community_cards"] if c != -1]
        opp_discards = [int(c) for c in obs["opp_discarded_cards"] if c != -1]
        blind_pos = int(obs.get("blind_position", 0))
        cards_known_count, cards_known_list = cards_known_for_player(
            my_cards, community, opp_discards, blind_pos
        )
        from collections import Counter
        suits = [
            c // 9
            for c in (my_cards + community + (opp_discards if blind_pos == 0 else []))
            if 0 <= c < 27
        ]
        sc = Counter(suits)
        suit_names = ["diamonds", "hearts", "spades"]
        cards_known_str = (
            ", ".join(f"{cnt} {suit_names[s]} seen" for s, cnt in sc.most_common()) or "0 cards seen"
        )
        valid_actions = list(obs.get("valid_actions", [1] * 5))
        if len(valid_actions) < 5:
            valid_actions = valid_actions + [0] * (5 - len(valid_actions))
        is_discard_turn = valid_actions[DISCARD] if len(valid_actions) > DISCARD else False
        analysis_lines = []
        if is_discard_turn and len(my_cards) == 5 and len(community) >= 3:
            analysis_list = analysis_for_discard_options(
                my_cards, community, opp_discards, [], env
            )
            analysis_lines = [a["label"] for a in analysis_list]

        street_name = get_street_name(obs.get("street", 0))
        pot = int(obs.get("pot_size", 0))
        my_bet = int(obs.get("my_bet", 0))
        opp_bet = int(obs.get("opp_bet", 0))
        min_raise = int(obs.get("min_raise", 2))
        max_raise = int(obs.get("max_raise", 100))

        return {
            "phase": "your_turn",
            "hand_number": self._hand_number,
            "num_hands": self._num_hands,
            "bankrolls": list(self._bankrolls),
            "street_name": street_name,
            "my_cards": [env.int_card_to_str(int(c)) for c in my_cards],
            "community": [env.int_card_to_str(int(c)) for c in community],
            "opp_discarded": [env.int_card_to_str(int(c)) for c in opp_discards],
            "blind_position": blind_pos,
            "position_str": "SB (discard 2nd)" if blind_pos == 0 else "BB (discard 1st)",
            "cards_known_count": cards_known_count,
            "cards_known_str": cards_known_str,
            "analysis_lines": analysis_lines,
            "is_discard_turn": is_discard_turn,
            "valid_actions": valid_actions,
            "pot": pot,
            "my_bet": my_bet,
            "opp_bet": opp_bet,
            "min_raise": min_raise,
            "max_raise": max_raise,
            "bot_name": self._bot_module_name,
        }

    def submit_action(
        self,
        action_type: int,
        amount: int = 0,
        keep_i: int = 0,
        keep_j: int = 0,
        human_reason: str = "",
        human_confidence: str = "",
        human_read: str = "",
        human_discard_reason: str = "",
        opp_keep_inference: str = "",
    ):
        if self._phase != "your_turn":
            raise ValueError("Not your turn")
        action = (int(action_type), int(amount), int(keep_i), int(keep_j))
        env, session = self._env, self._session
        obs_before = self._obs0
        derived = compute_derived_state(obs_before, env)
        cards_known_count_pre = derived["cards_known"]
        cards_known_list_pre = derived.get("cards_known_list", [])

        (self._obs0, self._obs1), (self._reward0, self._reward1), self._terminated, self._truncated, self._info = env.step(
            action
        )
        self._info["hand_number"] = self._hand_number
        self._step_in_hand += 1
        self._last_move_0 = action[0]

        self._obs0["time_used"] = self._time_used_0
        self._obs1["time_used"] = self._time_used_1
        self._obs0["time_left"] = TIME_LIMIT_SECONDS - self._time_used_0
        self._obs1["time_left"] = TIME_LIMIT_SECONDS - self._time_used_1
        self._obs0["opp_last_action"] = PokerEnv.ActionType(action[0]).name
        self._obs1["opp_last_action"] = (
            "None" if self._last_move_1 is None else PokerEnv.ActionType(self._last_move_1).name
        )

        num_board = 0 if self._obs0["street"] == 0 else self._obs0["street"] + 2
        session.log_decision(
            hand_number=self._hand_number,
            street=get_street_name(self._obs0["street"]),
            active_team=0,
            team_0_bankroll=self._bankrolls[0],
            team_1_bankroll=self._bankrolls[1],
            team_0_cards=_fmt_cards(env, env.player_cards[0]),
            team_1_cards=_fmt_cards(env, env.player_cards[1]),
            board_cards=_fmt_cards(env, env.community_cards[:num_board]),
            team_0_discarded=_fmt_cards(env, env.discarded_cards[0]),
            team_1_discarded=_fmt_cards(env, env.discarded_cards[1]),
            team_0_bet=self._obs0["my_bet"],
            team_1_bet=self._obs0["opp_bet"],
            action_type=PokerEnv.ActionType(action[0]).name,
            action_amount=action[1],
            action_keep_1=action[2],
            action_keep_2=action[3],
            actor="Human",
            decision_time_sec=0.0,
            human_reason=human_reason,
            human_confidence=human_confidence,
            human_read=human_read,
            human_discard_reason=human_discard_reason,
            opp_keep_inference=opp_keep_inference,
            valid_actions=list(obs_before.get("valid_actions", [])),
            cards_known_count=cards_known_count_pre,
            cards_known_list=cards_known_list_pre,
        )
        obs_native = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in obs_before.items()
        }
        session.log_rl_transition(
            hand_id=self._hand_number,
            step_in_hand=self._step_in_hand,
            player=0,
            obs=obs_native,
            action=list(action),
            reward=self._reward0,
            done=self._terminated,
            derived_state=derived,
            is_human=True,
            info=self._info,
        )

        if self._terminated:
            self._bankrolls[0] += self._reward0
            self._bankrolls[1] += self._reward1
            self._phase = "hand_over"
            return self.get_state()

        self._run_bot_until_human_turn()
        return self.get_state()

    def next_hand(self, post_hand_changed: bool = False, post_hand_street: str = "", post_hand_note: str = ""):
        if self._phase != "hand_over":
            raise ValueError("No hand over")
        if post_hand_changed and post_hand_note:
            self._session.record_post_hand_review(
                self._hand_number, post_hand_street or "?", post_hand_note
            )
        self._hand_number += 1
        if self._hand_number >= self._num_hands:
            self._session.end_session(
                self._hand_number,
                self._bankrolls[0],
                self._bankrolls[1],
                self._bot_module_name,
            )
            self._phase = "game_over"
            return self.get_state()
        self._phase = "your_turn"
        self._start_hand()
        return self.get_state()
