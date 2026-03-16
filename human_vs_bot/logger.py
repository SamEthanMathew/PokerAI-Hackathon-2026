"""
Decision CSV, RL JSONL, and session summary logging for human_vs_bot.
Logs every decision with match.py columns plus actor, timing, human annotations,
cards_known, derived_state (in RL), and post-hand review.
"""
import csv
import json
import os
from datetime import datetime
from typing import Any


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _serialize_cards(cards: list) -> str:
    return json.dumps(cards) if cards else "[]"


class SessionLogger:
    def __init__(self, logs_dir: str | None = None, session_id: str | None = None):
        self.logs_dir = logs_dir or os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.session_id = session_id or _ts()
        self._csv_path = os.path.join(self.logs_dir, f"decisions_{self.session_id}.csv")
        self._rl_path = os.path.join(self.logs_dir, f"rl_{self.session_id}.jsonl")
        self._csv_file = None
        self._writer = None
        self._csv_headers = None
        self._time_used_0 = 0.0
        self._time_used_1 = 0.0
        self._step_in_hand = 0
        self._hand_id = 0
        self._post_hand_annotations: list[dict] = []

    def _csv_headers_list(self) -> list[str]:
        return [
            "hand_number", "street", "active_team",
            "team_0_bankroll", "team_1_bankroll",
            "team_0_cards", "team_1_cards", "board_cards",
            "team_0_discarded", "team_1_discarded",
            "team_0_bet", "team_1_bet",
            "action_type", "action_amount", "action_keep_1", "action_keep_2",
            "actor",
            "decision_time_sec", "time_used_player_0", "time_used_player_1",
            "human_reason", "human_confidence", "human_read",
            "human_discard_reason", "opp_keep_inference",
            "valid_actions", "cards_known_count", "cards_known_list",
            "bot_recommendation_keep",
            "post_hand_changed", "post_hand_street", "post_hand_note",
        ]

    def start_session(self, bot_name: str):
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_file.write(f"# Bot: {bot_name}, Session: {self.session_id}\n")
        self._csv_headers = self._csv_headers_list()
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_headers, extrasaction="ignore")
        self._writer.writeheader()
        self._time_used_0 = 0.0
        self._time_used_1 = 0.0

    def log_decision(
        self,
        hand_number: int,
        street: str,
        active_team: int,
        team_0_bankroll: int,
        team_1_bankroll: int,
        team_0_cards: list,
        team_1_cards: list,
        board_cards: list,
        team_0_discarded: list,
        team_1_discarded: list,
        team_0_bet: int,
        team_1_bet: int,
        action_type: str,
        action_amount: int,
        action_keep_1: int,
        action_keep_2: int,
        actor: str,
        decision_time_sec: float = 0.0,
        human_reason: str = "",
        human_confidence: str = "",
        human_read: str = "",
        human_discard_reason: str = "",
        opp_keep_inference: str = "",
        valid_actions: list | None = None,
        cards_known_count: int = 0,
        cards_known_list: list | None = None,
        bot_recommendation_keep: str = "",
        is_discard: bool = False,
    ):
        if self._writer is None:
            return
        row = {
            "hand_number": hand_number,
            "street": street,
            "active_team": active_team,
            "team_0_bankroll": team_0_bankroll,
            "team_1_bankroll": team_1_bankroll,
            "team_0_cards": json.dumps(team_0_cards) if isinstance(team_0_cards, list) else str(team_0_cards),
            "team_1_cards": json.dumps(team_1_cards) if isinstance(team_1_cards, list) else str(team_1_cards),
            "board_cards": json.dumps(board_cards) if isinstance(board_cards, list) else str(board_cards),
            "team_0_discarded": json.dumps(team_0_discarded) if isinstance(team_0_discarded, list) else str(team_0_discarded),
            "team_1_discarded": json.dumps(team_1_discarded) if isinstance(team_1_discarded, list) else str(team_1_discarded),
            "team_0_bet": team_0_bet,
            "team_1_bet": team_1_bet,
            "action_type": action_type,
            "action_amount": action_amount,
            "action_keep_1": action_keep_1,
            "action_keep_2": action_keep_2,
            "actor": actor,
            "decision_time_sec": round(decision_time_sec, 3) if actor == "Human" else "",
            "time_used_player_0": round(self._time_used_0, 3),
            "time_used_player_1": round(self._time_used_1, 3),
            "human_reason": human_reason,
            "human_confidence": human_confidence,
            "human_read": human_read,
            "human_discard_reason": human_discard_reason,
            "opp_keep_inference": opp_keep_inference,
            "valid_actions": json.dumps(valid_actions) if valid_actions is not None else "",
            "cards_known_count": cards_known_count,
            "cards_known_list": _serialize_cards(cards_known_list or []),
            "bot_recommendation_keep": bot_recommendation_keep,
            "post_hand_changed": "",
            "post_hand_street": "",
            "post_hand_note": "",
        }
        self._writer.writerow(row)

    def add_time_used(self, player: int, sec: float):
        if player == 0:
            self._time_used_0 += sec
        else:
            self._time_used_1 += sec

    def log_rl_transition(
        self,
        hand_id: int,
        step_in_hand: int,
        player: int,
        obs: dict,
        action: list | tuple,
        reward: float,
        done: bool,
        derived_state: dict | None,
        is_human: bool,
        info: dict | None = None,
    ):
        """Append one JSON object per line to RL JSONL."""
        o = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in obs.items()}
        for k, v in list(o.items()):
            if hasattr(v, "item"):
                o[k] = v.item()
            elif isinstance(v, (list, tuple)) and v and hasattr(v[0], "item"):
                o[k] = [x.item() if hasattr(x, "item") else x for x in v]
        ds = derived_state or {}
        ds_serializable = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ds.items()}
        rec = {
            "hand_id": hand_id,
            "step_in_hand": step_in_hand,
            "player": player,
            "obs": o,
            "action": list(action),
            "reward": reward,
            "done": done,
            "derived_state": ds_serializable,
            "is_human": is_human,
            "info": info or {},
        }
        with open(self._rl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def record_post_hand_review(self, hand_number: int, street: str, note: str):
        self._post_hand_annotations.append({"hand_number": hand_number, "street": street, "note": note})

    def end_session(self, total_hands: int, bankroll_0: int, bankroll_1: int, bot_name: str):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        summary = {
            "session_id": self.session_id,
            "bot_name": bot_name,
            "total_hands": total_hands,
            "bankroll_0": bankroll_0,
            "bankroll_1": bankroll_1,
            "time_used_0": round(self._time_used_0, 2),
            "time_used_1": round(self._time_used_1, 2),
            "post_hand_annotations": self._post_hand_annotations,
        }
        summary_path = os.path.join(self.logs_dir, f"session_{self.session_id}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary_path
