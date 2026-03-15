"""
Decision-Tree poker agent combining exact equity enumeration with a
parameterized decision tree. All parameters live in a theta JSON file
and are tunable via CMA-ES.
"""

from typing import Tuple, Optional

from agents.agent import Agent
from agents.libratus.fast_eval import load_rank7_table
from agents.libratus.exact_equity import equity_discard, equity_postflop
from agents.libratus.features import (
    hole2_features,
    board_features,
    OpponentStats,
)
from agents.libratus.decision_tree import (
    load_theta,
    choose_discard,
    choose_bet_action,
    FOLD, RAISE, CHECK, CALL, DISCARD,
)


class DecisionTreeAgent(Agent):
    def __name__(self):
        return "DecisionTreeAgent"

    def __init__(self, stream: bool = True, theta_path: Optional[str] = None):
        super().__init__(stream)
        self._theta = load_theta(theta_path)

        self._rank7 = None
        try:
            self._rank7 = load_rank7_table()
        except Exception:
            self.logger.warning("rank7 table not found; exact equity disabled")

        self._opp_stats = OpponentStats()
        self._last_my_action = None
        self._hand_count = 0

    def act(self, observation, reward, terminated, truncated, info) -> Tuple[int, int, int, int]:
        obs = observation
        va = obs["valid_actions"]

        if va[DISCARD]:
            return self._act_discard(obs)

        return self._act_bet(obs)

    def _act_discard(self, obs: dict) -> Tuple[int, int, int, int]:
        eq_fn = None
        if self._rank7 is not None:
            eq_fn = lambda my5, f3, od: equity_discard(my5, f3, od, self._rank7)

        action = choose_discard(
            obs, self._theta,
            exact_equity_fn=eq_fn,
            rank7_table=self._rank7,
        )
        self._last_my_action = "DISCARD"
        return action

    def _act_bet(self, obs: dict) -> Tuple[int, int, int, int]:
        hole = [c for c in obs["my_cards"] if c != -1]
        board = [c for c in obs["community_cards"] if c != -1]
        opp_disc = [c for c in obs.get("opp_discarded_cards", []) if c >= 0]

        if self._rank7 is not None and len(hole) == 2 and len(board) >= 3:
            equity = equity_postflop(tuple(hole), board, opp_disc, self._rank7)
        else:
            equity = 0.5

        bf = board_features(board)
        hf = {}
        if len(hole) == 2:
            hf = hole2_features(hole[0], hole[1])

        hand_feat = {**bf, **hf, **self._opp_stats.to_features()}

        action = choose_bet_action(obs, self._theta, equity, hand_feat)
        action_type = action[0]

        if action_type == FOLD:
            self._last_my_action = "FOLD"
        elif action_type == RAISE:
            self._last_my_action = "RAISE"
        elif action_type == CALL:
            self._last_my_action = "CALL"
        else:
            self._last_my_action = "CHECK"

        return action

    def observe(self, observation, reward, terminated, truncated, info) -> None:
        if terminated:
            self._hand_count += 1
            return

        opp_action = observation.get("opp_last_action", "")
        if opp_action:
            was_facing_raise = self._last_my_action == "RAISE"
            self._opp_stats.update(opp_action, was_facing_raise)


if __name__ == "__main__":
    DecisionTreeAgent.run(stream=True)
