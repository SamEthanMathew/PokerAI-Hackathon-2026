"""
PlayerAgent v15 — v14 + multi-street equity, compute budget, nit/sticky types, preflop dynamics.
"""

import os
import pickle
from pathlib import Path

from agents.agent import Agent
from gym_env import PokerEnv

from submission.constants import (
    FOLD, RAISE, CHECK, CALL, DISCARD,
    STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER,
    NUM_RANKS,
)
from submission.card_utils import CardUtils
from submission.time_manager import TimeManager
from submission.equity_engine import EquityEngine
from submission.equity_net_optional import OptionalEquityNetPredictor
from submission.discard_inference import DiscardInference
from submission.discard_optimizer import DiscardOptimizer
from submission.opponent_model import OpponentModel
from submission.preflop_strategy import PreFlopStrategy
from submission.betting_strategy import BettingStrategy
from submission.phase_controller import PhaseController
from submission.bet_sizing_tracker import BetSizingTracker


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.card_utils = CardUtils()
        self.equity_engine = EquityEngine()
        self.equity_net = OptionalEquityNetPredictor()
        self.discard_inference = DiscardInference(self.equity_engine)
        self.discard_optimizer = DiscardOptimizer(self.equity_engine)
        self.opponent_model = OpponentModel()
        preflop_table, preflop_sorted = self._load_preflop_table()
        self.preflop_strategy = PreFlopStrategy(
            preflop_table=preflop_table,
            preflop_sorted_values=preflop_sorted,
        )
        self.betting_strategy = BettingStrategy()
        self.time_manager = TimeManager(total_budget=1900.0, num_hands=1000)
        self.phase_controller = PhaseController(match_max_hands=1000)
        self.bet_sizing_tracker = BetSizingTracker()

        self.hand_number = -1
        self._acted_this_hand = False
        self._cached_weighted_range = None
        self._bankroll = 0
        self._total_hands = 1000

    def _load_preflop_table(self):
        """Load 2-card equity table from pickle: dict (c0,c1) -> float, c0 < c1. Returns (table, sorted_values)."""
        default_path = Path(__file__).resolve().parent / "data" / "preflop_table.pkl"
        path = Path(os.getenv("PREFLOP_TABLE_PATH", str(default_path)))
        if not path.exists():
            return {}, []
        try:
            with open(path, "rb") as f:
                raw = pickle.load(f)
        except Exception:
            return {}, []
        if not isinstance(raw, dict):
            return {}, []
        normalized = {}
        for key, val in raw.items():
            try:
                k = tuple(sorted(int(c) for c in key))
                if len(k) != 2:
                    continue
                normalized[k] = float(val)
            except (TypeError, ValueError):
                continue
        if not normalized:
            return {}, []
        sorted_vals = sorted(normalized.values())
        if self.logger:
            self.logger.info("Loaded %s preflop table entries from %s", len(normalized), path)
        return normalized, sorted_vals

    def __name__(self):
        return "PlayerAgent"

    def act(self, observation, reward, terminated, truncated, info):
        try:
            return self._decide(observation, reward, terminated, truncated, info)
        except Exception as e:
            self.logger.error(f"Error in act: {e}")
            return self._safe_fallback(observation)

    def _decide(self, observation, reward, terminated, truncated, info):
        current_hand = info.get("hand_number", self.hand_number + 1)
        if current_hand != self.hand_number:
            if self._acted_this_hand:
                self.time_manager.stop_hand()
            self.hand_number = current_hand
            self._acted_this_hand = False
            self._cached_weighted_range = None

        if not self._acted_this_hand:
            self.time_manager.update_from_observation(observation)
            self.time_manager.start_hand()
            self._acted_this_hand = True

        valid = observation["valid_actions"]
        street = observation["street"]

        opp_last = observation.get("opp_last_action", "None")
        if opp_last and opp_last != "None":
            self.opponent_model.update_action(observation, opp_last)
        self.bet_sizing_tracker.record_observation(observation, info)

        if self._is_victory_secured(observation):
            return self._victory_secure_action(observation)

        if valid[DISCARD]:
            return self._handle_discard(observation)
        if street == STREET_PREFLOP:
            return self._handle_preflop(observation)
        return self._handle_postflop_bet(observation, street)

    def _is_victory_secured(self, observation) -> bool:
        hands_remaining = self._total_hands - self.hand_number
        if hands_remaining <= 0:
            return False
        pos = observation.get("blind_position", 0)
        full_cycles = hands_remaining // 2
        remainder = hands_remaining % 2
        cost = full_cycles * 3
        if remainder == 1:
            cost += (1 if pos == 0 else 2)
        return self._bankroll > (cost + 5)

    def _victory_secure_action(self, observation):
        valid = observation["valid_actions"]
        if valid[DISCARD]:
            return (DISCARD, 0, 0, 1)
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _handle_discard(self, observation):
        my_cards = CardUtils.get_my_cards(observation)
        board = CardUtils.get_community_cards(observation)
        opp_discards = CardUtils.get_opp_discards(observation)
        if len(my_cards) != 5:
            return (DISCARD, 0, 0, 1)
        opp_type = self.opponent_model.classify()
        opp_range = None
        if opp_discards:
            try:
                dead_for_inference = set(my_cards) | set(board) | set(opp_discards)
                opp_range = self.discard_inference.infer_opponent_range(
                    opp_discards, board, dead_for_inference, opp_type=opp_type,
                )
                self._cached_weighted_range = opp_range
            except Exception:
                opp_range = None
        time_budget = self.time_manager.get_decision_budget("discard")
        if self.time_manager.should_use_fast_path():
            time_budget = 0.05
        try:
            keep_indices = self.discard_optimizer.choose_best_keep(
                my_cards, board,
                opp_discards=opp_discards if opp_discards else None,
                opp_range=opp_range,
                time_budget=time_budget,
            )
        except Exception:
            keep_indices = (0, 1)
        return (DISCARD, 0, keep_indices[0], keep_indices[1])

    def _handle_preflop(self, observation):
        my_cards = CardUtils.get_my_cards(observation)
        opp_tendencies = self.opponent_model.get_tendencies()
        hands_remaining = max(1, self._total_hands - self.hand_number)
        phase = self.phase_controller.get_phase(self.hand_number)
        variance_profile = self.phase_controller.get_variance_profile(self.hand_number)
        try:
            action = self.preflop_strategy.decide(
                my_cards, observation,
                opp_tendencies=opp_tendencies,
                bankroll=self._bankroll,
                hands_remaining=hands_remaining,
                phase=phase,
                variance_profile=variance_profile,
            )
        except Exception:
            action = self._safe_fallback(observation)
        return self._validate_action(observation, action)

    def _handle_postflop_bet(self, observation, street):
        my_cards = CardUtils.get_my_cards(observation)
        board = CardUtils.get_community_cards(observation)
        opp_discards = CardUtils.get_opp_discards(observation)
        if len(my_cards) != 2 or len(board) < 3:
            return self._safe_fallback(observation)

        weighted_range = self._cached_weighted_range
        if weighted_range is None and opp_discards:
            try:
                dead_for_inference = CardUtils.get_dead_cards(observation)
                weighted_range = self.discard_inference.infer_opponent_range(
                    opp_discards, board, dead_for_inference,
                    opp_type=self.opponent_model.classify(),
                )
                self._cached_weighted_range = weighted_range
            except Exception:
                weighted_range = None

        dead_cards = CardUtils.get_dead_cards(observation)
        net_equity = None
        if self.equity_net.available:
            try:
                net_dead = set(opp_discards) | set(c for c in observation.get("my_discarded_cards", []) if c != -1)
                net_equity = self.equity_net.predict(
                    hole_cards=my_cards, board_cards=board,
                    dead_cards=sorted(net_dead), street=street,
                )
            except Exception:
                pass

        weighted_equity = None
        max_samples = 1500 if net_equity is not None else 2500
        if self.time_manager.should_use_fast_path():
            max_samples = 200
        try:
            weighted_equity = self.equity_engine.compute_equity(
                my_cards=my_cards, board=board, dead_cards=dead_cards,
                opp_range=weighted_range, max_samples=max_samples,
            )
        except Exception:
            pass

        if net_equity is not None and weighted_equity is not None:
            community = [c for c in observation.get("community_cards", []) if c != -1]
            is_dangerous = False
            if len(community) >= 3:
                from collections import Counter
                board_ranks = [c % NUM_RANKS for c in community]
                if any(v >= 2 for v in Counter(board_ranks).values()):
                    is_dangerous = True
            nn_weight = 0.1 if is_dangerous else 0.3
            equity = nn_weight * net_equity + (1.0 - nn_weight) * weighted_equity
        elif weighted_equity is not None:
            equity = weighted_equity
        elif net_equity is not None:
            equity = net_equity
        else:
            if self.time_manager.is_time_critical():
                equity = self._quick_equity(my_cards, board)
            else:
                try:
                    equity = self.equity_engine.compute_equity(
                        my_cards=my_cards, board=board, dead_cards=dead_cards,
                        opp_range=None, max_samples=900,
                    )
                except Exception:
                    equity = self._quick_equity(my_cards, board)

        self.phase_controller.update_bankrolls(self._bankroll, 0)
        variance_profile = self.phase_controller.get_variance_profile(self.hand_number)
        opp_tendencies = self.opponent_model.get_tendencies()
        opp_tendencies["turn_barrel_prob"] = self.bet_sizing_tracker.get_turn_barrel_probability()
        opp_tendencies["river_barrel_prob"] = self.bet_sizing_tracker.get_river_barrel_probability()
        opp_tendencies["avg_bet_ratio_turn"] = self.bet_sizing_tracker.get_avg_bet_ratio(2)
        opp_tendencies["avg_bet_ratio_river"] = self.bet_sizing_tracker.get_avg_bet_ratio(3)

        try:
            action = self.betting_strategy.decide(
                equity=equity, observation=observation,
                opp_tendencies=opp_tendencies,
                variance_profile=variance_profile,
                street=street,
            )
        except Exception:
            action = self._safe_fallback(observation)
        return self._validate_action(observation, action)

    def observe(self, observation, reward, terminated, truncated, info):
        try:
            self._bankroll += reward
            if terminated:
                self.opponent_model.end_hand(info)
                if self._acted_this_hand:
                    self.time_manager.stop_hand()
                    self._acted_this_hand = False
                self._cached_weighted_range = None
        except Exception:
            pass

    def _quick_equity(self, my_cards, board):
        try:
            if len(board) >= 3:
                score = self.equity_engine.quick_hand_score(my_cards, board)
                return min(max(score / 10.0, 0.0), 1.0)
        except Exception:
            pass
        return 0.4

    def _validate_action(self, observation, action):
        valid = observation["valid_actions"]
        action_type = action[0]
        if action_type < len(valid) and valid[action_type]:
            if action_type == RAISE:
                amount = max(observation["min_raise"], min(action[1], observation["max_raise"]))
                self.opponent_model.record_raise_opportunity(observation.get("street", 0))
                return (RAISE, amount, action[2], action[3])
            return action
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        if valid[CALL]:
            return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _safe_fallback(self, observation):
        try:
            valid = observation["valid_actions"]
            if valid[DISCARD]:
                return (DISCARD, 0, 0, 1)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
        except Exception:
            pass
        return (FOLD, 0, 0, 0)
