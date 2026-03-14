"""
Libratus-style agent: blueprint strategy lookup + action history + equity-based discard.
"""

import os
import random
from typing import List, Tuple, Optional, Dict, Any

from agents.agent import Agent
from gym_env import PokerEnv
from agents.libratus.game_model import (
    PublicSequence,
    ActionRecord,
    infoset_key,
    parse_opponent_action_from_obs,
    action_record_from_obs,
    FOLD,
    RAISE,
    CHECK,
    CALL,
    DISCARD,
)
from agents.libratus.abstraction import CardAbstraction, get_abstract_raise_amounts
from agents.libratus.strategy_store import load_strategy

ActionType = PokerEnv.ActionType
DECK_SIZE = 27


class LibratusAgent(Agent):
    """
    Phase 1 Libratus: precomputed blueprint strategy + action-history tracking.
    Discard phase uses Monte Carlo equity (like ProbabilityAgent).
    """

    def __name__(self):
        return "LibratusAgent"

    def __init__(self, stream: bool = True, strategy_path: Optional[str] = None):
        super().__init__(stream)
        self.evaluator = PokerEnv().evaluator
        self.card_abstraction = CardAbstraction(num_buckets_5=30, num_buckets_2=15, evaluator=self.evaluator)
        self.strategy: Dict[str, List[float]] = {}
        self.infoset_actions: Dict[str, List[Tuple[int, int, int, int]]] = {}
        path = strategy_path or os.path.join(
            os.path.dirname(__file__), "libratus", "blueprint_strategy.json"
        )
        if os.path.isfile(path):
            self.strategy, self.infoset_actions = load_strategy(path)
            self.logger.info(f"Loaded blueprint from {path} ({len(self.strategy)} infosets)")
        else:
            self.logger.info(f"No strategy file at {path}; using fallback (equity/fold)")

        self.action_history: List[ActionRecord] = []
        self._last_obs: Optional[dict] = None

    def _my_player_id(self, observation: dict) -> int:
        return observation.get("blind_position", 0)

    def _compute_equity(
        self,
        my_cards: List[int],
        community_cards: List[int],
        opp_discarded_cards: List[int],
        num_simulations: int = 200,
    ) -> float:
        shown = set(my_cards)
        for c in community_cards:
            if c != -1:
                shown.add(c)
        for c in opp_discarded_cards:
            if c != -1:
                shown.add(c)
        non_shown = [i for i in range(DECK_SIZE) if i not in shown]
        opp_needed = 2
        board_needed = 5 - len([c for c in community_cards if c != -1])
        wins = 0
        valid = 0
        for _ in range(num_simulations):
            sample_size = opp_needed + board_needed
            if sample_size > len(non_shown):
                continue
            sample = random.sample(non_shown, sample_size)
            opp_cards = sample[:opp_needed]
            full_board = list(community_cards) + sample[opp_needed : opp_needed + board_needed]
            full_board = [c for c in full_board if c != -1]
            if len(full_board) != 5:
                continue
            my_hand = list(map(PokerEnv.int_to_card, my_cards))
            opp_hand = list(map(PokerEnv.int_to_card, opp_cards))
            board = list(map(PokerEnv.int_to_card, full_board))
            my_rank = self.evaluator.evaluate(my_hand, board)
            opp_rank = self.evaluator.evaluate(opp_hand, board)
            if my_rank < opp_rank:
                wins += 1
            valid += 1
        return wins / valid if valid > 0 else 0.0

    def _build_public_sequence(self, observation: dict) -> PublicSequence:
        street = observation["street"]
        my_discard = observation.get("my_discarded_cards", [])
        opp_discard = observation.get("opp_discarded_cards", [])
        discard_done = (
            len([c for c in my_discard if c != -1]) >= 3
            and len([c for c in opp_discard if c != -1]) >= 3
        )
        seq = PublicSequence(street=street, discard_done=discard_done)
        for a in self.action_history:
            seq.append(a)
        return seq

    def _get_infoset_key(self, observation: dict) -> str:
        my_cards = [c for c in observation["my_cards"] if c != -1]
        community_cards = tuple(c for c in observation["community_cards"] if c != -1)
        street = observation["street"]
        my_discard = observation.get("my_discarded_cards", [])
        opp_discard = observation.get("opp_discarded_cards", [])
        discard_done = (
            len([c for c in my_discard if c != -1]) >= 3
            and len([c for c in opp_discard if c != -1]) >= 3
        )
        seq = self._build_public_sequence(observation)
        bucket = self.card_abstraction.bucket_hand_for_infoset(
            tuple(my_cards) if my_cards else (0, 0),
            street,
            community_cards,
            discard_done,
        )
        me = self._my_player_id(observation)
        return infoset_key(me, seq, bucket)

    def _get_valid_abstract_actions(self, observation: dict) -> List[Tuple[int, int, int, int]]:
        """Return list of (action_type, raise_total, k1, k2) that are valid in current state."""
        valid = observation["valid_actions"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        pot = my_bet + opp_bet
        out = []
        if valid[ActionType.FOLD.value]:
            out.append((FOLD, 0, -1, -1))
        if valid[ActionType.CALL.value]:
            out.append((CALL, 0, -1, -1))
        if valid[ActionType.CHECK.value]:
            out.append((CHECK, 0, -1, -1))
        if valid[ActionType.DISCARD.value]:
            for i in range(5):
                for j in range(i + 1, 5):
                    out.append((DISCARD, 0, i, j))
            return out
        if valid[ActionType.RAISE.value]:
            max_inc = max_raise
            increments = get_abstract_raise_amounts(min_raise, max_inc, pot)
            for inc in increments:
                total_bet = opp_bet + inc
                if total_bet > my_bet and total_bet <= PokerEnv.MAX_PLAYER_BET:
                    out.append((RAISE, total_bet, -1, -1))
        return out

    def _lookup_strategy(self, infoset_key: str, valid_abstract: List[Tuple[int, int, int, int]]) -> List[float]:
        if infoset_key in self.strategy and infoset_key in self.infoset_actions:
            stored_actions = self.infoset_actions[infoset_key]
            stored_probs = self.strategy[infoset_key]
            # Map stored actions to valid_abstract indices (same order as in CFR)
            probs = []
            for va in valid_abstract:
                try:
                    idx = stored_actions.index(va)
                    probs.append(stored_probs[idx])
                except ValueError:
                    probs.append(0.0)
            if sum(probs) > 1e-10:
                s = sum(probs)
                return [p / s for p in probs]
        # Uniform over valid
        return [1.0 / len(valid_abstract)] * len(valid_abstract)

    def act(self, observation: dict, reward: float, terminated: bool, truncated: bool, info: dict) -> Tuple[int, int, int, int]:
        my_cards_raw = observation["my_cards"]
        my_cards = [c for c in my_cards_raw if c != -1]
        community_cards = [c for c in observation["community_cards"] if c != -1]
        opp_discarded_cards = list(observation.get("opp_discarded_cards", [-1] * 3))
        valid_actions = observation["valid_actions"]
        me = self._my_player_id(observation)

        # --- Discard phase: equity-based ---
        if valid_actions[ActionType.DISCARD.value]:
            if len(my_cards) == 5:
                best_keep = (0, 1)
                best_equity = -1.0
                for i in range(5):
                    for j in range(i + 1, 5):
                        keep_pair = [my_cards[i], my_cards[j]]
                        eq = self._compute_equity(keep_pair, community_cards, opp_discarded_cards, 200)
                        if eq > best_equity:
                            best_equity = eq
                            best_keep = (i, j)
                action = (ActionType.DISCARD.value, 0, best_keep[0], best_keep[1])
            else:
                action = (ActionType.DISCARD.value, 0, 0, 1)
            self.action_history.append(ActionRecord(player=me, action_type=DISCARD, keep_card_1=action[2], keep_card_2=action[3]))
            return action

        # --- Betting: blueprint lookup ---
        valid_abstract = self._get_valid_abstract_actions(observation)
        if not valid_abstract:
            self.action_history.append(ActionRecord(player=me, action_type=FOLD))
            return (ActionType.FOLD.value, 0, 0, 0)

        infoset = self._get_infoset_key(observation)
        probs = self._lookup_strategy(infoset, valid_abstract)
        r = random.random()
        cum = 0.0
        idx = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                idx = i
                break
        a_type, raise_total, k1, k2 = valid_abstract[idx]

        if a_type == FOLD:
            action = (ActionType.FOLD.value, 0, 0, 0)
        elif a_type == CALL:
            action = (ActionType.CALL.value, 0, 0, 0)
        elif a_type == CHECK:
            action = (ActionType.CHECK.value, 0, 0, 0)
        elif a_type == RAISE:
            opp_bet = observation["opp_bet"]
            my_bet = observation["my_bet"]
            increment = raise_total - my_bet
            increment = max(increment, observation["min_raise"])
            increment = min(increment, observation["max_raise"])
            action = (ActionType.RAISE.value, increment, 0, 0)
        else:
            action = (ActionType.FOLD.value, 0, 0, 0)

        self.action_history.append(ActionRecord(player=me, action_type=a_type, raise_amount=raise_total if a_type == RAISE else 0, keep_card_1=k1, keep_card_2=k2))
        return action

    def observe(self, observation: dict, reward: float, terminated: bool, truncated: bool, info: dict) -> None:
        if terminated:
            self.action_history.clear()
            if abs(reward) > 20:
                self.logger.info(f"Hand ended with reward: {reward}")
            if "player_0_cards" in info:
                self.logger.info(
                    f"Showdown: {info.get('player_0_cards')} vs {info.get('player_1_cards')} "
                    f"board {info.get('community_cards')}"
                )
            return
        opp_last = (observation.get("opp_last_action") or "").strip().upper()
        if not opp_last:
            return
        me = self._my_player_id(observation)
        opp = 1 - me
        rec = parse_opponent_action_from_obs(me, opp_last, observation["my_bet"], observation["opp_bet"], observation)
        if rec is not None:
            self.action_history.append(rec)
