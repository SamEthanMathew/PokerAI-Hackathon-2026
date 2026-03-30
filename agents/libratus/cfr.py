"""
MCCFR (Monte Carlo CFR) for the abstract 27-card poker game.

Uses outcome sampling: one trajectory per iteration, update regrets with
importance sampling. Produces a blueprint strategy (average strategy or
final regret-matching strategy) that can be serialized and loaded by the agent.
"""

from __future__ import annotations

import copy
import random
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

from gym_env import PokerEnv
from .game_model import (
    PublicSequence,
    ActionRecord,
    infoset_key,
    FOLD,
    RAISE,
    CHECK,
    CALL,
    DISCARD,
    DECK_SIZE,
    SMALL_BLIND,
    BIG_BLIND,
    MAX_BET,
)
from .abstraction import (
    CardAbstraction,
    get_abstract_raise_amounts,
)


class CFRGameState:
    """
    Lightweight copy of game state for CFR traversal (no Gym dependency).
    Can be deep-copied so we can try multiple actions from the same state.
    """
    __slots__ = (
        "street", "bets", "last_street_bet", "min_raise", "acting_player",
        "player_cards", "community_cards", "discard_completed", "small_blind_player",
        "deck_remaining",
    )

    def __init__(
        self,
        street: int = 0,
        bets: Optional[List[int]] = None,
        last_street_bet: int = 0,
        min_raise: int = BIG_BLIND,
        acting_player: int = 0,
        player_cards: Optional[List[List[int]]] = None,
        community_cards: Optional[List[int]] = None,
        discard_completed: Optional[List[bool]] = None,
        small_blind_player: int = 0,
        deck_remaining: Optional[List[int]] = None,
    ):
        self.street = street
        self.bets = list(bets) if bets is not None else [0, 0]
        self.last_street_bet = last_street_bet
        self.min_raise = min_raise
        self.acting_player = acting_player
        self.player_cards = [list(p) for p in (player_cards or [[], []])]
        self.community_cards = list(community_cards or [])
        self.discard_completed = list(discard_completed) if discard_completed is not None else [False, False]
        self.small_blind_player = small_blind_player
        self.deck_remaining = list(deck_remaining) if deck_remaining is not None else []

    def copy(self) -> "CFRGameState":
        return CFRGameState(
            street=self.street,
            bets=list(self.bets),
            last_street_bet=self.last_street_bet,
            min_raise=self.min_raise,
            acting_player=self.acting_player,
            player_cards=[list(p) for p in self.player_cards],
            community_cards=list(self.community_cards),
            discard_completed=list(self.discard_completed),
            small_blind_player=self.small_blind_player,
            deck_remaining=list(self.deck_remaining),
        )


def terminal_payoff(state: CFRGameState, evaluator) -> Optional[float]:
    """Return payoff for player 0 (positive = P0 wins). None if not terminal. Called only when street>3 and bets equal."""
    p0_hand = state.player_cards[0]
    p1_hand = state.player_cards[1]
    board = state.community_cards
    if len(p0_hand) != 2 or len(p1_hand) != 2 or len(board) != 5:
        return None
    h0 = list(map(PokerEnv.int_to_card, p0_hand))
    h1 = list(map(PokerEnv.int_to_card, p1_hand))
    b = list(map(PokerEnv.int_to_card, board))
    r0 = evaluator.evaluate(h0, b)
    r1 = evaluator.evaluate(h1, b)
    pot = min(state.bets)  # amount each put in for showdown
    if r0 < r1:
        return float(pot)
    if r1 < r0:
        return float(-pot)
    return 0.0


def step_cfr_state(
    state: CFRGameState,
    action_type: int,
    raise_total: int,
    keep_card_1: int,
    keep_card_2: int,
) -> Tuple[CFRGameState, Optional[float], bool]:
    """
    Apply action to state; return (new_state, payoff_if_terminal, terminated).
    raise_total = total bet for this player after the raise (for RAISE).
    """
    p = state.acting_player
    opp = 1 - p
    new_state = state.copy()

    if action_type == FOLD:
        # Opponent wins the pot
        pot = state.bets[0] + state.bets[1]
        payoff = -pot if p == 0 else pot  # P0 loses
        return new_state, float(payoff), True

    if action_type == CALL:
        new_state.bets[p] = new_state.bets[opp]
        new_street = True
        if state.street == 0 and p == new_state.small_blind_player and new_state.bets[p] == BIG_BLIND:
            new_street = False  # SB just called BB, more action on preflop
        if new_street:
            new_state.street += 1
            new_state.last_street_bet = new_state.bets[0]
            new_state.min_raise = BIG_BLIND
            new_state.acting_player = 1 - new_state.small_blind_player
            if new_state.street > 3:
                return new_state, None, True  # will compute payoff at end
        else:
            new_state.acting_player = opp
        return new_state, None, new_state.street > 3

    if action_type == CHECK:
        new_street = False
        if state.street == 0 and p == (1 - new_state.small_blind_player):
            new_street = True
        elif state.street >= 1 and p == new_state.small_blind_player:
            new_street = True
        if new_street:
            new_state.street += 1
            new_state.last_street_bet = new_state.bets[0]
            new_state.min_raise = BIG_BLIND
            new_state.acting_player = 1 - new_state.small_blind_player
            if new_state.street > 3:
                return new_state, None, True
        else:
            new_state.acting_player = opp
        return new_state, None, False

    if action_type == RAISE:
        new_state.bets[p] = raise_total
        raise_so_far = new_state.bets[p] - new_state.last_street_bet
        new_state.min_raise = min(raise_so_far, MAX_BET - max(new_state.bets))
        if new_state.min_raise < 1:
            new_state.min_raise = 1
        new_state.acting_player = opp
        return new_state, None, False

    if action_type == DISCARD:
        new_state.discard_completed[p] = True
        cards = list(new_state.player_cards[p])
        kept = [cards[keep_card_1], cards[keep_card_2]]
        new_state.player_cards[p] = kept
        new_state.acting_player = opp
        return new_state, None, False

    return new_state, None, False


def get_valid_actions_cfr(state: CFRGameState) -> List[Tuple[int, int, int, int]]:
    """Returns list of (action_type, raise_total, keep_card_1, keep_card_2). raise_total only for RAISE."""
    p = state.acting_player
    opp = 1 - p
    out = []
    # FOLD always allowed (except maybe not in some variants)
    out.append((FOLD, 0, -1, -1))
    if state.bets[p] < state.bets[opp]:
        out.append((CALL, 0, -1, -1))
    if state.bets[p] >= state.bets[opp]:
        out.append((CHECK, 0, -1, -1))
    if state.street == 1 and not state.discard_completed[p]:
        for i in range(5):
            for j in range(i + 1, 5):
                out.append((DISCARD, 0, i, j))
        return out
    pot = state.bets[0] + state.bets[1]
    max_raise_inc = MAX_BET - max(state.bets)
    if max_raise_inc > 0:
        # Abstract raise amounts as increments; total bet = opp_bet + increment (we match then add)
        increments = get_abstract_raise_amounts(state.min_raise, max_raise_inc, pot)
        for inc in increments:
            total_bet = state.bets[opp] + inc
            if total_bet > state.bets[p] and total_bet <= MAX_BET:
                out.append((RAISE, total_bet, -1, -1))
    return out


def build_public_sequence_from_state(state: CFRGameState, actions_so_far: List[ActionRecord]) -> PublicSequence:
    """Build public sequence from state (street, discard_done) and actions_so_far."""
    seq = PublicSequence(street=state.street, discard_done=any(state.discard_completed))
    for a in actions_so_far:
        seq.append(a)
    return seq


class MCCFR:
    """
    Outcome sampling MCCFR. One trajectory per iteration; update regrets with
    importance sampling. Stores cumulative regrets and strategy sum per (infoset, action).
    """

    def __init__(self, card_abstraction: CardAbstraction, evaluator=None):
        self.card_abstraction = card_abstraction
        self.evaluator = evaluator or PokerEnv().evaluator
        self.regrets: Dict[str, List[float]] = defaultdict(list)  # infoset_key -> list of regrets per action
        self.strategy_sum: Dict[str, List[float]] = defaultdict(list)  # infoset_key -> list of cumulative strategy
        self.infoset_actions: Dict[str, List[Tuple[int, int, int, int]]] = {}  # infoset_key -> list of (atype, raise_total, k1, k2)

    def _get_infoset_key(self, state: CFRGameState, player: int, actions_so_far: List[ActionRecord]) -> str:
        if state.street == 1 and not state.discard_completed[player]:
            cards = state.player_cards[player]
            if len(cards) == 5:
                bucket = self.card_abstraction.bucket_5card(tuple(cards))
            else:
                bucket = 0
        else:
            cards = state.player_cards[player]
            board = tuple(state.community_cards)
            if len(cards) >= 2:
                bucket = self.card_abstraction.bucket_2card((cards[0], cards[1]), board)
            else:
                bucket = 0
        seq = build_public_sequence_from_state(state, actions_so_far)
        return infoset_key(player, seq, bucket)

    def _regret_matching_probs(self, regrets: List[float]) -> List[float]:
        """Convert regrets to probabilities (regret matching)."""
        pos = [max(0, r) for r in regrets]
        s = sum(pos)
        if s <= 0:
            return [1.0 / len(regrets)] * len(regrets)
        return [x / s for x in pos]

    def _get_strategy(self, infoset_key: str) -> List[float]:
        if infoset_key not in self.regrets:
            return []
        return self._regret_matching_probs(self.regrets[infoset_key])

    def run_iteration(
        self,
        state: CFRGameState,
        actions_so_far: List[ActionRecord],
        traverser: int,
        reach_self: float,
        reach_opp: float,
    ) -> float:
        """
        One traversal. Returns payoff to traverser. Updates regrets.
        """
        # Terminal (showdown)
        if state.street > 3:
            payoff = terminal_payoff(state, self.evaluator)
            return payoff if payoff is not None else 0.0
        valid = get_valid_actions_cfr(state)
        if not valid:
            payoff = terminal_payoff(state, self.evaluator)
            return payoff if payoff is not None else 0.0

        player = state.acting_player
        infoset = self._get_infoset_key(state, player, actions_so_far)

        if infoset not in self.infoset_actions:
            self.infoset_actions[infoset] = valid
            self.regrets[infoset] = [0.0] * len(valid)
            self.strategy_sum[infoset] = [0.0] * len(valid)

        action_list = self.infoset_actions[infoset]
        strategy = self._get_strategy(infoset)

        # Update average strategy (current policy at this infoset)
        for i in range(len(action_list)):
            if i < len(strategy):
                self.strategy_sum[infoset][i] += strategy[i]

        if player == traverser:
            # Traverse all actions; accumulate value and update regrets
            values = []
            for i, (a_type, raise_tot, k1, k2) in enumerate(action_list):
                ns, pay, term = step_cfr_state(state, a_type, raise_tot, k1, k2)
                if term and pay is not None:
                    values.append(pay)
                else:
                    if ns.street > 3:
                        pay = terminal_payoff(ns, self.evaluator)
                        values.append(pay if pay is not None else 0.0)
                    else:
                        new_actions = list(actions_so_far)
                        new_actions.append(ActionRecord(player=player, action_type=a_type, raise_amount=raise_tot, keep_card_1=k1, keep_card_2=k2))
                        if ns.acting_player != player:
                            new_reach_opp = reach_opp * strategy[i] if i < len(strategy) else reach_opp
                            values.append(self.run_iteration(ns, new_actions, traverser, reach_self, new_reach_opp))
                        else:
                            values.append(self.run_iteration(ns, new_actions, traverser, reach_self, reach_opp))
            # Update regrets
            util = sum(strategy[j] * values[j] for j in range(len(values)) if j < len(strategy))
            for i in range(len(action_list)):
                if reach_opp > 1e-10:
                    self.regrets[infoset][i] += (values[i] - util) * (1.0 / reach_opp)
            return util
        else:
            # Sample one action
            if not strategy or len(strategy) != len(action_list):
                strategy = [1.0 / len(action_list)] * len(action_list)
            r = random.random()
            s = 0.0
            idx = 0
            for i, p in enumerate(strategy):
                s += p
                if r <= s:
                    idx = i
                    break
            a_type, raise_tot, k1, k2 = action_list[idx]
            ns, pay, term = step_cfr_state(state, a_type, raise_tot, k1, k2)
            new_actions = list(actions_so_far)
            new_actions.append(ActionRecord(player=player, action_type=a_type, raise_amount=raise_tot, keep_card_1=k1, keep_card_2=k2))
            new_reach_opp = reach_opp * strategy[idx]
            if term and pay is not None:
                return pay
            if ns.street > 3:
                pay = terminal_payoff(ns, self.evaluator)
                return pay if pay is not None else 0.0
            return self.run_iteration(ns, new_actions, traverser, reach_self, new_reach_opp)

    def get_average_strategy(self) -> Dict[str, List[float]]:
        """Return average strategy (normalized strategy sum) per infoset."""
        out = {}
        for key, ssum in self.strategy_sum.items():
            total = sum(ssum)
            if total > 1e-10:
                out[key] = [x / total for x in ssum]
            else:
                n = len(ssum)
                out[key] = [1.0 / n] * n
        return out

    def run(self, num_iterations: int, rng: Optional[random.Random] = None) -> Dict[str, List[float]]:
        """Run MCCFR for num_iterations; return average strategy."""
        rng = rng or random.Random()
        for it in range(num_iterations):
            # Sample deal
            deck = list(range(DECK_SIZE))
            rng.shuffle(deck)
            p0_cards = [deck.pop() for _ in range(5)]
            p1_cards = [deck.pop() for _ in range(5)]
            board = [deck.pop() for _ in range(5)]
            state = CFRGameState(
                street=0,
                bets=[SMALL_BLIND, BIG_BLIND],
                last_street_bet=0,
                min_raise=BIG_BLIND,
                acting_player=0,
                player_cards=[p0_cards, p1_cards],
                community_cards=board,
                discard_completed=[False, False],
                small_blind_player=0,
                deck_remaining=deck,
            )
            traverser = it % 2
            self.run_iteration(state, [], traverser, 1.0, 1.0)
        return self.get_average_strategy()
