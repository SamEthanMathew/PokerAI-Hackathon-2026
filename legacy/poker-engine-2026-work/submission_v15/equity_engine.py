"""
Equity Engine for v14 Alpha — WMC/exhaustive with weighted opponent ranges and cache.
"""

import random
from itertools import combinations

from gym_env import WrappedEval, PokerEnv
from submission.constants import NUM_CARDS


class EquityEngine:
    def __init__(self):
        self.evaluator = WrappedEval()
        self._int_to_card = PokerEnv.int_to_card
        self._eval_cache = {}
        self._cache_max = 200000

    def evaluate_hand(self, hand: list, board: list) -> int:
        """Evaluate 2-card hand vs 5-card board. Cached. Lower rank = better."""
        key = (frozenset(hand), frozenset(board))
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached
        treys_hand = [self._int_to_card(c) for c in hand]
        treys_board = [self._int_to_card(c) for c in board]
        rank = self.evaluator.evaluate(treys_hand, treys_board)
        self._eval_cache[key] = rank
        if len(self._eval_cache) > self._cache_max:
            self._eval_cache.clear()
        return rank

    def compute_equity(
        self,
        my_cards: list,
        board: list,
        dead_cards: set,
        opp_range=None,
        max_samples: int = 5000,
    ) -> float:
        """
        Compute win probability. opp_range: None (uniform), list of (c1,c2), or list of (c1,c2,weight).
        """
        my_set = set(my_cards)
        all_known = dead_cards | my_set
        unknown = [c for c in range(NUM_CARDS) if c not in all_known]

        weighted_hands = None
        if opp_range is None:
            opp_hands = list(combinations(unknown, 2))
        elif opp_range and len(opp_range[0]) == 3:
            weighted_hands = [
                (c1, c2, w) for c1, c2, w in opp_range
                if c1 not in all_known and c2 not in all_known
            ]
            opp_hands = [(c1, c2) for c1, c2, _ in weighted_hands]
        else:
            opp_hands = [
                (c1, c2) for c1, c2 in opp_range
                if c1 not in all_known and c2 not in all_known
            ]

        if not opp_hands:
            return 0.5

        cards_to_come = 5 - len(board)
        if cards_to_come == 0:
            return self._river_equity(my_cards, board, opp_hands, weighted_hands)
        elif cards_to_come <= 2:
            runout_combos = list(combinations(unknown, cards_to_come))
            total_work = len(runout_combos) * len(opp_hands)
            if total_work <= max_samples:
                return self._exhaustive(my_cards, board, opp_hands, runout_combos, weighted_hands)
            return self._monte_carlo(
                my_cards, board, unknown, opp_hands, cards_to_come, max_samples, weighted_hands
            )
        else:
            return self._monte_carlo(
                my_cards, board, unknown, opp_hands, cards_to_come, max_samples, weighted_hands
            )

    def _river_equity(self, my_cards, board, opp_hands, weighted_hands=None):
        my_rank = self.evaluate_hand(my_cards, board)
        if weighted_hands:
            weighted_wins = 0.0
            total_weight = 0.0
            for c1, c2, w in weighted_hands:
                opp_rank = self.evaluate_hand([c1, c2], board)
                if my_rank < opp_rank:
                    weighted_wins += w
                elif my_rank == opp_rank:
                    weighted_wins += w * 0.5
                total_weight += w
            return weighted_wins / total_weight if total_weight > 0 else 0.5
        wins = ties = total = 0
        for opp_hand in opp_hands:
            opp_rank = self.evaluate_hand(list(opp_hand), board)
            if my_rank < opp_rank:
                wins += 1
            elif my_rank == opp_rank:
                ties += 1
            total += 1
        return (wins + ties * 0.5) / total if total else 0.5

    def _exhaustive(self, my_cards, board, opp_hands, runout_combos, weighted_hands=None):
        weight_map = {}
        if weighted_hands:
            for c1, c2, w in weighted_hands:
                weight_map[(c1, c2)] = w
                weight_map[(c2, c1)] = w
        weighted_wins = 0.0
        total_weight = 0.0
        for runout in runout_combos:
            runout_set = set(runout)
            full_board = board + list(runout)
            my_rank = self.evaluate_hand(my_cards, full_board)
            for opp_hand in opp_hands:
                if opp_hand[0] in runout_set or opp_hand[1] in runout_set:
                    continue
                w = weight_map.get(opp_hand, 1.0) if weight_map else 1.0
                opp_rank = self.evaluate_hand(list(opp_hand), full_board)
                if my_rank < opp_rank:
                    weighted_wins += w
                elif my_rank == opp_rank:
                    weighted_wins += w * 0.5
                total_weight += w
        return weighted_wins / total_weight if total_weight > 0 else 0.5

    def _monte_carlo(self, my_cards, board, unknown, opp_hands, cards_to_come, max_samples, weighted_hands=None):
        unknown_list = list(unknown)
        if weighted_hands:
            total_w = sum(w for _, _, w in weighted_hands)
            if total_w <= 0:
                return 0.5
        weighted_wins = 0.0
        total_weight = 0.0
        for _ in range(max_samples):
            random.shuffle(unknown_list)
            runout = unknown_list[:cards_to_come]
            runout_set = set(runout)
            full_board = board + runout
            if weighted_hands:
                valid_hw = [(c1, c2, w) for c1, c2, w in weighted_hands
                             if c1 not in runout_set and c2 not in runout_set]
                if not valid_hw:
                    continue
                tw = sum(w for _, _, w in valid_hw)
                r = random.random() * tw
                acc = 0.0
                chosen = valid_hw[0]
                for c1, c2, w in valid_hw:
                    acc += w
                    if acc >= r:
                        chosen = (c1, c2, w)
                        break
                opp_hand = [chosen[0], chosen[1]]
                w = chosen[2]
            else:
                valid_opp = [h for h in opp_hands if h[0] not in runout_set and h[1] not in runout_set]
                if not valid_opp:
                    continue
                opp_hand = list(random.choice(valid_opp))
                w = 1.0
            my_rank = self.evaluate_hand(my_cards, full_board)
            opp_rank = self.evaluate_hand(opp_hand, full_board)
            if my_rank < opp_rank:
                weighted_wins += w
            elif my_rank == opp_rank:
                weighted_wins += w * 0.5
            total_weight += w
        return weighted_wins / total_weight if total_weight > 0 else 0.5

    def quick_hand_score(self, two_cards: list, board: list) -> float:
        """Fast approximate hand strength (higher = better, ~0-10)."""
        try:
            rank = self.evaluate_hand(two_cards, board)
            return max(0.0, (7500 - rank) / 750.0)
        except Exception:
            return 0.0
