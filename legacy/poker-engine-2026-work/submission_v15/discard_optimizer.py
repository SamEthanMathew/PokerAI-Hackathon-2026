"""
Discard Optimizer for v15 — more samples when time budget allows.
"""

import time
from itertools import combinations

from submission.card_utils import CardUtils
from submission.constants import NUM_RANKS, RANK_A


class DiscardOptimizer:
    def __init__(self, equity_engine):
        self.equity_engine = equity_engine

    def choose_best_keep(
        self,
        my_five_cards: list,
        board_three: list,
        opp_discards: list = None,
        opp_range: list = None,
        time_budget: float = 0.4,
    ) -> tuple:
        start = time.time()
        card_indices = list(range(5))
        all_keeps = list(combinations(card_indices, 2))
        heuristic_scores = []
        for keep_pair in all_keeps:
            keep_cards = [my_five_cards[i] for i in keep_pair]
            score = self._heuristic_score(keep_cards, board_three)
            heuristic_scores.append((score, keep_pair))
        heuristic_scores.sort(reverse=True)
        dead_base = set(board_three)
        if opp_discards:
            dead_base |= set(c for c in opp_discards if c != -1)
        best_equity = -1.0
        best_keep = heuristic_scores[0][1]
        max_samples = 2000
        if time_budget >= 0.55:
            max_samples = 4500
        elif time_budget >= 0.35:
            max_samples = 3200
        for _, keep_pair in heuristic_scores:
            if time.time() - start > time_budget:
                break
            keep_cards = [my_five_cards[i] for i in keep_pair]
            discard_indices = [i for i in card_indices if i not in keep_pair]
            discarded = [my_five_cards[i] for i in discard_indices]
            dead = dead_base | set(keep_cards) | set(discarded)
            try:
                equity = self.equity_engine.compute_equity(
                    my_cards=keep_cards,
                    board=board_three,
                    dead_cards=dead,
                    opp_range=opp_range,
                    max_samples=max_samples,
                )
            except Exception:
                equity = 0.0
            if equity > best_equity:
                best_equity = equity
                best_keep = keep_pair
        return best_keep

    def _heuristic_score(self, keep_cards: list, board: list) -> float:
        c1, c2 = keep_cards
        r1, r2 = CardUtils.get_rank(c1), CardUtils.get_rank(c2)
        s1, s2 = CardUtils.get_suit(c1), CardUtils.get_suit(c2)
        score = 0.0
        if r1 == r2:
            score += 15.0 + r1 * 0.8
            board_ranks = [CardUtils.get_rank(c) for c in board]
            if r1 in board_ranks:
                score += 30.0
        board_ranks = [CardUtils.get_rank(c) for c in board]
        for r in (r1, r2):
            board_matches = board_ranks.count(r)
            if board_matches >= 2:
                score += 20.0
            elif board_matches == 1:
                score += 8.0
        unique_ranks = sorted(set(board_ranks + [r1, r2]))
        if RANK_A in unique_ranks:
            ext_ranks = sorted(unique_ranks + [-1, NUM_RANKS])
        else:
            ext_ranks = unique_ranks
        max_run = 1
        current_run = 1
        for i in range(1, len(ext_ranks)):
            if ext_ranks[i] == ext_ranks[i - 1] + 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        if max_run >= 4:
            score += 12.0
        elif max_run == 3:
            score += 4.0
        elif max_run == 2:
            score += 1.0
        if s1 == s2:
            board_suits = [CardUtils.get_suit(c) for c in board]
            suited_on_board = board_suits.count(s1)
            if suited_on_board >= 2:
                score += 10.0
            elif suited_on_board == 1:
                score += 3.0
        score += (r1 + r2) * 0.2
        return score
