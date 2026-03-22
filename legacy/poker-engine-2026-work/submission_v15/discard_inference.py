"""
Discard Inference for v14 Alpha — infer opponent weighted range from their 3 discards.
"""

import math
from itertools import combinations

from submission.constants import NUM_CARDS, NUM_RANKS, RANK_A


class DiscardInference:
    TEMPERATURE_MAP = {
        "tag":            1.2,
        "tight_passive":  1.5,
        "unknown":        2.0,
        "calling_station": 3.0,
        "maniac":         3.5,
        "shover":         2.0,
    }

    def __init__(self, equity_engine):
        self.equity_engine = equity_engine

    def infer_opponent_range(
        self,
        opp_discards: list,
        board: list,
        dead_cards: set,
        opp_type: str = "unknown",
    ) -> list:
        """Returns list of (card1, card2, weight) — weighted range."""
        temperature = self.TEMPERATURE_MAP.get(opp_type, 2.0)
        all_known = dead_cards | set(opp_discards)
        unknown = [c for c in range(NUM_CARDS) if c not in all_known]
        possible_keeps = list(combinations(unknown, 2))
        if not possible_keeps:
            return []

        weighted = []
        for keep in possible_keeps:
            weight = self._rationality_weight(keep, opp_discards, board, temperature)
            if weight > 0:
                weighted.append((keep[0], keep[1], weight))

        if not weighted:
            n = len(possible_keeps)
            return [(c1, c2, 1.0 / max(n, 1)) for c1, c2 in possible_keeps]
        total_w = sum(w for _, _, w in weighted)
        if total_w <= 0:
            n = len(weighted)
            return [(c1, c2, 1.0 / max(n, 1)) for c1, c2, _ in weighted]
        return [(c1, c2, w / total_w) for c1, c2, w in weighted]

    def _rationality_weight(self, kept: tuple, discarded: list, board: list, temperature: float) -> float:
        original_five = list(kept) + list(discarded)
        if len(original_five) != 5:
            return 0.0
        card_indices = list(range(5))
        keep_combos = list(combinations(card_indices, 2))
        scores = []
        for kc in keep_combos:
            kc_cards = [original_five[i] for i in kc]
            score = self._quick_score(kc_cards, board)
            scores.append(score)
        kept_set = set(kept)
        kept_score = None
        for i, kc in enumerate(keep_combos):
            kc_cards = set(original_five[j] for j in kc)
            if kc_cards == kept_set:
                kept_score = scores[i]
                break
        if kept_score is None:
            return 0.0
        try:
            max_score = max(scores)
            exp_kept = math.exp((kept_score - max_score) / temperature)
            exp_total = sum(math.exp((s - max_score) / temperature) for s in scores)
            if exp_total == 0:
                return 0.1
            return exp_kept / exp_total
        except (OverflowError, ValueError):
            return 0.1

    def _quick_score(self, two_cards: list, board: list) -> float:
        try:
            score = self.equity_engine.quick_hand_score(two_cards, board)
        except Exception:
            score = 0.0
        c1, c2 = two_cards
        r1, r2 = c1 % NUM_RANKS, c2 % NUM_RANKS
        s1, s2 = c1 // NUM_RANKS, c2 // NUM_RANKS
        if r1 == r2:
            score += 2.5
        if s1 == s2:
            board_suit_count = sum(1 for b in board if b // NUM_RANKS == s1)
            score += 0.5 + board_suit_count * 0.8
        gap = abs(r1 - r2)
        if gap == 1:
            score += 0.6
        elif gap == 2:
            score += 0.3
        if (r1 == RANK_A and r2 == 0) or (r2 == RANK_A and r1 == 0):
            score += 0.4
        score += max(r1, r2) * 0.12
        board_ranks = [b % NUM_RANKS for b in board]
        for r in (r1, r2):
            matches = board_ranks.count(r)
            if matches >= 2:
                score += 5.0
            elif matches == 1:
                score += 2.0
        return score
