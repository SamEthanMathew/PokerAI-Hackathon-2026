"""
Preflop Strategy for v15 — preflop table + dynamic thresholds vs opponent and facing raises.
"""

import bisect
from itertools import combinations

from submission.card_utils import CardUtils
from submission.constants import FOLD, RAISE, CHECK, CALL, RANK_A, NUM_RANKS


class PreFlopStrategy:
    def __init__(self, preflop_table: dict = None, preflop_sorted_values: list = None):
        self.preflop_table = preflop_table if preflop_table else {}
        self._preflop_sorted_values = preflop_sorted_values if preflop_sorted_values else []

    def _dynamic_thresholds(self, observation: dict, opp_tendencies: dict, variance_profile: dict):
        r_shift = (variance_profile or {}).get("raise_threshold_shift", 0.0)
        raise_t = 0.8 - r_shift
        call_t = 0.65
        limp_t = 0.3
        ot = opp_tendencies or {}
        hands = ot.get("hands_seen", 0)
        ftr0 = ot.get("fold_to_raise", {}).get(0, 0.4)
        vpip = ot.get("vpip", 0.5)
        if hands >= 25 and ftr0 > 0.52:
            raise_t -= 0.03
        if vpip > 0.62:
            call_t -= 0.025
            limp_t -= 0.02
        to_call = observation["opp_bet"] - observation["my_bet"]
        if to_call >= 14:
            raise_t = max(raise_t, 0.88)
            call_t = max(call_t, 0.76)
        elif to_call >= 8:
            raise_t = max(raise_t, 0.84)
            call_t = max(call_t, 0.72)
        elif to_call >= 5:
            call_t = max(call_t, 0.68)
        return raise_t, call_t, limp_t

    def decide(
        self,
        my_five_cards: list,
        observation: dict,
        opp_tendencies: dict = None,
        bankroll: int = 0,
        hands_remaining: int = 500,
        phase: str = "exploit",
        variance_profile: dict = None,
    ) -> tuple:
        valid = observation["valid_actions"]
        strength = self._preflop_strength(my_five_cards)
        min_raise = observation.get("min_raise", 2)
        max_raise = observation.get("max_raise", 0)
        to_call = observation["opp_bet"] - observation["my_bet"]
        is_bb = observation.get("blind_position", 0) == 1
        variance_profile = variance_profile or {}
        raise_t, call_t, limp_t = self._dynamic_thresholds(
            observation, opp_tendencies, variance_profile
        )

        if self._should_desperation_shove(bankroll, hands_remaining, strength):
            if valid[RAISE] and max_raise >= 50:
                return (RAISE, max_raise, 0, 0)

        if to_call >= 50:
            return self._handle_facing_allin(strength, to_call, valid, opp_tendencies)

        if strength >= raise_t:
            raise_amt = self._raise_size(strength, min_raise, max_raise, observation, phase)
            if valid[RAISE]:
                return (RAISE, raise_amt, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            return (CHECK, 0, 0, 0)
        elif strength >= call_t:
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if valid[RAISE]:
                return (RAISE, min_raise, 0, 0)
            return (FOLD, 0, 0, 0)
        elif strength >= limp_t:
            if is_bb:
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
                if valid[CALL] and to_call <= 4:
                    return (CALL, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)
        else:
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

    def _handle_facing_allin(self, strength, to_call, valid, opp_tendencies):
        is_shover = (opp_tendencies or {}).get("is_shover", False)
        if is_shover:
            threshold = 0.80 if to_call >= 90 else 0.70
        else:
            threshold = 0.90 if to_call >= 90 else 0.80
        if strength >= threshold and valid[CALL]:
            return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _should_desperation_shove(self, bankroll: int, hands_remaining: int, strength: float) -> bool:
        if hands_remaining <= 0 or bankroll >= 0:
            return False
        deficit = abs(bankroll)
        if deficit >= 600 and hands_remaining < 100:
            return strength >= 0.40
        if deficit >= 400 and hands_remaining < 150:
            return strength >= 0.50
        if deficit >= 200 and hands_remaining < 200:
            return strength >= 0.60
        if deficit >= 100 and hands_remaining < 50:
            return strength >= 0.55
        return False

    def _raise_size(self, strength: float, min_raise: int, max_raise: int, observation: dict, phase: str) -> int:
        pot = observation.get("pot_size", 3)
        if strength >= 0.98:
            return max_raise
        if strength >= 0.90:
            return max(min_raise + 8, min(int(pot * 2.0), max_raise))
        if strength >= 0.80:
            return max(min_raise + 4, min(int(pot * 1.2), max_raise))
        return min_raise

    def _preflop_strength(self, cards: list) -> float:
        if len(cards) == 2 and self.preflop_table and self._preflop_sorted_values:
            key = tuple(sorted(int(c) for c in cards))
            equity = self.preflop_table.get(key)
            if equity is not None:
                rank = bisect.bisect_left(self._preflop_sorted_values, equity)
                percentile = rank / len(self._preflop_sorted_values)
                return 0.35 + percentile * 0.63
        return self._hand_strength(cards)

    def _hand_strength(self, five_cards: list) -> float:
        best = 0.0
        for keep in combinations(range(len(five_cards)), 2):
            c1, c2 = five_cards[keep[0]], five_cards[keep[1]]
            score = self._two_card_score(c1, c2)
            if score > best:
                best = score
        return min(best, 1.0)

    def _two_card_score(self, c1: int, c2: int) -> float:
        r1, r2 = CardUtils.get_rank(c1), CardUtils.get_rank(c2)
        s1, s2 = CardUtils.get_suit(c1), CardUtils.get_suit(c2)
        score = 0.0
        if r1 == r2:
            rank = r1
            base = 0.45 if rank <= 3 else (0.55 if rank <= 6 else 0.65)
            score += base + rank * 0.04
        if s1 == s2:
            score += 0.15
        gap = abs(r1 - r2)
        if gap == 1:
            score += 0.12
        elif gap == 2:
            score += 0.06
        if (r1 == RANK_A and r2 == 0) or (r2 == RANK_A and r1 == 0):
            score += 0.10
        high = max(r1, r2)
        if high == RANK_A:
            score += 0.18
        elif high == 7:
            score += 0.10
        elif high >= 4:
            score += 0.06
        return score
