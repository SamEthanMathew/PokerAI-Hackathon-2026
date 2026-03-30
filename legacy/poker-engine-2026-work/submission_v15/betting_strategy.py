"""
Betting Strategy for v15 — multi-street required equity, scaled future-cost discount, river thin value.
"""

import random
from collections import Counter

from submission.constants import (
    FOLD, RAISE, CHECK, CALL,
    STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER,
    NUM_RANKS,
)


class BettingStrategy:
    THRESHOLDS = {
        "unknown": (0.75, 0.58, 0.42, 0.25),
        "calling_station": (0.62, 0.45, 0.38, 0.22),
        "tight_passive": (0.72, 0.55, 0.42, 0.20),
        "maniac": (0.68, 0.52, 0.44, 0.30),
        "tag": (0.75, 0.58, 0.42, 0.28),
        "shover": (0.70, 0.52, 0.42, 0.25),
        "nit": (0.78, 0.60, 0.40, 0.22),
        "sticky_station": (0.68, 0.50, 0.40, 0.28),
    }

    def _future_cost_and_discount(self, street: int, pot: float, opp_tendencies: dict):
        turn_bp = opp_tendencies.get("turn_barrel_prob") or 0.5
        river_bp = opp_tendencies.get("river_barrel_prob") or 0.5
        avg_turn = opp_tendencies.get("avg_bet_ratio_turn") or 0.3
        avg_river = opp_tendencies.get("avg_bet_ratio_river") or 0.34
        if street == STREET_FLOP:
            future = turn_bp * avg_turn * pot + turn_bp * river_bp * avg_river * pot
        else:
            future = river_bp * avg_river * pot
        discount = min(0.08, 0.02 + (future / max(pot, 1.0)) * 0.15)
        return future, discount

    def _multi_street_should_fold(
        self, equity: float, pot: float, to_call: float, street: int, opp_tendencies: dict
    ) -> bool:
        if to_call <= pot * 0.18:
            return False
        future, _ = self._future_cost_and_discount(street, pot, opp_tendencies or {})
        eff_pot = pot + to_call + future
        if eff_pot <= 0:
            return False
        margin = 0.045
        req = to_call / eff_pot + margin
        return equity < req

    def decide(
        self,
        equity: float,
        observation: dict,
        opp_tendencies: dict = None,
        variance_profile: dict = None,
        street: int = 0,
    ) -> tuple:
        valid = observation["valid_actions"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        pot = my_bet + opp_bet
        to_call = opp_bet - my_bet
        min_raise = observation.get("min_raise", 2)
        max_raise = observation.get("max_raise", 0)
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
        opp_type = (opp_tendencies or {}).get("type", "unknown")
        variance_profile = variance_profile or {}
        strong_t, value_t, marginal_t, weak_t = self.THRESHOLDS.get(
            opp_type, self.THRESHOLDS["unknown"]
        )
        r_shift = variance_profile.get("raise_threshold_shift", 0.0)
        c_shift = variance_profile.get("call_threshold_shift", 0.0)
        strong_t += r_shift
        value_t += r_shift
        marginal_t += c_shift
        weak_t += c_shift
        bluff_mult = variance_profile.get("bluff_multiplier", 1.0)
        size_mult = variance_profile.get("raise_size_multiplier", 1.0)

        if street > STREET_PREFLOP:
            penalty = self._board_danger_penalty(observation)
            equity = max(0.0, equity - penalty)

        if 0 < to_call <= pot * 0.20 and street > STREET_PREFLOP:
            if equity >= 0.22 and valid[CALL]:
                return (CALL, 0, 0, 0)
            if to_call <= 3 and equity >= 0.18 and valid[CALL]:
                return (CALL, 0, 0, 0)

        if to_call >= 50 and street > STREET_PREFLOP:
            return self._handle_facing_big_bet(equity, to_call, pot, pot_odds, valid, opp_type)

        if to_call > 0 and street in (STREET_FLOP, STREET_TURN) and opp_tendencies:
            future, disc = self._future_cost_and_discount(street, pot, opp_tendencies)
            if future > pot * 0.08:
                equity = max(0.0, equity - disc)
            if self._multi_street_should_fold(equity, pot, to_call, street, opp_tendencies):
                return self._try_fold(valid)

        if street == STREET_RIVER:
            return self._river_decision(
                equity, pot, to_call, pot_odds, min_raise, max_raise,
                valid, observation, opp_type, opp_tendencies, bluff_mult, size_mult
            )

        if equity >= strong_t:
            action = self._value_bet(pot, min_raise, max_raise, valid, observation, opp_type, size_mult)
            if action[0] == RAISE:
                return action
            return self._try_call(valid)
        elif equity >= value_t:
            if to_call == 0:
                bet = self._calc_bet(pot, min_raise, max_raise, "value", opp_type, size_mult)
                return self._try_raise(bet, valid, observation)
            if equity > pot_odds + 0.05:
                return self._try_call(valid)
            return self._try_check_fold(valid)
        elif equity >= marginal_t:
            if to_call == 0:
                if self._should_bluff(pot, to_call, opp_tendencies, street, bluff_mult):
                    bet = self._calc_bet(pot, min_raise, max_raise, "bluff", opp_type, size_mult)
                    return self._try_raise(bet, valid, observation)
                return self._try_check(valid)
            if to_call <= pot * 0.35:
                if equity > pot_odds:
                    return self._try_call(valid)
                return self._try_check_fold(valid)
            return self._try_check_fold(valid)
        elif equity >= weak_t:
            if to_call == 0:
                return self._try_check(valid)
            if to_call <= 2 and pot >= 10:
                return self._try_call(valid)
            return self._try_fold(valid)
        else:
            if to_call == 0:
                return self._try_check(valid)
            return self._try_fold(valid)

    def _board_danger_penalty(self, observation: dict) -> float:
        community = [c for c in observation.get("community_cards", []) if c != -1]
        if len(community) < 3:
            return 0.0
        penalty = 0.0
        board_ranks = [c % NUM_RANKS for c in community]
        rank_counts = Counter(board_ranks)
        if any(v >= 3 for v in rank_counts.values()):
            penalty += 0.15
        elif any(v >= 2 for v in rank_counts.values()):
            penalty += 0.10 if len([v for v in rank_counts.values() if v >= 2]) >= 2 else 0.05
        board_suits = [c // NUM_RANKS for c in community]
        suit_counts = Counter(board_suits)
        if any(v >= 4 for v in suit_counts.values()):
            penalty += 0.10
        elif any(v >= 3 for v in suit_counts.values()):
            penalty += 0.04
        return penalty

    def _handle_facing_big_bet(self, equity, to_call, pot, pot_odds, valid, opp_type):
        if opp_type == "shover":
            threshold = max(pot_odds + 0.05, 0.40)
        elif opp_type == "nit":
            threshold = max(pot_odds + 0.12, 0.52)
        elif opp_type == "sticky_station":
            threshold = max(pot_odds + 0.06, 0.48)
        else:
            threshold = max(pot_odds + 0.08, 0.50)
        if equity >= threshold:
            return self._try_call(valid)
        return self._try_fold(valid)

    def _river_decision(self, equity, pot, to_call, pot_odds, min_raise, max_raise,
                        valid, observation, opp_type, opp_tendencies, bluff_mult=1.0, size_mult=1.0):
        if equity >= 0.80:
            bet = self._calc_bet(pot, min_raise, max_raise, "shove" if pot > 30 else "value", opp_type, size_mult)
            return self._try_raise(bet, valid, observation)
        elif equity >= 0.55:
            if to_call == 0:
                if 0.55 <= equity < 0.70 and opp_type not in ("nit", "tight_passive", "calling_station"):
                    thin = max(min_raise, min(int(pot * 0.40 * size_mult), max_raise))
                    if valid[RAISE] and thin <= max_raise:
                        return (RAISE, thin, 0, 0)
                bet = self._calc_bet(pot, min_raise, max_raise, "value" if opp_type == "calling_station" else "probe", opp_type, size_mult)
                return self._try_raise(bet, valid, observation)
            call_margin = 0.03
            if opp_type == "calling_station":
                call_margin = 0.01
            elif opp_type == "nit":
                call_margin = 0.06
            if equity > pot_odds + call_margin:
                return self._try_call(valid)
            return self._try_check_fold(valid)
        elif equity >= 0.35:
            if to_call == 0:
                return self._try_check(valid)
            river_margin = -0.05
            if opp_type == "nit":
                river_margin = 0.02
            elif opp_type in ("calling_station", "sticky_station"):
                river_margin = -0.08
            if to_call > 0 and equity > pot_odds + river_margin:
                return self._try_call(valid)
            return self._try_fold(valid)
        else:
            if to_call == 0 and self._should_bluff(pot, to_call, opp_tendencies, STREET_RIVER, bluff_mult):
                bet = self._calc_bet(pot, min_raise, max_raise, "bluff", opp_type, size_mult)
                return self._try_raise(bet, valid, observation)
            if to_call == 0:
                return self._try_check(valid)
            if to_call <= 3 and pot >= 8:
                return self._try_call(valid)
            return self._try_fold(valid)

    def _should_bluff(self, pot, to_call, opp_tendencies, street, bluff_mult=1.0):
        if not opp_tendencies or opp_tendencies.get("hands_seen", 0) < 30:
            return False
        opp_type = opp_tendencies.get("type", "unknown")
        if opp_type in ("calling_station", "shover", "sticky_station"):
            return False
        if opp_type == "tight_passive":
            bluff_mult *= 1.2
        fold_rates = opp_tendencies.get("fold_to_raise", {})
        fold_rate = fold_rates.get(street, 0.4)
        bet_size = pot * 0.5
        ev = fold_rate * pot - (1 - fold_rate) * bet_size
        ev *= bluff_mult
        return ev > 0 and fold_rate > 0.40

    def _calc_bet(self, pot, min_raise, max_raise, mode="value", opp_type="unknown", size_mult=1.0):
        if mode == "value":
            target = int(pot * (0.75 if opp_type == "calling_station" else 0.60))
        elif mode == "probe":
            target = int(pot * 0.30)
        elif mode == "bluff":
            target = int(pot * (0.35 if opp_type == "tight_passive" else 0.55))
        elif mode == "shove":
            return max_raise
        else:
            target = int(pot * 0.50)
        target = int(target * size_mult)
        return max(min_raise, min(target, max_raise))

    def _value_bet(self, pot, min_raise, max_raise, valid, obs, opp_type="unknown", size_mult=1.0):
        bet = self._calc_bet(pot, min_raise, max_raise, "value", opp_type, size_mult)
        if pot > 30 and opp_type == "calling_station":
            bet = max(min_raise, min(int(pot * 0.85 * size_mult), max_raise))
        result = self._try_raise(bet, valid, obs)
        if result[0] == RAISE:
            return result
        return self._try_call(valid)

    def _try_raise(self, amount, valid, obs):
        if valid[RAISE]:
            amount = max(obs["min_raise"], min(amount, obs["max_raise"]))
            return (RAISE, amount, 0, 0)
        return self._try_call(valid)

    def _try_call(self, valid):
        if valid[CALL]:
            return (CALL, 0, 0, 0)
        return self._try_check(valid)

    def _try_check(self, valid):
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        if valid[CALL]:
            return (CALL, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    def _try_fold(self, valid):
        if valid[FOLD]:
            return (FOLD, 0, 0, 0)
        return self._try_check(valid)

    def _try_check_fold(self, valid):
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)
