"""
Opponent Model for v15 — stats, classification including nit and sticky_station.
"""

from submission.constants import STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER


class OpponentModel:
    def __init__(self):
        self.hands_seen = 0
        self.vpip = 0
        self.pfr = 0
        self.fold_to_raise = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}
        self.bet_raise_count = 0
        self.call_count = 0
        self.check_count = 0
        self.showdown_hands = []
        self.raise_sizes = {0: [], 1: [], 2: [], 3: []}
        self.preflop_allin_count = 0
        self.preflop_big_raise_count = 0
        self.postflop_allin_count = 0
        self._current_hand_vpip = False
        self._current_hand_pfr = False
        self._last_opp_action = None
        self._opp_raised_this_street = False

    def update_action(self, observation: dict, opp_action_str: str):
        if opp_action_str in (None, "None", "DISCARD"):
            return
        street = observation.get("street", 0)
        if opp_action_str == "RAISE":
            self.bet_raise_count += 1
            self._opp_raised_this_street = True
            if street == 0:
                self._current_hand_pfr = True
                self._current_hand_vpip = True
            opp_bet = observation.get("opp_bet", 0)
            self.raise_sizes[street].append(opp_bet)
            if street == 0:
                if opp_bet >= 50:
                    self.preflop_allin_count += 1
                elif opp_bet >= 20:
                    self.preflop_big_raise_count += 1
            else:
                my_bet = observation.get("my_bet", 0)
                if opp_bet >= 80:
                    self.postflop_allin_count += 1
        elif opp_action_str == "CALL":
            self.call_count += 1
            if street == 0:
                self._current_hand_vpip = True
        elif opp_action_str == "CHECK":
            self.check_count += 1
        elif opp_action_str == "FOLD":
            self.fold_to_raise[street][0] += 1
        self._last_opp_action = opp_action_str

    def record_raise_opportunity(self, street: int):
        self.fold_to_raise[street][1] += 1

    def end_hand(self, info: dict = None):
        self.hands_seen += 1
        if self._current_hand_vpip:
            self.vpip += 1
        if self._current_hand_pfr:
            self.pfr += 1
        if info and "player_0_cards" in info:
            self.showdown_hands.append(info)
        self._current_hand_vpip = False
        self._current_hand_pfr = False
        self._last_opp_action = None
        self._opp_raised_this_street = False

    def get_vpip_rate(self) -> float:
        return self.vpip / self.hands_seen if self.hands_seen else 0.5

    def get_pfr_rate(self) -> float:
        return self.pfr / self.hands_seen if self.hands_seen else 0.3

    def get_aggression_factor(self) -> float:
        if self.call_count == 0:
            return 3.0 if self.bet_raise_count > 0 else 1.0
        return self.bet_raise_count / self.call_count

    def get_fold_to_raise_rate(self, street: int) -> float:
        folds, opps = self.fold_to_raise[street]
        return folds / opps if opps >= 5 else 0.4

    def get_avg_raise_size(self, street: int) -> float:
        sizes = self.raise_sizes[street]
        return sum(sizes) / len(sizes) if sizes else 4.0

    def get_preflop_allin_rate(self) -> float:
        return self.preflop_allin_count / self.hands_seen if self.hands_seen >= 10 else 0.0

    def is_shover(self) -> bool:
        return self.hands_seen >= 15 and self.get_preflop_allin_rate() > 0.05

    def classify(self) -> str:
        if self.is_shover():
            return "shover"
        if self.hands_seen < 30:
            return "unknown"
        vpip = self.get_vpip_rate()
        af = self.get_aggression_factor()
        ftr_flop_f, ftr_flop_o = self.fold_to_raise[1]
        ftr_flop = ftr_flop_f / ftr_flop_o if ftr_flop_o >= 8 else None

        if vpip < 0.38 and ftr_flop is not None and ftr_flop > 0.52:
            return "nit"
        if vpip > 0.62 and ftr_flop is not None and ftr_flop < 0.38:
            return "sticky_station"
        if vpip > 0.62 and ftr_flop_o < 8 and vpip > 0.68 and af > 1.0:
            return "sticky_station"

        if vpip > 0.65 and af < 0.8:
            return "calling_station"
        elif vpip < 0.35 and af < 1.0:
            return "tight_passive"
        elif vpip > 0.65 and af > 2.0:
            return "maniac"
        elif vpip < 0.45 and af > 1.5:
            return "tag"
        return "unknown"

    def get_tendencies(self) -> dict:
        return {
            "type": self.classify(),
            "vpip": self.get_vpip_rate(),
            "pfr": self.get_pfr_rate(),
            "af": self.get_aggression_factor(),
            "fold_to_raise": {s: self.get_fold_to_raise_rate(s) for s in range(4)},
            "hands_seen": self.hands_seen,
            "preflop_allin_rate": self.get_preflop_allin_rate(),
            "is_shover": self.is_shover(),
        }
