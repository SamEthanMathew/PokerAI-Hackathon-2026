"""
Bet Sizing Tracker for v14 Alpha — optional; tracks opponent first-action sizing and barrel tendency.
"""


class BetSizingTracker:
    """Tracks opponent postflop first-action bet ratio and multi-street barrels."""

    DEFAULT_BET_RATIO_PRIORS = {1: 0.42, 2: 0.30, 3: 0.34}
    DEFAULT_BARREL_PRIORS = {2: 0.56, 3: 0.50}

    def __init__(self):
        self.current_hand_number = -1
        self._seen_street_action_this_hand = set()
        self._aggressive_streets_this_hand = set()
        self.street_first_bet_ratios = {1: [], 2: [], 3: []}
        self.barrel_opportunities = {2: 0, 3: 0}
        self.barrel_count = {2: 0, 3: 0}

    def start_new_hand(self, hand_number: int):
        self.current_hand_number = hand_number
        self._seen_street_action_this_hand = set()
        self._aggressive_streets_this_hand = set()

    def record_observation(self, observation: dict, info: dict = None):
        hand_number = int(info.get("hand_number", self.current_hand_number)) if info else self.current_hand_number
        if hand_number != self.current_hand_number:
            self.start_new_hand(hand_number)
        street = int(observation.get("street", 0))
        if street < 1:
            return
        opp_last = observation.get("opp_last_action", "")
        my_bet = int(observation.get("my_bet", 0))
        opp_bet = int(observation.get("opp_bet", 0))
        pot_before = my_bet + opp_bet
        if street not in self._seen_street_action_this_hand and opp_last == "RAISE":
            self._seen_street_action_this_hand.add(street)
            self._aggressive_streets_this_hand.add(street)
            if pot_before > 0:
                increment = opp_bet - my_bet
                self.street_first_bet_ratios[street].append(increment / pot_before)
        if street in (2, 3) and (street - 1) in self._aggressive_streets_this_hand:
            self.barrel_opportunities[street] += 1
            if opp_last == "RAISE":
                self.barrel_count[street] += 1

    def get_turn_barrel_probability(self) -> float:
        opps = self.barrel_opportunities.get(2, 0)
        if opps < 5:
            return self.DEFAULT_BARREL_PRIORS.get(2, 0.5)
        return self.barrel_count.get(2, 0) / opps

    def get_river_barrel_probability(self) -> float:
        opps = self.barrel_opportunities.get(3, 0)
        if opps < 5:
            return self.DEFAULT_BARREL_PRIORS.get(3, 0.5)
        return self.barrel_count.get(3, 0) / opps

    def get_avg_bet_ratio(self, street: int, prior: float = 0.4) -> float:
        ratios = self.street_first_bet_ratios.get(street, [])
        if not ratios:
            return prior
        return (sum(ratios) + prior * 4) / (len(ratios) + 4)
