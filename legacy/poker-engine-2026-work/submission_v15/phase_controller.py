"""
Match Phase Controller for v14 Alpha — scout/exploit/aggro/closer + variance knobs.
"""


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(val, hi))


class PhaseController:
    def __init__(self, match_max_hands: int = 1000):
        self.match_max_hands = match_max_hands
        self.my_bankroll = 0
        self.opp_bankroll = 0

    def update_bankrolls(self, my_bankroll: int, opp_bankroll: int):
        self.my_bankroll = my_bankroll
        self.opp_bankroll = opp_bankroll

    def get_phase(self, hand_number: int) -> str:
        """Returns scout | exploit | aggro | closer | closer_winning | closer_losing."""
        h = hand_number
        b = self.my_bankroll - self.opp_bankroll
        remaining = max(1, self.match_max_hands - h)

        if h < 150:
            phase = "scout"
            if b <= -300 and h >= 50:
                phase = "exploit"
        elif h < 500:
            phase = "exploit"
        elif h < 750:
            phase = "aggro"
            if b >= 500:
                phase = "closer"
        else:
            phase = "closer"

        if phase == "closer":
            if b > 0:
                phase = "closer_winning"
            else:
                phase = "closer_losing"
        return phase

    def get_variance_profile(self, hand_number: int) -> dict:
        """Variance knobs: raise_threshold_shift, call_threshold_shift, bluff_multiplier, raise_size_multiplier. Applied from hand 0."""
        hands_remaining = max(1, self.match_max_hands - hand_number)
        blind_budget = max(30.0, 1.5 * hands_remaining)
        lead_ratio = (self.my_bankroll - self.opp_bankroll) / blind_budget
        late_factor = clamp((500 - hands_remaining) / 250.0, 0.0, 1.0)
        protect = late_factor * clamp((lead_ratio - 0.12) / 0.55, 0.0, 1.0)
        press = late_factor * clamp(((-lead_ratio) - 0.12) / 0.55, 0.0, 1.0)
        raise_threshold_shift = 0.06 * protect - 0.05 * press
        call_threshold_shift = 0.03 * protect - 0.02 * press
        bluff_mult = clamp(1.0 + 0.2 * press - 0.15 * protect, 0.7, 1.3)
        size_mult = clamp(1.0 + 0.1 * press - 0.08 * protect, 0.85, 1.15)
        return {
            "raise_threshold_shift": raise_threshold_shift,
            "call_threshold_shift": call_threshold_shift,
            "bluff_multiplier": bluff_mult,
            "raise_size_multiplier": size_mult,
        }
