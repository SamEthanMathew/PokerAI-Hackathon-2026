"""
Time budget management for v15 — uses full match budget for better equity/discard.
"""

import time


class TimeManager:
    def __init__(self, total_budget: float = 1900.0, num_hands: int = 1000):
        self.total_budget = total_budget
        self.num_hands = num_hands
        self.hands_played = 0
        self.remaining = total_budget
        self.safety_buffer = total_budget * 0.05
        self._hand_start: float | None = None

    def start_hand(self):
        self._hand_start = time.time()

    def stop_hand(self):
        if self._hand_start is not None:
            elapsed = time.time() - self._hand_start
            self.remaining -= elapsed
            self.hands_played += 1
            self._hand_start = None

    def elapsed_this_hand(self) -> float:
        if self._hand_start is None:
            return 0.0
        return time.time() - self._hand_start

    def update_from_observation(self, observation: dict):
        if "time_left" in observation:
            self.remaining = observation["time_left"]

    def get_hand_budget(self) -> float:
        hands_left = max(self.num_hands - self.hands_played, 1)
        available = max(self.remaining - self.safety_buffer, 0.1)
        base = available / hands_left
        return min(base * 2.0, 4.0)

    def get_decision_budget(self, decision_type: str = "bet") -> float:
        hand_budget = self.get_hand_budget()
        budgets = {
            "preflop": min(hand_budget * 0.02, 0.008),
            "discard": min(hand_budget * 0.55, 0.85),
            "bet": min(hand_budget * 0.30, 0.55),
            "river": min(hand_budget * 0.12, 0.15),
        }
        return budgets.get(decision_type, budgets["bet"])

    def should_use_fast_path(self) -> bool:
        hands_left = max(self.num_hands - self.hands_played, 1)
        avg_remaining = self.remaining / hands_left
        return avg_remaining < 0.35

    def is_time_critical(self) -> bool:
        return self.remaining < 35.0
