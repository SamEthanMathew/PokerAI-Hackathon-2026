"""
Real-time Monte Carlo for critical decisions.

Time budget: 1500 seconds / 1000 hands = 1.5s average per hand.
Most hands are simple (fold preflop, small pots) and take <5ms.
This leaves a TIME BANK of surplus seconds for critical decisions.

Strategy:
  - Track remaining time budget
  - For small pots (<10 chips): use neural net only (~2ms)
  - For medium pots (10-30 chips): run 1000 MC simulations (~50ms)
  - For large pots (30+ chips) or all-in: run 10,000 MC sims (~200ms)
  - Never exceed 500ms on any single decision
  - Reserve 100 seconds as safety buffer
"""

import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Repo root for gym_env
_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_BOT_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gym_env import PokerEnv, WrappedEval

MAX_DECISION_MS = 0.5  # Never exceed 500ms on any single decision
DECK_SIZE = 27


class RealtimeSearch:
    def __init__(
        self,
        evaluator: Optional[WrappedEval] = None,
        equity_table: Optional[Any] = None,
        range_table: Optional[Any] = None,
        time_budget: float = 1500.0,
        safety_buffer: float = 100.0,
    ):
        self.evaluator = evaluator or WrappedEval()
        self.equity_table = equity_table
        self.range_table = range_table
        self.time_budget = time_budget
        self.time_used = 0.0
        self.safety_buffer = safety_buffer
        self.hands_remaining = 1000

    def consume_hand(self) -> None:
        """Call when a hand terminates (from observe(terminated=True))."""
        self.hands_remaining = max(0, self.hands_remaining - 1)

    def should_search(self, pot_size: float, my_stack: float) -> Tuple[bool, int]:
        """
        Decide whether to spend time on MC search.
        Returns (do_search, num_sims). num_sims capped so we don't exceed 500ms.
        """
        available = self.time_budget - self.time_used - self.safety_buffer
        if available <= 0:
            return False, 0
        avg_remaining = available / max(self.hands_remaining, 1)

        # Always search on all-in decisions
        if my_stack <= pot_size:
            n = min(10000, int(avg_remaining * 50000))
            n = min(n, 10000)  # cap sims
            return True, max(100, n)

        # Search on large pots
        if pot_size > 30:
            n = min(5000, int(avg_remaining * 25000))
            return True, max(100, n)

        # Search on medium pots if we have time
        if pot_size > 10 and avg_remaining > 0.05:
            return True, 1000

        return False, 0

    def monte_carlo_equity(
        self,
        my_cards: List[int],
        community: List[int],
        known_removed: List[int],
        opp_range: Optional[Dict[Tuple[int, ...], float]] = None,
        num_sims: int = 1000,
        max_time_sec: float = MAX_DECISION_MS,
    ) -> float:
        """
        Run Monte Carlo simulations to estimate equity.
        Card-removal-aware: only sample from cards not in my_cards, community, or known_removed.
        opp_range: optional dict (hand_tuple -> probability); if None, opponent is uniform random.
        Stops early if max_time_sec (default 0.5s) is exceeded.
        """
        start = time.time()
        used = set(my_cards) | set(community) | set(c for c in known_removed if c >= 0)
        remaining_deck = [c for c in range(DECK_SIZE) if c not in used]

        int_to_card = PokerEnv.int_to_card
        ev = self.evaluator

        wins = 0.0
        total = 0

        for _ in range(num_sims):
            if time.time() - start >= max_time_sec:
                break
            opp_hand = self._sample_opponent_hand(remaining_deck, opp_range)
            if opp_hand is None:
                continue

            deck_minus_opp = [c for c in remaining_deck if c not in opp_hand]
            cards_needed = 5 - len(community)
            if cards_needed > 0:
                if len(deck_minus_opp) < cards_needed:
                    continue
                board_completion = random.sample(deck_minus_opp, cards_needed)
                full_board_ints = list(community) + board_completion
            else:
                full_board_ints = list(community)

            my_treys = [int_to_card(c) for c in my_cards]
            opp_treys = [int_to_card(c) for c in opp_hand]
            board_treys = [int_to_card(c) for c in full_board_ints]

            my_rank = ev.evaluate(my_treys, board_treys)
            opp_rank = ev.evaluate(opp_treys, board_treys)

            if my_rank < opp_rank:
                wins += 1.0
            elif my_rank == opp_rank:
                wins += 0.5
            total += 1

        elapsed = time.time() - start
        self.time_used += elapsed

        return wins / max(total, 1)

    def _sample_opponent_hand(
        self,
        remaining: List[int],
        opp_range: Optional[Dict[Tuple[int, ...], float]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Sample 2 cards from remaining. If opp_range given, weight by range (keys are (c1,c2) with c1<c2)."""
        if opp_range and len(opp_range) > 0:
            hands = list(opp_range.keys())
            probs = list(opp_range.values())
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]
                idx = random.choices(range(len(hands)), weights=probs, k=1)[0]
                h = hands[idx]
                if len(h) >= 2 and h[0] in remaining and h[1] in remaining:
                    return (h[0], h[1])

        if len(remaining) >= 2:
            return tuple(random.sample(remaining, 2))
        return None

    def override_neural_net(
        self,
        nn_action: str,
        nn_confidence: float,
        mc_equity: float,
        pot_size: float,
    ) -> str:
        """
        Compare neural net's suggestion with MC equity.
        Override if there's significant disagreement on big pots.
        """
        if nn_action == "fold" and mc_equity > 0.40 and pot_size > 15:
            return "call"
        if nn_action == "raise" and mc_equity < 0.25 and pot_size > 20:
            return "fold"
        if nn_action == "call" and mc_equity > 0.70 and pot_size > 20:
            return "raise"
        return nn_action


def action_type_to_str(action_type: int) -> str:
    """Map action_type int to string for override_neural_net."""
    return ["fold", "raise", "check", "call"][action_type] if 0 <= action_type < 4 else "fold"


def str_to_action_type(s: str, min_raise: int, max_raise: int) -> Tuple[int, int]:
    """Map override result back to (action_type, raise_amount)."""
    a = s.lower()
    if a == "fold":
        return (0, 0)
    if a == "raise":
        return (1, min_raise)  # default to min_raise; bot may apply position scaling
    if a == "call":
        return (3, 0)
    if a == "check":
        return (2, 0)
    return (0, 0)
