"""
Action and card abstraction for 27-card poker.

- Action abstraction: limit raise amounts to a small set (e.g. pot fractions or fixed sizes).
- Card abstraction: bucket 5-card hands (preflop/discard) and 2-card hands (post-discard) by strength.
"""

from typing import List, Tuple, Dict, Optional
from itertools import combinations

from gym_env import PokerEnv
from .game_model import DECK_SIZE, MAX_BET, BIG_BLIND


# ---- Action abstraction ----

# Default abstract raise sizes (as total bet or increment; we use total bet for clarity in lookup)
DEFAULT_RAISE_SIZES = [2, 4, 8, 16, 32, 50, 100]


def get_abstract_raise_amounts(
    min_raise: int,
    max_raise: int,
    pot_size: int,
    custom_sizes: Optional[List[int]] = None,
) -> List[int]:
    """
    Return list of abstract raise *increments* in [min_raise, max_raise].
    Total bet after raising = opponent_bet + chosen_increment.
    """
    if custom_sizes:
        sizes = [s for s in custom_sizes if min_raise <= s <= max_raise]
        if sizes:
            return sorted(set(sizes))
    # Pot fractions: 0.5p, 1p, 2p, all-in
    if pot_size <= 0:
        pot_size = 4  # preflop default
    candidates = [
        min_raise,
        min_raise + (pot_size // 2),
        min_raise + pot_size,
        min_raise + 2 * pot_size,
        max_raise,
    ]
    candidates = [c for c in candidates if min_raise <= c <= max_raise]
    for s in DEFAULT_RAISE_SIZES:
        if min_raise <= s <= max_raise and s not in candidates:
            candidates.append(s)
    return sorted(set(candidates))


def round_raise_to_abstract(amount: int, abstract_amounts: List[int]) -> int:
    """Round a concrete raise amount to nearest in-abstraction size."""
    if not abstract_amounts:
        return amount
    best = abstract_amounts[0]
    for a in abstract_amounts:
        if abs(a - amount) < abs(best - amount):
            best = a
    return best


def is_raise_in_abstraction(amount: int, abstract_amounts: List[int], tolerance: int = 0) -> bool:
    """Check if raise amount is in (or within tolerance of) the abstraction."""
    if tolerance == 0:
        return amount in abstract_amounts
    for a in abstract_amounts:
        if abs(a - amount) <= tolerance:
            return True
    return False


# ---- Card abstraction (bucketing) ----

def _hand_strength_5card(cards: Tuple[int, ...], evaluator, board: Optional[List[int]] = None) -> float:
    """Crude strength for 5 cards: average rank index (high card) or use evaluator if board given."""
    if len(cards) != 5:
        return 0.0
    # Without board we use average rank (2=0 .. A=8) as proxy
    ranks = [c % 9 for c in cards]
    return sum(ranks) / 5.0 + (max(ranks) * 0.1)


def _hand_strength_2card(cards: Tuple[int, int], evaluator, board: List[int]) -> float:
    """Strength of 2-card hand on given board: use evaluator rank (lower = better)."""
    board = [c for c in board if c >= 0]
    if len(board) < 3:
        ranks = [cards[0] % 9, cards[1] % 9]
        return -(sum(ranks) + max(ranks) * 2)  # negate so higher rank = stronger
    hand = list(map(PokerEnv.int_to_card, list(cards)))
    b = list(map(PokerEnv.int_to_card, board))
    rank = evaluator.evaluate(hand, b)
    return -rank  # lower rank = better hand, so we negate for "strength"


class CardAbstraction:
    """
    Bucket 5-card hands (preflop) and 2-card hands (post-discard) by strength.
    Uses a fixed number of buckets; hands in same bucket are treated identically in CFR.
    """

    def __init__(self, num_buckets_5: int = 50, num_buckets_2: int = 20, evaluator=None):
        self.num_buckets_5 = num_buckets_5
        self.num_buckets_2 = num_buckets_2
        self.evaluator = evaluator or PokerEnv().evaluator
        self._bucket_5_cache: Dict[Tuple[int, ...], int] = {}
        self._bucket_2_cache: Dict[Tuple[Tuple[int, int], Tuple[int, ...]], int] = {}

    def bucket_5card(self, cards: Tuple[int, ...]) -> int:
        """Bucket a 5-card hand (preflop). Cards must be 5 indices from 0..26."""
        if len(cards) != 5:
            cards = tuple(sorted(cards)[:5]) if len(cards) >= 5 else (0, 0, 0, 0, 0)
        key = tuple(sorted(cards))
        if key in self._bucket_5_cache:
            return self._bucket_5_cache[key]
        strength = _hand_strength_5card(key, self.evaluator, None)
        # Simple linear bucket from 0 to num_buckets_5-1
        b = int((strength / 10.0) * self.num_buckets_5) % self.num_buckets_5
        if b < 0:
            b = 0
        self._bucket_5_cache[key] = b
        return b

    def bucket_2card(self, cards: Tuple[int, int], board: Tuple[int, ...]) -> int:
        """Bucket a 2-card hand given the visible board (3, 4, or 5 cards)."""
        board_list = tuple(c for c in board if c >= 0)
        key = (tuple(sorted(cards)), board_list)
        if key in self._bucket_2_cache:
            return self._bucket_2_cache[key]
        strength = _hand_strength_2card(cards, self.evaluator, [c for c in board_list if c >= 0])
        b = int((strength + 10000) / 1000 * self.num_buckets_2) % self.num_buckets_2
        if b < 0:
            b = 0
        self._bucket_2_cache[key] = b
        return b

    def bucket_hand_for_infoset(
        self,
        cards: Tuple[int, ...],
        street: int,
        board: Tuple[int, ...],
        discard_done: bool,
    ) -> int:
        """
        Return bucket id for current infoset: 5-card bucket preflop or before discard,
        2-card bucket after discard (given board).
        """
        if street == 0 or (street == 1 and not discard_done):
            if len(cards) >= 5:
                return self.bucket_5card(tuple(cards[:5]))
            return self.bucket_5card(tuple(list(cards) + [0] * (5 - len(cards))))
        # Post-discard: 2 cards
        two = (cards[0], cards[1]) if len(cards) >= 2 else (cards[0], 0)
        return self.bucket_2card(two, board)


def get_all_5card_hands() -> List[Tuple[int, ...]]:
    """All possible 5-card hands from 27-card deck (for precomputation)."""
    return list(combinations(range(DECK_SIZE), 5))


def get_all_2card_hands() -> List[Tuple[int, int]]:
    """All possible 2-card hands from 27-card deck."""
    return list(combinations(range(DECK_SIZE), 2))
