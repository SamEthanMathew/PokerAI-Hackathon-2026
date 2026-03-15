"""
Fast hand evaluation via precomputed rank7 lookup table and bitmask utilities.

- Combinadic indexing: map sorted 7-card combo to table index in O(1).
- rank7 table: uint16 array of C(27,7)=888030 entries (~1.78 MB).
- Pair arrays: all C(27,2)=351 two-card combos with bitmasks for dead-card filtering.
"""

import os
import numpy as np
from typing import Tuple, Optional, List

DECK_SIZE = 27
NUM_PAIRS = 351   # C(27,2)
NUM_7CARD = 888030  # C(27,7)

_MAX_N = 28
_MAX_K = 8
COMB = np.zeros((_MAX_N, _MAX_K), dtype=np.int32)
for _n in range(_MAX_N):
    COMB[_n, 0] = 1
    for _k in range(1, min(_n + 1, _MAX_K)):
        COMB[_n, _k] = COMB[_n - 1, _k - 1] + COMB[_n - 1, _k]


def combinadic_index_7(cards) -> int:
    """Combinadic rank for a sorted 7-card combo from a 27-card deck."""
    s = sorted(cards)
    return int(
        COMB[s[0], 1] + COMB[s[1], 2] + COMB[s[2], 3]
        + COMB[s[3], 4] + COMB[s[4], 5] + COMB[s[5], 6] + COMB[s[6], 7]
    )


def cards_to_mask(cards) -> int:
    """OR-combine 27-bit masks for a collection of card ints."""
    m = 0
    for c in cards:
        if c >= 0:
            m |= (1 << c)
    return m


# ---- Pair tables (computed once at import) ----

_pair_cards = np.zeros((NUM_PAIRS, 2), dtype=np.uint8)
_pair_mask = np.zeros(NUM_PAIRS, dtype=np.uint32)
_idx = 0
for _a in range(DECK_SIZE):
    for _b in range(_a + 1, DECK_SIZE):
        _pair_cards[_idx, 0] = _a
        _pair_cards[_idx, 1] = _b
        _pair_mask[_idx] = (1 << _a) | (1 << _b)
        _idx += 1

PAIR_CARDS: np.ndarray = _pair_cards
PAIR_MASK: np.ndarray = _pair_mask


def legal_pair_indices(dead_mask: int) -> np.ndarray:
    """Return indices into PAIR_CARDS where neither card is dead."""
    return np.where((PAIR_MASK & dead_mask) == 0)[0]


# ---- Rank7 table ----

_rank7_table: Optional[np.ndarray] = None
_TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")


def load_rank7_table(path: Optional[str] = None) -> np.ndarray:
    """Load the precomputed rank7 table from .npy file."""
    global _rank7_table
    if _rank7_table is not None:
        return _rank7_table
    if path is None:
        path = os.path.join(_TABLES_DIR, "rank7_uint16.npy")
    _rank7_table = np.load(path)
    return _rank7_table


def rank7_lookup(hand2: Tuple[int, int], board5: Tuple[int, ...],
                 rank7: Optional[np.ndarray] = None) -> int:
    """Look up rank for 2 hole cards + 5 board cards."""
    if rank7 is None:
        rank7 = load_rank7_table()
    seven = tuple(sorted(list(hand2) + list(board5)))
    return int(rank7[combinadic_index_7(seven)])


def batch_rank7_for_pairs(pair_indices: np.ndarray,
                          board5: Tuple[int, ...],
                          rank7: np.ndarray) -> np.ndarray:
    """Compute ranks for many 2-card holdings against a fixed 5-card board."""
    board_sorted = sorted(board5)
    n = len(pair_indices)
    ranks = np.zeros(n, dtype=np.uint16)
    for i in range(n):
        c0 = int(PAIR_CARDS[pair_indices[i], 0])
        c1 = int(PAIR_CARDS[pair_indices[i], 1])
        seven = tuple(sorted([c0, c1] + board_sorted))
        ranks[i] = rank7[combinadic_index_7(seven)]
    return ranks
