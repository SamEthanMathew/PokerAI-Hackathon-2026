import os
import struct
import numpy as np

from submission.data_path import get_data_dir

KEEP2 = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

def _py_to_cpp(py_card):
    rank = py_card % 9
    suit = py_card // 9
    return rank * 3 + suit

def _cpp_to_py(cpp_card):
    rank = cpp_card // 3
    suit = cpp_card % 3
    return suit * 9 + rank

def _canonicalize_cpp(cards_cpp):
    """Canonicalize 8 cards in C++ encoding space (must match C++ exactly)."""
    hole_sorted = sorted(cards_cpp[:5])
    flop_sorted = sorted(cards_cpp[5:8])
    ordered = hole_sorted + flop_sorted

    suit_map = [-1, -1, -1]
    next_suit = 0
    for c in ordered:
        suit = c % 3
        if suit_map[suit] == -1:
            suit_map[suit] = next_suit
            next_suit += 1

    canon = []
    for c in ordered:
        rank = c // 3
        suit = c % 3
        canon.append(rank * 3 + suit_map[suit])

    hole = sorted(canon[:5])
    flop = sorted(canon[5:8])
    return hole + flop

def _canonicalize_py(hole_py, flop_py):
    """Canonicalize (5-hole, 3-flop) from Python encoding. Returns tuple of 8 Python-encoded cards."""
    cards_cpp = [_py_to_cpp(c) for c in hole_py] + [_py_to_cpp(c) for c in flop_py]
    canon_cpp = _canonicalize_cpp(cards_cpp)
    return tuple(_cpp_to_py(c) for c in canon_cpp)


class BBTable:
    """Fast BB discard table using sorted numpy arrays + binary search.

    New format: 8 key bytes + 10 uint8 raw equities per entry.
    Equity byte 0-255 maps linearly to [0.0, 1.0].
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            n_entries = struct.unpack("<Q", f.read(8))[0]
            data = f.read(n_entries * 18)
        raw = np.frombuffer(data, dtype=np.uint8).reshape(n_entries, 18)
        self._keys = np.zeros(n_entries, dtype=np.uint64)
        for i in range(8):
            self._keys += raw[:, i].astype(np.uint64) << (i * 8)
        self._equities = raw[:, 8:].copy()  # uint8[n, 10] raw equities
        order = np.argsort(self._keys)
        self._keys = self._keys[order]
        self._equities = self._equities[order]

    def lookup(self, canon_key_tuple):
        """Look up 8-card canonical key. Returns uint8[10] equities or None."""
        packed = np.uint64(0)
        for i, c in enumerate(canon_key_tuple):
            packed += np.uint64(c) << np.uint64(i * 8)
        idx = np.searchsorted(self._keys, packed)
        if idx < len(self._keys) and self._keys[idx] == packed:
            return self._equities[idx]
        return None

def _load_bb_table():
    """Load BB discard table. Returns BBTable or None."""
    path = os.path.join(get_data_dir(), "bb_discard_table.bin")
    if not os.path.exists(path):
        return None
    return BBTable(path)
