"""
Exact equity enumeration for the 27-card poker variant.

Key optimization at discard time: opponent rank arrays are computed once per
runout and reused across all 10 keep candidates (~10x speedup).
"""

import numpy as np
from itertools import combinations
from typing import List, Tuple, Optional

from .fast_eval import (
    DECK_SIZE,
    PAIR_CARDS,
    PAIR_MASK,
    combinadic_index_7,
    load_rank7_table,
    cards_to_mask,
    legal_pair_indices,
)


def equity_discard(
    my5: List[int],
    flop3: List[int],
    opp_discards: Optional[List[int]] = None,
    rank7: Optional[np.ndarray] = None,
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Exact equity for all 10 keep-2 candidates at discard time.

    Returns list of ((keep_i, keep_j), equity) sorted best-first.
    Indices reference positions in my5 (0..4).
    """
    if rank7 is None:
        rank7 = load_rank7_table()

    dead_mask = cards_to_mask(my5) | cards_to_mask(flop3)
    if opp_discards:
        dead_mask |= cards_to_mask([c for c in opp_discards if c >= 0])

    pool = [c for c in range(DECK_SIZE) if not (dead_mask & (1 << c))]

    keep_pairs = []
    keep_cards = []
    for i in range(5):
        for j in range(i + 1, 5):
            keep_pairs.append((i, j))
            keep_cards.append((my5[i], my5[j]))

    n_keeps = len(keep_pairs)
    wins = np.zeros(n_keeps, dtype=np.float64)
    ties = np.zeros(n_keeps, dtype=np.float64)
    total = np.zeros(n_keeps, dtype=np.float64)

    flop_sorted = sorted(flop3)

    for runout in combinations(pool, 2):
        board5 = tuple(sorted(flop_sorted + list(runout)))

        opp_dead = dead_mask | cards_to_mask(runout)
        opp_indices = legal_pair_indices(opp_dead)
        if len(opp_indices) == 0:
            continue

        opp_ranks = np.zeros(len(opp_indices), dtype=np.uint16)
        for oi, pi in enumerate(opp_indices):
            c0 = int(PAIR_CARDS[pi, 0])
            c1 = int(PAIR_CARDS[pi, 1])
            seven = tuple(sorted([c0, c1] + list(board5)))
            opp_ranks[oi] = rank7[combinadic_index_7(seven)]

        n_opp = len(opp_ranks)

        for k in range(n_keeps):
            c0, c1 = keep_cards[k]
            seven = tuple(sorted([c0, c1] + list(board5)))
            my_rank = rank7[combinadic_index_7(seven)]

            w = int(np.sum(opp_ranks > my_rank))
            t = int(np.sum(opp_ranks == my_rank))
            wins[k] += w
            ties[k] += t
            total[k] += n_opp

    results = []
    for k in range(n_keeps):
        eq = (wins[k] + 0.5 * ties[k]) / total[k] if total[k] > 0 else 0.5
        results.append((keep_pairs[k], float(eq)))

    results.sort(key=lambda x: -x[1])
    return results


def equity_postflop(
    hole2: Tuple[int, int],
    board: List[int],
    dead_cards: Optional[List[int]] = None,
    rank7: Optional[np.ndarray] = None,
) -> float:
    """
    Exact equity for a 2-card holding against the visible board (3, 4, or 5 cards).
    Enumerates remaining board cards + opponent hands.
    """
    if rank7 is None:
        rank7 = load_rank7_table()

    board_visible = [c for c in board if c >= 0]
    board_needed = 5 - len(board_visible)

    dead_mask = cards_to_mask(hole2) | cards_to_mask(board_visible)
    if dead_cards:
        dead_mask |= cards_to_mask([c for c in dead_cards if c >= 0])

    pool = [c for c in range(DECK_SIZE) if not (dead_mask & (1 << c))]

    wins = 0.0
    ties = 0.0
    total = 0.0

    runouts = list(combinations(pool, board_needed)) if board_needed > 0 else [()]

    for runout in runouts:
        board5 = tuple(sorted(board_visible + list(runout)))

        opp_dead = dead_mask | cards_to_mask(runout)
        opp_indices = legal_pair_indices(opp_dead)
        if len(opp_indices) == 0:
            continue

        my_seven = tuple(sorted(list(hole2) + list(board5)))
        my_rank = rank7[combinadic_index_7(my_seven)]

        opp_ranks = np.zeros(len(opp_indices), dtype=np.uint16)
        for oi, pi in enumerate(opp_indices):
            c0 = int(PAIR_CARDS[pi, 0])
            c1 = int(PAIR_CARDS[pi, 1])
            seven = tuple(sorted([c0, c1] + list(board5)))
            opp_ranks[oi] = rank7[combinadic_index_7(seven)]

        w = int(np.sum(opp_ranks > my_rank))
        t = int(np.sum(opp_ranks == my_rank))
        wins += w
        ties += t
        total += len(opp_indices)

    if total > 0:
        return (wins + 0.5 * ties) / total
    return 0.5
