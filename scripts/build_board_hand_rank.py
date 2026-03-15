#!/usr/bin/env python3
"""
Build the (board5, hand2) -> rank precomputed table.

Generates:
  board_hand_rank_uint16.npy  shape (80730, 351), dtype uint16
  where 80730 = C(27,5) boards, 351 = C(27,2) hole pairs.
  Overlapping (board, hand) entries get sentinel 65535.

Run from project root:
    python scripts/build_board_hand_rank.py
"""

import os
import sys
import time
import math
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv, WrappedEval

DECK_SIZE = 27
NUM_BOARDS = math.comb(27, 5)   # 80730
NUM_HANDS = math.comb(27, 2)    # 351

_NCR = [[0] * 8 for _ in range(28)]
for n in range(28):
    for k in range(0, min(7, n) + 1):
        _NCR[n][k] = math.comb(n, k)

def comb_index(sorted_cards, k):
    idx = 0
    for i, c in enumerate(sorted_cards):
        idx += _NCR[int(c)][i + 1]
    return idx

def main():
    evaluator = WrappedEval()
    table = np.full((NUM_BOARDS, NUM_HANDS), 65535, dtype=np.uint16)

    all_boards = list(combinations(range(DECK_SIZE), 5))
    all_hands = list(combinations(range(DECK_SIZE), 2))

    t0 = time.time()
    for bi, board in enumerate(all_boards):
        board_set = set(board)
        bid = comb_index(sorted(board), 5)
        board_treys = [PokerEnv.int_to_card(c) for c in board]

        for hand in all_hands:
            if hand[0] in board_set or hand[1] in board_set:
                continue
            hid = comb_index(sorted(hand), 2)
            hand_treys = [PokerEnv.int_to_card(c) for c in hand]
            r = evaluator.evaluate(hand_treys, board_treys)
            table[bid, hid] = min(r, 65535)

        if (bi + 1) % 10000 == 0:
            elapsed = time.time() - t0
            pct = (bi + 1) / len(all_boards) * 100
            print(f"  {bi+1}/{len(all_boards)} ({pct:.1f}%) elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"Built board_hand_rank table in {elapsed:.1f}s")

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "board_hand_rank_uint16.npy",
    )
    np.save(out_path, table)
    print(f"Saved {out_path} ({os.path.getsize(out_path)} bytes)")

    # Verify
    board_test = (0, 1, 2, 3, 4)
    hand_test = (5, 6)
    bid = comb_index(sorted(board_test), 5)
    hid = comb_index(sorted(hand_test), 2)
    expected = evaluator.evaluate(
        [PokerEnv.int_to_card(c) for c in hand_test],
        [PokerEnv.int_to_card(c) for c in board_test],
    )
    got = table[bid, hid]
    print(f"  Verify board={board_test} hand={hand_test}: expected={expected} got={got} [{'OK' if got == expected else 'MISMATCH'}]")


if __name__ == "__main__":
    main()
