#!/usr/bin/env python3
"""
Build the rank7 precomputed table for the 27-card poker variant.

Generates:
  agents/libratus/tables/rank7_uint16.npy  (~1.78 MB, C(27,7)=888030 entries)

Run from project root:
    python scripts/build_rank_tables.py
"""

import os
import sys
import time
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv, WrappedEval
from agents.libratus.fast_eval import combinadic_index_7, NUM_7CARD

DECK_SIZE = 27
TABLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "agents", "libratus", "tables",
)


def build_rank7():
    evaluator = WrappedEval()
    rank7 = np.zeros(NUM_7CARD, dtype=np.uint16)

    t0 = time.time()
    count = 0

    for combo in combinations(range(DECK_SIZE), 7):
        idx = combinadic_index_7(combo)
        cards = list(combo)
        hand_treys = [PokerEnv.int_to_card(c) for c in cards[:2]]
        board_treys = [PokerEnv.int_to_card(c) for c in cards[2:]]
        r = evaluator.evaluate(hand_treys, board_treys)
        rank7[idx] = min(r, 65535)

        count += 1
        if count % 100000 == 0:
            elapsed = time.time() - t0
            pct = count / NUM_7CARD * 100
            print(f"  {count}/{NUM_7CARD} ({pct:.1f}%) elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"Built rank7 table: {count} entries in {elapsed:.1f}s")
    return rank7


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)

    print("Building rank7 table...")
    rank7 = build_rank7()
    path = os.path.join(TABLES_DIR, "rank7_uint16.npy")
    np.save(path, rank7)
    print(f"Saved {path} ({os.path.getsize(path)} bytes)")

    evaluator = WrappedEval()
    test_combos = [(0, 1, 2, 3, 4, 5, 6), (0, 5, 10, 15, 20, 25, 26)]
    for combo in test_combos:
        idx = combinadic_index_7(combo)
        hand_treys = [PokerEnv.int_to_card(c) for c in combo[:2]]
        board_treys = [PokerEnv.int_to_card(c) for c in combo[2:]]
        expected = evaluator.evaluate(hand_treys, board_treys)
        got = rank7[idx]
        status = "OK" if got == expected else "MISMATCH"
        print(f"  Verify {combo}: expected={expected} got={got} [{status}]")

    print("Done.")


if __name__ == "__main__":
    main()
