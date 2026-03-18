#!/usr/bin/env python3
"""
Build the evaluation lookup table for OMICRoN V1.2.

Iterates all C(27,7) = 888,030 seven-card combinations from the 27-card deck.
For each, computes the hand rank using both treys representations (normal and
ace-low alt) and stores min(rank, alt_rank) in a numpy int16 array indexed by
the combinatorial number system.

Output: eval_table.npy (~1.7 MB)
Run once locally: python build_eval_table.py
"""

import os
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from treys import Card, Evaluator
from gym_env import PokerEnv

NUM_RANKS = 9
DECK_SIZE = 27
TOTAL_COMBOS = 888030  # C(27,7)

_C = [[0] * 28 for _ in range(28)]
for n in range(28):
    _C[n][0] = 1
    for k in range(1, n + 1):
        _C[n][k] = _C[n - 1][k - 1] + _C[n - 1][k]


def combo_index(c):
    return (_C[c[0]][1] + _C[c[1]][2] + _C[c[2]][3] +
            _C[c[3]][4] + _C[c[4]][5] + _C[c[5]][6] + _C[c[6]][7])


def main():
    int_to_treys = [PokerEnv.int_to_card(i) for i in range(DECK_SIZE)]
    int_to_treys_alt = []
    for tc in int_to_treys:
        s = Card.int_to_str(tc)
        int_to_treys_alt.append(Card.new(s.replace("A", "T")))

    evaluator = Evaluator()

    lut = np.zeros(TOTAL_COMBOS, dtype=np.int16)

    t0 = time.time()
    done = 0
    for combo in combinations(range(DECK_SIZE), 7):
        cards = [int_to_treys[c] for c in combo]
        cards_alt = [int_to_treys_alt[c] for c in combo]

        r = evaluator.evaluate(cards[:2], cards[2:])
        a = evaluator.evaluate(cards_alt[:2], cards_alt[2:])
        rank = a if a < r else r

        idx = combo_index(combo)
        lut[idx] = rank

        done += 1
        if done % 100000 == 0:
            elapsed = time.time() - t0
            pct = done / TOTAL_COMBOS * 100
            rate = done / elapsed
            eta = (TOTAL_COMBOS - done) / rate
            print(f"  {pct:5.1f}%  {done:>7d}/{TOTAL_COMBOS}  "
                  f"{rate:.0f} combos/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    out_path = os.path.join(os.path.dirname(__file__), "eval_table.npy")
    np.save(out_path, lut)
    sz = os.path.getsize(out_path)
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Saved {out_path} ({sz / 1024:.1f} KB)")
    print(f"Max rank value: {lut.max()}, fits int16: {lut.max() < 32767}")


if __name__ == "__main__":
    main()
