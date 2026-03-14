"""
Train the Libratus blueprint strategy by running MCCFR and saving the result.
Run from project root: python train_libratus.py
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.libratus.abstraction import CardAbstraction
from agents.libratus.cfr import MCCFR
from agents.libratus.strategy_store import save_strategy


DEFAULT_STRATEGY_PATH = os.path.join(
    os.path.dirname(__file__), "agents", "libratus", "blueprint_strategy.json"
)


def main():
    parser = argparse.ArgumentParser(description="Train Libratus blueprint via MCCFR")
    parser.add_argument("--iterations", type=int, default=2000, help="MCCFR iterations")
    parser.add_argument("--output", type=str, default=DEFAULT_STRATEGY_PATH, help="Output JSON path")
    parser.add_argument("--buckets5", type=int, default=30, help="5-card buckets")
    parser.add_argument("--buckets2", type=int, default=15, help="2-card buckets")
    args = parser.parse_args()

    card_abstraction = CardAbstraction(num_buckets_5=args.buckets5, num_buckets_2=args.buckets2)
    mccfr = MCCFR(card_abstraction=card_abstraction)
    print(f"Running MCCFR for {args.iterations} iterations...")
    strategy = mccfr.run(num_iterations=args.iterations)
    print(f"Infosets: {len(strategy)}")
    save_strategy(strategy, mccfr.infoset_actions, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
