"""
Precompute equity lookup tables for the 27-card poker variant.

Run once before training (~30 min on Jetson GPU / 12 CPU cores).

Tables generated:
  tables/equity_vs_random.npy  — (729,) float32, indexed by card pair key
  tables/optimal_discards.npy  — dict: (hand5_key, flop3_key) -> (keep_i, keep_j)
  tables/opponent_ranges.npy   — dict: discard3_key -> list[(hand_pair, prob)]

Usage:
    python precompute.py                   # generate all tables
    python precompute.py --table equity    # equity only
    python precompute.py --table discard   # optimal discards only
    python precompute.py --table ranges    # opponent ranges only
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import numpy as np

# Add repo root to path so gym_env is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gym_env import PokerEnv, WrappedEval

# ── Constants ───────────────────────────────────────────────────────────────

DECK_SIZE = 27
NUM_RANKS = 9
NUM_SUITS = 3
TABLES_DIR = os.path.join(_HERE, "tables")
NUM_SIMS_EQUITY = 5000          # Monte Carlo sims per 2-card hand

evaluator = None  # module-level for subprocess reuse


def _init_evaluator():
    global evaluator
    evaluator = WrappedEval()


def _card_pair_key(c1: int, c2: int) -> int:
    """Deterministic key for an unordered pair of cards. Range: [0, 729)."""
    a, b = min(c1, c2), max(c1, c2)
    return a * DECK_SIZE + b


def _hand5_key(cards: tuple) -> tuple:
    return tuple(sorted(cards))


def _flop3_key(cards: tuple) -> tuple:
    return tuple(sorted(cards))


def _eval_hand(hole_cards: list, board: list) -> int:
    """Evaluate a 2-card hold'em hand against board using WrappedEval."""
    treys_hand = [PokerEnv.int_to_card(c) for c in hole_cards]
    treys_board = [PokerEnv.int_to_card(c) for c in board]
    return evaluator.evaluate(treys_hand, treys_board)


# ── Table 1: equity_vs_random ────────────────────────────────────────────────

def _compute_equity_for_pair(args):
    """Worker: compute equity for one 2-card hand vs random opponent."""
    c1, c2, n_sims = args
    _init_evaluator()

    all_cards = list(range(DECK_SIZE))
    known = {c1, c2}

    wins = 0
    valid = 0

    rng = np.random.default_rng()

    for _ in range(n_sims):
        remaining = [c for c in all_cards if c not in known]
        rng.shuffle(remaining)

        # Opponent gets 2 cards, board gets 5
        if len(remaining) < 7:
            continue

        opp_cards = remaining[:2]
        board = remaining[2:7]
        used = known | set(opp_cards) | set(board)
        if len(used) != 2 + 2 + 5:
            continue  # collision (should not happen)

        my_rank = _eval_hand([c1, c2], board)
        opp_rank = _eval_hand(opp_cards, board)

        if my_rank < opp_rank:   # lower rank = better hand in treys
            wins += 1
        elif my_rank == opp_rank:
            wins += 0.5
        valid += 1

    return (c1, c2, wins / valid if valid > 0 else 0.5)


def build_equity_vs_random(n_workers: int = 12, n_sims: int = NUM_SIMS_EQUITY) -> np.ndarray:
    """
    Build equity_vs_random table.
    Returns ndarray of shape (729,) indexed by _card_pair_key(c1, c2).
    """
    all_pairs = [(c1, c2) for c1 in range(DECK_SIZE) for c2 in range(c1 + 1, DECK_SIZE)]
    print(f"  Computing equity for {len(all_pairs)} card pairs ({n_sims} sims each) …")

    table = np.full(DECK_SIZE * DECK_SIZE, 0.5, dtype=np.float32)

    args = [(c1, c2, n_sims) for c1, c2 in all_pairs]

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for fut in as_completed(pool.submit(_compute_equity_for_pair, a) for a in args):
            c1, c2, eq = fut.result()
            key = _card_pair_key(c1, c2)
            table[key] = eq
            table[_card_pair_key(c2, c1)] = eq  # symmetric
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                remaining = (len(args) - done) / done * elapsed
                print(f"    {done}/{len(args)} pairs done  ({remaining:.0f}s remaining)")

    print(f"  equity_vs_random done in {time.time() - t0:.1f}s")
    return table


# ── Table 2: optimal_discards ────────────────────────────────────────────────

def _best_keep_from_5(hand5: list, flop3: list) -> tuple:
    """
    Given 5 hole cards and a 3-card flop, find the keep pair (i, j) maximising
    simulated equity. Returns indices into hand5.
    """
    _init_evaluator()
    all_cards = set(range(DECK_SIZE))
    known = set(hand5) | set(flop3)

    best_eq = -1.0
    best_pair = (0, 1)

    remaining_deck = [c for c in all_cards if c not in known]

    for (i, j) in combinations(range(5), 2):
        my_cards = [hand5[i], hand5[j]]
        wins = 0
        total = 0
        rng = np.random.default_rng()

        for _ in range(400):   # fast: 400 sims per combo
            deck = remaining_deck.copy()
            rng.shuffle(deck)
            if len(deck) < 4:
                continue
            opp_cards = deck[:2]
            turn_river = deck[2:4]
            board = flop3 + turn_river
            if len(set(board) | set(opp_cards) | set(my_cards)) != 9:
                continue

            my_rank = _eval_hand(my_cards, board)
            opp_rank = _eval_hand(opp_cards, board)
            if my_rank < opp_rank:
                wins += 1
            elif my_rank == opp_rank:
                wins += 0.5
            total += 1

        eq = wins / total if total > 0 else 0.5
        if eq > best_eq:
            best_eq = eq
            best_pair = (i, j)

    return best_pair


def _discard_worker(args):
    hand5_key, flop3_key = args
    hand5 = list(hand5_key)
    flop3 = list(flop3_key)
    return hand5_key, flop3_key, _best_keep_from_5(hand5, flop3)


def build_optimal_discards(n_workers: int = 12) -> dict:
    """
    Build optimal_discards dict: (hand5_key, flop3_key) -> (keep_i, keep_j).
    Only compute for a representative sample (full table would be too large).
    """
    print("  Building optimal_discards table …")

    rng = np.random.default_rng(42)
    all_cards = list(range(DECK_SIZE))
    table = {}

    # Sample 50K random (hand, flop) combinations
    n_samples = 50_000
    t0 = time.time()
    work_items = []
    seen = set()
    attempts = 0

    while len(work_items) < n_samples and attempts < n_samples * 10:
        attempts += 1
        cards = rng.permutation(all_cards)
        hand5 = _hand5_key(tuple(cards[:5].tolist()))
        flop3 = _flop3_key(tuple(cards[5:8].tolist()))
        key = (hand5, flop3)
        if key not in seen:
            seen.add(key)
            work_items.append((hand5, flop3))

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for fut in as_completed(pool.submit(_discard_worker, item) for item in work_items):
            h5k, f3k, keep = fut.result()
            table[(h5k, f3k)] = keep
            done += 1
            if done % 5000 == 0:
                elapsed = time.time() - t0
                remaining = (n_samples - done) / max(done, 1) * elapsed
                print(f"    {done}/{n_samples} done  ({remaining:.0f}s remaining)")

    print(f"  optimal_discards done: {len(table)} entries in {time.time() - t0:.1f}s")
    return table


# ── Table 3: opponent_ranges ─────────────────────────────────────────────────

def build_opponent_ranges() -> dict:
    """
    For each possible set of 3 opponent discards, compute the probability
    distribution over all possible 2-card hands the opponent could hold.

    key:   tuple(sorted(3 discard cards))
    value: list of (hand_pair_tuple, probability) sorted by probability descending
    """
    print("  Building opponent_ranges table …")
    t0 = time.time()
    table = {}
    all_cards = list(range(DECK_SIZE))

    discard_combos = list(combinations(range(DECK_SIZE), 3))
    total = len(discard_combos)

    for idx, discards in enumerate(discard_combos):
        discard_set = set(discards)
        remaining = [c for c in all_cards if c not in discard_set]
        possible_hands = list(combinations(remaining, 2))
        n = len(possible_hands)
        if n == 0:
            continue
        # Uniform prior — each remaining 2-card combination is equally likely
        prob = 1.0 / n
        table[discards] = [(hand, prob) for hand in possible_hands]

        if idx % 500 == 0:
            print(f"    {idx}/{total} discard combos …")

    print(f"  opponent_ranges done: {len(table)} entries in {time.time() - t0:.1f}s")
    return table


# ── Save / load helpers ──────────────────────────────────────────────────────

def save_tables(equity: np.ndarray, discards: dict, ranges: dict, tables_dir: str = TABLES_DIR):
    os.makedirs(tables_dir, exist_ok=True)

    equity_path = os.path.join(tables_dir, "equity_vs_random.npy")
    np.save(equity_path, equity)
    print(f"  Saved {equity_path}  shape={equity.shape}")

    discard_path = os.path.join(tables_dir, "optimal_discards.npy")
    np.save(discard_path, discards, allow_pickle=True)
    print(f"  Saved {discard_path}  entries={len(discards)}")

    ranges_path = os.path.join(tables_dir, "opponent_ranges.npy")
    np.save(ranges_path, ranges, allow_pickle=True)
    print(f"  Saved {ranges_path}  entries={len(ranges)}")


def load_tables(tables_dir: str = TABLES_DIR) -> dict:
    """
    Load precomputed tables. Returns a dict with keys:
        "equity_vs_random"  (ndarray or None)
        "optimal_discards"  (dict or None)
        "opponent_ranges"   (dict or None)
    If a table file is missing, the corresponding value is None (features fall back to 0).
    """
    tables = {
        "equity_vs_random": None,
        "optimal_discards": None,
        "opponent_ranges": None,
    }

    ep = os.path.join(tables_dir, "equity_vs_random.npy")
    dp = os.path.join(tables_dir, "optimal_discards.npy")
    rp = os.path.join(tables_dir, "opponent_ranges.npy")

    if os.path.exists(ep):
        tables["equity_vs_random"] = np.load(ep)
        print(f"[tables] Loaded equity_vs_random  shape={tables['equity_vs_random'].shape}")
    else:
        print(f"[tables] equity_vs_random not found at {ep} — equity features will be 0")

    if os.path.exists(dp):
        tables["optimal_discards"] = np.load(dp, allow_pickle=True).item()
        print(f"[tables] Loaded optimal_discards  entries={len(tables['optimal_discards'])}")
    else:
        print(f"[tables] optimal_discards not found — discard reward shaping disabled")

    if os.path.exists(rp):
        tables["opponent_ranges"] = np.load(rp, allow_pickle=True).item()
        print(f"[tables] Loaded opponent_ranges  entries={len(tables['opponent_ranges'])}")
    else:
        print(f"[tables] opponent_ranges not found — range features will be 0")

    return tables


def lookup_equity(tables: dict, c1: int, c2: int) -> float:
    """Look up equity for a 2-card hand. Returns 0.5 if table missing."""
    ev = tables.get("equity_vs_random")
    if ev is None:
        return 0.5
    key = _card_pair_key(c1, c2)
    if key >= len(ev):
        return 0.5
    return float(ev[key])


def lookup_optimal_discard(tables: dict, hand5: list, flop3: list):
    """Return (keep_i, keep_j) for optimal discard, or None if table missing."""
    od = tables.get("optimal_discards")
    if od is None:
        return None
    key = (_hand5_key(tuple(hand5)), _flop3_key(tuple(flop3)))
    return od.get(key)


def lookup_opponent_range(tables: dict, discards: list):
    """Return list of (hand_pair, probability) or None if table missing."""
    or_ = tables.get("opponent_ranges")
    if or_ is None:
        return None
    key = tuple(sorted(discards))
    return or_.get(key)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute poker equity tables")
    parser.add_argument(
        "--table",
        choices=["equity", "discard", "ranges", "all"],
        default="all",
        help="Which table to compute",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--sims", type=int, default=NUM_SIMS_EQUITY)
    parser.add_argument("--tables-dir", default=TABLES_DIR)
    args = parser.parse_args()

    os.makedirs(args.tables_dir, exist_ok=True)

    equity, discards, ranges = None, None, None

    # Load existing tables so we don't regenerate unnecessarily
    existing = load_tables(args.tables_dir)

    if args.table in ("equity", "all"):
        ep = os.path.join(args.tables_dir, "equity_vs_random.npy")
        if os.path.exists(ep):
            print("equity_vs_random already exists — skipping (delete to regenerate)")
            equity = existing["equity_vs_random"]
        else:
            print("=== Computing equity_vs_random ===")
            equity = build_equity_vs_random(n_workers=args.workers, n_sims=args.sims)
            np.save(ep, equity)
            print(f"Saved {ep}")

    if args.table in ("discard", "all"):
        dp = os.path.join(args.tables_dir, "optimal_discards.npy")
        if os.path.exists(dp):
            print("optimal_discards already exists — skipping")
        else:
            print("=== Computing optimal_discards ===")
            discards = build_optimal_discards(n_workers=args.workers)
            np.save(dp, discards, allow_pickle=True)
            print(f"Saved {dp}")

    if args.table in ("ranges", "all"):
        rp = os.path.join(args.tables_dir, "opponent_ranges.npy")
        if os.path.exists(rp):
            print("opponent_ranges already exists — skipping")
        else:
            print("=== Computing opponent_ranges ===")
            ranges = build_opponent_ranges()
            np.save(rp, ranges, allow_pickle=True)
            print(f"Saved {rp}")

    print("\nDone. Tables are in:", args.tables_dir)


if __name__ == "__main__":
    main()
