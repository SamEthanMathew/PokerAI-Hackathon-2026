"""
Opponent discard posterior model.
Bayesian-ish lookup: P(opp_keep_bucket | opp_discard_bucket, flop_bucket).
Built from simulation: deal 5 to opponent, try all 10 keeps, pick "best" by equity,
record the (discard_bucket, keep_bucket) pair.
"""
import random
from collections import defaultdict

from libratus.deck import deal, DECK
from libratus.evaluator import mc_equity
from libratus.abstractions import (
    bucket_opp_discard, bucket_keep, bucket_flop_simple,
    RANK_A, RANK_9, RANK_8,
)
from libratus.deck import rank, suit, same_suit, are_connected


KEEP_BUCKETS = [
    "premium_pair", "medium_pair", "low_pair",
    "suited_connector", "suited_semi", "suited_gapper",
    "offsuit_connector", "offsuit_other",
]

OPP_DISCARD_BUCKETS = [
    "low_junk", "high_junk", "suited_cluster",
    "connected_cluster", "discarded_pair", "mixed_discard",
]


def _quick_keep_score(keep2, toss3, community, rng):
    """Fast heuristic for which keep the opponent would choose."""
    dead = set(toss3)
    eq = mc_equity(keep2, community, dead, num_sims=30, rng=rng)
    # Structural bonus (simplified)
    r1, r2 = rank(keep2[0]), rank(keep2[1])
    bonus = 0.0
    if r1 == r2:
        bonus += 0.05
        if r1 in (RANK_A, RANK_9):
            bonus += 0.05
    elif same_suit(keep2[0], keep2[1]):
        bonus += 0.04
        if are_connected(keep2[0], keep2[1]):
            bonus += 0.04
    return eq + bonus


def build_posterior_table(num_hands=5000, seed=42):
    """
    Simulate opponent hands and build posterior table.
    Returns dict: {(opp_discard_bucket, flop_bucket): {keep_bucket: count}}
    """
    rng = random.Random(seed)
    from itertools import combinations

    posterior = defaultdict(lambda: defaultdict(int))
    total_by_key = defaultdict(int)

    for _ in range(num_hands):
        hand5 = deal(5, rng=rng)
        dead_hand = set(hand5)
        board3 = deal(3, exclude=dead_hand, rng=rng)
        flop_b = bucket_flop_simple(board3)

        # Find best keep
        best_score = -999
        best_keep_idx = (0, 1)
        for i, j in combinations(range(5), 2):
            keep = [hand5[i], hand5[j]]
            toss = [hand5[k] for k in range(5) if k not in (i, j)]
            sc = _quick_keep_score(keep, toss, board3, rng)
            if sc > best_score:
                best_score = sc
                best_keep_idx = (i, j)

        keep_cards = [hand5[best_keep_idx[0]], hand5[best_keep_idx[1]]]
        toss_cards = [hand5[k] for k in range(5) if k not in best_keep_idx]

        disc_b = bucket_opp_discard(toss_cards)
        keep_b = bucket_keep(keep_cards)

        key = (disc_b, flop_b)
        posterior[key][keep_b] += 1
        total_by_key[key] += 1

    # Normalize to probabilities
    prob_table = {}
    for key, counts in posterior.items():
        total = total_by_key[key]
        if total > 0:
            prob_table[key] = {kb: cnt / total for kb, cnt in counts.items()}
    return prob_table


def infer_opp_keep_distribution(opp_discards, community, posterior_table):
    """
    Given observed opponent discards and community cards, return
    probability distribution over opponent keep buckets.
    """
    disc_b = bucket_opp_discard(opp_discards)
    flop_b = bucket_flop_simple(community)
    key = (disc_b, flop_b)
    if key in posterior_table:
        return posterior_table[key]
    # Fall back to discard bucket only (marginalize over flop)
    fallback = defaultdict(float)
    total = 0
    for (db, fb), dist in posterior_table.items():
        if db == disc_b:
            for kb, p in dist.items():
                fallback[kb] += p
                total += p
    if total > 0:
        return {k: v / total for k, v in fallback.items()}
    # Uniform prior
    n = len(KEEP_BUCKETS)
    return {kb: 1.0 / n for kb in KEEP_BUCKETS}


if __name__ == "__main__":
    print("Building posterior table (5000 hands)...")
    table = build_posterior_table(5000, seed=42)
    print(f"\nGenerated {len(table)} posterior entries.\n")
    for key in sorted(table.keys()):
        dist = table[key]
        top = sorted(dist.items(), key=lambda x: -x[1])[:3]
        print(f"  {str(key):50s} -> {top}")
