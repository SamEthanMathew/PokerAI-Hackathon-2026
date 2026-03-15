"""
Offline Monte Carlo simulation pipeline.
Generates equity tables, hand distributions, and showdown stats by keep archetype.
"""
import random
from collections import defaultdict
from itertools import combinations

from libratus.deck import DECK, DECK_SIZE, rank, suit, card_str, deal, NUM_RANKS
from libratus.evaluator import mc_equity, evaluate_hand
from libratus.abstractions import bucket_keep, bucket_flop_simple


KEEP_ARCHETYPES = [
    "premium_pair", "medium_pair", "low_pair",
    "suited_connector", "suited_semi", "suited_gapper",
    "offsuit_connector", "offsuit_other",
]


def _generate_keep_samples(archetype, num_samples=200, rng=None):
    """Generate random 2-card hands matching a given archetype."""
    r = rng if rng else random
    samples = []
    attempts = 0
    while len(samples) < num_samples and attempts < num_samples * 50:
        attempts += 1
        cards = r.sample(DECK, 2)
        if bucket_keep(cards) == archetype:
            samples.append(cards)
    return samples


def simulate_keep_equity(archetype, flop_bucket_filter=None, num_sims=5000, rng=None):
    """Average MC equity for a keep archetype, optionally filtered by flop texture."""
    r = rng if rng else random
    keeps = _generate_keep_samples(archetype, num_samples=200, rng=r)
    if not keeps:
        return 0.5

    equities = []
    for keep2 in keeps:
        dead = set(keep2)
        board3 = deal(3, exclude=dead, rng=r)
        if flop_bucket_filter and bucket_flop_simple(board3) != flop_bucket_filter:
            continue
        eq = mc_equity(keep2, board3, dead=set(), num_sims=25, rng=r)
        equities.append(eq)
        if len(equities) >= num_sims // 25:
            break
    return sum(equities) / len(equities) if equities else 0.5


def simulate_hand_distribution(archetype, num_sims=5000, rng=None):
    """Estimate hand class probabilities by river for a keep archetype."""
    r = rng if rng else random
    keeps = _generate_keep_samples(archetype, num_samples=300, rng=r)
    if not keeps:
        return {}

    # treys hand class ranges (lower = better):
    # 1-10: straight flush, 11-166: four of a kind (impossible), 167-322: full house,
    # 323-1599: flush, 1600-1609: straight, ... etc.
    # Simplified: just count wins/equity ranges
    class_counts = defaultdict(int)
    total = 0
    for keep2 in keeps:
        dead = set(keep2)
        board5 = deal(5, exclude=dead, rng=r)
        hand_rank = evaluate_hand(keep2, board5)
        # Classify by rank range (treys-specific for this deck)
        if hand_rank <= 10:
            class_counts["straight_flush"] += 1
        elif hand_rank <= 322:
            class_counts["full_house"] += 1
        elif hand_rank <= 1599:
            class_counts["flush"] += 1
        elif hand_rank <= 1609:
            class_counts["straight"] += 1
        elif hand_rank <= 2467:
            class_counts["trips"] += 1
        elif hand_rank <= 3325:
            class_counts["two_pair"] += 1
        elif hand_rank <= 6185:
            class_counts["pair"] += 1
        else:
            class_counts["high_card"] += 1
        total += 1
        if total >= num_sims:
            break

    return {k: v / total for k, v in class_counts.items()} if total else {}


def simulate_showdown(arch1, arch2, num_sims=2000, rng=None):
    """Simulate heads-up showdown win rate for arch1 vs arch2."""
    r = rng if rng else random
    keeps1 = _generate_keep_samples(arch1, 200, r)
    keeps2 = _generate_keep_samples(arch2, 200, r)
    if not keeps1 or not keeps2:
        return 0.5

    wins = 0.0
    total = 0
    for _ in range(num_sims):
        k1 = r.choice(keeps1)
        k2 = r.choice(keeps2)
        if set(k1) & set(k2):
            continue
        dead = set(k1) | set(k2)
        board5 = deal(5, exclude=dead, rng=r)
        r1 = evaluate_hand(k1, board5)
        r2 = evaluate_hand(k2, board5)
        if r1 < r2:
            wins += 1
        elif r1 == r2:
            wins += 0.5
        total += 1
    return wins / total if total else 0.5


def run_full_simulation(num_sims=2000, seed=42):
    """Run the full simulation pipeline and return results dict."""
    rng = random.Random(seed)
    results = {}

    print("Simulating keep archetype equities...")
    equity_table = {}
    for arch in KEEP_ARCHETYPES:
        eq = simulate_keep_equity(arch, num_sims=num_sims, rng=rng)
        equity_table[arch] = round(eq, 4)
        print(f"  {arch:20s} equity = {eq:.4f}")
    results["equity"] = equity_table

    print("\nSimulating hand distributions...")
    dist_table = {}
    for arch in KEEP_ARCHETYPES:
        dist = simulate_hand_distribution(arch, num_sims=num_sims, rng=rng)
        dist_table[arch] = {k: round(v, 4) for k, v in dist.items()}
        print(f"  {arch:20s} {dist_table[arch]}")
    results["distributions"] = dist_table

    print("\nSimulating showdown matchups (premium_pair vs each)...")
    matchup_table = {}
    for arch in KEEP_ARCHETYPES:
        if arch == "premium_pair":
            matchup_table[("premium_pair", arch)] = 0.5
            continue
        wr = simulate_showdown("premium_pair", arch, num_sims=num_sims, rng=rng)
        matchup_table[("premium_pair", arch)] = round(wr, 4)
        print(f"  premium_pair vs {arch:20s} = {wr:.4f}")
    results["matchups"] = matchup_table

    return results


if __name__ == "__main__":
    run_full_simulation(num_sims=1000, seed=42)
