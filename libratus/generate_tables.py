"""
Generate lookup tables for the runtime bot.
Outputs submission/libratus_tables.py with POLICY, KEEP_EQUITY, and POSTERIOR dicts.
Also outputs libratus/odds_tables.csv for human analysis.
"""
import os
import sys
import csv
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libratus.simulate import KEEP_ARCHETYPES, simulate_keep_equity, simulate_hand_distribution, simulate_showdown
from libratus.policy_train import build_full_policy, policy_to_source
from libratus.posterior_model import build_posterior_table
from libratus.abstractions import bucket_flop_simple


FLOP_BUCKETS = ["wet", "medium", "dry"]


def generate_equity_table(num_sims=1000, seed=42):
    """Generate equity table: {archetype: equity} and {(archetype, flop_bucket): equity}."""
    rng = random.Random(seed)
    table = {}
    print("Generating equity table...")
    for arch in KEEP_ARCHETYPES:
        eq = simulate_keep_equity(arch, num_sims=num_sims, rng=rng)
        table[arch] = round(eq, 4)
        print(f"  {arch:20s} = {eq:.4f}")
    return table


def generate_distribution_table(num_sims=1000, seed=42):
    """Generate hand distribution table."""
    rng = random.Random(seed)
    table = {}
    print("\nGenerating hand distribution table...")
    for arch in KEEP_ARCHETYPES:
        dist = simulate_hand_distribution(arch, num_sims=num_sims, rng=rng)
        table[arch] = {k: round(v, 4) for k, v in dist.items()}
        top = sorted(dist.items(), key=lambda x: -x[1])[:3]
        print(f"  {arch:20s} -> {top}")
    return table


def generate_matchup_table(num_sims=500, seed=42):
    """Generate head-to-head matchup table."""
    rng = random.Random(seed)
    table = {}
    print("\nGenerating matchup table...")
    for i, a1 in enumerate(KEEP_ARCHETYPES):
        for a2 in KEEP_ARCHETYPES[i:]:
            if a1 == a2:
                table[(a1, a2)] = 0.5
                continue
            wr = simulate_showdown(a1, a2, num_sims=num_sims, rng=rng)
            table[(a1, a2)] = round(wr, 4)
            table[(a2, a1)] = round(1.0 - wr, 4)
            print(f"  {a1:20s} vs {a2:20s} = {wr:.4f}")
    return table


def write_tables_py(equity, policy, posterior, matchups, outpath):
    """Write the tables as Python source to submission/libratus_tables.py."""
    lines = [
        '"""',
        'Auto-generated lookup tables for Libratus runtime bot.',
        'Do not edit manually. Regenerate with: python libratus/generate_tables.py',
        '"""',
        '',
        '# Equity by keep archetype (higher = stronger)',
        f'KEEP_EQUITY = {equity!r}',
        '',
    ]

    # Posterior table: convert tuple keys to string for safety
    post_str = {}
    for (disc_b, flop_b), dist in posterior.items():
        post_str[f"{disc_b}|{flop_b}"] = dist
    lines.append('# Opponent posterior: P(keep_bucket | discard_bucket, flop_bucket)')
    lines.append(f'POSTERIOR = {post_str!r}')
    lines.append('')

    # Matchup table
    match_str = {}
    for (a1, a2), wr in matchups.items():
        match_str[f"{a1}|{a2}"] = wr
    lines.append('# Head-to-head matchup win rates')
    lines.append(f'MATCHUPS = {match_str!r}')
    lines.append('')

    # Policy table: convert tuple keys to string for Python compatibility
    policy_str = {}
    for key, val in policy.items():
        policy_str[str(key)] = val
    lines.append('# Betting policy: (street, position, strength, board, to_call) -> action probs')
    lines.append(f'POLICY = {policy_str!r}')
    lines.append('')

    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote tables to {outpath}")


def write_odds_csv(equity, distributions, matchups, outpath):
    """Write human-readable CSV odds tables."""
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow(["=== KEEP ARCHETYPE EQUITY ==="])
        w.writerow(["archetype", "equity"])
        for arch in KEEP_ARCHETYPES:
            w.writerow([arch, equity.get(arch, "")])
        w.writerow([])

        w.writerow(["=== HAND DISTRIBUTIONS BY RIVER ==="])
        hand_classes = ["straight_flush", "full_house", "flush", "straight", "trips", "two_pair", "pair", "high_card"]
        w.writerow(["archetype"] + hand_classes)
        for arch in KEEP_ARCHETYPES:
            dist = distributions.get(arch, {})
            row = [arch] + [dist.get(hc, 0) for hc in hand_classes]
            w.writerow(row)
        w.writerow([])

        w.writerow(["=== MATCHUP TABLE ==="])
        w.writerow([""] + KEEP_ARCHETYPES)
        for a1 in KEEP_ARCHETYPES:
            row = [a1]
            for a2 in KEEP_ARCHETYPES:
                row.append(matchups.get((a1, a2), ""))
            w.writerow(row)

    print(f"Wrote odds CSV to {outpath}")


def main():
    num_sims = 1000
    seed = 42

    if "--fast" in sys.argv:
        num_sims = 200
        print("Running in FAST mode (200 sims)...\n")

    equity = generate_equity_table(num_sims, seed)
    distributions = generate_distribution_table(num_sims, seed)
    matchups = generate_matchup_table(min(num_sims, 500), seed)

    policy = build_full_policy()
    print(f"\nBuilt {len(policy)} policy entries.")

    posterior = build_posterior_table(min(num_sims, 2000), seed)
    print(f"Built {len(posterior)} posterior entries.")

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tables_path = os.path.join(base, "submission", "libratus_tables.py")
    csv_path = os.path.join(base, "libratus", "odds_tables.csv")

    write_tables_py(equity, policy, posterior, matchups, tables_path)
    write_odds_csv(equity, distributions, matchups, csv_path)

    print("\nDone! Tables ready for runtime bot.")


if __name__ == "__main__":
    main()
