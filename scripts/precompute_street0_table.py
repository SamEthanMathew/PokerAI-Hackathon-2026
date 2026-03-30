"""
Precompute Street 0 base scores for all C(27,5) = 80,730 hands
or per-bucket (240 buckets) when --by-bucket.
Full flop and turn-river enumeration for maximum accuracy.
Output: pickle with table, global_min, global_max; optional CSV and .md scores table.
"""

import argparse
import csv
import os
import pickle
import sys
from collections import defaultdict
from itertools import combinations

# Project root so we can import submission
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from submission.street0_score import DECK_SIZE, compute_base_street0_score, hand5_to_bucket_id_slight

TOTAL_HANDS = 80730  # C(27, 5)
NUM_BUCKETS_SLIGHT = 240  # 80 base x 3 tiers
PROGRESS_INTERVAL = 2000
PARTIAL_SUFFIX = ".partial.pkl"
TIER_NAMES = ("low", "mid", "high")


def _group_label(bucket_id: int) -> str:
    base = bucket_id // 3
    tier = bucket_id % 3
    return f"B{base}_T{TIER_NAMES[tier]}"


def _write_scores_table(table: dict, global_min: float, global_max: float, bucket_hand_count: dict, out_dir: str, base_name: str) -> None:
    """Write CSV and Markdown table of bucket_id, group_label, raw_base, s_base, v_future, v_optionality, c_discard, c_reveal, hand_count."""
    if not table:
        return
    span = global_max - global_min
    if span < 1e-9:
        span = 1.0
    rows = []
    for bid in sorted(table.keys()):
        raw, v_future, v_opt, c_disc, c_rev, _ = table[bid]
        s_base = max(0.0, min(1.0, (raw - global_min) / span))
        label = _group_label(bid)
        hand_count = bucket_hand_count.get(bid, 0)
        rows.append({
            "bucket_id": bid,
            "group_label": label,
            "raw_base": raw,
            "s_base": s_base,
            "v_future": v_future,
            "v_optionality": v_opt,
            "c_discard": c_disc,
            "c_reveal": c_rev,
            "hand_count": hand_count,
        })
    csv_path = os.path.join(out_dir, base_name + "_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    md_path = os.path.join(out_dir, base_name + "_scores.md")
    with open(md_path, "w") as f:
        f.write("| bucket_id | group_label | raw_base | s_base | v_future | v_optionality | c_discard | c_reveal | hand_count |\n")
        f.write("|-----------|-------------|----------|--------|----------|---------------|-----------|----------|-------------|\n")
        for r in rows:
            f.write(f"| {r['bucket_id']} | {r['group_label']} | {r['raw_base']:.6f} | {r['s_base']:.4f} | {r['v_future']:.4f} | {r['v_optionality']:.4f} | {r['c_discard']:.4f} | {r['c_reveal']:.4f} | {r['hand_count']} |\n")
    print(f"Wrote {md_path}")


def run_by_bucket(output_path: str) -> None:
    table = {}
    bucket_hand_count = defaultdict(int)
    n_evals = 0
    for hand5_tuple in combinations(range(DECK_SIZE), 5):
        hand5 = list(hand5_tuple)
        bid = hand5_to_bucket_id_slight(hand5)
        bucket_hand_count[bid] += 1
        if bid not in table:
            raw, bd = compute_base_street0_score(hand5, n_flop_samples=None, n_tr_samples=None)
            table[bid] = (
                raw,
                bd.v_future,
                bd.v_optionality,
                bd.c_discard,
                bd.c_reveal,
                bd.best_keep_counts,
            )
            n_evals += 1
            if n_evals % 20 == 0:
                print(f"  Evaluated {n_evals} buckets (seen {sum(bucket_hand_count.values())} hands)")

    raw_values = [v[0] for v in table.values()]
    global_min = min(raw_values)
    global_max = max(raw_values)
    print(f"Global raw range: [{global_min:.6f}, {global_max:.6f}]")
    print(f"Total buckets with data: {len(table)}")

    result = {
        "table": table,
        "global_min": global_min,
        "global_max": global_max,
        "by_bucket": True,
    }
    with open(output_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Wrote {output_path} ({len(table)} buckets)")

    out_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    _write_scores_table(table, global_min, global_max, bucket_hand_count, out_dir, base_name)


def main():
    parser = argparse.ArgumentParser(description="Precompute Street 0 lookup table")
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(_PROJECT_ROOT, "submission", "street0_precomputed.pkl"),
        help="Output pickle path",
    )
    parser.add_argument(
        "--by-bucket",
        action="store_true",
        help="Precompute by hand abstraction (240 buckets); output CSV and .md scores table",
    )
    parser.add_argument(
        "--partial-dir",
        default=None,
        help="Directory for partial saves (default: same as output dir); enables resume",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing partial file and start from scratch",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=None,
        help="Stop after this many hands (for testing); default = all 80730",
    )
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)

    if args.by_bucket:
        print("Precomputing by bucket (240 buckets)...")
        run_by_bucket(output_path)
        return

    partial_dir = args.partial_dir or os.path.dirname(output_path)
    partial_path = os.path.join(partial_dir, os.path.basename(output_path) + PARTIAL_SUFFIX)

    table = {}
    if not args.no_resume and os.path.exists(partial_path):
        print(f"Resuming from {partial_path}")
        with open(partial_path, "rb") as f:
            data = pickle.load(f)
        table = data.get("table", {})
        print(f"  Loaded {len(table)} hands already computed")

    n_done = len(table)
    max_hands = args.max_hands
    target = min(TOTAL_HANDS, n_done + max_hands) if max_hands is not None else TOTAL_HANDS
    for i, hand5_tuple in enumerate(combinations(range(DECK_SIZE), 5)):
        if n_done >= target:
            break
        key = hand5_tuple  # already sorted
        if key in table:
            continue
        hand5 = list(hand5_tuple)
        raw, bd = compute_base_street0_score(hand5, n_flop_samples=None, n_tr_samples=None)
        table[key] = (
            raw,
            bd.v_future,
            bd.v_optionality,
            bd.c_discard,
            bd.c_reveal,
            bd.best_keep_counts,
        )
        n_done += 1
        if n_done % PROGRESS_INTERVAL == 0:
            print(f"  {n_done}/{target} hands")
            # Save partial for resume (only when doing full run)
            if max_hands is None:
                try:
                    with open(partial_path, "wb") as f:
                        pickle.dump({"table": table, "partial": True}, f)
                except Exception as e:
                    print(f"  (could not save partial: {e})")

    # Global min/max over raw scores
    raw_values = [v[0] for v in table.values()]
    global_min = min(raw_values)
    global_max = max(raw_values)
    print(f"Global raw range: [{global_min:.6f}, {global_max:.6f}]")

    result = {
        "table": table,
        "global_min": global_min,
        "global_max": global_max,
    }
    with open(output_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Wrote {output_path} ({len(table)} hands)")

    if os.path.exists(partial_path):
        try:
            os.remove(partial_path)
            print(f"Removed partial file {partial_path}")
        except Exception as e:
            print(f"Could not remove partial: {e}")


if __name__ == "__main__":
    main()
