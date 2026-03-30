"""
Analyze Genesis bot log (HAND_RESULT lines) for loss patterns.
Run from repo root: python scripts/analyze_genesis_match.py [bot_log_path]
Writes visualizer/data/M1_analysis_report.md and prints summary tables.
"""
import argparse
import os
import sys
from collections import defaultdict

# Allow importing visualizer parsers when run from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISUALIZER_DIR = os.path.join(REPO_ROOT, "visualizer")
if VISUALIZER_DIR not in sys.path:
    sys.path.insert(0, VISUALIZER_DIR)

from parsers.bot_log import parse_bot_log


BIG_POT_THRESHOLD = 50


def _safe_int(x, default=0):
    try:
        return int(x) if x is not None else default
    except (TypeError, ValueError):
        return default


def _safe_float(x, default=0.0):
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


def analyze(hand_results: list) -> dict:
    """Compute aggregations from hand_results (list of HAND_RESULT dicts)."""
    by_street_end_type = defaultdict(lambda: {"count": 0, "total_reward": 0.0})
    by_position = defaultdict(lambda: {"count": 0, "total_reward": 0.0})
    we_fold_big_pot = []
    showdown_losses = []
    by_opp_type = defaultdict(lambda: {"count": 0, "total_reward": 0.0})

    for r in hand_results:
        hand = r.get("hand")
        if hand is None:
            continue
        street_ended = r.get("street_ended")
        end_type = (r.get("end_type") or "").strip()
        position = (r.get("position") or "").strip()
        reward = _safe_float(r.get("reward"))
        pot = _safe_int(r.get("pot"))
        opp_type = (r.get("opp_type") or "").strip()

        key_se = (street_ended, end_type)
        by_street_end_type[key_se]["count"] += 1
        by_street_end_type[key_se]["total_reward"] += reward

        by_position[position]["count"] += 1
        by_position[position]["total_reward"] += reward

        if opp_type:
            by_opp_type[opp_type]["count"] += 1
            by_opp_type[opp_type]["total_reward"] += reward

        if end_type == "we_fold" and pot >= BIG_POT_THRESHOLD:
            we_fold_big_pot.append({
                "hand": hand,
                "pot": pot,
                "reward": reward,
                "position": position,
                "street_ended": street_ended,
                "flop_line": r.get("flop_line") or "",
                "our_discard_class": r.get("our_discard_class") or "",
            })

        if end_type == "showdown" and reward < 0:
            showdown_losses.append({
                "hand": hand,
                "pot": pot,
                "reward": reward,
                "position": position,
                "our_discard_class": r.get("our_discard_class") or "",
                "flop_line": r.get("flop_line") or "",
                "flop_texture": r.get("flop_texture") or "",
                "turn_texture": r.get("turn_texture") or "",
                "river_texture": r.get("river_texture") or "",
            })

    return {
        "by_street_end_type": dict(by_street_end_type),
        "by_position": dict(by_position),
        "we_fold_big_pot": we_fold_big_pot,
        "showdown_losses": showdown_losses,
        "by_opp_type": dict(by_opp_type),
    }


def print_tables(agg: dict, total_hands: int) -> None:
    """Print summary tables to stdout."""
    print("\n=== Loss by (street_ended, end_type) ===")
    rows = sorted(agg["by_street_end_type"].items(), key=lambda x: (-x[1]["count"], x[0]))
    for (se, et), v in rows:
        print(f"  street_ended={se} end_type={et!r}: count={v['count']}, total_reward={v['total_reward']:.1f}")
    print("\n=== Loss by position ===")
    for pos, v in sorted(agg["by_position"].items(), key=lambda x: -x[1]["count"]):
        print(f"  {pos!r}: count={v['count']}, total_reward={v['total_reward']:.1f}")
    print("\n=== We folded in big pots (pot >= %d) ===" % BIG_POT_THRESHOLD)
    for row in agg["we_fold_big_pot"][:30]:
        print(f"  hand={row['hand']} pot={row['pot']} reward={row['reward']} pos={row['position']} "
              f"street_ended={row['street_ended']} flop_line={row['flop_line']!r} discard_class={row['our_discard_class']!r}")
    if len(agg["we_fold_big_pot"]) > 30:
        print(f"  ... and {len(agg['we_fold_big_pot']) - 30} more")
    print("\n=== Showdown losses (sample) ===")
    for row in agg["showdown_losses"][:25]:
        print(f"  hand={row['hand']} pot={row['pot']} reward={row['reward']} pos={row['position']} "
              f"discard={row['our_discard_class']!r} flop_line={row['flop_line']!r} "
              f"flop_tex={row['flop_texture']!r} turn_tex={row['turn_texture']!r} river_tex={row['river_texture']!r}")
    if len(agg["showdown_losses"]) > 25:
        print(f"  ... and {len(agg['showdown_losses']) - 25} more")
    print("\n=== By opp_type (at hand end) ===")
    for ot, v in sorted(agg["by_opp_type"].items(), key=lambda x: -x[1]["count"]):
        print(f"  {ot!r}: count={v['count']}, total_reward={v['total_reward']:.1f}")
    total_reward = sum(r["total_reward"] for r in agg["by_street_end_type"].values())
    print(f"\nTotal hands: {total_hands}, Total reward (all hands): {total_reward:.1f}")


def write_report(agg: dict, total_hands: int, out_path: str) -> None:
    """Write markdown report to out_path."""
    total_reward = sum(r["total_reward"] for r in agg["by_street_end_type"].values())
    lines = [
        "# Genesis match analysis report (M1)",
        "",
        f"- Total hands: {total_hands}",
        f"- Total reward: {total_reward:.1f}",
        "",
        "## Loss by (street_ended, end_type)",
        "",
        "| street_ended | end_type | count | total_reward |",
        "|--------------|----------|-------|--------------|",
    ]
    for (se, et), v in sorted(agg["by_street_end_type"].items(), key=lambda x: (-x[1]["count"], x[0])):
        lines.append(f"| {se} | {et!r} | {v['count']} | {v['total_reward']:.1f} |")
    lines.extend([
        "",
        "## Loss by position",
        "",
        "| position | count | total_reward |",
        "|----------|-------|--------------|",
    ])
    for pos, v in sorted(agg["by_position"].items(), key=lambda x: -x[1]["count"]):
        lines.append(f"| {pos!r} | {v['count']} | {v['total_reward']:.1f} |")
    lines.extend([
        "",
        "## We folded in big pots (pot >= %d)" % BIG_POT_THRESHOLD,
        "",
        "| hand | pot | reward | position | street_ended | flop_line | our_discard_class |",
        "|------|-----|--------|----------|--------------|-----------|-------------------|",
    ])
    for row in agg["we_fold_big_pot"]:
        lines.append(f"| {row['hand']} | {row['pot']} | {row['reward']} | {row['position']!r} | {row['street_ended']} | {row['flop_line']!r} | {row['our_discard_class']!r} |")
    lines.extend([
        "",
        "## Showdown losses",
        "",
        "| hand | pot | reward | position | our_discard_class | flop_line | flop_texture | turn_texture | river_texture |",
        "|------|-----|--------|----------|-------------------|-----------|--------------|--------------|---------------|",
    ])
    for row in agg["showdown_losses"]:
        lines.append(f"| {row['hand']} | {row['pot']} | {row['reward']} | {row['position']!r} | {row['our_discard_class']!r} | {row['flop_line']!r} | {row['flop_texture']!r} | {row['turn_texture']!r} | {row['river_texture']!r} |")
    lines.extend([
        "",
        "## By opp_type (at hand end)",
        "",
        "| opp_type | count | total_reward |",
        "|----------|-------|--------------|",
    ])
    for ot, v in sorted(agg["by_opp_type"].items(), key=lambda x: -x[1]["count"]):
        lines.append(f"| {ot!r} | {v['count']} | {v['total_reward']:.1f} |")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Analyze Genesis bot log for loss patterns")
    parser.add_argument(
        "bot_log_path",
        nargs="?",
        default=os.path.join(REPO_ROOT, "visualizer", "data", "M1_Bot.md"),
        help="Path to bot log (default: visualizer/data/M1_Bot.md)",
    )
    parser.add_argument(
        "--report",
        default=os.path.join(REPO_ROOT, "visualizer", "data", "M1_analysis_report.md"),
        help="Output report path",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.bot_log_path):
        print(f"Error: bot log not found: {args.bot_log_path}")
        sys.exit(1)

    bot_data = parse_bot_log(args.bot_log_path)
    hand_results = bot_data.get("hand_results") or []
    total_hands = len(hand_results)
    print(f"Parsed {total_hands} HAND_RESULT lines from {args.bot_log_path}")

    agg = analyze(hand_results)
    print_tables(agg, total_hands)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    write_report(agg, total_hands, args.report)
    print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
