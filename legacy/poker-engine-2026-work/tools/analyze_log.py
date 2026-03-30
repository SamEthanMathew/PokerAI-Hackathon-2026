"""
Post-match log analyzer for OMICRON_V1.

Usage:
    python tools/analyze_log.py <log_file>

Example:
    python tools/analyze_log.py agent_logs/match_unknown_unknown.log
"""

import json
import re
import sys
import math
from collections import defaultdict


def load_log(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"- INFO - (.+)$", line)
            if m:
                try:
                    entries.append(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    pass
    return entries


def group_by(entries, key):
    d = defaultdict(list)
    for e in entries:
        d[e.get(key)].append(e)
    return d


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def stddev(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def pct(num, den):
    return f"{100.0 * num / den:.1f}%" if den else "N/A"


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/analyze_log.py <log_file>")
        sys.exit(1)

    entries = load_log(sys.argv[1])
    if not entries:
        print("No parseable log entries found.")
        sys.exit(0)

    by_event = group_by(entries, "event")
    by_hand = defaultdict(list)
    for e in entries:
        by_hand[e.get("hand", -1)].append(e)

    # ── 1. Overall summary ────────────────────────────────────────────────────
    results = by_event.get("hand_result", [])
    wins = sum(1 for r in results if r.get("outcome") == "win")
    losses = sum(1 for r in results if r.get("outcome") == "loss")
    ties = sum(1 for r in results if r.get("outcome") == "tie")
    total = len(results)
    net_pnl = results[-1].get("running_pnl", 0) if results else 0
    final_pnl_vals = [r.get("pnl", 0) for r in results]

    print("=" * 60)
    print("MATCH SUMMARY")
    print("=" * 60)
    print(f"  Total hands:  {total}")
    print(f"  Win / Loss / Tie:  {wins} / {losses} / {ties}  ({pct(wins, total)} win rate)")
    print(f"  Net PnL:      {net_pnl:+d} chips")
    print(f"  Avg PnL/hand: {mean(final_pnl_vals):+.2f}")

    # ── 2. PnL trajectory (min/max/midpoint) ──────────────────────────────────
    if results:
        pnls = [r.get("running_pnl", 0) for r in results]
        print(f"\nPnL Trajectory:")
        print(f"  Min: {min(pnls):+d}  Max: {max(pnls):+d}  Final: {pnls[-1]:+d}")
        milestones = [0, len(pnls) // 4, len(pnls) // 2, 3 * len(pnls) // 4, len(pnls) - 1]
        print("  Milestones (hand -> cumulative PnL):")
        for idx in milestones:
            r = results[idx]
            print(f"    hand {r.get('hand', idx):4d} -> {pnls[idx]:+d}")

    # ── 3. Showdown analysis ──────────────────────────────────────────────────
    showdowns = [r for r in results if r.get("showdown")]
    sd_wins = sum(1 for r in showdowns if r.get("outcome") == "win")
    sd_losses = [r for r in showdowns if r.get("outcome") == "loss"]
    print(f"\nShowdown Analysis:")
    print(f"  Showdowns: {len(showdowns)} ({pct(len(showdowns), total)} of hands)")
    print(f"  Showdown win rate: {pct(sd_wins, len(showdowns))}")
    if sd_losses:
        print(f"  Showdown losses (sample of up to 5):")
        for r in sd_losses[:5]:
            print(f"    hand {r.get('hand'):4d}  us={r.get('our_kept_cards')}  opp={r.get('opp_kept_cards')}  board={r.get('community')}  pnl={r.get('pnl'):+d}")

    fold_wins = sum(1 for r in results if r.get("opp_folded") and r.get("outcome") == "win")
    we_fold = sum(1 for r in results if r.get("we_folded"))
    print(f"  We folded: {we_fold} ({pct(we_fold, total)})")
    print(f"  Opp folded (our wins): {fold_wins} ({pct(fold_wins, total)})")
    large_swings = [r for r in results if r.get("large_swing")]
    print(f"  Large swings (|pnl|>=20): {len(large_swings)}")

    # ── 4. Preflop analysis ───────────────────────────────────────────────────
    pf = by_event.get("preflop_decision", [])
    if pf:
        pf_folds = [e for e in pf if e.get("action") == "FOLD"]
        near_fold = [e for e in pf_folds if e.get("equity") is not None and e.get("eq_gate") is not None
                     and e["equity"] >= e["eq_gate"] - 0.03]
        pf_raises = [e for e in pf if e.get("action") == "RAISE"]

        print(f"\nPreflop Analysis:")
        print(f"  Total preflop decisions: {len(pf)}")
        print(f"  Raises: {len(pf_raises)} ({pct(len(pf_raises), len(pf))})")
        print(f"  Folds: {len(pf_folds)} ({pct(len(pf_folds), len(pf))})")
        print(f"  Near-threshold folds (equity within 3% of gate): {len(near_fold)}")
        if near_fold:
            print(f"  Near-fold sample (up to 3):")
            for e in near_fold[:3]:
                print(f"    hand {e.get('hand'):4d}  eq={e.get('equity'):.3f}  gate={e.get('eq_gate'):.3f}  cards={e.get('hole_cards')}  reason={e.get('reason')}")
        reasons = defaultdict(int)
        for e in pf:
            reasons[e.get("reason", "unknown")] += 1
        print("  Decision reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # ── 5. Discard analysis ───────────────────────────────────────────────────
    disc = by_event.get("discard_decision", [])
    if disc:
        modes = defaultdict(int)
        for e in disc:
            modes[e.get("mode", "unknown")] += 1
        margins = [e.get("equity_margin", 0) for e in disc]
        flip_decisions = [e for e in disc if e.get("equity_margin", 1) < 0.02]

        print(f"\nDiscard Analysis:")
        print(f"  Total discards: {len(disc)}")
        print(f"  Sim mode breakdown: " + ", ".join(f"{k}={v}" for k, v in sorted(modes.items())))
        print(f"  Emergency mode: {modes.get('emergency', 0)} ({pct(modes.get('emergency', 0), len(disc))})")
        print(f"  Avg equity margin: {mean(margins):.4f}  stddev: {stddev(margins):.4f}")
        print(f"  Flip decisions (margin < 0.02): {len(flip_decisions)} ({pct(len(flip_decisions), len(disc))})")
        if flip_decisions:
            print(f"  Flip decision sample (up to 3):")
            for e in flip_decisions[:3]:
                print(f"    hand {e.get('hand'):4d}  margin={e.get('equity_margin'):.4f}  chosen={e.get('chosen_keep')}  equity={e.get('chosen_equity'):.4f}")

        fallbacks = by_event.get("discard_pool_fallback", [])
        if fallbacks:
            print(f"  Process pool fallbacks: {len(fallbacks)}")

    # ── 6. Postflop analysis ──────────────────────────────────────────────────
    post = by_event.get("postflop_decision", [])
    if post:
        changed = [e for e in post if e.get("baseline_changed")]
        semi_bluffs = [e for e in post if e.get("semi_bluff_fired")]
        bleedouts = [e for e in post if e.get("bleedout_lock")]

        # Win rates: exploit changed vs unchanged
        hand_results = {r.get("hand"): r.get("outcome") for r in results}
        changed_hands = set(e.get("hand") for e in changed)
        unchanged_hands = set(e.get("hand") for e in post if not e.get("baseline_changed"))
        changed_wins = sum(1 for h in changed_hands if hand_results.get(h) == "win")
        unchanged_wins = sum(1 for h in unchanged_hands if hand_results.get(h) == "win")

        print(f"\nPostflop Analysis:")
        print(f"  Total postflop decisions: {len(post)}")
        print(f"  Exploit changed decision: {len(changed)} ({pct(len(changed), len(post))})")
        if changed_hands:
            print(f"  Win rate (exploit changed hands):   {pct(changed_wins, len(changed_hands))}")
        if unchanged_hands:
            print(f"  Win rate (unchanged hands):         {pct(unchanged_wins, len(unchanged_hands))}")
        print(f"  Semi-bluffs fired: {len(semi_bluffs)}")
        print(f"  Bleedout locks: {len(bleedouts)}")

        # Texture penalty over-folding
        texture_folds = [e for e in post
                         if e.get("texture_adj", 0) < -0.10
                         and e.get("final_action") == "FOLD"
                         and e.get("raw_equity", 0) > 0.50]
        print(f"  Texture-penalty over-folds (adj<-0.10, raw_eq>0.50, fold): {len(texture_folds)}")
        if texture_folds:
            for e in texture_folds[:3]:
                print(f"    hand {e.get('hand'):4d}  street={e.get('street_name')}  raw_eq={e.get('raw_equity'):.3f}  texture={e.get('texture_adj'):.3f}  cards={e.get('my_cards')}  board={e.get('community')}")

        # Equity distribution by decision
        by_action = defaultdict(list)
        for e in post:
            by_action[e.get("final_action", "?")].append(e.get("adj_equity", 0))
        print(f"  Avg equity by action:")
        for action, eqs in sorted(by_action.items()):
            print(f"    {action}: {mean(eqs):.3f} (n={len(eqs)})")

    # ── 7. Bleedout locks ─────────────────────────────────────────────────────
    locks = by_event.get("bleedout_lock", [])
    if locks:
        print(f"\nBleedout Locks:")
        print(f"  Total activations: {len(locks)}")
        earliest = min(locks, key=lambda e: e.get("hand", 9999))
        print(f"  Earliest: hand {earliest.get('hand')}  pnl={earliest.get('running_pnl'):+d}  hands_left={earliest.get('hands_remaining')}")
        late_locks = [e for e in locks if e.get("hands_remaining", 0) > 200]
        print(f"  Early locks (>200 hands remaining): {len(late_locks)}")

    # ── 8. Opponent profile evolution ─────────────────────────────────────────
    snapshots = by_event.get("opp_profile_snapshot", [])
    if snapshots:
        print(f"\nOpponent Profile Snapshots ({len(snapshots)} total):")
        print(f"  {'Hand':>6}  {'fold_med_flop':>13}  {'call_dn_river':>13}  {'barrel_river':>12}  {'bluff_freq_adj':>14}  {'regime':>8}")
        for s in snapshots:
            fold_mf = s.get("fold_rates", {}).get("fold_medium_flop", 0)
            cdn_r = s.get("call_down", {}).get("river", 0)
            bar_r = s.get("barrel_rates", {}).get("river", 0)
            bfa = s.get("exploit_state", {}).get("bluff_freq_adj", 0)
            regime = s.get("regime_shift", 0)
            print(f"  {s.get('hand', '?'):>6}  {fold_mf:>13.3f}  {cdn_r:>13.3f}  {bar_r:>12.3f}  {bfa:>14.4f}  {regime:>8.3f}")

    print("\n" + "=" * 60)
    print(f"Log entries parsed: {len(entries)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
