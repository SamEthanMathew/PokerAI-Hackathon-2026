#!/usr/bin/env python3
"""Analyze poker match bot logs + CSV for Ctrl+Alt+Defeat diagnostics."""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

JSON_RE = re.compile(r"\{.*\}\s*$")


def parse_bot_log(path: Path):
    events = []
    for line in path.open(encoding="utf-8", errors="replace"):
        m = JSON_RE.search(line)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        events.append(obj)
    return events


def index_hands(events):
    """Per hand: postflop list, preflop list, hand_result (last wins), hand_start."""
    by_hand = defaultdict(
        lambda: {
            "postflop": [],
            "preflop": [],
            "hand_results": [],
            "hand_start": None,
        }
    )
    for e in events:
        ev = e.get("event")
        h = e.get("hand")
        if h is None:
            continue
        if ev == "postflop_decision":
            by_hand[h]["postflop"].append(e)
        elif ev == "preflop_decision":
            by_hand[h]["preflop"].append(e)
        elif ev == "hand_result":
            by_hand[h]["hand_results"].append(e)
        elif ev == "hand_start":
            by_hand[h]["hand_start"] = e
    return by_hand


def last_hand_result(hdata):
    if not hdata["hand_results"]:
        return None
    return hdata["hand_results"][-1]


def parse_csv_hands(path: Path, our_team: int):
    """Rows grouped by hand_number; fold-to-raise uses active_team indices."""
    rows_by_hand = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in r:
            try:
                hn = int(row["hand_number"])
            except (KeyError, ValueError):
                continue
            rows_by_hand[hn].append(row)
    return rows_by_hand


def fold_to_raise_stats(rows_by_hand, our_team: int, hand_filter: set):
    """
    When our bot RAISEs on a street, does the next action (any team) show opponent folding
    immediately after our raise? Standard: next row has active_team == opponent and FOLD.
    """
    streets = ["Pre-Flop", "Flop", "Turn", "River"]
    stats = {s: {"raise": 0, "fold": 0} for s in streets}
    opp = 1 - our_team

    for hn, rows in rows_by_hand.items():
        if hn not in hand_filter:
            continue
        for i, row in enumerate(rows):
            if row.get("action_type") != "RAISE":
                continue
            try:
                at = int(row["active_team"])
            except (KeyError, ValueError):
                continue
            if at != our_team:
                continue
            st = row.get("street", "")
            if st not in stats:
                continue
            stats[st]["raise"] += 1
            if i + 1 < len(rows):
                nxt = rows[i + 1]
                try:
                    nat = int(nxt["active_team"])
                except (KeyError, ValueError):
                    continue
                if nat == opp and nxt.get("action_type") == "FOLD":
                    stats[st]["fold"] += 1
    return stats


def main():
    matches = [
        {
            "id": "74088",
            "bot": Path("match_74088_bot.txt"),
            "csv": Path("match_74088_csv.txt"),
            "our_team": 1,
            "exclude_from": 998,
        },
        {
            "id": "74751",
            "bot": Path("match_74751_bot.txt"),
            "csv": Path("match_74751_csv.txt"),
            "our_team": 0,
            "exclude_from": 552,
        },
        {
            "id": "75338",
            "bot": Path("match_75338_bot.txt"),
            "csv": Path("match_75338_csv.txt"),
            "our_team": 1,
            "exclude_from": None,
        },
    ]
    base = Path(__file__).resolve().parent

    STRENGTH_ORDER = ("monster", "strong", "medium", "draw", "weak")

    for m in matches:
        bot_path = base / m["bot"]
        csv_path = base / m["csv"]
        our_team = m["our_team"]
        excl = m["exclude_from"]

        events = parse_bot_log(bot_path)
        by_hand = index_hands(events)
        rows_by_hand = parse_csv_hands(csv_path, our_team)

        all_hands = set(by_hand.keys())
        if excl is not None:
            hand_filter = {h for h in all_hands if h < excl}
        else:
            hand_filter = all_hands

        # --- 1. PnL by last postflop strength
        strength_pnl = defaultdict(lambda: [0, 0])  # sum, count
        no_postflop = 0
        for h in sorted(hand_filter):
            hdata = by_hand[h]
            pf = hdata["postflop"]
            hr = last_hand_result(hdata)
            if not hr:
                continue
            pnl = hr.get("pnl")
            if pnl is None:
                continue
            if not pf:
                no_postflop += 1
                continue
            last_pf = pf[-1]
            st = last_pf.get("strength", "unknown")
            strength_pnl[st][0] += pnl
            strength_pnl[st][1] += 1

        # --- 2. Showdown
        sd_win_pots = []
        sd_loss_pots = []
        sd_loss_monster = 0
        for h in sorted(hand_filter):
            hdata = by_hand[h]
            hr = last_hand_result(hdata)
            if not hr or not hr.get("showdown"):
                continue
            pf = hdata["postflop"]
            pnl = hr.get("pnl", 0)
            pot = None
            if pf:
                pot = pf[-1].get("pot")
            if pot is None:
                continue
            if pnl > 0:
                sd_win_pots.append(pot)
            elif pnl < 0:
                sd_loss_pots.append(pot)
                if pf and pf[-1].get("strength") == "monster":
                    sd_loss_monster += 1

        def avg(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        # --- 3. Semi-bluff
        semi_events = []
        semi_hands = set()
        semi_equities = []
        for h in hand_filter:
            for e in by_hand[h]["postflop"]:
                if e.get("semi_bluff_fired"):
                    semi_events.append((h, e))
                    semi_hands.add(h)
                    if e.get("adj_equity") is not None:
                        semi_equities.append(e["adj_equity"])
        semi_pnl = 0
        for h in semi_hands:
            hr = last_hand_result(by_hand[h])
            if hr:
                semi_pnl += hr.get("pnl", 0)

        # --- 4. Large |pnl| >= 50
        large = {"sd_win": 0, "sd_loss": 0, "fold_win": 0, "we_fold_loss": 0}
        for h in sorted(hand_filter):
            hr = last_hand_result(by_hand[h])
            if not hr:
                continue
            pnl = abs(hr.get("pnl", 0))
            if pnl < 50:
                continue
            sd = hr.get("showdown")
            wf = hr.get("we_folded")
            of = hr.get("opp_folded")
            if sd:
                if hr.get("pnl", 0) > 0:
                    large["sd_win"] += 1
                else:
                    large["sd_loss"] += 1
            else:
                if of and not wf:
                    large["fold_win"] += 1
                elif wf:
                    large["we_fold_loss"] += 1

        # --- 5. Equity calibration (showdown only)
        buckets = {
            "0-0.3": [0, 0],
            "0.3-0.5": [0, 0],
            "0.5-0.7": [0, 0],
            "0.7-1.0": [0, 0],
        }

        def bucket_eq(x):
            if x < 0.3:
                return "0-0.3"
            if x < 0.5:
                return "0.3-0.5"
            if x < 0.7:
                return "0.5-0.7"
            return "0.7-1.0"

        for h in sorted(hand_filter):
            hdata = by_hand[h]
            hr = last_hand_result(hdata)
            if not hr or not hr.get("showdown"):
                continue
            pf = hdata["postflop"]
            if not pf:
                continue
            adj = pf[-1].get("adj_equity")
            if adj is None:
                continue
            b = bucket_eq(adj)
            buckets[b][1] += 1
            if hr.get("pnl", 0) > 0:
                buckets[b][0] += 1

        # --- 6. Fold-to-raise (CSV)
        ftr = fold_to_raise_stats(rows_by_hand, our_team, hand_filter)

        # --- 7. Bet sizing: avg raise amount by street
        raise_sums = {"Pre-Flop": [0, 0], "Flop": [0, 0], "Turn": [0, 0], "River": [0, 0]}
        for h in hand_filter:
            for e in by_hand[h]["preflop"]:
                if e.get("action") == "RAISE" and e.get("amount") is not None:
                    raise_sums["Pre-Flop"][0] += e["amount"]
                    raise_sums["Pre-Flop"][1] += 1
            for e in by_hand[h]["postflop"]:
                if e.get("final_action") == "RAISE" and e.get("final_amount") is not None:
                    stname = e.get("street_name", "").lower()
                    key = {"flop": "Flop", "turn": "Turn", "river": "River"}.get(stname)
                    if key:
                        raise_sums[key][0] += e["final_amount"]
                        raise_sums[key][1] += 1

        # --- 8. Comeback mode (hand_start)
        cb_hands = set()
        for h in hand_filter:
            hs = by_hand[h].get("hand_start")
            if hs and hs.get("comeback_mode"):
                cb_hands.add(h)
        cb_pnl = 0
        norm_pnl = 0
        for h in hand_filter:
            hr = last_hand_result(by_hand[h])
            if not hr:
                continue
            p = hr.get("pnl", 0)
            if h in cb_hands:
                cb_pnl += p
            else:
                norm_pnl += p

        # Output
        print("=" * 72)
        print(f"MATCH {m['id']}  |  Our team: {our_team}  |  Real hands: {len([h for h in hand_filter if last_hand_result(by_hand[h])])}")
        if excl is not None:
            print(f"Bleed-out excluded: hands >= {excl}")
        else:
            print("No bleed-out exclusion")

        # verify pnl sum
        total_pnl = sum(
            last_hand_result(by_hand[h]).get("pnl", 0)
            for h in hand_filter
            if last_hand_result(by_hand[h])
        )
        print(f"Sum PnL (filtered): {total_pnl}")

        print("\n### 1. PnL BY LAST POSTFLOP STRENGTH")
        print("| Strength | Hands | Total PnL | Avg PnL |")
        print("|----------|-------|-----------|---------|")
        for st in STRENGTH_ORDER:
            s, c = strength_pnl.get(st, [0, 0])
            avg_p = s / c if c else float("nan")
            print(f"| {st} | {c} | {s} | {avg_p:.4f} |")
        other = [(k, v) for k, v in strength_pnl.items() if k not in STRENGTH_ORDER]
        for k, (s, c) in sorted(other):
            avg_p = s / c if c else float("nan")
            print(f"| {k} | {c} | {s} | {avg_p:.4f} |")
        print(f"(Hands with no postflop: {no_postflop})")

        print("\n### 2. SHOWDOWN")
        print(f"| Avg pot @ showdown WIN | {avg(sd_win_pots):.2f} | (n={len(sd_win_pots)}) |")
        print(f"| Avg pot @ showdown LOSS | {avg(sd_loss_pots):.2f} | (n={len(sd_loss_pots)}) |")
        print(f"| Showdown losses with last strength=monster | {sd_loss_monster} |")

        print("\n### 3. SEMI-BLUFF")
        print(f"| semi_bluff_fired events | {len(semi_events)} |")
        print(f"| Distinct hands with ≥1 semi-bluff | {len(semi_hands)} |")
        print(f"| Total PnL those hands | {semi_pnl} |")
        print(f"| Avg adj_equity when fired | {avg(semi_equities):.4f} | (n={len(semi_equities)}) |")

        print("\n### 4. LARGE POT (|pnl| ≥ 50)")
        for k, v in large.items():
            print(f"| {k} | {v} |")

        print("\n### 5. EQUITY CALIBRATION (showdown, last postflop adj_equity)")
        print("| Bucket | Wins | Total | Win rate |")
        print("|--------|------|-------|----------|")
        for bname in ["0-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0"]:
            w, t = buckets[bname]
            wr = w / t if t else float("nan")
            print(f"| {bname} | {w} | {t} | {wr:.4f} |")

        print("\n### 6. OPPONENT FOLD-TO-RAISE (next action after our RAISE)")
        print("| Street | Folds | Raises | Rate |")
        print("|--------|-------|--------|------|")
        for st in ["Pre-Flop", "Flop", "Turn", "River"]:
            r = ftr[st]["raise"]
            f = ftr[st]["fold"]
            rate = f / r if r else float("nan")
            print(f"| {st} | {f} | {r} | {rate:.4f} |")

        print("\n### 7. BET SIZING (avg raise amount)")
        print("| Street | Avg amount | Count |")
        print("|--------|------------|-------|")
        for st in ["Pre-Flop", "Flop", "Turn", "River"]:
            s, c = raise_sums[st]
            av = s / c if c else float("nan")
            print(f"| {st} | {av:.2f} | {c} |")

        print("\n### 8. COMEBACK MODE (hand_start.comeback_mode)")
        print(f"| Hands with comeback_mode | {len(cb_hands)} |")
        print(f"| PnL in comeback hands | {cb_pnl} |")
        print(f"| PnL in normal hands | {norm_pnl} |")

        print()


if __name__ == "__main__":
    main()
