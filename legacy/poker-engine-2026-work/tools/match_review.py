"""
Post-match review tool - combines bot log, CSV, and summary to show
where we lost and what patterns to fix.

Usage:
    python tools/match_review.py <match_id_or_base_path>

Examples:
    python tools/match_review.py 70807
    python tools/match_review.py "match logs/match_70807"
    python tools/match_review.py "match logs/match_70807.txt"
"""

import ast
import csv
import json
import re
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path


# -- Path resolution ------------------------------------------------------------

def resolve_base(arg):
    """Return base path (no extension) from user arg (ID, full path, etc.)."""
    p = Path(arg)
    if p.suffix in (".txt", ".csv"):
        p = p.with_suffix("")
    stem = p.stem
    for sfx in ("_bot", "_csv"):
        if stem.endswith(sfx):
            p = p.with_name(stem[: -len(sfx)])
            break
    if p.name.isdigit():
        p = Path("match logs") / f"match_{p.name}"
    return p


# -- Loaders --------------------------------------------------------------------

def load_summary(base):
    return base.with_suffix(".txt").read_text(encoding="utf-8")


def parse_summary(text):
    info = {}
    m = re.search(r"Final results - (.+?) bankroll: (-?\d+), (.+?) bankroll: (-?\d+)", text)
    if m:
        info["team0_name"] = m.group(1)
        info["team0_final"] = int(m.group(2))
        info["team1_name"] = m.group(3)
        info["team1_final"] = int(m.group(4))
    m = re.search(r"Time used - .+?: ([\d.]+) seconds, .+?: ([\d.]+) seconds", text)
    if m:
        info["team0_time"] = float(m.group(1))
        info["team1_time"] = float(m.group(2))
    m = re.search(r"Time limit: (\d+)", text)
    if m:
        info["time_limit"] = int(m.group(1))
    m = re.search(r"- (match_\w+) - INFO", text)
    if m:
        info["match_id"] = m.group(1)
    return info


def load_bot_events(base):
    events = []
    path = Path(str(base) + "_bot.txt")
    with path.open(encoding="utf-8") as f:
        for line in f:
            m = re.search(r"- INFO - (\{.+)$", line)
            if m:
                try:
                    events.append(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    pass
    return events


def _parse_cards(s):
    s = s.strip()
    if not s or s == "[]":
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def load_csv_rows(base):
    path = Path(str(base) + "_csv.txt")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    header_idx = next(i for i, l in enumerate(lines) if not l.startswith("#"))
    card_cols = {"team_0_cards", "team_1_cards", "board_cards", "team_0_discarded", "team_1_discarded"}
    int_cols = {"hand_number", "active_team", "action_amount",
                "team_0_bankroll", "team_1_bankroll", "team_0_bet", "team_1_bet",
                "action_keep_1", "action_keep_2"}
    rows = []
    reader = csv.DictReader(StringIO("".join(lines[header_idx:])))
    for row in reader:
        for col in int_cols:
            if col in row:
                row[col] = int(row[col])
        for col in card_cols:
            if col in row:
                row[col] = _parse_cards(row[col])
        rows.append(row)
    return rows


# -- Per-hand join --------------------------------------------------------------

def build_hands(bot_events, csv_rows):
    hands = defaultdict(lambda: {
        "hand_start": None,
        "preflop_decisions": [],
        "discard_decision": None,
        "postflop_decisions": [],
        "hand_result": None,
        "bleedout_lock": None,
        "csv_rows": [],
    })
    for e in bot_events:
        h = e.get("hand", -1)
        ev = e.get("event")
        if ev == "hand_start":
            hands[h]["hand_start"] = e
        elif ev == "preflop_decision":
            hands[h]["preflop_decisions"].append(e)
        elif ev == "discard_decision":
            hands[h]["discard_decision"] = e
        elif ev == "postflop_decision":
            hands[h]["postflop_decisions"].append(e)
        elif ev == "hand_result":
            hands[h]["hand_result"] = e
        elif ev == "bleedout_lock":
            hands[h]["bleedout_lock"] = e
    for row in csv_rows:
        hands[row["hand_number"]]["csv_rows"].append(row)
    return hands


# -- Formatting helpers ---------------------------------------------------------

def fc(cards):
    return "[" + ",".join(cards) + "]" if cards else "[]"


def pct(num, den):
    return f"{100.0 * num / den:.1f}%" if den else "N/A"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


# -- Section 1: Match Header ----------------------------------------------------

def print_header(info, base):
    print("=" * 70)
    mid = info.get("match_id", base.name)
    t0 = info.get("team0_name", "Us")
    t1 = info.get("team1_name", "Opp")
    f0 = info.get("team0_final", None)
    f1 = info.get("team1_final", None)
    if f0 is not None and f1 is not None:
        winner = t0 if f0 > f1 else (t1 if f1 > f0 else "TIE")
        result_str = "  Result: {} {:+d}  vs  {} {:+d}  =>  Winner: {}".format(
            t0, f0, t1, f1, winner)
    else:
        result_str = "  Result: unknown"
    print("  {}  |  {} vs {}".format(mid, t0, t1))
    print(result_str)
    print("  Time:   us {:.1f}s  |  opp {:.1f}s  |  limit {}s".format(
        info.get("team0_time", 0),
        info.get("team1_time", 0),
        info.get("time_limit", 1500)))
    print("=" * 70)


# -- Section 2: Worst Losses ----------------------------------------------------

def loss_category(h):
    r = h["hand_result"]
    if not r or r["outcome"] != "loss":
        return None
    if r.get("we_folded"):
        return "FOLD_LOSS"
    if r.get("showdown"):
        return "SHOWDOWN_LOSS"
    return "LOSS"


def print_losses(hands, n=10):
    print(f"\n-- TOP {n} LOSING HANDS ----------------------------------------------")
    losing = [(hnum, h) for hnum, h in hands.items()
              if h["hand_result"] and h["hand_result"]["outcome"] == "loss"]
    losing.sort(key=lambda x: x[1]["hand_result"]["pnl"])

    if not losing:
        print("  No losing hands.")
        return

    for hnum, h in losing[:n]:
        r = h["hand_result"]
        cat = loss_category(h)
        pnl = r["pnl"]
        swing = " [BIG SWING]" if r.get("large_swing") else ""
        print(f"\n  Hand {hnum:4d}  {cat}  pnl={pnl:+d}{swing}")
        print(f"           us={fc(r.get('our_kept_cards',[]))}  "
              f"opp={fc(r.get('opp_kept_cards',[]))}  "
              f"board={fc(r.get('community',[]))}")

        # Preflop summary (last decision = final action)
        pf_decisions = h["preflop_decisions"]
        if pf_decisions:
            last = pf_decisions[-1]
            eq = last.get("equity")
            gate = last.get("eq_gate")
            action = last.get("action", "-")
            reason = last.get("reason", "-")
            eq_str = f"eq={eq:.3f} " if eq is not None else ""
            gate_str = f"gate={gate:.2f} " if gate is not None else ""
            print(f"           preflop: {action} ({eq_str}{gate_str}reason={reason})")

        # Discard summary
        dd = h["discard_decision"]
        if dd:
            kept = fc(dd.get("chosen_keep", []))
            ceq = dd.get("chosen_equity", 0)
            margin = dd.get("equity_margin", 0)
            combos = dd.get("keep_combos", [])
            alt = ""
            if len(combos) >= 2:
                s2 = combos[1]
                alt = f"  (2nd: {fc(s2['keep'])} eq={s2['equity']:.3f})"
            print(f"           discard: kept {kept} eq={ceq:.3f} margin={margin:.3f}{alt}")

        # Postflop streets
        for pd in h["postflop_decisions"]:
            sname = pd.get("street_name", "-")
            raw = pd.get("raw_equity", 0)
            adj = pd.get("adj_equity", 0)
            tex = pd.get("texture_adj", 0)
            action = pd.get("final_action", "-")
            tex_str = f" tex={tex:+.2f}" if abs(tex) > 0.01 else ""
            # Dominant texture component
            tb = pd.get("texture_breakdown", {})
            if tb:
                dom_k, dom_v = max(tb.items(), key=lambda x: abs(x[1]))
                dom_str = f" ({dom_k})" if abs(dom_v) > 0.05 else ""
            else:
                dom_str = ""
            print(f"           {sname:5s}:  {action}  raw={raw:.3f} adj={adj:.3f}{tex_str}{dom_str}")

        if r.get("showdown") and r["outcome"] == "loss":
            print(f"           -> lost at showdown")


# -- Section 3: Systematic Patterns --------------------------------------------

def print_patterns(hands):
    print("\n-- SYSTEMATIC MISTAKE PATTERNS --------------------------------------")
    results = [h["hand_result"] for h in hands.values() if h["hand_result"]]
    total = len(results)

    # -- a) Discard quality
    discards = [h["discard_decision"] for h in hands.values() if h["discard_decision"]]
    print(f"\n  a) Discard Quality ({len(discards)} hands with discard):")
    if discards:
        margins = [d.get("equity_margin", 0) for d in discards]
        flip = [d for d in discards if d.get("equity_margin", 1) < 0.02]
        emergency = [d for d in discards if d.get("mode") == "emergency"]
        # Coin-flip win rate
        flip_wins = sum(
            1 for d in flip
            if hands.get(d["hand"], {}).get("hand_result", {}) and
               hands[d["hand"]]["hand_result"]["outcome"] == "win"
        )
        # Suboptimal picks (sanity check)
        suboptimal = [d for d in discards
                      if d.get("keep_combos") and
                      d["keep_combos"][0]["equity"] > d.get("chosen_equity", 0) + 0.001]
        print(f"     Avg margin: {mean(margins):.4f}  |  "
              f"Coin-flip (<0.02): {len(flip)} ({pct(len(flip), len(discards))})  "
              f"win rate on flips: {pct(flip_wins, len(flip))}")
        print(f"     Emergency mode: {len(emergency)} ({pct(len(emergency), len(discards))})")
        if suboptimal:
            print(f"     *** SUBOPTIMAL PICKS: {len(suboptimal)} hands (bug!)")
            for d in suboptimal[:3]:
                print(f"         hand {d['hand']}  chose eq={d.get('chosen_equity'):.3f}  best={d['keep_combos'][0]['equity']:.3f}")
        # Show worst coin-flip hands
        if flip:
            print(f"     Coin-flip discard sample (up to 3):")
            for d in flip[:3]:
                combos = d.get("keep_combos", [])
                top2 = combos[:2]
                r = hands.get(d["hand"], {}).get("hand_result")
                outcome = r["outcome"] if r else "-"
                print(f"       hand {d['hand']:4d}  "
                      f"best={fc(top2[0]['keep']) if top2 else '-'} eq={top2[0]['equity']:.3f}  "
                      f"2nd={fc(top2[1]['keep']) if len(top2)>1 else '-'} eq={top2[1]['equity']:.3f}  "
                      f"-> {outcome}")
    else:
        print("     No discard decisions found.")

    # -- b) Texture over-folds
    postflop_all = [pd for h in hands.values() for pd in h["postflop_decisions"]]
    over_folds = [e for e in postflop_all
                  if e.get("texture_adj", 0) < -0.10
                  and e.get("final_action") == "FOLD"
                  and e.get("raw_equity", 0) > 0.50]
    print(f"\n  b) Texture Over-folds (tex<-0.10, raw_eq>0.50, FOLD): {len(over_folds)} "
          f"({pct(len(over_folds), total)} of hands)")
    for e in over_folds[:5]:
        r = hands.get(e["hand"], {}).get("hand_result")
        outcome = r["outcome"] if r else "-"
        print(f"     hand {e['hand']:4d}  {e.get('street_name'):5s}  "
              f"raw={e.get('raw_equity'):.3f}  tex={e.get('texture_adj'):.3f}  "
              f"{fc(e.get('my_cards',[]))} board={fc(e.get('community',[]))}  -> {outcome}")

    # -- c) Preflop
    pf_all = [pf for h in hands.values() for pf in h["preflop_decisions"]]
    pf_folds = [e for e in pf_all if e.get("action") == "FOLD"]
    near_folds = [e for e in pf_folds
                  if e.get("equity") is not None and e.get("eq_gate") is not None
                  and e["equity"] >= e["eq_gate"] - 0.03]
    pf_raise_hands = set(e["hand"] for e in pf_all if e.get("action") == "RAISE")
    raise_sd_losses = [hnum for hnum in pf_raise_hands
                       if hands[hnum]["hand_result"] and
                       hands[hnum]["hand_result"]["outcome"] == "loss" and
                       hands[hnum]["hand_result"].get("showdown")]
    print(f"\n  c) Preflop:")
    print(f"     Folds: {len(pf_folds)}  near-gate (within 3%): {len(near_folds)}")
    print(f"     Raise hands: {len(pf_raise_hands)}  -> showdown losses: {len(raise_sd_losses)} "
          f"({pct(len(raise_sd_losses), len(pf_raise_hands))})")
    if near_folds:
        print(f"     Near-gate fold sample (up to 3):")
        for e in near_folds[:3]:
            print(f"       hand {e['hand']:4d}  eq={e.get('equity'):.3f}  "
                  f"gate={e.get('eq_gate'):.3f}  cards={e.get('hole_cards')}")

    # -- d) Call-down losses
    call_down_losses = []
    for hnum, h in hands.items():
        r = h["hand_result"]
        if not r or r["outcome"] != "loss" or not r.get("showdown"):
            continue
        called_streets = [e for e in h["postflop_decisions"]
                          if e.get("final_action") == "CALL" and e.get("to_call", 0) > 0]
        if len(called_streets) >= 2:
            call_down_losses.append((hnum, h, called_streets))
    print(f"\n  d) Call-down Losses (called 2+ streets, lost showdown): {len(call_down_losses)}")
    for hnum, h, streets in call_down_losses[:5]:
        r = h["hand_result"]
        streets_str = "+".join(e.get("street_name", "-") for e in streets)
        print(f"     hand {hnum:4d}  called [{streets_str}]  "
              f"us={fc(r.get('our_kept_cards',[]))}  "
              f"opp={fc(r.get('opp_kept_cards',[]))}  pnl={r['pnl']:+d}")


# -- Section 4: Opponent Analysis ----------------------------------------------

def print_opponent(hands, csv_rows):
    print("\n-- OPPONENT PATTERN ANALYSIS -----------------------------------------")

    opp_actions = [r for r in csv_rows
                   if r["active_team"] == 1 and r["action_type"] != "DISCARD"]
    by_street = defaultdict(list)
    for r in opp_actions:
        by_street[r["street"]].append(r["action_type"])

    print("\n  Opponent action rates by street:")
    for street in ["Pre-Flop", "Flop", "Turn", "River"]:
        acts = by_street[street]
        if not acts:
            continue
        n = len(acts)
        folds = acts.count("FOLD")
        raises = acts.count("RAISE")
        calls = acts.count("CALL")
        checks = acts.count("CHECK")
        print(f"    {street:8s}: n={n:4d}  "
              f"fold={pct(folds,n):6s}  raise={pct(raises,n):6s}  "
              f"call={pct(calls,n):6s}  check={pct(checks,n):6s}")

    # Opponent raise sizes
    opp_raises = [r for r in opp_actions
                  if r["action_type"] == "RAISE" and r["action_amount"] > 0]
    if opp_raises:
        avg_raise = mean([r["action_amount"] for r in opp_raises])
        print(f"\n  Opp avg raise amount: {avg_raise:.1f}")

    # Showdown stats
    showdown_hands = [(hnum, h) for hnum, h in hands.items()
                      if h["hand_result"] and h["hand_result"].get("showdown")]
    opp_cards_sd = [h["hand_result"].get("opp_kept_cards", []) for _, h in showdown_hands]

    def has_pair(cards):
        ranks = [c[:-1] for c in cards]
        return len(ranks) != len(set(ranks))

    opp_pairs = sum(1 for c in opp_cards_sd if has_pair(c))
    print(f"\n  Opp showdown stats: {len(opp_cards_sd)} showdowns  "
          f"pair at showdown: {opp_pairs} ({pct(opp_pairs, len(opp_cards_sd))})")

    # Big swings we lost
    big_losses = [(hnum, h) for hnum, h in hands.items()
                  if h["hand_result"] and
                  h["hand_result"]["outcome"] == "loss" and
                  h["hand_result"].get("large_swing")]
    print(f"\n  Large swings we lost (pnl <= -20): {len(big_losses)}")
    for hnum, h in big_losses[:8]:
        r = h["hand_result"]
        print(f"    hand {hnum:4d}  pnl={r['pnl']:+d}  "
              f"us={fc(r.get('our_kept_cards',[]))}  "
              f"opp={fc(r.get('opp_kept_cards',[]))}  "
              f"board={fc(r.get('community',[]))}")

    # Hands where opponent won without going to showdown (they bluffed us out)
    opp_fold_wins = [(hnum, h) for hnum, h in hands.items()
                     if h["hand_result"] and
                     h["hand_result"]["outcome"] == "loss" and
                     h["hand_result"].get("we_folded")]
    print(f"\n  Hands we folded and lost chips: {len(opp_fold_wins)}")
    if opp_fold_wins:
        fold_pnls = [h["hand_result"]["pnl"] for _, h in opp_fold_wins]
        print(f"    Avg fold loss: {mean(fold_pnls):.1f}  Total: {sum(fold_pnls):+d}")


# -- Section 5: Suggestions ----------------------------------------------------

def print_suggestions(hands, csv_rows):
    print("\n-- IMPROVEMENT SUGGESTIONS -------------------------------------------")
    suggestions = []

    results = [h["hand_result"] for h in hands.values() if h["hand_result"]]
    total = len(results)
    showdowns = [r for r in results if r.get("showdown")]
    sd_losses = [r for r in showdowns if r["outcome"] == "loss"]

    # Showdown loss rate
    if showdowns and len(sd_losses) / len(showdowns) > 0.55:
        suggestions.append(
            f"Showdown loss rate {pct(len(sd_losses), len(showdowns))} - "
            f"calling too often with losing hands; tighten river call threshold")

    # Texture over-folds
    postflop_all = [pd for h in hands.values() for pd in h["postflop_decisions"]]
    over_folds = [e for e in postflop_all
                  if e.get("texture_adj", 0) < -0.10
                  and e.get("final_action") == "FOLD"
                  and e.get("raw_equity", 0) > 0.50]
    if total > 0 and len(over_folds) / total > 0.05:
        suggestions.append(
            f"Texture over-fold rate {pct(len(over_folds), total)} - "
            f"reduce texture_adj magnitude on monotone boards when raw equity > 50%")

    # Discard coin-flips
    discards = [h["discard_decision"] for h in hands.values() if h["discard_decision"]]
    if discards:
        flip = [d for d in discards if d.get("equity_margin", 1) < 0.02]
        if len(flip) / len(discards) > 0.20:
            suggestions.append(
                f"Coin-flip discard rate {pct(len(flip), len(discards))} - "
                f"increase MC sim count for tighter equity resolution at discard")
        emergency = [d for d in discards if d.get("mode") == "emergency"]
        if len(emergency) / len(discards) > 0.15:
            suggestions.append(
                f"Emergency discard mode {pct(len(emergency), len(discards))} - "
                f"time pressure degrading discard quality; profile sim budget allocation")

    # Opponent fold vs our preflop fold
    opp_actions = [r for r in csv_rows if r["active_team"] == 1 and r["action_type"] != "DISCARD"]
    opp_folds = [r for r in opp_actions if r["action_type"] == "FOLD"]
    opp_fold_rate = len(opp_folds) / len(opp_actions) if opp_actions else 0
    pf_all = [pf for h in hands.values() for pf in h["preflop_decisions"]]
    pf_folds = [e for e in pf_all if e.get("action") == "FOLD"]
    pf_fold_rate = len(pf_folds) / len(pf_all) if pf_all else 0
    if opp_fold_rate > 0.40 and pf_fold_rate > 0.35:
        suggestions.append(
            f"Opp folds {pct(len(opp_folds), len(opp_actions))} - "
            f"we fold preflop {pct(len(pf_folds), len(pf_all))} vs a passive opp; "
            f"lower preflop fold threshold to exploit")

    # Early bleedout locks
    bleedout_hands = [h for h in hands.values() if h.get("bleedout_lock")]
    early = [h for h in bleedout_hands
             if h["bleedout_lock"] and h["bleedout_lock"].get("hands_remaining", 0) > 300]
    if early:
        suggestions.append(
            f"Bleedout lock triggered early ({len(early)} times with 300+ hands left) - "
            f"may be capping upside in a recoverable deficit")

    # Near-gate preflop folds
    near_folds = [e for e in pf_folds
                  if e.get("equity") is not None and e.get("eq_gate") is not None
                  and e["equity"] >= e["eq_gate"] - 0.03]
    if len(near_folds) > 5:
        suggestions.append(
            f"{len(near_folds)} preflop folds were within 3% of the equity gate - "
            f"consider lowering the gate by 0.01-0.02 to capture marginal edges")

    if not suggestions:
        suggestions.append("No major systematic issues detected.")

    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")


# -- Main -----------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/match_review.py <match_id_or_base_path>")
        print("Example: python tools/match_review.py 70807")
        sys.exit(1)

    base = resolve_base(Path(sys.argv[1]))

    summary_text = load_summary(base)
    info = parse_summary(summary_text)
    bot_events = load_bot_events(base)
    csv_rows = load_csv_rows(base)
    hands = build_hands(bot_events, csv_rows)

    print_header(info, base)
    print_losses(hands)
    print_patterns(hands)
    print_opponent(hands, csv_rows)
    print_suggestions(hands, csv_rows)

    print("\n" + "=" * 70)
    print(f"  Hands: {len(hands)}  |  Bot events: {len(bot_events)}  |  CSV rows: {len(csv_rows)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
