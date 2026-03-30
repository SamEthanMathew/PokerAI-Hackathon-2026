"""
Analyses equity trajectory and actions street by street across both matches.
Focuses on: where equity dropped, what action we took at that point,
and whether we were already losing ground before the river.
"""
import json, re, csv, ast
from collections import defaultdict
from pathlib import Path


def load_bot_log(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"- INFO - (.+)$", line)
            if m:
                try:
                    entries.append(json.loads(m.group(1)))
                except:
                    pass
    return entries


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        try:
            row["hand_number"] = int(row["hand_number"])
        except:
            pass
        rows.append(row)
    return rows


def build_hands(events, csv_rows):
    hands = defaultdict(lambda: {
        "preflop": None, "discard": None,
        "postflop": [], "result": None, "csv": []
    })
    for e in events:
        h = e.get("hand", -1)
        ev = e.get("event")
        if ev == "preflop_decision":
            hands[h]["preflop"] = e
        elif ev == "discard_decision":
            hands[h]["discard"] = e
        elif ev == "postflop_decision":
            hands[h]["postflop"].append(e)
        elif ev == "hand_result":
            hands[h]["result"] = e
    for row in csv_rows:
        hands[row["hand_number"]]["csv"].append(row)
    return hands


def avg(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def get_street_data(postflop):
    by_street = {}
    for p in postflop:
        sn = (p.get("street_name") or p.get("street") or "?").lower()
        # keep last decision per street (there can be multiple per street due to reraises)
        by_street[sn] = p
    return by_street.get("flop"), by_street.get("turn"), by_street.get("river")


def chips_in_by_street(csv_rows, team=0):
    """Sum up how many chips we put in per street from CSV."""
    street_chips = defaultdict(int)
    prev_bet = 0
    for row in csv_rows:
        if int(row.get("active_team", -1)) != team:
            continue
        action = row.get("action_type", "")
        if action in ("RAISE", "BET", "CALL", "ALL_IN"):
            try:
                cur = int(row.get("team_0_bet", 0) if team == 0 else row.get("team_1_bet", 0))
                diff = max(0, cur - prev_bet)
                street_chips[row.get("street", "?")] += diff
                prev_bet = cur
            except:
                pass
    return street_chips


def analyze_match(match_id):
    base = Path("match logs") / f"match_{match_id}"
    events = load_bot_log(str(base) + "_bot.txt")
    csv_rows = load_csv(str(base) + "_csv.txt")
    hands = build_hands(events, csv_rows)
    return hands, match_id


def run_analysis(hands_dict, match_id):
    print()
    print("=" * 72)
    print(f"MATCH {match_id} - Street-by-Street Equity & Action Analysis")
    print("=" * 72)

    sd_losses = []
    sd_wins = []

    for hnum in sorted(hands_dict.keys()):
        h = hands_dict[hnum]
        r = h["result"]
        if not r or not r.get("showdown"):
            continue

        disc = h["discard"]
        flop, turn, river = get_street_data(h["postflop"])
        disc_eq   = disc.get("chosen_equity") if disc else None
        flop_adj  = flop.get("adj_equity")  if flop  else None
        flop_act  = flop.get("final_action") if flop  else None
        turn_adj  = turn.get("adj_equity")  if turn  else None
        turn_act  = turn.get("final_action") if turn  else None
        river_adj = river.get("adj_equity") if river else None
        river_act = river.get("final_action") if river else None

        row = {
            "hand": hnum, "outcome": r.get("outcome"), "pnl": r.get("pnl", 0),
            "disc_eq": disc_eq,
            "flop_adj": flop_adj, "flop_act": flop_act,
            "turn_adj": turn_adj, "turn_act": turn_act,
            "river_adj": river_adj, "river_act": river_act,
        }

        if r.get("outcome") == "loss":
            sd_losses.append(row)
        else:
            sd_wins.append(row)

    all_sd = sd_losses + sd_wins
    print(f"\nShowdown hands: {len(all_sd)}  Losses: {len(sd_losses)}  Wins: {len(sd_wins)}")

    # ── Where did equity drop most? ────────────────────────────────────────
    print("\n--- Where Equity Dropped on Losing Showdown Hands ---")
    drop_at_flop  = 0
    drop_at_turn  = 0
    drop_at_river = 0
    already_below50_at_flop = 0
    already_below50_at_turn = 0
    raised_while_dropping   = 0

    for r in sd_losses:
        d = r["disc_eq"] or 0
        f = r["flop_adj"]
        t = r["turn_adj"]
        rv = r["river_adj"]

        # Biggest single-street drop
        drops = []
        if f is not None and d > 0:
            drops.append(("flop",  d - f))
        if t is not None and f is not None:
            drops.append(("turn",  f - t))
        if rv is not None and t is not None:
            drops.append(("river", t - rv))
        elif rv is not None and f is not None:
            drops.append(("river", f - rv))

        if drops:
            worst = max(drops, key=lambda x: x[1])
            if worst[0] == "flop":   drop_at_flop  += 1
            elif worst[0] == "turn": drop_at_turn  += 1
            else:                    drop_at_river += 1

        if f is not None and f < 0.50:
            already_below50_at_flop += 1
        elif t is not None and t < 0.50 and (f is None or f >= 0.50):
            already_below50_at_turn += 1

        # Did we RAISE/BET while equity was already declining?
        if (f is not None and d > 0 and f < d and
                r["flop_act"] in ("RAISE", "BET")):
            raised_while_dropping += 1
        if (t is not None and f is not None and t < f and
                r["turn_act"] in ("RAISE", "BET")):
            raised_while_dropping += 1

    n = len(sd_losses)
    print(f"  Biggest drop on flop:   {drop_at_flop:3d} / {n}  ({100*drop_at_flop/n:.0f}%)")
    print(f"  Biggest drop on turn:   {drop_at_turn:3d} / {n}  ({100*drop_at_turn/n:.0f}%)")
    print(f"  Biggest drop on river:  {drop_at_river:3d} / {n}  ({100*drop_at_river/n:.0f}%)")
    print(f"  Already <50% by flop:   {already_below50_at_flop:3d} / {n}  ({100*already_below50_at_flop/n:.0f}%)")
    print(f"  Already <50% by turn:   {already_below50_at_turn:3d} / {n}  ({100*already_below50_at_turn/n:.0f}%)")
    print(f"  Raised while eq falling:{raised_while_dropping:3d} / {n}  ({100*raised_while_dropping/n:.0f}%)")

    # ── Flop equity comparison: losses vs wins ─────────────────────────────
    print("\n--- Flop Adj Equity: Losses vs Wins (showdown only) ---")
    for label, subset in [("Losses", sd_losses), ("Wins", sd_wins)]:
        f_vals = [r["flop_adj"] for r in subset if r["flop_adj"] is not None]
        below40 = sum(1 for v in f_vals if v < 0.40)
        r4_5    = sum(1 for v in f_vals if 0.40 <= v < 0.50)
        r5_6    = sum(1 for v in f_vals if 0.50 <= v < 0.60)
        above60 = sum(1 for v in f_vals if v >= 0.60)
        a = avg(f_vals)
        print(f"  {label} (n={len(subset)}):  avg={a:.3f}  "
              f"<40%={below40}  40-50%={r4_5}  50-60%={r5_6}  >=60%={above60}")

    # ── Turn equity comparison ─────────────────────────────────────────────
    print("\n--- Turn Adj Equity: Losses vs Wins ---")
    for label, subset in [("Losses", sd_losses), ("Wins", sd_wins)]:
        t_vals = [r["turn_adj"] for r in subset if r["turn_adj"] is not None]
        below40 = sum(1 for v in t_vals if v < 0.40)
        r4_5    = sum(1 for v in t_vals if 0.40 <= v < 0.50)
        r5_6    = sum(1 for v in t_vals if 0.50 <= v < 0.60)
        above60 = sum(1 for v in t_vals if v >= 0.60)
        a = avg(t_vals)
        print(f"  {label} (n={len(subset)}):  avg={a:.3f}  "
              f"<40%={below40}  40-50%={r4_5}  50-60%={r5_6}  >=60%={above60}")

    # ── Action breakdown on each street for losses ─────────────────────────
    print("\n--- Our Actions on Losing Showdown Hands ---")
    for street, key in [("Flop", "flop_act"), ("Turn", "turn_act"), ("River", "river_act")]:
        acts = defaultdict(int)
        for r in sd_losses:
            acts[r[key] or "no_data"] += 1
        parts = "  ".join(f"{k}={v}" for k, v in sorted(acts.items(), key=lambda x: -x[1]))
        print(f"  {street}: {parts}")

    # ── Hands where we raised/bet AND equity was already below 50% ─────────
    print("\n--- Hands Where We Raised/Bet With Adj Equity <50% (losses) ---")
    problem_hands = []
    for r in sd_losses:
        issues = []
        if r["flop_adj"] is not None and r["flop_adj"] < 0.50 and r["flop_act"] in ("RAISE", "BET"):
            issues.append(f"flop RAISE eq={r['flop_adj']:.3f}")
        if r["turn_adj"] is not None and r["turn_adj"] < 0.50 and r["turn_act"] in ("RAISE", "BET"):
            issues.append(f"turn RAISE eq={r['turn_adj']:.3f}")
        if issues:
            problem_hands.append((r, issues))

    print(f"  Count: {len(problem_hands)} / {n} losing hands ({100*len(problem_hands)/n:.0f}%)")
    for r, issues in sorted(problem_hands, key=lambda x: x[0]["pnl"])[:8]:
        print(f"  hand {r['hand']:4d}  pnl={r['pnl']:+d}  disc={r['disc_eq']:.3f}  " + "  ".join(issues))

    # ── Hands where we called with declining equity ────────────────────────
    print("\n--- Hands Where We Called on 2+ Streets With Declining Adj Equity ---")
    call_decline = []
    for r in sd_losses:
        d = r["disc_eq"] or 0
        f = r["flop_adj"]
        t = r["turn_adj"]
        rv = r["river_adj"]
        fa = r["flop_act"]
        ta = r["turn_act"]
        ra = r["river_act"]

        streets_called_declining = 0
        if f is not None and f < d and fa in ("CALL",):
            streets_called_declining += 1
        if t is not None and f is not None and t < f and ta in ("CALL",):
            streets_called_declining += 1
        if rv is not None and t is not None and rv < t and ra in ("CALL",):
            streets_called_declining += 1

        if streets_called_declining >= 2:
            call_decline.append((r, streets_called_declining))

    print(f"  Count: {len(call_decline)} / {n} ({100*len(call_decline)/n:.0f}%)")
    for r, cnt in sorted(call_decline, key=lambda x: x[0]["pnl"])[:8]:
        seq = []
        for street, eq_key, act_key in [("flop","flop_adj","flop_act"),("turn","turn_adj","turn_act"),("river","river_adj","river_act")]:
            eq = r[eq_key]
            act = r[act_key]
            if eq is not None:
                seq.append(f"{street}:{act}({eq:.2f})")
        print(f"  hand {r['hand']:4d}  pnl={r['pnl']:+d}  disc={r['disc_eq'] or 0:.3f}  " + " -> ".join(seq))

    # ── Trajectory pattern summary ─────────────────────────────────────────
    print("\n--- Equity Pattern on Losses: Were We Already Losing Before River? ---")
    patterns = defaultdict(int)
    for r in sd_losses:
        d  = r["disc_eq"] or 0.5
        f  = r["flop_adj"]
        t  = r["turn_adj"]
        rv = r["river_adj"]

        # determine if we were ahead/behind at each point
        fa = "A" if (f  is not None and f  >= 0.50) else ("B" if f  is not None else "?")
        ta = "A" if (t  is not None and t  >= 0.50) else ("B" if t  is not None else "?")
        ra = "A" if (rv is not None and rv >= 0.50) else ("B" if rv is not None else "?")
        da = "A" if d >= 0.50 else "B"
        patterns[f"disc:{da} flop:{fa} turn:{ta} river:{ra}"] += 1

    print(f"  (A=above 50% adj equity, B=below 50%, ?=no data)")
    for pat, cnt in sorted(patterns.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cnt:3d}x  {pat}")

    return sd_losses, sd_wins


# Run both matches
h1, mid1 = analyze_match(70807)
h2, mid2 = analyze_match(70860)
l1, w1 = run_analysis(h1, mid1)
l2, w2 = run_analysis(h2, mid2)

# Combined trajectory patterns
print()
print("=" * 72)
print("COMBINED - Trajectory Patterns on ALL Showdown Losses (both matches)")
print("=" * 72)

all_losses = l1 + l2
n = len(all_losses)
patterns = defaultdict(list)
for r in all_losses:
    d  = r["disc_eq"] or 0.5
    f  = r["flop_adj"]
    t  = r["turn_adj"]
    rv = r["river_adj"]
    da = "A" if d  >= 0.50 else "B"
    fa = "A" if (f  is not None and f  >= 0.50) else ("B" if f  is not None else "?")
    ta = "A" if (t  is not None and t  >= 0.50) else ("B" if t  is not None else "?")
    ra = "A" if (rv is not None and rv >= 0.50) else ("B" if rv is not None else "?")
    patterns[f"disc:{da} flop:{fa} turn:{ta} river:{ra}"].append(r["pnl"])

print(f"\nTotal showdown losses: {n}")
print(f"  (A=adj_equity>=50%, B=adj_equity<50%)\n")
print(f"  {'Pattern':<38} {'Count':>6} {'% of losses':>12} {'Total PnL':>10} {'Avg PnL':>9}")
for pat, pnls in sorted(patterns.items(), key=lambda x: -len(x[1])):
    cnt = len(pnls)
    tp  = sum(pnls)
    ap  = tp / cnt
    print(f"  {pat:<38} {cnt:>6} {100*cnt/n:>11.1f}% {tp:>+10} {ap:>+9.1f}")

# Summary stats
above_50_at_flop_and_lost = sum(1 for r in all_losses if r["flop_adj"] and r["flop_adj"] >= 0.50)
above_50_at_turn_and_lost = sum(1 for r in all_losses if r["turn_adj"] and r["turn_adj"] >= 0.50)
print(f"\nOf {n} total losses:")
print(f"  Still >=50% adj equity on FLOP: {above_50_at_flop_and_lost} ({100*above_50_at_flop_and_lost/n:.0f}%)")
print(f"  Still >=50% adj equity on TURN: {above_50_at_turn_and_lost} ({100*above_50_at_turn_and_lost/n:.0f}%)")
print(f"  Were already <50% at discard: {sum(1 for r in all_losses if r['disc_eq'] and r['disc_eq'] < 0.50)} ({100*sum(1 for r in all_losses if r['disc_eq'] and r['disc_eq'] < 0.50)/n:.0f}%)")
