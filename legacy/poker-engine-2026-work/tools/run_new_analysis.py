import json, re, csv, math, os, glob
from collections import defaultdict
from pathlib import Path

def load_bot_log(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"- INFO - (\{.+\})$", line)
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
            row["active_team"] = int(row["active_team"])
            row["team_0_bet"]  = int(row.get("team_0_bet", 0) or 0)
            row["team_1_bet"]  = int(row.get("team_1_bet", 0) or 0)
        except:
            pass
        rows.append(row)
    return rows

def build_hands(events, csv_rows, our_team):
    hands = defaultdict(lambda: {
        "preflop": None, "discard": None,
        "postflop": [], "result": None, "csv": []
    })
    for e in events:
        h  = e.get("hand", -1)
        ev = e.get("event")
        if   ev == "preflop_decision": hands[h]["preflop"]  = e
        elif ev == "discard_decision": hands[h]["discard"]  = e
        elif ev == "postflop_decision":hands[h]["postflop"].append(e)
        elif ev == "hand_result":      hands[h]["result"]   = e
    for row in csv_rows:
        hands[row["hand_number"]]["csv"].append(row)
    return hands

def avg(vals):
    v = [x for x in vals if x is not None]
    return sum(v)/len(v) if v else None

def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "N/A"

def get_streets(postflop):
    by = {}
    for p in postflop:
        sn = (p.get("street_name") or p.get("street") or "?").lower()
        by[sn] = p
    return by.get("flop"), by.get("turn"), by.get("river")

def bet_amount(row, our_team):
    key = f"team_{our_team}_bet"
    try:
        return int(row.get(key, 0) or 0)
    except:
        return 0

def extract_records(hands, match_id, our_team):
    records = []
    for hnum in sorted(hands.keys()):
        h   = hands[hnum]
        r   = h["result"]
        if not r:
            continue
        pf   = h["preflop"]
        disc = h["discard"]
        fl, tu, ri = get_streets(h["postflop"])

        street_chips = defaultdict(int)
        prev_bet = 0
        for row in sorted(h["csv"], key=lambda x: x.get("hand_number", 0)):
            if row.get("active_team") != our_team:
                continue
            act = row.get("action_type", "")
            if act in ("RAISE", "BET", "CALL", "ALL_IN"):
                cur  = bet_amount(row, our_team)
                diff = max(0, cur - prev_bet)
                street_chips[row.get("street", "?")] += diff
                prev_bet = cur

        opp_raises = sum(1 for row in h["csv"]
                         if row.get("active_team") != our_team
                         and row.get("action_type") in ("RAISE", "BET", "ALL_IN"))

        pnl   = r.get("pnl", 0)
        phase = "early" if hnum < 334 else ("mid" if hnum < 667 else "late")

        records.append({
            "match":      match_id,
            "hand":       hnum,
            "phase":      phase,
            "outcome":    r.get("outcome"),
            "pnl":        pnl,
            "showdown":   r.get("showdown", False),
            "pf_eq":      pf.get("equity")         if pf   else None,
            "pf_action":  pf.get("action")         if pf   else None,
            "pf_reason":  pf.get("reason")         if pf   else None,
            "disc_eq":    disc.get("chosen_equity") if disc else None,
            "disc_margin":disc.get("equity_margin") if disc else None,
            "disc_mode":  disc.get("mode")          if disc else None,
            "flop_adj":   fl.get("adj_equity")      if fl   else None,
            "flop_raw":   fl.get("raw_equity")      if fl   else None,
            "flop_act":   fl.get("final_action")    if fl   else None,
            "flop_tex":   fl.get("texture_adj", 0)  if fl   else 0,
            "turn_adj":   tu.get("adj_equity")      if tu   else None,
            "turn_raw":   tu.get("raw_equity")      if tu   else None,
            "turn_act":   tu.get("final_action")    if tu   else None,
            "turn_tex":   tu.get("texture_adj", 0)  if tu   else 0,
            "river_adj":  ri.get("adj_equity")      if ri   else None,
            "river_raw":  ri.get("raw_equity")      if ri   else None,
            "river_act":  ri.get("final_action")    if ri   else None,
            "chips_preflop": street_chips.get("Pre-Flop", 0),
            "chips_flop":    street_chips.get("Flop", 0),
            "chips_turn":    street_chips.get("Turn", 0),
            "chips_river":   street_chips.get("River", 0),
            "opp_raises":    opp_raises,
            "running_pnl":   r.get("running_pnl", 0),
        })
    return records

all_records = []
match_summaries = []

for fpath in glob.glob("match logs/match_*_csv.txt"):
    base = fpath.replace("_csv.txt", "")
    mid = re.search(r"match_(\d+)", base).group(1)
    
    summary_path = base + ".txt"
    if not os.path.exists(summary_path): continue
    with open(summary_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines: continue
        last_line = lines[-1]
        m = re.search(r"Pulp Ficshun bankroll: ([-\d]+), (.+) bankroll", last_line)
        if m:
            final_pnl = int(m.group(1))
            opp_name = m.group(2)
        else:
            final_pnl = 0
            opp_name = "Unknown"
    
    csv_r = load_csv(fpath)
    our_team = 0
    for r in csv_r:
        if r.get("team_0_name") == "Pulp Ficshun":
            our_team = 0; break
        if r.get("team_1_name") == "Pulp Ficshun":
            our_team = 1; break
            
    events = load_bot_log(base + "_bot.txt")
    hands  = build_hands(events, csv_r, our_team)
    recs   = extract_records(hands, mid, our_team)
    all_records.extend(recs)
    match_summaries.append((mid, opp_name, final_pnl, len(recs)))

print("=" * 72)
print("SECTION 1 — MATCH OVERVIEW")
print("=" * 72)

SD   = [r for r in all_records if r["showdown"]]
wins = [r for r in SD if r["outcome"] == "win"]
loss = [r for r in SD if r["outcome"] == "loss"]

print(f"\n{'Match':<8} {'Opponent':<28} {'Result':>8} {'Hands':>6} {'SD':>5} {'SD Win%':>8}")
for mid, opp, pnl, nhands in match_summaries:
    m_sd   = [r for r in SD   if r["match"] == mid]
    m_wins = sum(1 for r in m_sd if r["outcome"] == "win")
    print(f"  {mid:<6} {opp:<28} {pnl:>+8} {nhands:>6} {len(m_sd):>5} {pct(m_wins,len(m_sd)):>8}")

total_pnl = sum(p for _,_,p,_ in match_summaries)
total_sd  = len(SD)
total_w   = len(wins)
print(f"\n  TOTAL  All matches           {total_pnl:>+8} {len(all_records):>6} {total_sd:>5} {pct(total_w,total_sd):>8}")

print()
print("=" * 72)
print("SECTION 2 — EQUITY DISTRIBUTION AT DISCARD (all showdown hands)")
print("=" * 72)
eq_buckets = [(0,.40),(.40,.50),(.50,.55),(.55,.60),(.60,.70),(.70,.80),(.80,1.01)]
print(f"\n  {'Equity':<12} {'Total':>6} {'Wins':>6} {'Losses':>6} {'Win%':>8} {'Avg PnL':>9} {'Expected Win%':>14}")
expected = {(0,.40):"~40%",(.40,.50):"~45%",(.50,.55):"~52%",(.55,.60):"~57%",
            (.60,.70):"~65%",(.70,.80):"~75%",(.80,1.01):"~85%"}
for lo, hi in eq_buckets:
    b  = [r for r in SD if r["disc_eq"] is not None and lo <= r["disc_eq"] < hi]
    if not b: continue
    bw = sum(1 for r in b if r["outcome"] == "win")
    ap = avg([r["pnl"] for r in b])
    exp = expected.get((lo,hi), "")
    print(f"  {lo:.0%}-{hi:.0%}      {len(b):>6} {bw:>6} {len(b)-bw:>6} {pct(bw,len(b)):>8} {ap:>+9.1f} {exp:>14}")

print()
print("=" * 72)
print("SECTION 3 — LOSS TRAJECTORY PATTERNS")
print("=" * 72)

patterns = defaultdict(list)
for r in loss:
    da = "A" if (r["disc_eq"]  or 0) >= 0.50 else "B"
    fa = "A" if (r["flop_adj"] or -1) >= 0.50 else ("B" if r["flop_adj"] is not None else "?")
    ta = "A" if (r["turn_adj"] or -1) >= 0.50 else ("B" if r["turn_adj"] is not None else "?")
    ra = "A" if (r["river_adj"]or -1) >= 0.50 else ("B" if r["river_adj"]is not None else "?")
    patterns[f"disc:{da} flop:{fa} turn:{ta} river:{ra}"].append(r["pnl"])

total_loss = len(loss)
print(f"\n  Total showdown losses: {total_loss}")
print(f"  (A = adj_equity >= 50%, B = below 50%)\n")
print(f"  {'Pattern':<38} {'Count':>6} {'%':>7} {'Total PnL':>11} {'Avg PnL':>9} {'Interpretation'}")
for pat, pnls in sorted(patterns.items(), key=lambda x: -len(x[1])):
    cnt = len(pnls)
    tp  = sum(pnls)
    ap  = tp/cnt
    print(f"  {pat:<38} {cnt:>6} {100*cnt/total_loss:>6.1f}% {tp:>+11} {ap:>+9.1f}")

print()
print("=" * 72)
print("SECTION 4 — BET SIZING vs CONFIDENCE (chips committed per street)")
print("=" * 72)

for street, eq_key, chips_key in [
    ("Flop",  "flop_adj",  "chips_flop"),
    ("Turn",  "turn_adj",  "chips_turn"),
    ("River", "river_adj", "chips_river"),
]:
    has_data = [r for r in SD if r[eq_key] is not None and r[chips_key] > 0]
    if not has_data:
        continue
    print(f"\n  {street} — avg chips committed by adj_equity bucket (showdown hands):")
    print(f"    {'Equity':>12}  {'Hands':>6}  {'Avg Chips In':>13}  {'Win%':>8}  {'Avg PnL':>9}")
    for lo, hi in [(0,.30),(.30,.40),(.40,.50),(.50,.60),(.60,.70),(.70,.80),(.80,1.01)]:
        b = [r for r in has_data if lo <= r[eq_key] < hi]
        if not b: continue
        ac = avg([r[chips_key] for r in b])
        bw = sum(1 for r in b if r["outcome"] == "win")
        ap = avg([r["pnl"] for r in b])
        print(f"    {lo:.0%}-{hi:.0%}        {len(b):>6}  {ac:>13.1f}  {pct(bw,len(b)):>8}  {ap:>+9.1f}")
