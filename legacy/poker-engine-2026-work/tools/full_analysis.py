"""
Full 5-match comprehensive analysis.
Covers: equity distribution, showdown win rates, street-by-street trajectory,
bet sizing vs confidence, late-game patterns, and root cause breakdown.
"""
import json, re, csv, math
from collections import defaultdict
from pathlib import Path

# Match config: (match_id, our_team_index, opponent_name, our_final_pnl)
MATCHES = [
    (70807, 0, "super-quantum-frogtron",  -69),
    (70860, 1, "never fold 67",           -75),
    (70985, 1, "Mind the raise",          -73),
    (71569, 1, "super-quantum-frogtron2", -110),
    (71708, 0, "AA998",                    -9),
]


# ── Loaders ───────────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Per-hand record builder ───────────────────────────────────────────────────

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

        # chips we put in per street (from CSV)
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

        # opponent actions
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


# ── Load all matches ──────────────────────────────────────────────────────────

all_records = []
match_summaries = []

for mid, our_team, opp_name, final_pnl in MATCHES:
    base   = Path("match logs") / f"match_{mid}"
    events = load_bot_log(str(base) + "_bot.txt")
    csv_r  = load_csv(str(base) + "_csv.txt")
    hands  = build_hands(events, csv_r, our_team)
    recs   = extract_records(hands, mid, our_team)
    all_records.extend(recs)
    match_summaries.append((mid, opp_name, final_pnl, len(recs)))
    print(f"Loaded match {mid}: {len(recs)} hands  (us=Team{our_team}, opp={opp_name}, result={final_pnl:+d})")

print()

# ── Section helpers ───────────────────────────────────────────────────────────

SD   = [r for r in all_records if r["showdown"]]
wins = [r for r in SD if r["outcome"] == "win"]
loss = [r for r in SD if r["outcome"] == "loss"]

def bucket_stats(records, key, buckets, label=""):
    rows = []
    for lo, hi in buckets:
        b  = [r for r in records if r[key] is not None and lo <= r[key] < hi]
        if not b: continue
        bw = sum(1 for r in b if r["outcome"] == "win")
        bl = len(b) - bw
        ap = avg([r["pnl"] for r in b])
        rows.append((f"{lo:.0%}-{hi:.0%}", len(b), bw, bl,
                     f"{100*bw/len(b):.1f}%", f"{ap:+.1f}"))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("SECTION 1 — MATCH OVERVIEW (5 matches, 5000 hands)")
print("=" * 72)
print(f"\n{'Match':<8} {'Opponent':<28} {'Result':>8} {'Hands':>6} {'SD':>5} {'SD Win%':>8}")
for mid, opp, pnl, nhands in match_summaries:
    m_sd   = [r for r in SD   if r["match"] == mid]
    m_wins = sum(1 for r in m_sd if r["outcome"] == "win")
    print(f"  {mid:<6} {opp:<28} {pnl:>+8} {nhands:>6} {len(m_sd):>5} {pct(m_wins,len(m_sd)):>8}")

total_pnl = sum(p for _,_,p,_ in match_summaries)
total_sd  = len(SD)
total_w   = len(wins)
print(f"\n  TOTAL  5 matches lost        {total_pnl:>+8} {len(all_records):>6} {total_sd:>5} {pct(total_w,total_sd):>8}")
print(f"\n  Expected SD win rate if equity estimates accurate: ~50-55%")
print(f"  Actual:  {pct(total_w, total_sd)}  =>  {total_w} wins / {len(loss)} losses")


# ═══════════════════════════════════════════════════════════════════════════════
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

# Underperformance summary
print(f"\n  Key anomaly — 55-60% equity bucket:")
b5560 = [r for r in SD if r["disc_eq"] is not None and 0.55 <= r["disc_eq"] < 0.60]
w5560 = sum(1 for r in b5560 if r["outcome"] == "win")
print(f"    {len(b5560)} hands, {w5560} wins = {pct(w5560,len(b5560))} (expected ~57%)")


# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("SECTION 3 — WHERE EQUITY GOES: AVERAGE TRAJECTORY BY OUTCOME")
print("=" * 72)
print(f"\n  {'':12} {'Disc':>7} {'Flop(adj)':>10} {'Turn(adj)':>10} {'River(adj)':>11} {'River(raw)':>11}")
for label, subset in [("Winners", wins), ("Losers", loss)]:
    d  = avg([r["disc_eq"]  for r in subset])
    f  = avg([r["flop_adj"] for r in subset])
    t  = avg([r["turn_adj"] for r in subset])
    rv = avg([r["river_adj"]for r in subset])
    rr = avg([r["river_raw"]for r in subset])
    fmt = lambda x: f"{x:.3f}" if x is not None else "  N/A"
    print(f"  {label:<14} {fmt(d):>7} {fmt(f):>10} {fmt(t):>10} {fmt(rv):>11} {fmt(rr):>11}")

# How fast does equity drop on losses?
print(f"\n  Equity DROP from discard to turn on losses:")
drops = [(r["disc_eq"] or 0) - (r["turn_adj"] or r["disc_eq"] or 0)
         for r in loss if r["disc_eq"] and r["turn_adj"]]
print(f"    Avg drop:  {avg(drops):+.3f}")
print(f"    >0.10 drop (significant): {sum(1 for d in drops if d > 0.10)} / {len(drops)} ({pct(sum(1 for d in drops if d>0.10),len(drops))})")
print(f"    >0.20 drop (major):       {sum(1 for d in drops if d > 0.20)} / {len(drops)} ({pct(sum(1 for d in drops if d>0.20),len(drops))})")


# ═══════════════════════════════════════════════════════════════════════════════
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

# Big bets with low equity
print(f"\n  PROBLEM BETS: committed >=20 chips with adj_equity <40%")
problem = [r for r in all_records
           if (r["chips_flop"]  > 20 and r["flop_adj"]  is not None and r["flop_adj"]  < 0.40)
           or (r["chips_turn"]  > 20 and r["turn_adj"]  is not None and r["turn_adj"]  < 0.40)
           or (r["chips_river"] > 20 and r["river_adj"] is not None and r["river_adj"] < 0.40)]
pw = sum(1 for r in problem if r["outcome"] == "win")
print(f"    Count: {len(problem)}  Win rate: {pct(pw,len(problem))}  Total PnL: {sum(r['pnl'] for r in problem):+d}")


# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("SECTION 5 — LOSS TRAJECTORY PATTERNS (all 5 matches)")
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
interp = {
    "disc:B flop:B turn:B river:B": "Never had edge — shouldn't be here",
    "disc:A flop:A turn:A river:A": "Had edge all way — pure bad beat",
    "disc:A flop:A turn:B river:B": "Turn killed us — board ran bad",
    "disc:A flop:B turn:B river:B": "Flop killed us — board ran bad",
    "disc:A flop:A turn:A river:B": "River suck-out — bad beat",
    "disc:A flop:A turn:B river:A": "Recovered on river but still lost",
}
for pat, pnls in sorted(patterns.items(), key=lambda x: -len(x[1])):
    cnt = len(pnls)
    tp  = sum(pnls)
    ap  = tp/cnt
    note = interp.get(pat, "")
    print(f"  {pat:<38} {cnt:>6} {100*cnt/total_loss:>6.1f}% {tp:>+11} {ap:>+9.1f}  {note}")

# Avoidable vs unavoidable
avoidable = sum(len(v) for k,v in patterns.items() if k.startswith("disc:B"))
bad_beats  = sum(len(v) for k,v in patterns.items() if "turn:A river:A" in k or k == "disc:A flop:A turn:A river:A")
board_ran  = total_loss - avoidable - bad_beats
print(f"\n  Summary:")
print(f"    Already behind at discard (avoidable?): {avoidable} ({pct(avoidable,total_loss)})")
print(f"    Had edge everywhere (bad beats):         {bad_beats} ({pct(bad_beats,total_loss)})")
print(f"    Board ran bad mid-hand:                  {board_ran} ({pct(board_ran,total_loss)})")


# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("SECTION 6 — LATE-GAME ANALYSIS (hands 667-999)")
print("=" * 72)

for phase in ["early", "mid", "late"]:
    ph = [r for r in SD if r["phase"] == phase]
    pw = sum(1 for r in ph if r["outcome"] == "win")
    pnl_all = [r for r in all_records if r["phase"] == phase]
    print(f"\n  {phase.upper()} phase (hands {'0-333' if phase=='early' else '334-666' if phase=='mid' else '667-999'}):")
    print(f"    Showdowns: {len(ph)}   Win rate: {pct(pw,len(ph))}")
    print(f"    Total PnL (all hands): {sum(r['pnl'] for r in pnl_all):+d}")
    print(f"    Avg disc_eq (showdowns): {avg([r['disc_eq'] for r in ph]):.3f}" if ph else "")

# Late-game bet sizing vs equity
late_sd = [r for r in SD if r["phase"] == "late"]
print(f"\n  Late-game avg chips committed vs equity on turn:")
for lo, hi in [(.40,.50),(.50,.60),(.60,.70),(.70,1.01)]:
    b = [r for r in late_sd if r["turn_adj"] is not None and r["chips_turn"] > 0
         and lo <= r["turn_adj"] < hi]
    if not b: continue
    bw = sum(1 for r in b if r["outcome"]=="win")
    print(f"    {lo:.0%}-{hi:.0%}  n={len(b)}  avg_chips={avg([r['chips_turn'] for r in b]):.1f}  win%={pct(bw,len(b))}")


# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("SECTION 7 — RAISE/BET WITH DECLINING EQUITY (all matches)")
print("=" * 72)

raised_declining = []
for r in all_records:
    d  = r["disc_eq"] or 0
    f  = r["flop_adj"]
    t  = r["turn_adj"]
    issues = []
    if f is not None and f < d and f < 0.50 and r["flop_act"] in ("RAISE", "BET"):
        issues.append(f"flop RAISE eq={f:.3f}")
    if t is not None and (f or d) > t and t < 0.50 and r["turn_act"] in ("RAISE", "BET"):
        issues.append(f"turn RAISE eq={t:.3f}")
    if issues:
        raised_declining.append((r, issues))

rw = sum(1 for r,_ in raised_declining if r["outcome"]=="win")
print(f"\n  Raised/bet with adj_equity <50% AND equity was falling: {len(raised_declining)}")
print(f"  Win rate: {pct(rw,len(raised_declining))}   Total PnL: {sum(r['pnl'] for r,_ in raised_declining):+d}")
print(f"\n  Worst examples:")
for r, issues in sorted(raised_declining, key=lambda x: x[0]["pnl"])[:10]:
    print(f"    match {r['match']} hand {r['hand']:4d}  pnl={r['pnl']:+d}  " + "  ".join(issues))


# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("SECTION 8 — KEY FINDINGS SUMMARY")
print("=" * 72)

total_chips_lost = sum(r["pnl"] for r in all_records)
sd_chips_lost    = sum(r["pnl"] for r in loss)
fold_loss        = [r for r in all_records if r.get("outcome") == "loss" and not r["showdown"]]
fold_chips       = sum(r["pnl"] for r in fold_loss)

# Bad beats: >=50% on river and lost
river_fav_loss = [r for r in loss if r["river_adj"] and r["river_adj"] >= 0.50]
rfpnl = sum(r["pnl"] for r in river_fav_loss)

# Underdog showdowns
underdog_sd = [r for r in SD if r["disc_eq"] and r["disc_eq"] < 0.50]
udw = sum(1 for r in underdog_sd if r["outcome"]=="win")

# Coin-flip discards
coin_flip = [r for r in all_records if r["disc_margin"] is not None and r["disc_margin"] < 0.02]
cfsd = [r for r in coin_flip if r["showdown"]]
cfsdw = sum(1 for r in cfsd if r["outcome"]=="win")

print(f"""
  5 MATCHES — ALL LOSSES:
    Total net PnL:           {total_chips_lost:+d} chips
    Lost at showdown:        {sd_chips_lost:+d} chips  ({len(loss)} hands)
    Lost by folding:         {fold_chips:+d} chips  ({len(fold_loss)} hands)

  FINDING 1 — Showdown win rate is catastrophically low:
    Expected ~50-55%,  actual: {pct(total_w, total_sd)} ({total_w}W/{len(loss)}L)
    The 55-60% equity bucket wins only {pct(w5560,len(b5560))}  (should be ~57%)
    This is the single biggest signal of a systematic problem.

  FINDING 2 — Bad beats are real but not the main story:
    {len(river_fav_loss)} losses where we had >=50% adj equity on the river
    Total chips lost in those hands: {rfpnl:+d}
    These are legitimate bad beats but over 5000 hands they should average out.

  FINDING 3 — We go to showdown as underdogs too often:
    {len(underdog_sd)} showdowns entered with <50% disc equity
    Win rate: {pct(udw,len(underdog_sd))} (expected ~40-45%)
    Total PnL: {sum(r['pnl'] for r in underdog_sd):+d}

  FINDING 4 — Raised/bet with falling equity <50%: {len(raised_declining)} times
    Win rate: {pct(rw,len(raised_declining))}   PnL: {sum(r['pnl'] for r,_ in raised_declining):+d}
    These are the most directly fixable chip leaks.

  FINDING 5 — Coin-flip discards (~29% of hands):
    {len(coin_flip)} discards with margin <0.02
    When these go to showdown: {pct(cfsdw,len(cfsd))} win rate
    Low MC sim resolution creates random discard choices on marginal hands.

  ROOT CAUSE HYPOTHESIS:
    The equity calculation (especially in the 55-60% range) is likely
    overestimating our actual win probability. Possible causes:
    - MC sim not accounting for opponent's discard choices correctly
    - Texture penalty may be underweighted in equity estimate (not just adj)
    - Sample size at sim time too low for marginal hands
""")
