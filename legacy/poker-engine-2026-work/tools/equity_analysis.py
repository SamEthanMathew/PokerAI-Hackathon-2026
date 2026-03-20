import json, re, ast
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


def build_hands(events):
    hands = defaultdict(lambda: {
        "preflop": None, "discard": None,
        "postflop": [], "result": None
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
    return hands


def avg(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def analyze(match_id):
    base = Path("match logs") / f"match_{match_id}"
    events = load_bot_log(str(base) + "_bot.txt")
    hands = build_hands(events)

    print()
    print("=" * 70)
    print(f"MATCH {match_id} - Full Equity Progression Analysis")
    print("=" * 70)

    showdowns = []
    all_results = []

    for hnum in sorted(hands.keys()):
        h = hands[hnum]
        r = h["result"]
        if not r:
            continue

        pf   = h["preflop"]
        disc = h["discard"]
        posts = h["postflop"]

        pf_eq   = pf.get("equity") if pf else None
        disc_eq = disc.get("chosen_equity") if disc else None

        by_street = {}
        for p in posts:
            sn = (p.get("street_name") or p.get("street") or "?").lower()
            by_street[sn] = p

        flop  = by_street.get("flop")
        turn  = by_street.get("turn")
        river = by_street.get("river")

        row = {
            "hand":      hnum,
            "pf_eq":     pf_eq,
            "disc_eq":   disc_eq,
            "flop_raw":  flop.get("raw_equity")  if flop  else None,
            "flop_adj":  flop.get("adj_equity")  if flop  else None,
            "turn_raw":  turn.get("raw_equity")  if turn  else None,
            "turn_adj":  turn.get("adj_equity")  if turn  else None,
            "river_raw": river.get("raw_equity") if river else None,
            "river_adj": river.get("adj_equity") if river else None,
            "outcome":   r.get("outcome"),
            "pnl":       r.get("pnl", 0),
            "showdown":  r.get("showdown", False),
            "pf_action": pf.get("action")  if pf else None,
            "pf_reason": pf.get("reason")  if pf else None,
        }
        all_results.append(row)
        if row["showdown"]:
            showdowns.append(row)

    total    = len(all_results)
    sd_total = len(showdowns)
    sd_wins  = sum(1 for r in showdowns if r["outcome"] == "win")

    print(f"\nTotal hands: {total}  |  Showdowns: {sd_total}  |  SD win rate: {sd_wins}/{sd_total} = {100*sd_wins/sd_total:.1f}%")

    # Showdown win rate by post-discard equity bucket
    print("\n--- Showdown Win Rate by Post-Discard Equity Bucket ---")
    print(f"  {'Equity Range':<18} {'Hands':>6} {'Wins':>6} {'Losses':>6} {'Win%':>8} {'Avg PnL':>9}")
    for lo, hi in [(0,0.4),(0.4,0.5),(0.5,0.55),(0.55,0.6),(0.6,0.7),(0.7,0.8),(0.8,1.01)]:
        b = [r for r in showdowns if r["disc_eq"] is not None and lo <= r["disc_eq"] < hi]
        if not b:
            continue
        bw = sum(1 for r in b if r["outcome"] == "win")
        bl = len(b) - bw
        ap = sum(r["pnl"] for r in b) / len(b)
        print(f"  {lo:.0%} - {hi:.0%}            {len(b):>6} {bw:>6} {bl:>6} {100*bw/len(b):>7.1f}% {ap:>+9.1f}")

    # Showdown win rate by river adj equity bucket
    has_river = [r for r in showdowns if r["river_adj"] is not None]
    print(f"\n--- Showdown Win Rate by River Adj Equity ({len(has_river)} hands with river data) ---")
    print(f"  {'River Adj Eq':<18} {'Hands':>6} {'Wins':>6} {'Losses':>6} {'Win%':>8}")
    for lo, hi in [(0,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,1.01)]:
        b = [r for r in has_river if lo <= r["river_adj"] < hi]
        if not b:
            continue
        bw = sum(1 for r in b if r["outcome"] == "win")
        print(f"  {lo:.0%} - {hi:.0%}            {len(b):>6} {bw:>6} {len(b)-bw:>6} {100*bw/len(b):>7.1f}%")

    # Average equity journey: winners vs losers
    print("\n--- Avg Equity Journey: Winners vs Losers at Showdown ---")
    print(f"  {'':18} {'Disc':>7} {'Flop':>8} {'Turn':>8} {'River':>8} {'RiverRaw':>10}")
    for label, subset in [("Winners", [r for r in showdowns if r["outcome"] == "win"]),
                           ("Losers",  [r for r in showdowns if r["outcome"] == "loss"])]:
        d  = avg([r["disc_eq"]   for r in subset])
        f  = avg([r["flop_adj"]  for r in subset])
        t  = avg([r["turn_adj"]  for r in subset])
        rv = avg([r["river_adj"] for r in subset])
        rr = avg([r["river_raw"] for r in subset])
        fmt = lambda x: f"{x:.3f}" if x is not None else "  N/A"
        print(f"  {label:<20} {fmt(d):>7} {fmt(f):>8} {fmt(t):>8} {fmt(rv):>8} {fmt(rr):>10}")

    # All-in preflop
    allin = [r for r in all_results if r.get("pf_reason") == "premium_pair_allin"]
    if allin:
        aw = sum(1 for r in allin if r["outcome"] == "win")
        de = avg([r["disc_eq"] for r in allin])
        print(f"\n--- Premium Pair All-In Preflop: {len(allin)} hands ---")
        print(f"  Win rate: {aw}/{len(allin)} = {100*aw/len(allin):.1f}%")
        print(f"  Avg post-discard equity: {de:.3f}")
        print(f"  Total PnL: {sum(r['pnl'] for r in allin):+d}")

    # Big pots
    big = [r for r in all_results if abs(r["pnl"]) >= 50]
    bw  = sum(1 for r in big if r["outcome"] == "win")
    bl  = sum(1 for r in big if r["outcome"] == "loss")
    bde_win  = avg([r["disc_eq"] for r in big if r["outcome"] == "win"  and r["disc_eq"] is not None])
    bde_loss = avg([r["disc_eq"] for r in big if r["outcome"] == "loss" and r["disc_eq"] is not None])
    print(f"\n--- Big Pot Hands (|pnl| >= 50): {len(big)} total ---")
    print(f"  Wins: {bw}  Losses: {bl}  Net: {sum(r['pnl'] for r in big):+d}")
    print(f"  Avg disc_eq on wins:   {bde_win:.3f}" if bde_win else "  Avg disc_eq wins: N/A")
    print(f"  Avg disc_eq on losses: {bde_loss:.3f}" if bde_loss else "  Avg disc_eq losses: N/A")

    return all_results


r1 = analyze(70807)
r2 = analyze(70860)

print()
print("=" * 70)
print("COMBINED ANALYSIS (both matches, 2000 hands total)")
print("=" * 70)

combined = r1 + r2
all_sd = [r for r in combined if r["showdown"]]
wins   = sum(1 for r in all_sd if r["outcome"] == "win")
losses = sum(1 for r in all_sd if r["outcome"] == "loss")
print(f"\nTotal showdowns: {len(all_sd)}  Wins: {wins}  Losses: {losses}  Win rate: {100*wins/len(all_sd):.1f}%")

# Hands with >=60% disc_eq that we lost
high_eq_losses = [r for r in all_sd if r["disc_eq"] and r["disc_eq"] >= 0.60 and r["outcome"] == "loss"]
print(f"\nBad beats (>=60% post-discard eq, lost at showdown): {len(high_eq_losses)}")
for r in sorted(high_eq_losses, key=lambda x: -(x["disc_eq"] or 0))[:12]:
    rv = f"{r['river_adj']:.3f}" if r["river_adj"] is not None else "  N/A"
    print(f"  hand {r['hand']:4d}  disc={r['disc_eq']:.3f}  river_adj={rv}  pnl={r['pnl']:+d}")

# Hands with <50% disc_eq at showdown
weak_sd = [r for r in all_sd if r["disc_eq"] and r["disc_eq"] < 0.50]
ww = sum(1 for r in weak_sd if r["outcome"] == "win")
print(f"\nShowdown hands entered with <50% post-discard equity: {len(weak_sd)}")
print(f"  Win rate: {ww}/{len(weak_sd)} = {100*ww/max(1,len(weak_sd)):.1f}%")
print(f"  Total PnL: {sum(r['pnl'] for r in weak_sd):+d}")

# Hands with >70% disc_eq
strong_sd = [r for r in all_sd if r["disc_eq"] and r["disc_eq"] >= 0.70]
sw = sum(1 for r in strong_sd if r["outcome"] == "win")
print(f"\nShowdown hands entered with >=70% post-discard equity: {len(strong_sd)}")
print(f"  Win rate: {sw}/{len(strong_sd)} = {100*sw/max(1,len(strong_sd)):.1f}%")
print(f"  Total PnL: {sum(r['pnl'] for r in strong_sd):+d}")

# River adj equity distribution for showdown losses
river_losses = [r for r in all_sd if r["outcome"] == "loss" and r["river_adj"] is not None]
above_50_river_loss = [r for r in river_losses if r["river_adj"] >= 0.50]
print(f"\nShowdown losses where river adj_equity was still >=50%: {len(above_50_river_loss)}")
print(f"  (opponent outdrawn us despite us being river favorite)")
for r in sorted(above_50_river_loss, key=lambda x: -(x["river_adj"] or 0))[:8]:
    print(f"  hand {r['hand']:4d}  river_adj={r['river_adj']:.3f}  river_raw={r['river_raw'] or 0:.3f}  pnl={r['pnl']:+d}")
