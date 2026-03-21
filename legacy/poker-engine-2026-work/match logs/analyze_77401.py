#!/usr/bin/env python3
"""One-off analysis for match 77401."""
import csv
import json
import os
import re
from collections import defaultdict, Counter

_BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_BASE, "match_77401_csv.txt")
BOT_PATH = os.path.join(_BASE, "match_77401_bot.txt")
OUR_TEAM = 1
OPP = 0

# ---------- CSV ----------
rows = []
with open(CSV_PATH, newline="") as f:
    r = csv.reader(f)
    for row in r:
        if not row or row[0].startswith("#"):
            continue
        if row[0] == "hand_number":
            header = row
            col = {name: i for i, name in enumerate(header)}
            break
    for row in r:
        if not row or row[0].startswith("#"):
            continue
        rows.append(row)


def get(r, name, default=None):
    i = col.get(name)
    if i is None or i >= len(r):
        return default
    return r[i]


hands_data = defaultdict(list)
for r in rows:
    try:
        h = int(r[0])
    except ValueError:
        continue
    hands_data[h].append(r)

all_hands = sorted(hands_data.keys())
last_hand = max(all_hands)


def opponent_folded_preflop(hand_rows):
    for rr in hand_rows:
        if get(rr, "street") != "Pre-Flop":
            continue
        if int(get(rr, "active_team", -1)) == OPP and get(rr, "action_type") == "FOLD":
            return True
    return False


# Longest suffix: walk back from last_hand
idx = last_hand
while idx >= 0 and opponent_folded_preflop(hands_data[idx]):
    idx -= 1
# Suffix 999,998,... while opp folds preflop; first hand of run = idx+1
bleed_H = idx + 1 if idx < last_hand else None

exclude = set(range(bleed_H, last_hand + 1)) if bleed_H is not None else set()

# PnL per hand (team 1)
def pnl_for_hand(h):
    rs = hands_data[h]
    if not rs:
        return 0
    last = rs[-1]
    t1_end = int(get(last, "team_1_bankroll", 0))
    if h == 0:
        t1_start = 0
    else:
        prev = hands_data[h - 1][-1]
        t1_start = int(get(prev, "team_1_bankroll", 0))
    return t1_end - t1_start


pnl_bankroll = {h: pnl_for_hand(h) for h in all_hands}

# Prefer bot log hand_result.pnl (CSV bankrolls can lag within-hand)
hand_results_pre = {}
with open(BOT_PATH, encoding="utf-8", errors="replace") as f:
    for line in f:
        if '"event": "hand_result"' not in line:
            continue
        m = re.search(r"\{.*\}\s*$", line)
        if not m:
            continue
        try:
            d = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        hand_results_pre[d["hand"]] = d

pnl_by_hand = {}
for h in all_hands:
    hr = hand_results_pre.get(h)
    if hr is not None:
        pnl_by_hand[h] = hr.get("pnl", pnl_bankroll[h])
    else:
        pnl_by_hand[h] = pnl_bankroll[h]

real_hands = [h for h in all_hands if h not in exclude]
net_pnl = sum(pnl_by_hand[h] for h in real_hands)

# CSV: our bot actions (exclude DISCARD)
pf_counts = Counter()
post_counts = Counter()
for h in real_hands:
    for rr in hands_data[h]:
        if int(get(rr, "active_team", -1)) != OUR_TEAM:
            continue
        act = get(rr, "action_type")
        if act == "DISCARD":
            continue
        st = get(rr, "street")
        if st == "Pre-Flop":
            pf_counts[act] += 1
        else:
            post_counts[act] += 1

# Win rate, showdown from bot log
hand_results = dict(hand_results_pre)
hand_start_info = {}
postflop_decisions = []
preflop_reasons = Counter()
texture_river = []

with open(BOT_PATH, encoding="utf-8", errors="replace") as f:
    for line in f:
        if '"event":' not in line:
            continue
        m = re.search(r"\{.*\}\s*$", line)
        if not m:
            continue
        try:
            d = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        ev = d.get("event")
        h = d.get("hand")
        if ev == "hand_start":
            hand_start_info[h] = d
        elif ev == "preflop_decision":
            if h not in exclude:
                preflop_reasons[str(d.get("reason") or "")] += 1
        elif ev == "postflop_decision":
            if h in exclude:
                continue
            postflop_decisions.append(d)
            if d.get("street_name") == "river":
                texture_river.append(d.get("texture_adj"))

# Last postflop decision per hand (for E)
last_pf_by_hand = {}
for d in postflop_decisions:
    last_pf_by_hand[d["hand"]] = d

wins = losses = 0
sd = 0
sd_wins = 0
for h in real_hands:
    hr = hand_results.get(h)
    if not hr:
        continue
    pnl = hr.get("pnl", 0)
    if pnl > 0:
        wins += 1
    elif pnl < 0:
        losses += 1
    if hr.get("showdown"):
        sd += 1
        if pnl > 0:
            sd_wins += 1

played = wins + losses
win_rate = wins / played if played else 0
sd_wr = sd_wins / sd if sd else 0

# Pot at showdown (last row team bets sum)
def pot_size(h):
    rs = hands_data[h]
    if not rs:
        return 0
    last = rs[-1]
    t0 = int(get(last, "team_0_bet", 0) or 0)
    t1 = int(get(last, "team_1_bet", 0) or 0)
    return t0 + t1


sd_pots = []
for h in real_hands:
    hr = hand_results.get(h)
    if hr and hr.get("showdown"):
        sd_pots.append(pot_size(h))

avg_sd_pot = sum(sd_pots) / len(sd_pots) if sd_pots else 0

big_loss = sum(1 for h in real_hands if pnl_by_hand[h] <= -50)
big_win = sum(1 for h in real_hands if pnl_by_hand[h] >= 50)

# PnL SB vs BB
pnl_sb = pnl_bb = 0
n_sb = n_bb = 0
for h in real_hands:
    hs = hand_start_info.get(h)
    if not hs:
        continue
    pos = hs.get("position")
    p = pnl_by_hand[h]
    if pos == "SB":
        pnl_sb += p
        n_sb += 1
    elif pos == "BB":
        pnl_bb += p
        n_bb += 1

# D) Strength distribution + avg adj_equity
strength_eq = defaultdict(list)
for d in postflop_decisions:
    st = d.get("strength") or "?"
    strength_eq[st].append(d.get("adj_equity"))

street_raw = defaultdict(list)
street_adj = defaultdict(list)
for d in postflop_decisions:
    sn = d.get("street_name") or "?"
    if d.get("raw_equity") is not None:
        street_raw[sn].append(d.get("raw_equity"))
    if d.get("adj_equity") is not None:
        street_adj[sn].append(d.get("adj_equity"))

monster_low = sum(
    1
    for d in postflop_decisions
    if d.get("strength") == "monster" and (d.get("adj_equity") or 0) < 0.50
)
raise_low = sum(
    1
    for d in postflop_decisions
    if d.get("final_action") == "RAISE" and (d.get("adj_equity") or 0) < 0.30
)

avg_tex_river = sum(texture_river) / len(texture_river) if texture_river else 0

# E) PnL by strength at last postflop decision (hands that reached postflop only)
pnl_by_strength = defaultdict(float)
n_by_strength = Counter()
for h in real_hands:
    d = last_pf_by_hand.get(h)
    if not d:
        continue
    st = d.get("strength") or "?"
    pnl_by_strength[st] += pnl_by_hand[h]
    n_by_strength[st] += 1

# F) Showdown avg pot win vs loss
sd_w_pots = []
sd_l_pots = []
for h in real_hands:
    hr = hand_results.get(h)
    if not hr or not hr.get("showdown"):
        continue
    p = hr.get("pnl", 0)
    ps = pot_size(h)
    if p > 0:
        sd_w_pots.append(ps)
    elif p < 0:
        sd_l_pots.append(ps)

sd_loss_monster = []
for h in real_hands:
    hr = hand_results.get(h)
    if not hr or not hr.get("showdown"):
        continue
    if hr.get("pnl", 0) >= 0:
        continue
    d = last_pf_by_hand.get(h)
    if d and d.get("strength") == "monster":
        sd_loss_monster.append(h)

# G) Semi-bluff
semi_fired = sum(1 for d in postflop_decisions if d.get("semi_bluff_fired"))
semi_raise = sum(
    1
    for d in postflop_decisions
    if d.get("semi_bluff_fired") and d.get("final_action") == "RAISE"
)

# H) Large pots
large = [(h, pnl_by_hand[h]) for h in real_hands if abs(pnl_by_hand[h]) >= 50]

# I) Equity calibration at showdown: bucket adj_equity from last postflop
calib = defaultdict(lambda: {"w": 0, "l": 0})
for h in real_hands:
    hr = hand_results.get(h)
    if not hr or not hr.get("showdown"):
        continue
    d = last_pf_by_hand.get(h)
    if not d:
        continue
    ae = d.get("adj_equity")
    if ae is None:
        continue
    bucket = int(ae * 10) / 10.0  # 0.0, 0.1, ...
    if hr.get("pnl", 0) > 0:
        calib[bucket]["w"] += 1
    else:
        calib[bucket]["l"] += 1

# J) Opponent fold to our raise by street
# When we RAISE, track if opponent next action on same street is FOLD
fold_to_raise = defaultdict(lambda: {"raise": 0, "fold": 0})
for h in real_hands:
    rs = hands_data[h]
    for i, rr in enumerate(rs):
        if int(get(rr, "active_team", -1)) != OUR_TEAM:
            continue
        if get(rr, "action_type") != "RAISE":
            continue
        st = get(rr, "street")
        if st == "Pre-Flop":
            sk = "preflop"
        elif st == "Flop":
            sk = "flop"
        elif st == "Turn":
            sk = "turn"
        elif st == "River":
            sk = "river"
        else:
            continue
        fold_to_raise[sk]["raise"] += 1
        # look for next action by opponent on same street
        for j in range(i + 1, len(rs)):
            if get(rs[j], "street") != st:
                break
            if int(get(rs[j], "active_team", -1)) == OPP:
                if get(rs[j], "action_type") == "FOLD":
                    fold_to_raise[sk]["fold"] += 1
                break

# K) Avg raise amount by street (our raises)
raise_amts = defaultdict(list)
for h in real_hands:
    for rr in hands_data[h]:
        if int(get(rr, "active_team", -1)) != OUR_TEAM:
            continue
        if get(rr, "action_type") != "RAISE":
            continue
        st = get(rr, "street")
        amt = int(get(rr, "action_amount", 0) or 0)
        if st == "Pre-Flop":
            raise_amts["preflop"].append(amt)
        elif st == "Flop":
            raise_amts["flop"].append(amt)
        elif st == "Turn":
            raise_amts["turn"].append(amt)
        elif st == "River":
            raise_amts["river"].append(amt)

# L) Comeback mode
comeback_hs = 0
for h in real_hands:
    d = hand_start_info.get(h)
    if d and d.get("comeback_mode"):
        comeback_hs += 1

# M) Top 5 losing hands
losing = sorted([(h, pnl_by_hand[h]) for h in real_hands if pnl_by_hand[h] < 0], key=lambda x: x[1])[:5]

# Print structured output
print("=== A) MATCH SUMMARY ===")
print(f"Team 0 (Poker? I Barely Know Her) final bankroll: 36")
print(f"Team 1 (Ctrl+Alt+Defeat) final bankroll: -36")
print(f"Winner: Team 0 (Poker? I Barely Know Her)")
print(f"Our bot team number: 1")

print("\n=== B) BLEED-OUT ===")
print(f"H = {bleed_H if bleed_H is not None else 'none'}")
print(f"Bleed-out hands excluded: {len(exclude)}")

print("\n=== C) CSV (excl bleed) ===")
print(f"Total real-play hands: {len(real_hands)}")
print(f"Net PnL (team 1): {net_pnl}")
print(f"Win rate: {win_rate:.4f} ({wins}/{played})")
print(f"Showdown count: {sd}, SD win rate: {sd_wr:.4f} ({sd_wins}/{sd})")
print("Our action freq PRE:", dict(pf_counts))
print("Our action freq POST:", dict(post_counts))
print(f"Avg pot (showdown hands): {avg_sd_pot:.2f}")
print(f"Hands PnL <= -50: {big_loss}, >= +50: {big_win}")
print(f"PnL as SB: {pnl_sb} (n={n_sb}), as BB: {pnl_bb} (n={n_bb})")

print("\n=== D) BOT LOG (excl bleed) ===")
print("Strength distribution (postflop decisions, count):")
for st in sorted(strength_eq.keys()):
    eqs = [x for x in strength_eq[st] if x is not None]
    avg = sum(eqs) / len(eqs) if eqs else 0
    print(f"  {st}: n={len(strength_eq[st])}, avg_adj_equity={avg:.4f}")
print("Preflop reasons (top 15):")
for k, v in preflop_reasons.most_common(15):
    print(f"  {k!r}: {v}")
print("Avg raw_equity / adj_equity by street:")
for sn in sorted(street_raw.keys()):
    ravg = sum(street_raw[sn]) / len(street_raw[sn])
    aavg = sum(street_adj[sn]) / len(street_adj[sn])
    print(f"  {sn}: raw={ravg:.4f}, adj={aavg:.4f}, n={len(street_raw[sn])}")
print(f"strength=monster & adj_equity<0.50: {monster_low}")
print(f"RAISE & adj_equity<0.30: {raise_low}")
print(f"Avg texture_adj on river: {avg_tex_river:.4f}")

print("\n=== E) PNL BY STRENGTH (last postflop decision) ===")
for st in sorted(pnl_by_strength.keys()):
    tot = pnl_by_strength[st]
    n = n_by_strength[st]
    print(f"  {st}: total={tot:.0f}, avg={tot/n:.2f} ({n} hands)")

print("\n=== F) SHOWDOWN ===")
print(f"Avg pot on SD win: {sum(sd_w_pots)/len(sd_w_pots) if sd_w_pots else 0:.2f} (n={len(sd_w_pots)})")
print(f"Avg pot on SD loss: {sum(sd_l_pots)/len(sd_l_pots) if sd_l_pots else 0:.2f} (n={len(sd_l_pots)})")
print(f"SD losses with last strength=monster: {len(sd_loss_monster)} hands {sd_loss_monster[:20]}")

print("\n=== G) SEMI-BLUFF ===")
print(f"semi_bluff_fired count: {semi_fired}")
print(f"semi_bluff_fired & RAISE: {semi_raise}")

print("\n=== H) LARGE POTS |pnl|>=50 ===")
print(f"Count: {len(large)}")
for h, p in sorted(large, key=lambda x: -abs(x[1]))[:25]:
    print(f"  hand {h}: pnl={p}")

print("\n=== I) EQUITY CALIBRATION (showdown, bucket=adj_equity last PF) ===")
for b in sorted(calib.keys()):
    w, l = calib[b]["w"], calib[b]["l"]
    tot = w + l
    wr = w / tot if tot else 0
    print(f"  [{b:.1f},{b+0.1:.1f}): n={tot}, win_rate={wr:.3f}")

print("\n=== J) OPP FOLD TO OUR RAISE ===")
for sk in ["preflop", "flop", "turn", "river"]:
    d = fold_to_raise[sk]
    r, f = d["raise"], d["fold"]
    pct = f / r if r else 0
    print(f"  {sk}: fold={f}/{r} ({pct:.3f})")

print("\n=== K) BET SIZING avg raise amount ===")
for sk in ["preflop", "flop", "turn", "river"]:
    xs = raise_amts[sk]
    print(f"  {sk}: avg={sum(xs)/len(xs) if xs else 0:.2f}, n={len(xs)}")

print("\n=== L) COMEBACK MODE ===")
print(f" hands with comeback_mode=true at hand_start: {comeback_hs}/{len(real_hands)}")

print("\n=== M) TOP 5 LOSING HANDS ===")
for h, p in losing:
    hr = hand_results.get(h, {})
    hs = hand_start_info.get(h, {})
    lp = last_pf_by_hand.get(h)
    print(
        f"  hand {h}: pnl={p}, pos={hs.get('position')}, SD={hr.get('showdown')}, "
        f"last_strength={lp.get('strength') if lp else None}, adj_eq={lp.get('adj_equity') if lp else None}"
    )
