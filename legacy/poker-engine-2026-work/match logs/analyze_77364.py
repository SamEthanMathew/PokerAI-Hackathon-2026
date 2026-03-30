#!/usr/bin/env python3
"""One-off analysis for match 77364 — run: python3 analyze_77364.py"""
import csv
import json
import re
from collections import defaultdict, Counter

OUR_TEAM = 1
OPP_TEAM = 0
CSV_PATH = "/Users/rudrakshsingh/Downloads/poker-engine-2026-work/match logs/match_77364_csv.txt"
BOT_PATH = "/Users/rudrakshsingh/Downloads/poker-engine-2026-work/match logs/match_77364_bot.txt"

# --- CSV ---
rows = []
with open(CSV_PATH) as f:
    f.readline()
    for r in csv.DictReader(f):
        r["hand_number"] = int(r["hand_number"])
        r["active_team"] = int(r["active_team"])
        r["action_amount"] = int(r["action_amount"] or 0)
        rows.append(r)

hand_by = defaultdict(list)
for r in rows:
    hand_by[r["hand_number"]].append(r)

last_hand = max(hand_by.keys())


def opp_folded_preflop(hrs):
    for x in hrs:
        if x["street"] != "Pre-Flop":
            continue
        if x["active_team"] == OPP_TEAM and x["action_type"] == "FOLD":
            return True
    return False


bleed_H = None
for H in range(0, last_hand + 1):
    if all(opp_folded_preflop(hand_by[h]) for h in range(H, last_hand + 1)):
        bleed_H = H
        break

real_hands = list(range(0, last_hand + 1))
if bleed_H is not None:
    real_hands = [h for h in real_hands if h < bleed_H]
real_set = set(real_hands)

# --- Bot log ---
json_re = re.compile(r"\{.*\}\s*$")
hand_results = {}
hand_starts = {}
postflop_by_hand = defaultdict(list)
preflop_reasons = []
strength_adj = defaultdict(list)
raw_by_street = defaultdict(list)
adj_by_street = defaultdict(list)
river_texture = []
monster_low_adj = 0
raise_low_adj = 0
semi_bluff_hands = set()
semi_bluff_eq = []
last_postflop_per_hand = {}
comeback_hands = set()
all_hand_pnls = {}

with open(BOT_PATH) as f:
    for line in f:
        m = json_re.search(line)
        if not m:
            continue
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        ev = d.get("event")
        h = d.get("hand")
        if h is None or not isinstance(h, int):
            continue
        if h not in real_set and ev != "hand_result":
            continue
        if ev == "hand_start":
            hand_starts[h] = d
            if d.get("comeback_mode"):
                comeback_hands.add(h)
        elif ev == "hand_result":
            hand_results[h] = d
            all_hand_pnls[h] = d.get("pnl", 0)
        elif ev == "preflop_decision":
            if h in real_set:
                preflop_reasons.append(d.get("reason", ""))
        elif ev == "postflop_decision":
            if h not in real_set:
                continue
            postflop_by_hand[h].append(d)
            st = d.get("strength")
            if st and d.get("adj_equity") is not None:
                strength_adj[st].append(d["adj_equity"])
            sn = d.get("street_name")
            if sn in ("flop", "turn", "river"):
                if d.get("raw_equity") is not None:
                    raw_by_street[sn].append(d["raw_equity"])
                if d.get("adj_equity") is not None:
                    adj_by_street[sn].append(d["adj_equity"])
            if sn == "river" and d.get("texture_adj") is not None:
                river_texture.append(d["texture_adj"])
            if st == "monster" and d.get("adj_equity") is not None and d["adj_equity"] < 0.50:
                monster_low_adj += 1
            if d.get("final_action") == "RAISE" and d.get("adj_equity") is not None and d["adj_equity"] < 0.30:
                raise_low_adj += 1
            if d.get("semi_bluff_fired"):
                semi_bluff_hands.add(h)
                semi_bluff_eq.append(d.get("adj_equity"))
            if d.get("comeback_mode"):
                comeback_hands.add(h)

street_order = {"flop": 1, "turn": 2, "river": 3}
for h, plist in postflop_by_hand.items():
    if not plist:
        continue
    last = max(plist, key=lambda x: (street_order.get(x.get("street_name"), 0), x.get("ts", 0)))
    last_postflop_per_hand[h] = last

pnl_by_strength = defaultdict(float)
count_by_strength = defaultdict(int)
for h, dec in last_postflop_per_hand.items():
    if h not in real_set or h not in hand_results:
        continue
    st = dec.get("strength")
    if not st:
        continue
    pnl_by_strength[st] += hand_results[h].get("pnl", 0)
    count_by_strength[st] += 1

losses = sorted(
    ((h, all_hand_pnls[h]) for h in real_set if h in all_hand_pnls),
    key=lambda x: x[1],
)[:5]

semi_bluff_pnls = [all_hand_pnls[h] for h in semi_bluff_hands if h in real_set and h in all_hand_pnls]
semi_eq_avg = sum(semi_bluff_eq) / len(semi_bluff_eq) if semi_bluff_eq else 0

comeback_real = [h for h in comeback_hands if h in real_set]
cb_pnl = sum(all_hand_pnls[h] for h in comeback_real if h in all_hand_pnls)
norm_pnl = sum(all_hand_pnls[h] for h in real_set if h in all_hand_pnls and h not in comeback_hands)

# --- CSV actions (real hands) ---


def our_actions(h):
    out = []
    for x in hand_by[h]:
        if x["active_team"] != OUR_TEAM:
            continue
        if x["action_type"] == "DISCARD":
            continue
        out.append(x)
    return out


def final_pot(h):
    last = hand_by[h][-1]
    return int(last["team_0_bet"]) + int(last["team_1_bet"])


total_real = len(real_set)
net_pnl = sum(all_hand_pnls.get(h, 0) for h in real_set if h in all_hand_pnls)

wins = sum(1 for h in real_set if hand_results.get(h, {}).get("outcome") == "win")
ties = sum(1 for h in real_set if hand_results.get(h, {}).get("outcome") == "tie")
win_rate = wins / total_real if total_real else 0

sd_hands = [h for h in real_set if hand_results.get(h, {}).get("showdown")]
sd_count = len(sd_hands)
sd_wins = sum(1 for h in sd_hands if hand_results[h].get("outcome") == "win")
sd_win_rate = sd_wins / sd_count if sd_count else 0

pf_fold = pf_check = pf_call = pf_raise = 0
post_fold = post_check = post_call = post_raise = 0

for h in real_set:
    for x in our_actions(h):
        st = x["street"]
        at = x["action_type"]
        is_pf = st == "Pre-Flop"
        if is_pf:
            if at == "FOLD":
                pf_fold += 1
            elif at == "CHECK":
                pf_check += 1
            elif at == "CALL":
                pf_call += 1
            elif at == "RAISE":
                pf_raise += 1
        else:
            if at == "FOLD":
                post_fold += 1
            elif at == "CHECK":
                post_check += 1
            elif at == "CALL":
                post_call += 1
            elif at == "RAISE":
                post_raise += 1

avg_pot_sd = sum(final_pot(h) for h in sd_hands) / len(sd_hands) if sd_hands else 0

le50 = sum(1 for h in real_set if all_hand_pnls.get(h, 0) <= -50)
ge50 = sum(1 for h in real_set if all_hand_pnls.get(h, 0) >= 50)

pnl_sb = pnl_bb = 0
sb_c = bb_c = 0
for h in real_set:
    hs = hand_starts.get(h, {})
    pos = hs.get("position")
    p = all_hand_pnls.get(h, 0)
    if pos == "SB":
        pnl_sb += p
        sb_c += 1
    elif pos == "BB":
        pnl_bb += p
        bb_c += 1

sd_win_pots = [final_pot(h) for h in sd_hands if hand_results[h].get("outcome") == "win"]
sd_loss_pots = [final_pot(h) for h in sd_hands if hand_results[h].get("outcome") == "loss"]
avg_sd_win_pot = sum(sd_win_pots) / len(sd_win_pots) if sd_win_pots else 0
avg_sd_loss_pot = sum(sd_loss_pots) / len(sd_loss_pots) if sd_loss_pots else 0

sd_loss_monster = 0
for h in sd_hands:
    if hand_results[h].get("outcome") != "loss":
        continue
    lp = last_postflop_per_hand.get(h)
    if lp and lp.get("strength") == "monster":
        sd_loss_monster += 1

large = [h for h in real_set if abs(all_hand_pnls.get(h, 0)) >= 50]
h_sd_win = h_sd_loss = h_fold_win = h_we_fold = 0
for h in large:
    hr = hand_results.get(h, {})
    pnl = all_hand_pnls.get(h, 0)
    if hr.get("showdown"):
        if hr.get("outcome") == "win":
            h_sd_win += 1
        elif hr.get("outcome") == "loss":
            h_sd_loss += 1
    else:
        if pnl > 0 and hr.get("opp_folded"):
            h_fold_win += 1
        if hr.get("we_folded"):
            h_we_fold += 1

cal_buckets = {"0-0.3": [], "0.3-0.5": [], "0.5-0.7": [], "0.7-1.0": []}
for h in sd_hands:
    lp = last_postflop_per_hand.get(h)
    if not lp or lp.get("adj_equity") is None:
        continue
    ae = lp["adj_equity"]
    won = 1 if hand_results[h].get("outcome") == "win" else 0
    if ae < 0.3:
        cal_buckets["0-0.3"].append(won)
    elif ae < 0.5:
        cal_buckets["0.3-0.5"].append(won)
    elif ae < 0.7:
        cal_buckets["0.5-0.7"].append(won)
    else:
        cal_buckets["0.7-1.0"].append(won)

opp_fold_after_our_raise = defaultdict(lambda: {"raise": 0, "fold": 0})

for h in real_set:
    hrs = hand_by[h]
    by_street = defaultdict(list)
    for x in hrs:
        by_street[x["street"]].append(x)
    for st_name, actions in by_street.items():
        if st_name == "Pre-Flop":
            street_key = "preflop"
        else:
            street_key = st_name.lower()
        for i, x in enumerate(actions):
            if x["active_team"] != OUR_TEAM:
                continue
            if x["action_type"] != "RAISE":
                continue
            opp_fold_after_our_raise[street_key]["raise"] += 1
            folded = False
            for y in actions[i + 1 :]:
                if y["active_team"] == OPP_TEAM and y["action_type"] == "FOLD":
                    folded = True
                    break
            if folded:
                opp_fold_after_our_raise[street_key]["fold"] += 1

raise_amt = defaultdict(list)
for h in real_set:
    for x in hand_by[h]:
        if x["active_team"] != OUR_TEAM:
            continue
        if x["action_type"] != "RAISE":
            continue
        st = x["street"]
        if st == "Pre-Flop":
            key = "preflop"
        else:
            key = st.lower()
        raise_amt[key].append(x["action_amount"])

out = {
    "bleed_H": bleed_H,
    "total_real": total_real,
    "net_pnl": net_pnl,
    "win_rate": win_rate,
    "ties": ties,
    "sd_count": sd_count,
    "sd_win_rate": sd_win_rate,
    "pf_actions": {"FOLD": pf_fold, "CHECK": pf_check, "CALL": pf_call, "RAISE": pf_raise},
    "post_actions": {"FOLD": post_fold, "CHECK": post_check, "CALL": post_call, "RAISE": post_raise},
    "avg_pot_sd": avg_pot_sd,
    "le50": le50,
    "ge50": ge50,
    "pnl_sb": (pnl_sb, sb_c),
    "pnl_bb": (pnl_bb, bb_c),
    "strength_dist": {k: {"n": len(v), "avg_adj": sum(v) / len(v)} for k, v in strength_adj.items()},
    "preflop_reason_dist": dict(Counter(preflop_reasons)),
    "raw_adj_street": {
        s: {"avg_raw": sum(raw_by_street[s]) / len(raw_by_street[s]), "avg_adj": sum(adj_by_street[s]) / len(adj_by_street[s])}
        for s in ("flop", "turn", "river")
        if raw_by_street[s]
    },
    "monster_low_adj": monster_low_adj,
    "raise_low_adj": raise_low_adj,
    "avg_river_texture": sum(river_texture) / len(river_texture) if river_texture else 0,
    "top5_losses": [(h, all_hand_pnls[h]) for h, _ in losses],
    "pnl_by_strength": dict(pnl_by_strength),
    "count_by_strength": dict(count_by_strength),
    "avg_sd_win_pot": avg_sd_win_pot,
    "avg_sd_loss_pot": avg_sd_loss_pot,
    "sd_loss_monster": sd_loss_monster,
    "semi_bluff": {"count_hands": len(semi_bluff_hands), "total_pnl": sum(semi_bluff_pnls), "avg_eq_when_fired": semi_eq_avg},
    "large_pots": {"sd_win": h_sd_win, "sd_loss": h_sd_loss, "fold_win": h_fold_win, "we_folded": h_we_fold},
    "cal_buckets": {k: {"n": len(v), "win_rate": sum(v) / len(v) if v else None} for k, v in cal_buckets.items()},
    "opp_fold_raise": dict(opp_fold_after_our_raise),
    "raise_amt_avg": {k: sum(v) / len(v) for k, v in raise_amt.items()},
    "comeback": {"hands": len(comeback_real), "pnl": cb_pnl, "normal_pnl": norm_pnl, "normal_hands": total_real - len(comeback_real)},
}

print(json.dumps(out, indent=2))

print("\n--- TOP 5 LOSS DETAILS ---")
for h, pnl in sorted(((h, all_hand_pnls[h]) for h in real_set if h in all_hand_pnls), key=lambda x: x[1])[:5]:
    hr = hand_results.get(h, {})
    print(
        json.dumps(
            {
                "hand": h,
                "pnl": pnl,
                "outcome": hr.get("outcome"),
                "showdown": hr.get("showdown"),
                "we_folded": hr.get("we_folded"),
                "opp_folded": hr.get("opp_folded"),
                "large_swing": hr.get("large_swing"),
            },
            indent=2,
        )
    )
