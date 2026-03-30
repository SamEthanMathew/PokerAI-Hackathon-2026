import json
import csv
import sys
from collections import defaultdict, Counter

BOT_FILE = "/Users/rudrakshsingh/Downloads/poker-engine-2026-work/match logs/match_71783_bot.txt"
CSV_FILE = "/Users/rudrakshsingh/Downloads/poker-engine-2026-work/match logs/match_71783_csv.txt"
MAX_HAND = 522

events = []
with open(BOT_FILE, "r") as f:
    for line in f:
        line = line.strip()
        idx = line.find("{")
        if idx == -1:
            continue
        try:
            obj = json.loads(line[idx:])
        except json.JSONDecodeError:
            continue
        if "hand" in obj and obj["hand"] <= MAX_HAND:
            events.append(obj)

hand_results = [e for e in events if e["event"] == "hand_result"]
preflop = [e for e in events if e["event"] == "preflop_decision"]
discard = [e for e in events if e["event"] == "discard_decision"]
postflop = [e for e in events if e["event"] == "postflop_decision"]

# ============================================================
# 1. MATCH OVERVIEW
# ============================================================
print("=" * 70)
print("1. MATCH OVERVIEW")
print("=" * 70)
total_hands = len(hand_results)
real_pnl = hand_results[-1]["running_pnl"] if hand_results else 0
print(f"Real hands analyzed: {total_hands}")
print(f"Final running_pnl at hand {hand_results[-1]['hand']}: {real_pnl}")
total_pnl = sum(e["pnl"] for e in hand_results)
print(f"Sum of PnL (hands 0-{MAX_HAND}): {total_pnl}")

# ============================================================
# 2. HAND RESULTS
# ============================================================
print("\n" + "=" * 70)
print("2. HAND RESULTS")
print("=" * 70)
wins = [e for e in hand_results if e["outcome"] == "win"]
losses = [e for e in hand_results if e["outcome"] == "loss"]
ties = [e for e in hand_results if e["outcome"] == "tie"]
print(f"Total hands: {len(hand_results)}")
print(f"Wins:   {len(wins)}")
print(f"Losses: {len(losses)}")
print(f"Ties:   {len(ties)}")

showdowns = [e for e in hand_results if e["showdown"]]
sd_wins = [e for e in showdowns if e["outcome"] == "win"]
sd_losses = [e for e in showdowns if e["outcome"] == "loss"]
sd_ties = [e for e in showdowns if e["outcome"] == "tie"]
print(f"\nShowdowns:      {len(showdowns)}")
print(f"Showdown wins:  {len(sd_wins)}")
print(f"Showdown losses:{len(sd_losses)}")
print(f"Showdown ties:  {len(sd_ties)}")

we_folded = [e for e in hand_results if e["we_folded"]]
opp_folded = [e for e in hand_results if e["opp_folded"]]
print(f"\nWe folded:  {len(we_folded)}")
print(f"Opp folded: {len(opp_folded)}")

big_losses = [e for e in hand_results if e["pnl"] <= -50]
big_wins = [e for e in hand_results if e["pnl"] >= 50]
print(f"\nHands pnl <= -50: count={len(big_losses)}, total PnL={sum(e['pnl'] for e in big_losses)}")
print(f"Hands pnl >= 50:  count={len(big_wins)}, total PnL={sum(e['pnl'] for e in big_wins)}")

avg_win_pnl = sum(e["pnl"] for e in wins) / len(wins) if wins else 0
avg_loss_pnl = sum(e["pnl"] for e in losses) / len(losses) if losses else 0
print(f"\nAvg PnL per win:  {avg_win_pnl:.4f}")
print(f"Avg PnL per loss: {avg_loss_pnl:.4f}")

# ============================================================
# 3. PREFLOP
# ============================================================
print("\n" + "=" * 70)
print("3. PREFLOP DECISIONS")
print("=" * 70)
action_counts = Counter(e["action"] for e in preflop)
reason_counts = Counter(e["reason"] for e in preflop)
print(f"Total preflop decisions: {len(preflop)}")
print("\nBy action:")
for a, c in sorted(action_counts.items()):
    print(f"  {a}: {c}")
print("\nBy reason:")
for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print(f"  {r}: {c}")

equity_folds = [e for e in preflop if e["action"] == "FOLD" and e.get("equity") is not None]
equity_raises_eq = [e for e in preflop if e["action"] == "RAISE" and e.get("reason") in ("equity_raise",)]
equity_checks_eq = [e for e in preflop if e["action"] == "CHECK" and e.get("reason") in ("equity_check",)]

all_equity_based_folds = [e for e in preflop if e["action"] == "FOLD" and e.get("equity") is not None]
all_equity_based_raises = [e for e in preflop if e["action"] == "RAISE" and e.get("reason", "").startswith("equity")]

if all_equity_based_folds:
    avg_eq_fold = sum(e["equity"] for e in all_equity_based_folds) / len(all_equity_based_folds)
    print(f"\nAvg equity when FOLD: {avg_eq_fold:.5f} (n={len(all_equity_based_folds)})")
else:
    print("\nNo equity-based folds")

if all_equity_based_raises:
    avg_eq_raise = sum(e["equity"] for e in all_equity_based_raises) / len(all_equity_based_raises)
    print(f"Avg equity when RAISE (equity-based): {avg_eq_raise:.5f} (n={len(all_equity_based_raises)})")
else:
    print("No equity-based raises")

# ============================================================
# 4. DISCARD
# ============================================================
print("\n" + "=" * 70)
print("4. DISCARD DECISIONS")
print("=" * 70)
print(f"Total discard decisions: {len(discard)}")
if discard:
    avg_chosen_eq = sum(e["chosen_equity"] for e in discard) / len(discard)
    avg_margin = sum(e["equity_margin"] for e in discard) / len(discard)
    margin_lt_002 = sum(1 for e in discard if e["equity_margin"] < 0.02)
    margin_lt_001 = sum(1 for e in discard if e["equity_margin"] < 0.01)
    print(f"Avg chosen_equity: {avg_chosen_eq:.5f}")
    print(f"Avg equity_margin: {avg_margin:.5f}")
    print(f"Count margin < 0.02: {margin_lt_002}")
    print(f"Count margin < 0.01: {margin_lt_001}")

# ============================================================
# 5. POSTFLOP
# ============================================================
print("\n" + "=" * 70)
print("5. POSTFLOP DECISIONS")
print("=" * 70)
print(f"Total postflop decisions: {len(postflop)}")

# Count by street_name x final_action
street_action = Counter((e["street_name"], e["final_action"]) for e in postflop)
print("\nStreet x Final Action:")
for (s, a), c in sorted(street_action.items()):
    print(f"  {s:6s} x {a:6s}: {c}")

# Avg raw_equity and adj_equity by final_action
action_equities = defaultdict(lambda: {"raw": [], "adj": []})
for e in postflop:
    action_equities[e["final_action"]]["raw"].append(e["raw_equity"])
    action_equities[e["final_action"]]["adj"].append(e["adj_equity"])
print("\nAvg equity by final_action:")
for a in sorted(action_equities.keys()):
    raw = action_equities[a]["raw"]
    adj = action_equities[a]["adj"]
    print(f"  {a:6s}: avg_raw={sum(raw)/len(raw):.5f}, avg_adj={sum(adj)/len(adj):.5f} (n={len(raw)})")

semi_bluff_count = sum(1 for e in postflop if e.get("semi_bluff_fired"))
baseline_changed_count = sum(1 for e in postflop if e.get("baseline_changed"))
print(f"\nSemi bluff fired: {semi_bluff_count}")
print(f"Baseline changed: {baseline_changed_count}")

# Avg texture_adj by street
street_tex = defaultdict(list)
for e in postflop:
    street_tex[e["street_name"]].append(e.get("texture_adj", 0))
print("\nAvg texture_adj by street:")
for s in ["flop", "turn", "river"]:
    if s in street_tex:
        vals = street_tex[s]
        print(f"  {s:6s}: {sum(vals)/len(vals):.5f} (n={len(vals)})")

# Monster/trips_plus with adj_equity < 0.50
monster_low = [e for e in postflop if e.get("strength") in ("monster",) and e.get("hand_cat") in ("trips_plus",) and e["adj_equity"] < 0.50]
monster_low_all = [e for e in postflop if (e.get("strength") == "monster" or e.get("hand_cat") == "trips_plus") and e["adj_equity"] < 0.50]
print(f"\nMonster with adj_equity < 0.50: {len([e for e in postflop if e.get('strength') == 'monster' and e['adj_equity'] < 0.50])}")
print(f"trips_plus with adj_equity < 0.50: {len([e for e in postflop if e.get('hand_cat') == 'trips_plus' and e['adj_equity'] < 0.50])}")
print(f"monster OR trips_plus with adj_equity < 0.50: {len(monster_low_all)}")

# CHECK with adj_equity > 0.70
check_high = [e for e in postflop if e["final_action"] == "CHECK" and e["adj_equity"] > 0.70]
print(f"\nCHECK with adj_equity > 0.70: {len(check_high)}")
for e in check_high:
    print(f"  hand={e['hand']}, street={e['street_name']}, adj_equity={e['adj_equity']:.4f}, hand_cat={e.get('hand_cat')}, strength={e.get('strength')}")

# RAISE with adj_equity < 0.40
raise_low = [e for e in postflop if e["final_action"] == "RAISE" and e["adj_equity"] < 0.40]
print(f"\nRAISE with adj_equity < 0.40: {len(raise_low)}")
for e in raise_low:
    print(f"  hand={e['hand']}, street={e['street_name']}, adj_equity={e['adj_equity']:.4f}, hand_cat={e.get('hand_cat')}, strength={e.get('strength')}, semi_bluff={e.get('semi_bluff_fired')}")

# ============================================================
# 6. TOP 10 WORST LOSSES
# ============================================================
print("\n" + "=" * 70)
print("6. TOP 10 WORST LOSSES")
print("=" * 70)
sorted_losses = sorted(hand_results, key=lambda e: e["pnl"])
for i, e in enumerate(sorted_losses[:10]):
    print(f"  #{i+1}: hand={e['hand']}, pnl={e['pnl']}, outcome={e['outcome']}, showdown={e['showdown']}, "
          f"we_folded={e['we_folded']}, opp_folded={e['opp_folded']}, "
          f"our_cards={e.get('our_kept_cards')}, opp_cards={e.get('opp_kept_cards')}, "
          f"community={e.get('community')}")

# ============================================================
# 7. OPPONENT ACTIONS FROM CSV
# ============================================================
print("\n" + "=" * 70)
print("7. OPPONENT ACTIONS FROM CSV (hand_number <= 522)")
print("=" * 70)

opp_actions = []
with open(CSV_FILE, "r") as f:
    lines = f.readlines()

data_start = 0
header_line = None
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("#"):
        continue
    if stripped.startswith("hand_number"):
        header_line = stripped
        data_start = i + 1
        break

header = [h.strip() for h in header_line.split(",")]

for line in lines[data_start:]:
    line = line.strip()
    if not line:
        continue
    parts = []
    in_quotes = False
    current = []
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
            current.append(ch)
        elif ch == ',' and not in_quotes:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current))
    
    row = {}
    for idx, h in enumerate(header):
        if idx < len(parts):
            row[h] = parts[idx].strip()
    
    try:
        hand_num = int(row.get("hand_number", -1))
    except (ValueError, TypeError):
        continue
    if hand_num > MAX_HAND:
        continue
    
    active_team = row.get("active_team", "").strip()
    action_type = row.get("action_type", "").strip()
    street = row.get("street", "").strip()
    action_amount = row.get("action_amount", "0").strip()
    
    if active_team == "1":
        opp_actions.append({"hand": hand_num, "street": street, "action": action_type, "amount": action_amount})

opp_action_counts = Counter(a["action"] for a in opp_actions)
print(f"Total opponent actions: {len(opp_actions)}")
print("\nOpponent action counts:")
for a, c in sorted(opp_action_counts.items(), key=lambda x: -x[1]):
    print(f"  {a}: {c}")

opp_street_action = Counter((a["street"], a["action"]) for a in opp_actions)
print("\nOpponent Street x Action:")
for (s, a), c in sorted(opp_street_action.items()):
    print(f"  {s:10s} x {a:10s}: {c}")

# Opponent fold rate
opp_total_decisions = len(opp_actions)
opp_folds = opp_action_counts.get("FOLD", 0)
opp_fold_rate = opp_folds / opp_total_decisions * 100 if opp_total_decisions else 0
print(f"\nOpponent fold rate: {opp_folds}/{opp_total_decisions} = {opp_fold_rate:.2f}%")

# Opponent raise count and call count
opp_raises = opp_action_counts.get("RAISE", 0)
opp_calls = opp_action_counts.get("CALL", 0)
opp_checks = opp_action_counts.get("CHECK", 0)
print(f"Opponent RAISE: {opp_raises}")
print(f"Opponent CALL:  {opp_calls}")
print(f"Opponent CHECK: {opp_checks}")
print(f"Opponent FOLD:  {opp_folds}")
opp_discards = opp_action_counts.get("DISCARD", 0)
print(f"Opponent DISCARD: {opp_discards}")

# Opponent preflop actions
opp_preflop = [a for a in opp_actions if a["street"] == "Pre-Flop"]
opp_pf_counts = Counter(a["action"] for a in opp_preflop)
print(f"\nOpponent Preflop actions:")
for a, c in sorted(opp_pf_counts.items(), key=lambda x: -x[1]):
    print(f"  {a}: {c}")
opp_pf_fold_rate = opp_pf_counts.get("FOLD", 0) / len(opp_preflop) * 100 if opp_preflop else 0
print(f"Opponent preflop fold rate: {opp_pf_counts.get('FOLD', 0)}/{len(opp_preflop)} = {opp_pf_fold_rate:.2f}%")

# Opponent postflop (flop/turn/river)
for street_name in ["Flop", "Turn", "River"]:
    opp_street = [a for a in opp_actions if a["street"] == street_name]
    if opp_street:
        opp_st_counts = Counter(a["action"] for a in opp_street)
        fold_n = opp_st_counts.get("FOLD", 0)
        total_n = len(opp_street)
        print(f"\nOpponent {street_name} actions (n={total_n}):")
        for a, c in sorted(opp_st_counts.items(), key=lambda x: -x[1]):
            print(f"  {a}: {c}")
        print(f"  Fold rate: {fold_n}/{total_n} = {fold_n/total_n*100:.2f}%")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
