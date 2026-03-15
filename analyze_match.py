"""
Analyze a match CSV and show where bot0 (ALPHANiT) is losing/winning chips
and where it could improve.

Usage:  python analyze_match.py [match_csv_path]
        Default: ./match_50_hands.csv
"""
import csv
import sys
from collections import defaultdict


def load_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        # skip comment lines at top
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.DictReader(lines)
    for r in reader:
        rows.append(r)
    return rows


def analyze(path):
    rows = load_rows(path)
    if not rows:
        print("No data in CSV.")
        return

    # ---- per-hand summaries ----
    hands = defaultdict(lambda: {
        "actions": [],
        "final_bankroll_0": 0,
        "final_bankroll_1": 0,
        "ended_by": None,
        "end_street": None,
        "bot0_cards": None,
        "bot1_cards": None,
        "board_cards": None,
    })

    for r in rows:
        hnum = int(r["hand_number"])
        h = hands[hnum]
        h["actions"].append(r)
        h["final_bankroll_0"] = int(r["team_0_bankroll"])
        h["final_bankroll_1"] = int(r["team_1_bankroll"])
        h["end_street"] = r["street"]
        h["bot0_cards"] = r["team_0_cards"]
        h["bot1_cards"] = r["team_1_cards"]
        h["board_cards"] = r["board_cards"]
        if r["action_type"] == "FOLD":
            h["ended_by"] = "fold_by_%d" % int(r["active_team"])

    hand_nums = sorted(hands.keys())
    num_hands = len(hand_nums)

    # compute per-hand chip deltas for bot0
    prev_bank_0 = 0
    hand_deltas = {}
    for hnum in hand_nums:
        h = hands[hnum]
        cur = h["final_bankroll_0"]
        hand_deltas[hnum] = cur - prev_bank_0
        prev_bank_0 = cur

    wins_0 = [hnum for hnum, d in hand_deltas.items() if d > 0]
    losses_0 = [hnum for hnum, d in hand_deltas.items() if d < 0]
    ties = [hnum for hnum, d in hand_deltas.items() if d == 0]

    total_won = sum(hand_deltas[h] for h in wins_0)
    total_lost = sum(hand_deltas[h] for h in losses_0)
    net = total_won + total_lost

    # ---- street analysis ----
    folds_by_street_bot0 = defaultdict(int)  # bot0 folded on this street
    folds_by_street_bot1 = defaultdict(int)  # bot1 folded on this street
    raises_by_street_bot0 = defaultdict(list)
    raises_by_street_bot1 = defaultdict(list)

    for r in rows:
        act = r["action_type"]
        street = r["street"]
        team = int(r["active_team"])
        if act == "FOLD":
            if team == 0:
                folds_by_street_bot0[street] += 1
            else:
                folds_by_street_bot1[street] += 1
        if act == "RAISE":
            amt = int(r["action_amount"])
            if team == 0:
                raises_by_street_bot0[street].append(amt)
            else:
                raises_by_street_bot1[street].append(amt)

    # ---- blind leak: hands where bot0 folded pre-flop ----
    preflop_folds_0 = 0
    blind_leak = 0
    for hnum in hand_nums:
        h = hands[hnum]
        for a in h["actions"]:
            if a["action_type"] == "FOLD" and int(a["active_team"]) == 0 and a["street"] == "Preflop":
                preflop_folds_0 += 1
                d = hand_deltas[hnum]
                if d < 0:
                    blind_leak += abs(d)
                break

    # ---- showdown analysis ----
    showdowns_won = 0
    showdowns_lost = 0
    showdown_chips_won = 0
    showdown_chips_lost = 0
    for hnum in hand_nums:
        h = hands[hnum]
        ended_fold = h["ended_by"]
        d = hand_deltas[hnum]
        if ended_fold is None:
            if d > 0:
                showdowns_won += 1
                showdown_chips_won += d
            elif d < 0:
                showdowns_lost += 1
                showdown_chips_lost += abs(d)

    # ---- biggest wins/losses ----
    sorted_by_delta = sorted(hand_deltas.items(), key=lambda x: x[1])
    worst_5 = sorted_by_delta[:5]
    best_5 = sorted_by_delta[-5:][::-1]

    # ---- print report ----
    print("=" * 70)
    print("MATCH ANALYSIS (Bot0 = ALPHANiT, Bot1 = Opponent)")
    print("CSV: %s   |   Hands: %d" % (path, num_hands))
    print("=" * 70)

    print("\n--- OVERALL ---")
    print("  Hands won:  %d / %d  (%.0f%%)" % (len(wins_0), num_hands, 100 * len(wins_0) / num_hands if num_hands else 0))
    print("  Hands lost: %d / %d  (%.0f%%)" % (len(losses_0), num_hands, 100 * len(losses_0) / num_hands if num_hands else 0))
    print("  Hands tied: %d" % len(ties))
    print("  Chips won:  +%d" % total_won)
    print("  Chips lost: %d" % total_lost)
    print("  Net:        %s%d" % ("+" if net >= 0 else "", net))

    print("\n--- BLIND LEAK (bot0 folding pre-flop) ---")
    print("  Pre-flop folds by bot0: %d / %d hands (%.0f%%)" % (
        preflop_folds_0, num_hands, 100 * preflop_folds_0 / num_hands if num_hands else 0))
    print("  Chips leaked from blinds: %d" % blind_leak)

    print("\n--- FOLDS BY STREET ---")
    all_streets = ["Preflop", "Flop", "Turn", "River"]
    print("  %-10s  Bot0-folds  Bot1-folds" % "Street")
    for st in all_streets:
        print("  %-10s  %10d  %10d" % (st, folds_by_street_bot0.get(st, 0), folds_by_street_bot1.get(st, 0)))

    print("\n--- RAISES BY STREET (bot0) ---")
    for st in all_streets:
        amts = raises_by_street_bot0.get(st, [])
        if amts:
            avg = sum(amts) / len(amts)
            print("  %-10s  count=%d  avg=%.1f  min=%d  max=%d" % (st, len(amts), avg, min(amts), max(amts)))
        else:
            print("  %-10s  count=0" % st)

    print("\n--- RAISES BY STREET (bot1) ---")
    for st in all_streets:
        amts = raises_by_street_bot1.get(st, [])
        if amts:
            avg = sum(amts) / len(amts)
            print("  %-10s  count=%d  avg=%.1f  min=%d  max=%d" % (st, len(amts), avg, min(amts), max(amts)))
        else:
            print("  %-10s  count=0" % st)

    print("\n--- SHOWDOWNS ---")
    total_showdowns = showdowns_won + showdowns_lost
    print("  Reached showdown: %d / %d hands" % (total_showdowns, num_hands))
    if total_showdowns:
        print("  Won at showdown:  %d (%.0f%%)  -> +%d chips" % (
            showdowns_won, 100 * showdowns_won / total_showdowns, showdown_chips_won))
        print("  Lost at showdown: %d (%.0f%%)  -> -%d chips" % (
            showdowns_lost, 100 * showdowns_lost / total_showdowns, showdown_chips_lost))

    print("\n--- BIGGEST LOSSES (bot0) ---")
    for hnum, d in worst_5:
        h = hands[hnum]
        print("  Hand %3d: %+d chips  (street=%s, cards=%s, board=%s)" % (
            hnum, d, h["end_street"], h["bot0_cards"], h["board_cards"]))

    print("\n--- BIGGEST WINS (bot0) ---")
    for hnum, d in best_5:
        h = hands[hnum]
        print("  Hand %3d: %+d chips  (street=%s, cards=%s, board=%s)" % (
            hnum, d, h["end_street"], h["bot0_cards"], h["board_cards"]))

    # ---- opportunity analysis ----
    print("\n--- WHERE BOT0 COULD WIN MORE ---")
    bot1_folded_to_small_raise = 0
    for hnum in hand_nums:
        h = hands[hnum]
        if h["ended_by"] == "fold_by_1":
            for a in h["actions"]:
                if int(a["active_team"]) == 0 and a["action_type"] == "RAISE":
                    amt = int(a["action_amount"])
                    if amt < 10:
                        bot1_folded_to_small_raise += 1
                    break
    if bot1_folded_to_small_raise:
        print("  Opponent folded to a small raise (<10) %d times." % bot1_folded_to_small_raise)
        print("  -> Could raise bigger to extract more value before they fold.")

    free_flops_checked = 0
    for hnum in hand_nums:
        h = hands[hnum]
        d = hand_deltas[hnum]
        if d <= 0:
            continue
        for a in h["actions"]:
            if int(a["active_team"]) == 0 and a["action_type"] == "CHECK" and a["street"] == "Preflop":
                free_flops_checked += 1
                break
    if free_flops_checked:
        print("  Won %d hands after checking pre-flop." % free_flops_checked)
        print("  -> These hands had hidden value; consider raising pre-flop to build bigger pots.")

    check_folds = 0
    for hnum in hand_nums:
        h = hands[hnum]
        d = hand_deltas[hnum]
        if d >= 0:
            continue
        checked_then_folded = False
        bot0_checked = False
        for a in h["actions"]:
            if int(a["active_team"]) == 0:
                if a["action_type"] == "CHECK":
                    bot0_checked = True
                elif a["action_type"] == "FOLD" and bot0_checked:
                    checked_then_folded = True
        if checked_then_folded:
            check_folds += 1
    if check_folds:
        print("  Checked then folded in %d losing hands." % check_folds)
        print("  -> Could have folded earlier to save chips, or bet to take initiative.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "./match_50_hands.csv"
    analyze(csv_path)
