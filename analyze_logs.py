"""
Analyze match CSV(s) for Ctrl+Alt+Defeat (our bot). Supports multiple files,
infers "our team" from header, normalizes Pre-Flop/Preflop, and reports
blind leak, folded premium pair, and value-left-on-table.

Usage:  python analyze_logs.py [csv_path ...]
        Default: logs/M1.CSV logs/M2.CSV logs/M3.CSV logs/M4.CSV logs/M5.CSV
        With team name in header: "Team 0: Ctrl+Alt+Defeat" -> we are team 0
"""
import ast
import csv
import glob
import os
import sys
from collections import defaultdict

OUR_TEAM_NAME = "Ctrl+Alt+Defeat"
PREFLOP_ALIASES = ("Pre-Flop", "Preflop")
SMALL_RAISE_THRESHOLD = 10

# Premium pairs: AA (rank A=8 in engine), 99 (7), 88 (6). Cards in CSV are like 'As', '9h', '8d'.
def _rank_char(card_str):
    return card_str.strip("'\"")[0]

def is_premium_pair_card_strings(cards_str):
    """cards_str is e.g. \"['9s', '9d']\" or \"['Ad', 'Ah']\". Returns True if two cards form AA, 99, or 88."""
    try:
        cards = ast.literal_eval(cards_str)
    except (ValueError, SyntaxError):
        return False
    if not isinstance(cards, (list, tuple)) or len(cards) < 2:
        return False
    r1 = _rank_char(str(cards[0]))
    r2 = _rank_char(str(cards[1]))
    if r1 != r2:
        return False
    return r1 in ("A", "9", "8")


def normalize_street(street):
    return "Pre-Flop" if street in PREFLOP_ALIASES else street


def get_our_team_index(path):
    """Return 0 or 1 if our team name appears in the CSV header comment."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") and OUR_TEAM_NAME in line:
                if "Team 0:" in line and OUR_TEAM_NAME in line.split("Team 0:")[1].split(",")[0]:
                    return 0
                if "Team 1:" in line and OUR_TEAM_NAME in line.split("Team 1:")[1].split(",")[0]:
                    return 1
            if line.strip() and not line.startswith("#"):
                break
    return 0


def load_rows(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    if not lines:
        return rows
    reader = csv.DictReader(lines)
    for r in reader:
        rows.append(r)
    return rows


def analyze_one(path, our_team):
    rows = load_rows(path)
    if not rows:
        return None

    hands = defaultdict(lambda: {
        "actions": [],
        "final_bankroll_0": 0,
        "final_bankroll_1": 0,
        "ended_by": None,
        "end_street": None,
        "our_cards": None,
        "opp_cards": None,
        "board_cards": None,
    })

    for r in rows:
        hnum = int(r["hand_number"])
        h = hands[hnum]
        h["actions"].append(r)
        h["final_bankroll_0"] = int(r["team_0_bankroll"])
        h["final_bankroll_1"] = int(r["team_1_bankroll"])
        h["end_street"] = r["street"]
        h["our_cards"] = r["team_0_cards"] if our_team == 0 else r["team_1_cards"]
        h["opp_cards"] = r["team_1_cards"] if our_team == 0 else r["team_0_cards"]
        h["board_cards"] = r["board_cards"]
        if r["action_type"] == "FOLD":
            h["ended_by"] = "fold_by_%d" % int(r["active_team"])

    hand_nums = sorted(hands.keys())
    num_hands = len(hand_nums)

    our_bank_key = "final_bankroll_%d" % our_team
    prev_bank = 0
    hand_deltas = {}
    for hnum in hand_nums:
        h = hands[hnum]
        cur = h[our_bank_key]
        hand_deltas[hnum] = cur - prev_bank
        prev_bank = cur

    wins = [h for h in hand_nums if hand_deltas[h] > 0]
    losses = [h for h in hand_nums if hand_deltas[h] < 0]
    ties = [h for h in hand_nums if hand_deltas[h] == 0]
    total_won = sum(hand_deltas[h] for h in wins)
    total_lost = sum(hand_deltas[h] for h in losses)
    net = total_won + total_lost

    folds_by_street_us = defaultdict(int)
    folds_by_street_opp = defaultdict(int)
    raises_by_street_us = defaultdict(list)
    raises_by_street_opp = defaultdict(list)

    for r in rows:
        act = r["action_type"]
        street = normalize_street(r["street"])
        team = int(r["active_team"])
        if act == "FOLD":
            if team == our_team:
                folds_by_street_us[street] += 1
            else:
                folds_by_street_opp[street] += 1
        if act == "RAISE":
            amt = int(r["action_amount"])
            if team == our_team:
                raises_by_street_us[street].append(amt)
            else:
                raises_by_street_opp[street].append(amt)

    preflop_folds_us = 0
    blind_leak = 0
    for hnum in hand_nums:
        h = hands[hnum]
        for a in h["actions"]:
            if a["action_type"] == "FOLD" and int(a["active_team"]) == our_team:
                st = normalize_street(a["street"])
                if st == "Pre-Flop":
                    preflop_folds_us += 1
                    if hand_deltas[hnum] < 0:
                        blind_leak += abs(hand_deltas[hnum])
                break

    folded_premium_pair = 0
    for hnum in hand_nums:
        h = hands[hnum]
        if h["ended_by"] != "fold_by_%d" % our_team:
            continue
        for a in h["actions"]:
            if int(a["active_team"]) == our_team and a["action_type"] == "FOLD":
                st = normalize_street(a["street"])
                if st in ("Flop", "Turn", "River"):
                    our_cards = a["team_0_cards"] if our_team == 0 else a["team_1_cards"]
                    if is_premium_pair_card_strings(our_cards):
                        folded_premium_pair += 1
                break

    opp_folded_to_small_raise = 0
    for hnum in hand_nums:
        h = hands[hnum]
        if h["ended_by"] != "fold_by_%d" % (1 - our_team):
            continue
        for a in h["actions"]:
            if int(a["active_team"]) == our_team and a["action_type"] == "RAISE":
                amt = int(a["action_amount"])
                if amt < SMALL_RAISE_THRESHOLD:
                    opp_folded_to_small_raise += 1
                break

    showdowns_won = 0
    showdowns_lost = 0
    for hnum in hand_nums:
        h = hands[hnum]
        if h["ended_by"] is not None:
            continue
        d = hand_deltas[hnum]
        if d > 0:
            showdowns_won += 1
        elif d < 0:
            showdowns_lost += 1

    return {
        "path": path,
        "our_team": our_team,
        "num_hands": num_hands,
        "wins": len(wins),
        "losses": len(losses),
        "ties": len(ties),
        "total_won": total_won,
        "total_lost": total_lost,
        "net": net,
        "preflop_folds_us": preflop_folds_us,
        "blind_leak": blind_leak,
        "folded_premium_pair": folded_premium_pair,
        "opp_folded_to_small_raise": opp_folded_to_small_raise,
        "folds_by_street_us": dict(folds_by_street_us),
        "folds_by_street_opp": dict(folds_by_street_opp),
        "raises_by_street_us": dict(raises_by_street_us),
        "showdowns_won": showdowns_won,
        "showdowns_lost": showdowns_lost,
    }


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        base = os.path.join(os.path.dirname(__file__), "logs")
        a = sorted(glob.glob(os.path.join(base, "*.CSV")))
        b = sorted(glob.glob(os.path.join(base, "*.csv")))
        paths = list(dict.fromkeys(a + b))

    results = []
    for path in paths:
        our_team = get_our_team_index(path)
        r = analyze_one(path, our_team)
        if r is not None:
            r["our_team"] = our_team
            results.append(r)

    if not results:
        print("No data from any CSV.")
        return

    print("=" * 72)
    print("LOG ANALYSIS (our team: %s)" % OUR_TEAM_NAME)
    print("=" * 72)

    print("\n--- SUMMARY TABLE ---")
    print("%-12s  %5s  %5s  %6s  %8s  %6s  %6s  %6s" % (
        "File", "Wins", "Loss", "Net", "PFlopFold", "BlindLk", "FoldPP", "OppFold<10"))
    print("-" * 72)
    agg = defaultdict(int)
    for r in results:
        fname = r["path"].replace("\\", "/").split("/")[-1]
        print("%-12s  %5d  %5d  %+6d  %8d  %6d  %6d  %6d" % (
            fname,
            r["wins"],
            r["losses"],
            r["net"],
            r["preflop_folds_us"],
            r["blind_leak"],
            r["folded_premium_pair"],
            r["opp_folded_to_small_raise"],
        ))
        agg["wins"] += r["wins"]
        agg["losses"] += r["losses"]
        agg["net"] += r["net"]
        agg["preflop_folds_us"] += r["preflop_folds_us"]
        agg["blind_leak"] += r["blind_leak"]
        agg["folded_premium_pair"] += r["folded_premium_pair"]
        agg["opp_folded_to_small_raise"] += r["opp_folded_to_small_raise"]
        agg["num_hands"] += r["num_hands"]

    print("-" * 72)
    print("%-12s  %5d  %5d  %+6d  %8d  %6d  %6d  %6d" % (
        "(aggregate)",
        agg["wins"],
        agg["losses"],
        agg["net"],
        agg["preflop_folds_us"],
        agg["blind_leak"],
        agg["folded_premium_pair"],
        agg["opp_folded_to_small_raise"],
    ))

    print("\n--- IMPROVEMENT OPPORTUNITIES ---")
    if agg["folded_premium_pair"]:
        print("  Folded premium pair (AA/99/88) on Flop/Turn/River: %d times." % agg["folded_premium_pair"])
        print("  -> Add safeguard: do not fold premium pair to a single small raise.")
    if agg["opp_folded_to_small_raise"]:
        print("  Opponent folded to our raise < %d: %d times." % (SMALL_RAISE_THRESHOLD, agg["opp_folded_to_small_raise"]))
        print("  -> Consider raising larger with strong hands to extract more value.")
    if agg["blind_leak"]:
        print("  Chips lost from blinds (we folded pre-flop): %d." % agg["blind_leak"])
        print("  -> If high, consider slightly widening pre-flop range or defending blinds.")
    if not (agg["folded_premium_pair"] or agg["opp_folded_to_small_raise"] or agg["blind_leak"]):
        print("  (No specific opportunities flagged from these metrics.)")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
