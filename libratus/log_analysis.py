"""
Match log analysis and self-patching loop for Libratus-Lite.
Parses match CSVs, buckets hands by strength/action/outcome,
identifies EV leaks, and suggests parameter adjustments.
"""
import csv
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RANKS = "23456789A"
NUM_RANKS = 9
RANK_A = 8
RANK_9 = 7


def _rank_char(c_str):
    return RANKS.index(c_str[0]) if c_str and c_str[0] in RANKS else -1


def _suit_char(c_str):
    return c_str[1] if len(c_str) >= 2 else "?"


def parse_cards(cards_str):
    """Parse card string like "['9s', '9d']" into list of card strings."""
    if not cards_str or cards_str in ("[]", ""):
        return []
    cards_str = cards_str.strip("[]' ")
    parts = [p.strip().strip("'\"") for p in cards_str.split(",")]
    return [p for p in parts if len(p) >= 2 and p[0] in RANKS]


def get_our_team_index(path):
    """Infer which team index is 'Libratus' or 'Ctrl+Alt+Defeat' from file header comment."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                # e.g. "# Team 0: Ctrl+Alt+Defeat, Team 1: sandwiches"
                low = line.lower()
                for name in ("ctrl+alt+defeat", "libratus"):
                    if name in low:
                        if "team 0:" in low and name in low.split("team 1:")[0].lower():
                            return 0
                        if "team 1:" in low:
                            return 1
                break
    return 0


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Skip comment lines starting with #
    data_lines = [l for l in lines if not l.startswith("#")]
    if not data_lines:
        return rows
    reader = csv.DictReader(data_lines)
    for row in reader:
        # Filter out None keys from malformed CSV
        clean = {k: v for k, v in row.items() if k is not None}
        rows.append(clean)
    return rows


def normalize_street(street):
    s = street.strip().lower() if street else ""
    if "pre" in s:
        return "preflop"
    if "flop" in s:
        return "flop"
    if "turn" in s:
        return "turn"
    if "river" in s:
        return "river"
    return s


def classify_keep(cards):
    """Classify a 2-card keep into a bucket."""
    if len(cards) != 2:
        return "unknown"
    r1 = _rank_char(cards[0])
    r2 = _rank_char(cards[1])
    s1 = _suit_char(cards[0])
    s2 = _suit_char(cards[1])

    if r1 == r2:
        if r1 in (RANK_A, RANK_9):
            return "premium_pair"
        if r1 >= 6:
            return "medium_pair"
        return "low_pair"
    suited = s1 == s2
    gap = abs(r1 - r2)
    eg = gap if gap <= 4 else NUM_RANKS - gap
    if suited:
        if eg <= 1:
            return "suited_connector"
        if eg <= 3:
            return "suited_semi"
        return "suited_gapper"
    if eg <= 1:
        return "offsuit_connector"
    return "offsuit_other"


def analyze_match(path, our_team=None):
    """Analyze a single match CSV. Returns summary dict."""
    if our_team is None:
        our_team = get_our_team_index(path)
    opp_team = 1 - our_team

    rows = load_rows(path)
    if not rows:
        return {}

    stats = {
        "hands": 0,
        "net_chips": 0,
        "wins": 0,
        "losses": 0,
        "folds_by_us": 0,
        "folds_by_opp": 0,
        "preflop_folds": 0,
        "showdowns": 0,
        "bucket_results": defaultdict(lambda: {"wins": 0, "losses": 0, "chips": 0}),
        "action_by_street": defaultdict(lambda: defaultdict(int)),
        "big_losses": [],
        "blind_leak": 0,
    }

    # Group rows by hand_number
    hands = defaultdict(list)
    for row in rows:
        hid = row.get("hand_number", "")
        hands[hid].append(row)

    for hid in sorted(hands.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        hand_rows = hands[hid]
        _process_hand(hid, hand_rows, stats, our_team, opp_team)

    return stats


def _process_hand(hand_id, rows, stats, our_team, opp_team):
    """Process a single hand's rows and update stats."""
    stats["hands"] += 1

    our_cards = []
    our_delta = 0
    folded_by = None

    # Get bankroll from last row of hand to compute delta
    our_bank_col = f"team_{our_team}_bankroll"
    first_bank = None
    last_bank = None

    for row in rows:
        try:
            b = float(row.get(our_bank_col, 0))
        except (ValueError, TypeError):
            b = 0
        if first_bank is None:
            first_bank = b
        last_bank = b

        # Track actions
        st = normalize_street(row.get("street", ""))
        act = row.get("action_type", "").upper()
        try:
            acting = int(row.get("active_team", -1))
        except (ValueError, TypeError):
            acting = -1

        if acting == our_team:
            stats["action_by_street"][st][act.lower()] += 1
            if act == "FOLD":
                stats["folds_by_us"] += 1
                if st == "preflop":
                    stats["preflop_folds"] += 1
                folded_by = "us"
        elif acting == opp_team:
            if act == "FOLD":
                stats["folds_by_opp"] += 1
                folded_by = "opp"

        # Get our 2-card keep (look for rows where our cards column has exactly 2)
        our_cards_col = f"team_{our_team}_cards"
        cards_str = row.get(our_cards_col, "")
        parsed = parse_cards(cards_str)
        if len(parsed) == 2:
            our_cards = parsed

    # Compute net from bankroll change
    # Bankroll columns show cumulative totals. Find the bankroll at the END of this hand
    # vs the beginning. But the CSV format shows the bankroll at the start of each action.
    # Actually, the next hand's first row will show updated bankrolls.
    # Since we can't easily access next hand here, we use a different approach:
    # Look at final bets to estimate result
    if rows:
        last_row = rows[-1]
        try:
            our_bet = float(last_row.get(f"team_{our_team}_bet", 0))
            opp_bet = float(last_row.get(f"team_{opp_team}_bet", 0))
        except (ValueError, TypeError):
            our_bet = opp_bet = 0

        if folded_by == "opp":
            our_delta = opp_bet
        elif folded_by == "us":
            our_delta = -our_bet
        else:
            # Went to showdown or check-down -- need to figure out who won
            stats["showdowns"] += 1
            # Check bankroll change between this hand and next (not available here)
            # Approximate: assume bigger bet player committed more
            our_delta = 0  # Will be 0 for showdowns we can't determine

    if our_delta > 0:
        stats["wins"] += 1
    elif our_delta < 0:
        stats["losses"] += 1

    stats["net_chips"] += our_delta

    if our_delta <= -50:
        stats["big_losses"].append({
            "hand_id": hand_id,
            "delta": our_delta,
            "cards": our_cards,
        })

    if our_cards and len(our_cards) == 2:
        bucket = classify_keep(our_cards)
        if our_delta > 0:
            stats["bucket_results"][bucket]["wins"] += 1
        elif our_delta < 0:
            stats["bucket_results"][bucket]["losses"] += 1
        stats["bucket_results"][bucket]["chips"] += our_delta


def analyze_all(paths, our_team=0):
    """Analyze multiple match files."""
    combined = {
        "total_hands": 0,
        "total_net": 0,
        "total_wins": 0,
        "total_losses": 0,
        "folds_by_us": 0,
        "folds_by_opp": 0,
        "preflop_folds": 0,
        "bucket_results": defaultdict(lambda: {"wins": 0, "losses": 0, "chips": 0}),
        "action_by_street": defaultdict(lambda: defaultdict(int)),
        "big_losses": [],
        "matches": [],
    }

    for path in paths:
        print(f"Analyzing {os.path.basename(path)}...")
        match_stats = analyze_match(path, our_team)
        if not match_stats:
            continue

        combined["total_hands"] += match_stats.get("hands", 0)
        combined["total_net"] += match_stats.get("net_chips", 0)
        combined["total_wins"] += match_stats.get("wins", 0)
        combined["total_losses"] += match_stats.get("losses", 0)
        combined["folds_by_us"] += match_stats.get("folds_by_us", 0)
        combined["folds_by_opp"] += match_stats.get("folds_by_opp", 0)
        combined["preflop_folds"] += match_stats.get("preflop_folds", 0)
        combined["big_losses"].extend(match_stats.get("big_losses", []))

        for bucket, data in match_stats.get("bucket_results", {}).items():
            combined["bucket_results"][bucket]["wins"] += data["wins"]
            combined["bucket_results"][bucket]["losses"] += data["losses"]
            combined["bucket_results"][bucket]["chips"] += data["chips"]

        for street, actions in match_stats.get("action_by_street", {}).items():
            for act, count in actions.items():
                combined["action_by_street"][street][act] += count

        combined["matches"].append({
            "file": os.path.basename(path),
            "hands": match_stats.get("hands", 0),
            "net": match_stats.get("net_chips", 0),
        })

    return combined


def print_report(combined):
    """Print a detailed leak analysis report."""
    print("\n" + "=" * 60)
    print("LIBRATUS-LITE LOG ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nMatches analyzed: {len(combined['matches'])}")
    print(f"Total hands: {combined['total_hands']}")
    print(f"Total net: {combined['total_net']:+.0f} chips")
    print(f"Win rate: {combined['total_wins']}/{combined['total_hands']} "
          f"({100 * combined['total_wins'] / max(1, combined['total_hands']):.1f}%)")

    print(f"\nFolds by us: {combined['folds_by_us']} "
          f"(preflop: {combined['preflop_folds']})")
    print(f"Folds by opp: {combined['folds_by_opp']}")

    print("\n--- Match Breakdown ---")
    for m in combined["matches"]:
        status = "WIN" if m["net"] > 0 else "LOSS" if m["net"] < 0 else "DRAW"
        print(f"  {m['file']:15s}  {m['hands']:4d} hands  {m['net']:+6.0f} chips  [{status}]")

    print("\n--- Keep Bucket Performance ---")
    print(f"  {'Bucket':20s} {'W':>4s} {'L':>4s} {'Net':>8s}")
    for bucket in sorted(combined["bucket_results"].keys()):
        d = combined["bucket_results"][bucket]
        print(f"  {bucket:20s} {d['wins']:4d} {d['losses']:4d} {d['chips']:+8.0f}")

    print("\n--- Action Distribution by Street ---")
    for street in ["preflop", "flop", "turn", "river"]:
        actions = combined["action_by_street"].get(street, {})
        if actions:
            total = sum(actions.values())
            parts = [f"{act}: {cnt} ({100 * cnt / total:.0f}%)" for act, cnt in sorted(actions.items())]
            print(f"  {street:8s}: {', '.join(parts)}")

    # Leak detection
    print("\n--- Potential Leaks ---")
    leaks = []

    total_hands = max(1, combined["total_hands"])
    fold_rate = combined["folds_by_us"] / total_hands
    if fold_rate > 0.50:
        leaks.append(f"HIGH FOLD RATE: {fold_rate:.1%} - consider loosening pre/post-flop play")

    preflop_fold_rate = combined["preflop_folds"] / total_hands
    if preflop_fold_rate > 0.30:
        leaks.append(f"HIGH PREFLOP FOLD: {preflop_fold_rate:.1%} - too tight preflop for LAG strategy")

    if combined["big_losses"]:
        avg_loss = sum(l["delta"] for l in combined["big_losses"]) / len(combined["big_losses"])
        leaks.append(f"BIG LOSSES: {len(combined['big_losses'])} hands avg {avg_loss:.0f} chips - "
                     "review bluff/overbet spots")

    # Check for bucket-specific leaks
    for bucket, d in combined["bucket_results"].items():
        total = d["wins"] + d["losses"]
        if total >= 10 and d["chips"] < -50:
            wr = d["wins"] / total
            leaks.append(f"LOSING BUCKET: {bucket} ({wr:.0%} win rate, {d['chips']:+.0f} chips) - "
                         "review strategy for this archetype")

    opp_fold_rate = combined["folds_by_opp"] / total_hands
    if opp_fold_rate < 0.15:
        leaks.append(f"OPP LOW FOLD: {opp_fold_rate:.1%} - opponent rarely folds, reduce bluffs")

    if not leaks:
        print("  No major leaks detected.")
    else:
        for leak in leaks:
            print(f"  * {leak}")

    # Recommendations
    print("\n--- Recommendations ---")
    if fold_rate > 0.50:
        print("  - Widen preflop calling range slightly")
        print("  - Increase semi-bluff frequency postflop")
    if preflop_fold_rate > 0.30:
        print("  - Lower fold threshold for marginal/weak preflop hands")
    if combined["big_losses"]:
        print("  - Review sizing discipline: reduce overbet frequency")
        print("  - Add river caution against large bets")
    print("  - Run 'python libratus/generate_tables.py' to regenerate tables after adjustments")


def main():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        a = glob.glob(os.path.join(log_dir, "*.CSV"))
        b = glob.glob(os.path.join(log_dir, "*.csv"))
        paths = sorted(set(a + b))

    if not paths:
        print("No log files found. Place match CSVs in logs/ directory.")
        return

    our_team = 0
    if "--team1" in sys.argv:
        our_team = 1

    combined = analyze_all(paths, our_team)
    print_report(combined)


if __name__ == "__main__":
    main()
