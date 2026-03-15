"""
Data-driven analysis of match CSVs for ALPHANiT (Nit bot). Computes hand-type
and value-bet stats, then writes logs/bot_profile.json and prints a report.
Does not suggest widening pre-flop range (stay Nit).

Usage:  python analyze_for_bot.py [csv_path ...]
        Default: glob logs/*.CSV
"""
import ast
import glob
import json
import os
import sys
from collections import defaultdict

# Reuse from analyze_logs
OUR_TEAM_NAME = "Ctrl+Alt+Defeat"
PREFLOP_ALIASES = ("Pre-Flop", "Preflop")
SMALL_RAISE_THRESHOLD = 10

RANK_CHAR_TO_NUM = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7, "A": 8}
PREMIUM_PAIR_RANKS = {8, 7, 6}  # AA, 99, 88
PREMIUM_RANK_PAIRS = frozenset([frozenset([8, 7]), frozenset([8, 6]), frozenset([7, 6]), frozenset([6, 5]), frozenset([5, 4]), frozenset([5, 7])])  # A9, A8, 98, 87, 76, 79
PREMIUM_SUITED_ONLY = frozenset([frozenset([7, 6]), frozenset([6, 5]), frozenset([5, 4]), frozenset([5, 7])])  # 98s, 87s, 76s, 79s


def _rank_char(card_str):
    return card_str.strip("'\"")[0]


def _suit_char(card_str):
    s = card_str.strip("'\"")
    return s[1] if len(s) > 1 else ""


def parse_cards(cards_str):
    try:
        cards = ast.literal_eval(cards_str)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(cards, (list, tuple)):
        return []
    return [str(c) for c in cards]


def classify_hand_type(cards_str):
    """Return premium_pair | other_pair | premium_suited | premium_offsuit | other. Nit: we do not add new types."""
    cards = parse_cards(cards_str)
    if len(cards) < 2:
        return "other"
    r1 = RANK_CHAR_TO_NUM.get(_rank_char(cards[0]), -1)
    r2 = RANK_CHAR_TO_NUM.get(_rank_char(cards[1]), -1)
    s1 = _suit_char(cards[0])
    s2 = _suit_char(cards[1])
    suited = s1 == s2 and s1
    ranks = frozenset([r1, r2])
    if r1 == r2:
        if r1 in PREMIUM_PAIR_RANKS:
            return "premium_pair"
        return "other_pair"
    if ranks in PREMIUM_RANK_PAIRS:
        return "premium_suited" if suited else "premium_offsuit"
    return "other"


def parse_board(board_str):
    try:
        cards = ast.literal_eval(board_str)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(cards, (list, tuple)):
        return []
    return [str(c) for c in cards]


def board_is_paired(board_str):
    cards = parse_board(board_str)
    if len(cards) < 2:
        return False
    ranks = [_rank_char(c) for c in cards]
    return len(ranks) != len(set(ranks))


def board_max_suit_count(board_str):
    cards = parse_board(board_str)
    if not cards:
        return 0
    suits = [_suit_char(c) for c in cards]
    return max(suits.count(s) for s in set(suits)) if suits else 0


def normalize_street(street):
    return "Pre-Flop" if street in PREFLOP_ALIASES else street


def get_our_team_index(path):
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
    import csv
    reader = csv.DictReader(lines)
    for r in reader:
        rows.append(r)
    return rows


def analyze_all(paths):
    """Aggregate over all CSVs: hand_type stats, raise-when-opp-folded, showdown losses, big losses."""
    hand_type_wins = defaultdict(int)
    hand_type_losses = defaultdict(int)
    hand_type_chips_won = defaultdict(int)
    hand_type_chips_lost = defaultdict(int)
    raise_amounts_when_opp_folded = []
    showdown_losses_with_board_paired = 0
    showdown_losses_total = 0
    big_losses = []  # (delta, street, our_cards, board)
    preflop_fold_chips_lost = 0

    for path in paths:
        our_team = get_our_team_index(path)
        rows = load_rows(path)
        if not rows:
            continue

        hands = defaultdict(lambda: {"actions": [], "final_br0": 0, "final_br1": 0, "ended_by": None})
        for r in rows:
            hnum = int(r["hand_number"])
            h = hands[hnum]
            h["actions"].append(r)
            h["final_br0"] = int(r["team_0_bankroll"])
            h["final_br1"] = int(r["team_1_bankroll"])
            if r["action_type"] == "FOLD":
                h["ended_by"] = int(r["active_team"])

        hand_nums = sorted(hands.keys())
        prev_bank = 0
        our_br_key = "final_br%d" % our_team
        for hnum in hand_nums:
            h = hands[hnum]
            cur = h[our_br_key]
            delta = cur - prev_bank
            prev_bank = cur

            # Last row of hand has our final cards and board
            last_r = h["actions"][-1]
            our_cards_str = last_r["team_0_cards"] if our_team == 0 else last_r["team_1_cards"]
            board_str = last_r["board_cards"]
            street = normalize_street(last_r["street"])
            hand_type = classify_hand_type(our_cards_str)

            hand_type_wins[hand_type] += 1 if delta > 0 else 0
            hand_type_losses[hand_type] += 1 if delta < 0 else 0
            hand_type_chips_won[hand_type] += delta if delta > 0 else 0
            hand_type_chips_lost[hand_type] += abs(delta) if delta < 0 else 0

            if delta < 0 and h["ended_by"] is None:
                showdown_losses_total += 1
                if board_is_paired(board_str):
                    showdown_losses_with_board_paired += 1

            if delta <= -50:
                big_losses.append((delta, street, our_cards_str, board_str))

            if delta < 0 and h["ended_by"] == our_team:
                for a in h["actions"]:
                    if int(a["active_team"]) == our_team and a["action_type"] == "FOLD":
                        if normalize_street(a["street"]) == "Pre-Flop":
                            preflop_fold_chips_lost += abs(delta)
                        break

            if h["ended_by"] == 1 - our_team:
                for a in h["actions"]:
                    if int(a["active_team"]) == our_team and a["action_type"] == "RAISE":
                        try:
                            raise_amounts_when_opp_folded.append(int(a["action_amount"]))
                        except (ValueError, TypeError):
                            pass
                        break

    total_hands = sum(hand_type_wins[t] + hand_type_losses[t] for t in hand_type_wins)
    avg_raise_opp_fold = sum(raise_amounts_when_opp_folded) / len(raise_amounts_when_opp_folded) if raise_amounts_when_opp_folded else 0
    pct_raise_small = sum(1 for x in raise_amounts_when_opp_folded if x < SMALL_RAISE_THRESHOLD) / len(raise_amounts_when_opp_folded) * 100 if raise_amounts_when_opp_folded else 0

    # Nit: raise larger when we left value (opp folded to small raises)
    if avg_raise_opp_fold < 8:
        recommended_open = 10
    elif avg_raise_opp_fold < 12:
        recommended_open = 8
    else:
        recommended_open = 6
    # Tighter river when showdown losses high; tighter on paired boards
    river_call_ratio = 0.30 if showdown_losses_total > 500 else (0.35 if showdown_losses_total > 100 else 0.40)
    board_paired_penalty = 0.10 if showdown_losses_with_board_paired > 300 else 0.08
    # Extract more with strong non-pair premiums when they're losing
    premium_suited_total = hand_type_wins.get("premium_suited", 0) + hand_type_losses.get("premium_suited", 0)
    premium_offsuit_total = hand_type_wins.get("premium_offsuit", 0) + hand_type_losses.get("premium_offsuit", 0)
    wr_s = hand_type_wins.get("premium_suited", 0) / premium_suited_total if premium_suited_total else 1.0
    wr_o = hand_type_wins.get("premium_offsuit", 0) / premium_offsuit_total if premium_offsuit_total else 1.0
    strong_flop_hi = 0.78 if (wr_s < 0.45 or wr_o < 0.45) else 0.75

    profile = {
        "standard_open": recommended_open,
        "monster_bet_frac_flop_lo": 0.55,
        "monster_bet_frac_flop_hi": 0.72,
        "strong_bet_frac_flop_lo": 0.60,
        "strong_bet_frac_flop_hi": strong_flop_hi,
        "good_bet_frac_lo": 0.30,
        "good_bet_frac_hi": 0.50,
        "river_call_pot_ratio_max": river_call_ratio,
        "board_paired_equity_penalty": board_paired_penalty,
        "min_value_raise": 12,
    }

    return {
        "hand_type_wins": dict(hand_type_wins),
        "hand_type_losses": dict(hand_type_losses),
        "hand_type_chips_won": dict(hand_type_chips_won),
        "hand_type_chips_lost": dict(hand_type_chips_lost),
        "raise_amounts_when_opp_folded": raise_amounts_when_opp_folded,
        "avg_raise_opp_fold": avg_raise_opp_fold,
        "pct_raise_small": pct_raise_small,
        "showdown_losses_total": showdown_losses_total,
        "showdown_losses_board_paired": showdown_losses_with_board_paired,
        "big_losses": big_losses[:20],
        "preflop_fold_chips_lost": preflop_fold_chips_lost,
        "total_hands": total_hands,
        "profile": profile,
    }


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        base = os.path.join(os.path.dirname(__file__), "logs")
        a = sorted(glob.glob(os.path.join(base, "*.CSV")))
        b = sorted(glob.glob(os.path.join(base, "*.csv")))
        paths = list(dict.fromkeys(a + b))

    if not paths:
        print("No CSV files found. Use: python analyze_for_bot.py [csv_path ...] or add files in logs/")
        return

    data = analyze_all(paths)
    profile = data["profile"]

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    profile_path = os.path.join(log_dir, "bot_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print("Wrote %s" % profile_path)

    print("\n" + "=" * 72)
    print("BOT PROFILE REPORT (Nit: no range widening)")
    print("=" * 72)
    print("\n--- RECOMMENDED PARAMETERS ---")
    print("  standard_open: %d" % profile["standard_open"])
    print("  monster_bet_frac_flop: %.2f - %.2f" % (profile["monster_bet_frac_flop_lo"], profile["monster_bet_frac_flop_hi"]))
    print("  strong_bet_frac_flop: %.2f - %.2f" % (profile["strong_bet_frac_flop_lo"], profile["strong_bet_frac_flop_hi"]))
    print("  river_call_pot_ratio_max: %.2f (fold to large river bets without premium)" % profile["river_call_pot_ratio_max"])
    print("  min_value_raise: %d" % profile["min_value_raise"])

    print("\n--- WIN RATE BY HAND TYPE (confirm premium only; do not add types) ---")
    for ht in ["premium_pair", "other_pair", "premium_suited", "premium_offsuit", "other"]:
        w = data["hand_type_wins"].get(ht, 0)
        l = data["hand_type_losses"].get(ht, 0)
        total = w + l
        if total:
            print("  %-16s  wins=%d  losses=%d  win%%=%.0f" % (ht, w, l, 100 * w / total))

    print("\n--- VALUE BET WHEN OPPONENT FOLDED ---")
    print("  Count: %d  Avg raise: %.1f  Pct raise < %d: %.0f%%" % (
        len(data["raise_amounts_when_opp_folded"]),
        data["avg_raise_opp_fold"],
        SMALL_RAISE_THRESHOLD,
        data["pct_raise_small"],
    ))

    print("\n--- SHOWDOWN LOSSES ---")
    print("  Total: %d  With paired board: %d" % (data["showdown_losses_total"], data["showdown_losses_board_paired"]))
    print("  Pre-flop fold chips lost (report only): %d" % data["preflop_fold_chips_lost"])

    if data["big_losses"]:
        print("\n--- SAMPLE BIG LOSSES (<= -50) ---")
        for delta, street, cards, board in data["big_losses"][:5]:
            print("  %+d  street=%s  our_cards=%s  board=%s" % (delta, street, cards[:40], board[:40]))

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
