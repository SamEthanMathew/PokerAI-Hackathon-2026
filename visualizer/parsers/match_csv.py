"""
Parse match CSV (per-action log from match.py).
Bankrolls in each row are at start of that hand. Bankroll after hand N = first row of hand N+1.
"""
import ast
import csv
from typing import Any


def _safe_literal_eval(s: str) -> Any:
    if s is None or s == "":
        return []
    s = s.strip()
    if s.startswith("["):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    return s


def parse_match_csv(path: str) -> dict[str, Any]:
    """
    Parse match CSV. Skip lines starting with #.
    Returns:
        actions: list of per-action dicts (hand_number, street, active_team, action_type, team_0_bankroll, team_1_bankroll, ...)
        hands: list of per-hand dicts with hand_number, actions (list), bankroll_after (from first row of next hand)
        team_0_name, team_1_name from first # comment if present
    """
    actions = []
    team_0_name = team_1_name = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    data_lines = []
    for line in raw_lines:
        if line.strip().startswith("#"):
            if "Team 0:" in line and "Team 1:" in line:
                try:
                    rest = line.split("Team 0:")[-1]
                    parts = rest.split("Team 1:")
                    if len(parts) >= 2:
                        team_0_name = parts[0].strip().rstrip(",").strip()
                        team_1_name = parts[1].strip()
                except Exception:
                    pass
            continue
        data_lines.append(line)

    if not data_lines:
        return {"actions": [], "hands": [], "team_0_name": None, "team_1_name": None}

    reader = csv.DictReader(data_lines)
    for row in reader:
            hand_num = row.get("hand_number", "")
            try:
                hand_num = int(hand_num)
            except (ValueError, TypeError):
                continue
            for key in ("team_0_bankroll", "team_1_bankroll", "team_0_bet", "team_1_bet",
                        "action_amount", "action_keep_1", "action_keep_2"):
                if key in row and row[key] != "":
                    try:
                        row[key] = int(row[key])
                    except (ValueError, TypeError):
                        pass
            for key in ("team_0_cards", "team_1_cards", "board_cards", "team_0_discarded", "team_1_discarded"):
                if key in row:
                    row[key] = _safe_literal_eval(row[key])
            row["hand_number"] = hand_num
            actions.append(row)

    # Group by hand
    hands_by_num: dict[int, list[dict]] = {}
    for a in actions:
        h = a["hand_number"]
        if h not in hands_by_num:
            hands_by_num[h] = []
        hands_by_num[h].append(a)

    hand_nums = sorted(hands_by_num.keys())
    hands = []
    for i, h in enumerate(hand_nums):
        hand_actions = hands_by_num[h]
        # Bankroll after hand N = first row of hand N+1
        bankroll_after = None
        if i + 1 < len(hand_nums):
            next_h = hand_nums[i + 1]
            first_next = hands_by_num[next_h][0]
            bankroll_after = {
                "team_0_bankroll": first_next.get("team_0_bankroll"),
                "team_1_bankroll": first_next.get("team_1_bankroll"),
            }
        hands.append({
            "hand_number": h,
            "actions": hand_actions,
            "bankroll_after": bankroll_after,
        })

    return {
        "actions": actions,
        "hands": hands,
        "team_0_name": team_0_name,
        "team_1_name": team_1_name,
    }
