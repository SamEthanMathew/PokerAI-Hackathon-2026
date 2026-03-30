"""
Parse match runner log (e.g. match_25757.txt).
Hand number: K is logged after hand K completes; bankrolls are after hand K.
"""
import re
from typing import Any


def parse_match_runner_log(path: str) -> dict[str, Any]:
    """
    Parse match runner log file.
    Returns:
        bankroll_checkpoints: list of {hand_number, team_0_bankroll, team_1_bankroll}
        final_team_0_bankroll, final_team_1_bankroll (or None)
        time_used_0, time_used_1 (seconds, or None)
        team_0_name, team_1_name (from first checkpoint line, or None)
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    checkpoints = []
    final_team_0 = final_team_1 = None
    time_0 = time_1 = None
    team_0_name = team_1_name = None

    # Hand number: 50, WW bankroll: -432, Ctrl+Alt+Defeat bankroll: 432
    checkpoint_re = re.compile(
        r"Hand number:\s*(\d+),\s*(.+?)\s+bankroll:\s*(-?\d+),\s*(.+?)\s+bankroll:\s*(-?\d+)"
    )
    # Final results - WW bankroll: -52, Ctrl+Alt+Defeat bankroll: 52
    final_re = re.compile(
        r"Final results\s*-\s*(.+?)\s+bankroll:\s*(-?\d+),\s*(.+?)\s+bankroll:\s*(-?\d+)"
    )
    # Time used - WW: 83.41 seconds, Ctrl+Alt+Defeat: 580.24 seconds
    time_re = re.compile(
        r"Time used\s*-\s*(.+?):\s*([\d.]+)\s*seconds,\s*(.+?):\s*([\d.]+)\s*seconds"
    )

    for line in lines:
        if "Hand number:" in line and "bankroll:" in line:
            m = checkpoint_re.search(line)
            if m:
                hand_num = int(m.group(1))
                t0_name = m.group(2).strip()
                t0_br = int(m.group(3))
                t1_name = m.group(4).strip()
                t1_br = int(m.group(5))
                if team_0_name is None:
                    team_0_name, team_1_name = t0_name, t1_name
                checkpoints.append({
                    "hand_number": hand_num,
                    "team_0_bankroll": t0_br,
                    "team_1_bankroll": t1_br,
                })
        if "Final results" in line:
            m = final_re.search(line)
            if m:
                final_team_0 = int(m.group(2))
                final_team_1 = int(m.group(4))
        if "Time used" in line and "seconds" in line:
            m = time_re.search(line)
            if m:
                time_0 = float(m.group(2))
                time_1 = float(m.group(4))

    return {
        "bankroll_checkpoints": checkpoints,
        "final_team_0_bankroll": final_team_0,
        "final_team_1_bankroll": final_team_1,
        "time_used_0": time_0,
        "time_used_1": time_1,
        "team_0_name": team_0_name,
        "team_1_name": team_1_name,
    }
