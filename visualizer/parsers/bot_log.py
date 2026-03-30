"""
Parse bot log (genesis agent logs: STREET0, STREET1, STREET2, STREET3, OPP_RECON, HAND_RESULT).
Pipe-separated key=value format.
"""
import re
from typing import Any


def _parse_pipe_kv(line: str) -> dict[str, Any]:
    """Parse 'PREFIX | key=value | key2=value2' into dict. Values stay strings unless numeric."""
    out = {}
    parts = [p.strip() for p in line.split("|")]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k, v = k.strip(), v.strip()
        if v in ("True", "False"):
            out[k] = v == "True"
        elif re.match(r"^-?\d+$", v):
            out[k] = int(v)
        elif re.match(r"^-?[\d.]+$", v):
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        else:
            # Remove surrounding quotes if present
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            out[k] = v
    return out


def parse_bot_log(path: str) -> dict[str, Any]:
    """
    Parse bot log file.
    Returns:
        hand_results: list of dicts (one per hand, from HAND_RESULT lines), keyed by hand for easy lookup
        opp_recon: list of dicts (one per hand, from OPP_RECON lines)
        street0, street1, street2, street3: lists of decision lines (optional drill-down)
    """
    hand_results = []  # list in order; also build hand -> result map
    hand_result_by_hand: dict[int, dict] = {}
    opp_recon = []
    opp_recon_by_hand: dict[int, dict] = {}
    street0, street1, street2, street3 = [], [], [], []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if " - INFO - " in line:
                msg = line.split(" - INFO - ", 1)[-1].strip()
            else:
                msg = line.strip()
            if not msg:
                continue

            if "HAND_RESULT" in msg:
                d = _parse_pipe_kv(msg)
                h = d.get("hand")
                if h is not None:
                    hand_results.append(d)
                    hand_result_by_hand[int(h)] = d
            elif "OPP_RECON" in msg:
                d = _parse_pipe_kv(msg)
                h = d.get("hand")
                if h is not None:
                    opp_recon.append(d)
                    opp_recon_by_hand[int(h)] = d
            elif "STREET0" in msg and "street=0" in msg:
                street0.append(_parse_pipe_kv(msg))
            elif "STREET1" in msg:
                street1.append(_parse_pipe_kv(msg))
            elif "STREET2" in msg:
                street2.append(_parse_pipe_kv(msg))
            elif "STREET3" in msg:
                street3.append(_parse_pipe_kv(msg))

    return {
        "hand_results": hand_results,
        "hand_result_by_hand": hand_result_by_hand,
        "opp_recon": opp_recon,
        "opp_recon_by_hand": opp_recon_by_hand,
        "street0": street0,
        "street1": street1,
        "street2": street2,
        "street3": street3,
    }
