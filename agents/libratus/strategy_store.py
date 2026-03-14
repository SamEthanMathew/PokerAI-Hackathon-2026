"""
Strategy storage: serialize and load the blueprint strategy (and infoset actions)
so the agent can load a precomputed strategy without re-running CFR.
"""

import json
import os
from typing import Dict, List, Tuple, Any


def save_strategy(
    strategy: Dict[str, List[float]],
    infoset_actions: Dict[str, List[Tuple[int, int, int, int]]],
    path: str,
) -> None:
    """
    Save average strategy and action lists to a JSON file.
    Each key's value is a list of floats (strategy) or list of [action_type, raise_total, k1, k2] (actions).
    """
    # Convert infoset_actions to JSON-serializable form (tuple -> list)
    actions_ser = {k: [list(a) for a in v] for k, v in infoset_actions.items()}
    data = {
        "strategy": strategy,
        "infoset_actions": actions_ser,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=0)


def load_strategy(path: str) -> Tuple[Dict[str, List[float]], Dict[str, List[Tuple[int, int, int, int]]]]:
    """
    Load strategy and infoset_actions from JSON.
    Returns (strategy, infoset_actions) where infoset_actions values are lists of (int, int, int, int).
    """
    with open(path, "r") as f:
        data = json.load(f)
    strategy = data["strategy"]
    actions_ser = data["infoset_actions"]
    infoset_actions = {k: [tuple(a) for a in v] for k, v in actions_ser.items()}
    return strategy, infoset_actions
