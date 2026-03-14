"""
Optional real-time subgame solving (Phase 2).

When the opponent uses an off-tree bet size or we reach turn/river,
we can solve a finer-grained subgame with blueprint values as alternate payoffs.
This module provides the interface; a full implementation would:
- Build an augmented subgame (opponent can exit with blueprint value or play).
- Solve with CFR+ and return our strategy for the subgame.
- Be invoked from LibratusAgent when off-tree action or late street is detected.

For Phase 1, the agent uses blueprint lookup only; subgame solving is not wired.
"""

from typing import Dict, List, Optional

from .game_model import ActionRecord
from .abstraction import CardAbstraction
from .cfr import CFRGameState, MCCFR


def solve_subgame(
    state: CFRGameState,
    actions_so_far: List[ActionRecord],
    blueprint_values: Dict[str, float],
    card_abstraction: CardAbstraction,
    max_iterations: int = 500,
) -> Dict[str, List[float]]:
    """
    Solve a subgame from the given state with blueprint values as exit payoffs.
    Returns strategy for the subgame (our player only) keyed by infoset.

    Stub: runs a small number of CFR iterations from this state;
    a full implementation would build the augmented subgame with alternate payoffs.
    """
    mccfr = MCCFR(card_abstraction=card_abstraction)
    for _ in range(max_iterations):
        mccfr.run_iteration(state, actions_so_far, state.acting_player, 1.0, 1.0)
    return mccfr.get_average_strategy()
