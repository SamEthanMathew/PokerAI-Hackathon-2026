"""
Submission entry point. PlayerAgent is GenesisAgent with entry_point='player.py'
so STREET0 logs can be segregated by entry (grep STREET0 | entry=player.py).
"""

from submission.genesis import GenesisAgent


class PlayerAgent(GenesisAgent):
    """GenesisAgent exposed as PlayerAgent; logs use entry=player.py for street 0."""

    def __init__(self, stream: bool = True):
        super().__init__(stream=stream, entry_point="player.py")

