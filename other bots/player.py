"""
Submission entry point. PlayerAgent is GenesisAgent with entry_point='player.py'
so STREET0 logs can be segregated by entry (grep STREET0 | entry=player.py).
"""

import os
import sys

# Ensure "other bots" is on path so "genesis" and "support" are findable
_other_bots_root = os.path.abspath(os.path.dirname(__file__))
if _other_bots_root not in sys.path:
    sys.path.insert(0, _other_bots_root)

from genesis import GenesisAgent


class PlayerAgent(GenesisAgent):
    """GenesisAgent exposed as PlayerAgent; logs use entry=player.py for street 0."""

    def __init__(self, stream: bool = True):
        super().__init__(stream=stream, entry_point="player.py")

