"""
Submission entry point. PlayerAgent is GenesisV2Agent with entry_point='player.py'
so logs can be segregated by entry (grep HAND_RESULT | entry=player.py).
"""

import os

import uvicorn

from submission.genesisV2 import GenesisV2Agent


class PlayerAgent(GenesisV2Agent):
    """GenesisV2Agent exposed as PlayerAgent; logs use entry=player.py."""

    def __init__(self, stream: bool = True):
        super().__init__(stream=stream, entry_point="player.py")

    @classmethod
    def run(cls, stream: bool = False, port: int = 8000, host: str = "0.0.0.0", player_id: str = None):
        if player_id is not None:
            os.environ["PLAYER_ID"] = player_id
        bot = cls(stream)
        bot.logger.info(f"Starting agent server on {host}:{port}")
        uvicorn.run(
            bot.app,
            host=host,
            port=port,
            log_level="info",
            access_log=False,
            timeout_graceful_shutdown=0,
        )

