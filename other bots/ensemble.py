"""
EnsembleAgent — shared-context bot switcher.

All sub-bots observe every action throughout the match, so each maintains a
full opponent model regardless of which bot is currently active.  Switching
only changes which bot's act() is called; the incoming bot already has
complete context.

Switching modes
---------------
1. Manual   — POST /switch_bot  {"bot_name": "delta"}
2. Scheduled — pass schedule={0: "genesis", 300: "delta"} to constructor,
               or POST /schedule {"schedule": {"0": "genesis", "300": "delta"}}

Running
-------
    python -m ensemble          # port 8000 (default)
    python -m ensemble 9000     # custom port
"""

import importlib
import sys
import os

# Ensure support/ and this directory are on the path (same as genesis.py etc.)
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# Add agents/ directory so we can import the base Agent
_AGENTS_DIR = os.path.join(os.path.dirname(_DIR), "agents")
if _AGENTS_DIR not in sys.path:
    sys.path.insert(0, _AGENTS_DIR)

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from fastapi import HTTPException

from agent import Agent

# ---------------------------------------------------------------------------
# Sub-bot imports (each file uses PlayerAgent as the class name)
# ---------------------------------------------------------------------------
from genesis import GenesisAgent
from blambot import PlayerAgent as BlamBotAgent
from delta import PlayerAgent as DeltaAgent
from ALPHANiT import PlayerAgent as ALPHANiTAgent

# OmicronV1.2 has a dot in the filename — use importlib
_omicron_mod = importlib.import_module("OmicronV1.2")
OmicronAgent = _omicron_mod.PlayerAgent

# ---------------------------------------------------------------------------
# Registry: name -> class
# ---------------------------------------------------------------------------
BOT_REGISTRY: Dict[str, type] = {
    "genesis": GenesisAgent,
    "blambot": BlamBotAgent,
    "delta": DeltaAgent,
    "alphanit": ALPHANiTAgent,
    "omicron": OmicronAgent,
}

# Default hand-number schedule: {hand_number: bot_name}
# Customize this or override via /schedule endpoint.
DEFAULT_SCHEDULE: Dict[int, str] = {
    0: "genesis",
}


# ---------------------------------------------------------------------------
# Pydantic request models for the extra endpoints
# ---------------------------------------------------------------------------
class SwitchBotRequest(BaseModel):
    bot_name: str


class ScheduleRequest(BaseModel):
    schedule: Dict[str, str]  # hand_number (as str) -> bot_name


class ActiveBotResponse(BaseModel):
    active_bot: str
    available_bots: List[str]
    schedule: Dict[int, str]
    hands_seen: int


# ---------------------------------------------------------------------------
# EnsembleAgent
# ---------------------------------------------------------------------------
class EnsembleAgent(Agent):
    def __init__(self, stream: bool = False, schedule: Optional[Dict[int, str]] = None):
        # Instantiate all sub-bots with stream=False so they don't start servers.
        # They still build their own FastAPI apps, but those are never served.
        self.bots: Dict[str, Agent] = {
            name: cls(stream=False) for name, cls in BOT_REGISTRY.items()
        }

        # Sorted schedule thresholds (ascending)
        raw_schedule = schedule if schedule is not None else dict(DEFAULT_SCHEDULE)
        self.schedule: Dict[int, str] = {int(k): v for k, v in raw_schedule.items()}
        self._schedule_keys: List[int] = sorted(self.schedule.keys())

        self.active_bot_name: str = self.schedule.get(0, next(iter(BOT_REGISTRY)))
        self._hands_seen: int = 0
        self._last_scheduled_hand: int = -1

        # Super().__init__ creates self.app, self.logger, and calls add_routes()
        super().__init__(stream=stream)

    def __name__(self) -> str:
        return f"EnsembleAgent[{self.active_bot_name}]"

    # ------------------------------------------------------------------
    # Core act / observe
    # ------------------------------------------------------------------

    def act(self, observation, reward, terminated, truncated, info):
        hand_number = (info or {}).get("hand_number", self._hands_seen)
        self._apply_schedule(hand_number)
        self._hands_seen = hand_number
        return self.bots[self.active_bot_name].act(observation, reward, terminated, truncated, info)

    def observe(self, observation, reward, terminated, truncated, info) -> None:
        # ALL bots observe — this is what keeps every bot's opponent model current.
        for bot in self.bots.values():
            bot.observe(observation, reward, terminated, truncated, info)

    # ------------------------------------------------------------------
    # Schedule logic
    # ------------------------------------------------------------------

    def _apply_schedule(self, hand_number: int) -> None:
        """Switch to the bot mapped to the highest threshold <= hand_number."""
        target_bot = None
        for threshold in self._schedule_keys:
            if hand_number >= threshold:
                target_bot = self.schedule[threshold]
            else:
                break
        if target_bot and target_bot != self.active_bot_name:
            self._switch_to(target_bot, reason=f"schedule at hand {hand_number}")

    def _switch_to(self, bot_name: str, reason: str = "manual") -> None:
        if bot_name not in self.bots:
            raise ValueError(f"Unknown bot: {bot_name!r}. Available: {list(self.bots)}")
        prev = self.active_bot_name
        self.active_bot_name = bot_name
        self.logger.info(f"[EnsembleAgent] Switched {prev} -> {bot_name} ({reason})")

    # ------------------------------------------------------------------
    # Extra FastAPI routes
    # ------------------------------------------------------------------

    def add_routes(self):
        super().add_routes()  # keeps /get_action and /post_observation

        @self.app.post("/switch_bot")
        def switch_bot(request: SwitchBotRequest):
            if request.bot_name not in self.bots:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown bot {request.bot_name!r}. Available: {list(self.bots)}",
                )
            self._switch_to(request.bot_name, reason="POST /switch_bot")
            return {"active_bot": self.active_bot_name}

        @self.app.get("/active_bot", response_model=ActiveBotResponse)
        def active_bot():
            return ActiveBotResponse(
                active_bot=self.active_bot_name,
                available_bots=list(self.bots.keys()),
                schedule=self.schedule,
                hands_seen=self._hands_seen,
            )

        @self.app.post("/schedule")
        def set_schedule(request: ScheduleRequest):
            try:
                new_schedule = {int(k): v for k, v in request.schedule.items()}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid schedule keys: {e}")
            unknown = [v for v in new_schedule.values() if v not in self.bots]
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown bot names in schedule: {unknown}. Available: {list(self.bots)}",
                )
            self.schedule = new_schedule
            self._schedule_keys = sorted(new_schedule.keys())
            self.logger.info(f"[EnsembleAgent] Schedule updated: {self.schedule}")
            return {"schedule": self.schedule}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    EnsembleAgent.run(stream=True, port=port)
