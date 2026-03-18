"""
Opponent pool for RL training.

Loads all bots from the `other bots/` directory as in-process instances.
No HTTP overhead — act() and observe() are called directly.

Each rollout batch picks one opponent at random.
"""

import importlib
import os
import random
import sys
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_OTHER_BOTS_DIR = os.path.join(_REPO_ROOT, "other bots")


class InProcessOpponent:
    """
    Wraps a bot Agent instance for direct in-process use during training.

    The wrapped bot still builds its own FastAPI app on __init__, but that
    app is never served. We call act() and observe() directly.
    """

    def __init__(self, name: str, bot_instance):
        self.name = name
        self._bot = bot_instance

    def act(
        self,
        observation: dict,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info: Optional[dict] = None,
    ) -> Tuple[int, int, int, int]:
        """Returns (action_type, raise_amount, keep1, keep2)."""
        return self._bot.act(observation, reward, terminated, truncated, info or {})

    def observe(
        self,
        observation: dict,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info: Optional[dict] = None,
    ) -> None:
        """Called when it is NOT this bot's turn (opponent observation)."""
        self._bot.observe(observation, reward, terminated, truncated, info or {})

    def reset(self) -> None:
        """Reset per-hand state if the bot supports it."""
        if hasattr(self._bot, "reset"):
            self._bot.reset()

    def __repr__(self) -> str:
        return f"InProcessOpponent({self.name!r})"


def _import_bot(module_name: str, class_name: str, bots_dir: str):
    """Dynamically import a bot class from bots_dir."""
    if bots_dir not in sys.path:
        sys.path.insert(0, bots_dir)

    # Support filenames that aren't valid Python identifiers (e.g. OmicronV1.2)
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        # Try loading by file path
        file_path = os.path.join(bots_dir, f"{module_name}.py")
        if not os.path.exists(file_path):
            return None
        spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name, None)


def load_opponent_pool(other_bots_dir: str = _OTHER_BOTS_DIR) -> List[InProcessOpponent]:
    """
    Load all available opponent bots from other_bots_dir.

    Bots are instantiated with stream=False so they don't spam stdout and
    don't attempt to serve their FastAPI apps.

    Returns a list of InProcessOpponent wrappers. Order is deterministic.
    """
    if not os.path.isdir(other_bots_dir):
        raise FileNotFoundError(f"other_bots_dir not found: {other_bots_dir}")

    if other_bots_dir not in sys.path:
        sys.path.insert(0, other_bots_dir)

    pool: List[InProcessOpponent] = []

    # ── Genesis (class name is GenesisAgent, not PlayerAgent) ──────────────
    try:
        GenesisAgent = _import_bot("genesis", "GenesisAgent", other_bots_dir)
        if GenesisAgent is not None:
            pool.append(InProcessOpponent("genesis", GenesisAgent(stream=False)))
            print(f"  [opponent_pool] Loaded: genesis")
    except Exception as e:
        print(f"  [opponent_pool] WARNING: could not load genesis: {e}")

    # ── BlamBot ────────────────────────────────────────────────────────────
    try:
        BlamBotAgent = _import_bot("blambot", "PlayerAgent", other_bots_dir)
        if BlamBotAgent is not None:
            pool.append(InProcessOpponent("blambot", BlamBotAgent(stream=False)))
            print(f"  [opponent_pool] Loaded: blambot")
    except Exception as e:
        print(f"  [opponent_pool] WARNING: could not load blambot: {e}")

    # ── Delta ──────────────────────────────────────────────────────────────
    try:
        DeltaAgent = _import_bot("delta", "PlayerAgent", other_bots_dir)
        if DeltaAgent is not None:
            pool.append(InProcessOpponent("delta", DeltaAgent(stream=False)))
            print(f"  [opponent_pool] Loaded: delta")
    except Exception as e:
        print(f"  [opponent_pool] WARNING: could not load delta: {e}")

    # ── ALPHANiT ───────────────────────────────────────────────────────────
    try:
        ALPHANiTAgent = _import_bot("ALPHANiT", "PlayerAgent", other_bots_dir)
        if ALPHANiTAgent is not None:
            pool.append(InProcessOpponent("alphanit", ALPHANiTAgent(stream=False)))
            print(f"  [opponent_pool] Loaded: alphanit")
    except Exception as e:
        print(f"  [opponent_pool] WARNING: could not load ALPHANiT: {e}")

    # ── OmicronV1.2 (filename has a dot) ───────────────────────────────────
    try:
        omicron_path = os.path.join(other_bots_dir, "OmicronV1.2.py")
        spec = importlib.util.spec_from_file_location("OmicronV1_2", omicron_path)
        if spec is not None:
            omicron_mod = importlib.util.module_from_spec(spec)
            sys.modules["OmicronV1_2"] = omicron_mod
            spec.loader.exec_module(omicron_mod)
            OmicronAgent = omicron_mod.PlayerAgent
            pool.append(InProcessOpponent("omicron", OmicronAgent(stream=False)))
            print(f"  [opponent_pool] Loaded: omicron")
    except Exception as e:
        print(f"  [opponent_pool] WARNING: could not load OmicronV1.2: {e}")

    if not pool:
        raise RuntimeError(
            f"No opponent bots could be loaded from {other_bots_dir}. "
            "Check that the other bots/ directory is correct and bots are importable."
        )

    print(f"  [opponent_pool] Total opponents loaded: {len(pool)}")
    return pool


class OpponentPool:
    """
    Manages a list of opponents and selects one uniformly at random per rollout.
    """

    def __init__(self, opponents: List[InProcessOpponent]):
        if not opponents:
            raise ValueError("Opponent pool must contain at least one opponent.")
        self.opponents = opponents

    def sample(self) -> InProcessOpponent:
        """Return one opponent chosen uniformly at random."""
        return random.choice(self.opponents)

    def get(self, name: str) -> Optional[InProcessOpponent]:
        """Return opponent by name, or None."""
        for opp in self.opponents:
            if opp.name == name:
                return opp
        return None

    def names(self) -> List[str]:
        return [o.name for o in self.opponents]

    def __len__(self) -> int:
        return len(self.opponents)

    def __repr__(self) -> str:
        return f"OpponentPool({self.names()})"
