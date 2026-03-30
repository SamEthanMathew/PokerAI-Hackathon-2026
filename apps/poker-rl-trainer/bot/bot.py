"""
Tournament submission bot for the CMU Poker Bot Competition 2026.

Runs as a FastAPI server (inherits from Agent base class).
Uses neural net for fast decisions; real-time Monte Carlo for critical
(large pot / all-in) decisions; position-aware bet sizing (SB/BB).

Directory layout (this file lives in bot/):
    bot/
    ├── bot.py                  ← this file
    ├── realtime_search.py      ← MC search + NN override for big decisions
    ├── model.py                ← copy of poker-rl-trainer/model.py
    ├── features.py             ← copy of poker-rl-trainer/features.py
    ├── poker_final.pt          ← trained weights (no value_head)
    ├── genesis_knowledge.json  ← evolved strategy knowledge
    └── tables/
        ├── equity_vs_random.npy
        ├── optimal_discards.npy
        └── opponent_ranges.npy

Launch:
    python bot.py [port]      # default port 8000
"""

import json
import os
import sys
from collections import defaultdict
from typing import Tuple

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_BOT_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_BOT_DIR, "../../.."))

# Add bot/ to path so model.py and features.py are importable
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

# Add repo root so agents/agent.py and gym_env are importable
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_AGENTS_DIR = os.path.join(_REPO_ROOT, "agents")
if _AGENTS_DIR not in sys.path:
    sys.path.insert(0, _AGENTS_DIR)

from agent import Agent  # base Agent class with FastAPI server
from model import PokerNetV2
from features import (
    extract_features,
    split_features,
    update_opp_stats,
    CARD_DIM,
    CONTEXT_DIM,
)
from realtime_search import (
    RealtimeSearch,
    action_type_to_str,
    str_to_action_type,
)
from gym_env import PokerEnv, WrappedEval

# Optional: import table loader from co-located precompute (or inline load)
try:
    sys.path.insert(0, os.path.join(_BOT_DIR, ".."))
    from precompute import load_tables
except ImportError:
    def load_tables(tables_dir: str) -> dict:
        import numpy as np
        tables = {"equity_vs_random": None, "optimal_discards": None, "opponent_ranges": None}
        ep = os.path.join(tables_dir, "equity_vs_random.npy")
        dp = os.path.join(tables_dir, "optimal_discards.npy")
        rp = os.path.join(tables_dir, "opponent_ranges.npy")
        if os.path.exists(ep):
            tables["equity_vs_random"] = np.load(ep)
        if os.path.exists(dp):
            tables["optimal_discards"] = np.load(dp, allow_pickle=True).item()
        if os.path.exists(rp):
            tables["opponent_ranges"] = np.load(rp, allow_pickle=True).item()
        return tables


# ── PlayerAgent ───────────────────────────────────────────────────────────────

class PlayerAgent(Agent):
    """
    Tournament bot.

    Loads the trained PokerNetV2 model (value_head stripped) and runs
    inference on CPU using the 277-dim feature extractor.
    """

    def __init__(self, stream: bool = True):
        # Load model (CPU only — tournament environment has no GPU)
        model_path = os.path.join(_BOT_DIR, "poker_final.pt")
        self._model = PokerNetV2(
            card_dim=CARD_DIM,
            context_dim=CONTEXT_DIM,
            hidden_dim=256,
            num_residual_blocks=3,
        )
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location="cpu")
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            missing, _ = self._model.load_state_dict(state, strict=False)
            if missing:
                print(f"[bot] Missing model keys: {missing}")
        else:
            print(f"[bot] WARNING: model not found at {model_path}. Using random weights.")

        self._model.eval()

        # Load precomputed equity tables (gracefully handles missing files)
        tables_dir = os.path.join(_BOT_DIR, "tables")
        self._tables = load_tables(tables_dir)

        # Load genesis knowledge
        genesis_path = os.path.join(_BOT_DIR, "genesis_knowledge.json")
        if os.path.exists(genesis_path):
            with open(genesis_path) as f:
                self._genesis = json.load(f)
        else:
            self._genesis = {}

        # Per-match live opponent model (reset between matches, accumulated per hand)
        self._opp_stats: dict = defaultdict(float)

        # Real-time Monte Carlo for critical decisions (large pots / all-in)
        self._realtime_search = RealtimeSearch(
            evaluator=WrappedEval(),
            equity_table=self._tables.get("equity_vs_random"),
            range_table=self._tables.get("opponent_ranges"),
            time_budget=1500.0,
            safety_buffer=100.0,
        )

        # Call Agent.__init__ last (it calls add_routes and starts the FastAPI app)
        super().__init__(stream=stream)

    def __name__(self) -> str:
        return "PokerNetV2Bot"

    # ── Core act() ────────────────────────────────────────────────────────────

    def act(
        self,
        observation: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> Tuple[int, int, int, int]:
        """
        Given the current game state, return (action_type, raise_amount, keep1, keep2).
        Uses NN for speed; real-time MC + override for critical decisions; position-aware sizing.
        """
        valid = list(observation["valid_actions"])
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        pot_size = observation.get("pot_size", observation["my_bet"] + observation["opp_bet"])
        my_bet = observation["my_bet"]
        blind_position = observation.get("blind_position", 0)  # 0 = SB, 1 = BB
        street = observation.get("street", 0)

        feat = extract_features(
            observation,
            self._opp_stats,
            self._genesis,
            self._tables,
        )
        card_feat, ctx_feat = split_features(feat)

        cf_t  = torch.from_numpy(card_feat).unsqueeze(0)
        ctx_t = torch.from_numpy(ctx_feat).unsqueeze(0)

        with torch.no_grad():
            action = self._model.predict_action(
                cf_t,
                ctx_t,
                valid,
                min_raise,
                max_raise,
            )

        # Discard phase: no MC override, no position scaling
        if valid[4]:  # DISCARD
            update_opp_stats(self._opp_stats, observation, action)
            return action

        action_type, raise_amt, k1, k2 = action
        my_stack = PokerEnv.MAX_PLAYER_BET - my_bet

        # Real-time MC for critical decisions (postflop only, 2 hole cards + board)
        my_cards = [c for c in observation["my_cards"] if c >= 0]
        community = [c for c in observation["community_cards"] if c >= 0]
        if len(my_cards) == 2 and len(community) >= 3:
            do_search, num_sims = self._realtime_search.should_search(pot_size, my_stack)
            if do_search and num_sims > 0:
                known_removed = list(observation.get("my_discarded_cards", [])) + list(
                    observation.get("opp_discarded_cards", [])
                )
                known_removed = [c for c in known_removed if c >= 0]
                opp_range = self._tables.get("opponent_ranges")  # may be None or different format
                mc_equity = self._realtime_search.monte_carlo_equity(
                    my_cards,
                    community,
                    known_removed,
                    opp_range if isinstance(opp_range, dict) else None,
                    num_sims=num_sims,
                )
                nn_action_str = action_type_to_str(action_type)
                overridden = self._realtime_search.override_neural_net(
                    nn_action_str, 0.5, mc_equity, pot_size
                )
                if overridden != nn_action_str:
                    action_type, raise_amt = str_to_action_type(overridden, min_raise, max_raise)
                    action = (action_type, raise_amt, 0, 0)

        # Position-aware bet sizing (Optimization 7)
        # SB: tighter preflop, smaller sizing OOP. BB: wider defense, larger sizing in position.
        if action_type == 1 and raise_amt > 0:
            if blind_position == 0:  # Small blind — smaller opens and c-bets
                factor = 0.75 if street == 0 else 0.85
                raise_amt = max(min_raise, int(raise_amt * factor))
            else:  # Big blind — can size up when leading (we have information)
                raise_amt = min(max_raise, int(raise_amt * 1.05))
            raise_amt = max(min_raise, min(max_raise, raise_amt))
            action = (action_type, raise_amt, 0, 0)

        # Update our action in opp_stats so action history features stay current
        update_opp_stats(self._opp_stats, observation, action)

        return action

    # ── observe() — called when opponent acts ─────────────────────────────────

    def observe(
        self,
        observation: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> None:
        """
        Update live opponent model with opponent's action.
        When a hand ends (terminated), decrement real-time search hand counter.
        """
        if terminated:
            self._realtime_search.consume_hand()

        opp_last = observation.get("opp_last_action", "")
        if opp_last:
            # Map string action to int for opp_stats
            _action_map = {"fold": 0, "raise": 1, "check": 2, "call": 3, "discard": 4}
            action_type = _action_map.get(opp_last.lower().split()[0], -1)
            if action_type >= 0:
                raise_amt = 0
                if action_type == 1:
                    try:
                        raise_amt = int(opp_last.split()[-1])
                    except (ValueError, IndexError):
                        raise_amt = observation.get("opp_bet", 0)
                opp_action = (action_type, raise_amt, 0, 0)
                update_opp_stats(self._opp_stats, observation, opp_action)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    PlayerAgent.run(stream=True, port=port)
