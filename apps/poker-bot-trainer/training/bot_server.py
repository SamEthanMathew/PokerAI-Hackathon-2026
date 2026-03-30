"""
Bot server — serves genesisV2 (or a trained clone) to the Next.js UI.

Run from the poker-bot-trainer directory:
    python training/bot_server.py

Or from the training directory:
    python bot_server.py

Endpoints:
    POST /act           — get bot action for the current game state
    GET  /health        — health check, returns generation info
    POST /reset         — reset per-session state (new GenesisV2Agent instance)
    POST /hot_swap      — swap to a new trained model checkpoint

The server translates the TypeScript EngineState JSON into the gym_env
observation dict format that GenesisV2Agent expects, then calls act().
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

# ── Path setup: make repo root importable ────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # apps/poker-bot-trainer/training/
_TRAINER_ROOT = _HERE.parent                      # apps/poker-bot-trainer/
_REPO_ROOT = _TRAINER_ROOT.parent.parent          # poker-engine-2026/

for p in [str(_REPO_ROOT), str(_TRAINER_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from submission.genesisV2 import GenesisV2Agent

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_RANKS = 9
KEEP_COMBOS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
FEATURE_DIM = 98
STRAIGHT_WINDOWS = [
    [8, 0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8],
]
LINEAGE_PATH = _HERE / "lineage.json"
MODELS_DIR = _HERE / "models"

# ── Lineage ───────────────────────────────────────────────────────────────────
def load_lineage() -> dict:
    if LINEAGE_PATH.exists():
        return json.loads(LINEAGE_PATH.read_text())
    return {"current_generation": 0, "models": [{"gen": 0, "type": "genesis"}]}

# ── Clone model (mirrors model.py) ───────────────────────────────────────────
class PokerCloneNet(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(),
        )
        self.action_head = nn.Sequential(nn.Linear(hidden_dim // 2, 64), nn.GELU(), nn.Linear(64, 4))
        self.raise_head = nn.Sequential(nn.Linear(hidden_dim // 2, 64), nn.GELU(), nn.Linear(64, 10))
        self.discard_head = nn.Sequential(nn.Linear(hidden_dim // 2, 64), nn.GELU(), nn.Linear(64, 10))

    def forward(self, x):
        s = self.backbone(x)
        return self.action_head(s), self.raise_head(s), self.discard_head(s)


# ── Feature extraction from gym obs ──────────────────────────────────────────
from collections import Counter

def _cr(c): return c % NUM_RANKS
def _cs(c): return c // NUM_RANKS

def _eval5(cards):
    ranks = [_cr(c) for c in cards]; suits = [_cs(c) for c in cards]
    rc = Counter(ranks); counts = sorted(rc.values(), reverse=True)
    is_f = len(set(suits)) == 1
    rs = set(ranks); is_s, si = False, -1
    for i in range(len(STRAIGHT_WINDOWS)-1, -1, -1):
        if all(r in rs for r in STRAIGHT_WINDOWS[i]): is_s, si = True, i; break
    if is_f and is_s: return 8, si/(len(STRAIGHT_WINDOWS)-1)
    if counts[0]==3 and len(counts)>1 and counts[1]==2:
        t=[r for r,c in rc.items() if c==3][0]; p=[r for r,c in rc.items() if c==2][0]
        return 7,(t*NUM_RANKS+p)/(NUM_RANKS**2)
    if is_f:
        sr=sorted(ranks,reverse=True); return 6, sum(r/NUM_RANKS**(i+1) for i,r in enumerate(sr))/NUM_RANKS
    if counts[0]==3: return 5,[r for r,c in rc.items() if c==3][0]/(NUM_RANKS-1)
    if is_s: return 4, si/(len(STRAIGHT_WINDOWS)-1)
    if counts[0]==2 and len(counts)>1 and counts[1]==2:
        ps=sorted([r for r,c in rc.items() if c==2],reverse=True)
        k=[r for r,c in rc.items() if c==1][0]
        return 3,(ps[0]*NUM_RANKS**2+ps[1]*NUM_RANKS+k)/(NUM_RANKS**3)
    if counts[0]==2:
        p=[r for r,c in rc.items() if c==2][0]; ks=sorted([r for r,c in rc.items() if c==1],reverse=True)
        return 2,(p*NUM_RANKS**2+ks[0]*NUM_RANKS+(ks[1] if len(ks)>1 else 0))/(NUM_RANKS**3)
    sr=sorted(ranks,reverse=True); return 1, sum(r/NUM_RANKS**(i+1) for i,r in enumerate(sr))/NUM_RANKS

def _eval_best(hole, comm):
    all_c=[c for c in (hole+comm) if c>=0]
    if len(all_c)<5: return 1, 0.0
    best=(0,0.0)
    for combo in combinations(all_c,5):
        r,tb=_eval5(list(combo))
        if r>best[0] or (r==best[0] and tb>best[1]): best=(r,tb)
    return best

ACTION_TYPE_MAP = {"fold": 0, "raise": 1, "check": 2, "call": 3, "discard": 4}

def _get_nn_opp_features() -> list[float]:
    """Return the 10 opponent-tendency context features from the current exploit profile."""
    try:
        nn = _load_exploit_profile().get("nn_opponent_features", {})
        return [
            nn.get("vpip", 0.5),
            nn.get("fold_to_flop_bet", 0.1),
            nn.get("fold_to_turn_bet", 0.1),
            nn.get("fold_to_river_bet", 0.1),
            nn.get("river_trap_winrate", 0.5),
            nn.get("river_raise_ratio", 0.5),
            nn.get("showdown_pct", 0.3),
            nn.get("three_bet_pct", 0.0),
            nn.get("check_raise_pct", 0.0),
            nn.get("hands_seen_norm", 0.0),
        ]
    except Exception:
        return [0.5, 0.1, 0.1, 0.1, 0.5, 0.5, 0.3, 0.0, 0.0, 0.0]


def extract_features_from_gym_obs(obs: dict, action_history: list = None) -> np.ndarray:
    """Extract 108-dim features from a gym_env-style observation dict."""
    feat = np.zeros(108, dtype=np.float32)
    my_cards = [c for c in obs.get("my_cards", []) if c >= 0]
    comm = [c for c in obs.get("community_cards", []) if c >= 0]
    opp_disc = [c for c in obs.get("opp_discarded_cards", []) if c >= 0]
    position = "SB" if obs.get("blind_position", 0) == 0 else "BB"
    street = obs.get("street", 0)

    for c in my_cards:
        if 0 <= c < 27: feat[c] = 1
    for c in comm:
        if 0 <= c < 27: feat[27+c] = 1

    if street == 0: feat[54] = 1
    elif street <= 2: feat[55] = 1
    else: feat[56] = 1
    feat[57] = 1 if position == "SB" else 0

    all_k = my_cards + comm
    if len(my_cards) >= 2 and len(comm) >= 3:
        hr, tb = _eval_best(my_cards, comm); feat[58]=(hr-1)/7; feat[64]=tb
    sc = Counter(_cs(c) for c in all_k if c>=0)
    feat[59] = 1 if max(sc.values(), default=0) >= 4 else 0
    rs2 = set(_cr(c) for c in all_k if c>=0)
    feat[60] = 1 if any(sum(r in rs2 for r in w)>=4 for w in STRAIGHT_WINDOWS) else 0
    rc2 = Counter(_cr(c) for c in my_cards if c>=0)
    feat[61] = min(sum(1 for v in rc2.values() if v>=2), 2) / 2
    feat[62] = 1 if any(_cr(c)==8 for c in my_cards) else 0
    feat[63] = len(my_cards) / 5

    my_bet = obs.get("my_bet", 0); opp_bet = obs.get("opp_bet", 0)
    pot = obs.get("pot_size", 0); to_call = max(0, opp_bet - my_bet); ppc = pot + to_call
    feat[65]=pot/200; feat[66]=my_bet/100; feat[67]=opp_bet/100
    feat[68]=to_call/100; feat[69]=to_call/ppc if ppc>0 else 0
    feat[70]=obs.get("min_raise",2)/100; feat[71]=obs.get("max_raise",98)/100
    feat[72]=(100-my_bet)/100

    history = action_history or []
    for i in range(10):
        if i < len(history):
            entry = history[-(i+1)]
            feat[73+i] = ACTION_TYPE_MAP.get(entry.get("type",""), 0) / 4
        else:
            feat[73+i] = -0.25

    for i, c in enumerate(opp_disc[:10]):
        feat[83+i] = c/26 if c>=0 else 0

    feat[93]=0; feat[94]=0; feat[95]=0; feat[96]=0.5; feat[97]=0

    # [98-107] opponent tendency context — injected from exploit_profile.json
    for i, v in enumerate(_get_nn_opp_features()):
        feat[98 + i] = v

    return feat


# ── CloneBotAdapter ──────────────────────────────────────────────────────────
class CloneBotAdapter:
    """Wraps PokerCloneNet to look like a GenesisV2Agent from the server's POV."""

    def __init__(self, model_path: str, hidden_dim: int = 256):
        self.model = PokerCloneNet(hidden_dim=hidden_dim)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.label = Path(model_path).stem

    def act(self, observation, reward, terminated, truncated, info, action_history=None):
        feat = extract_features_from_gym_obs(observation, action_history)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        valid = observation.get("valid_actions", [1,1,1,1,1])
        with torch.no_grad():
            a_logits, r_logits, d_logits = self.model(x)
        if valid[4]:
            combo_idx = d_logits[0].argmax().item()
            k1, k2 = KEEP_COMBOS[combo_idx]
            return (4, 0, k1, k2)
        probs = torch.softmax(a_logits[0], dim=0).clone()
        for i in range(4):
            if not valid[i]: probs[i] = 0.0
        if probs.sum() == 0: return (0, 0, 0, 1)
        action = probs.argmax().item()
        raise_amt = 0
        if action == 1:
            bucket = r_logits[0].argmax().item()
            min_r = observation.get("min_raise", 2)
            max_r = observation.get("max_raise", 98)
            raise_amt = max(min_r, min(int(min_r + (bucket/9)*(max_r-min_r)), max_r))
        return (action, raise_amt, 0, 1)


# ── Exploit layer ─────────────────────────────────────────────────────────────
EXPLOIT_PROFILE_PATH = _HERE / "exploit_profile.json"

def _load_exploit_profile() -> dict:
    if EXPLOIT_PROFILE_PATH.exists():
        try:
            return json.loads(EXPLOIT_PROFILE_PATH.read_text())
        except Exception:
            pass
    return {"exploit_params": {}}


class ExploitLayer:
    """
    Wraps any agent and overrides its decisions with exploitative adjustments
    derived from the opponent profile built by exploit_profile.py.

    Exploit rules (data-driven from observed human tendencies):

    1. RIVER — always bet: human folds to river bets <15% of the time.
       Convert bot CHECK on river to a RAISE sized at river_bet_size_ratio * pot.

    2. PREFLOP — always raise: human calls preflop ~99% and never re-raises.
       Convert bot CALL/CHECK on preflop to a RAISE of preflop_raise_size_bb * BB.

    3. FLOP/TURN bluff suppression: human folds <5% flop, <10% turn.
       Convert thin bot RAISE on flop to CHECK (if raise_amount < 0.6 * pot).

    4. RIVER TRAP detection: when human checks turn then raises river, they
       have a strong hand 77%+ of the time. If bot holds a marginal position
       and faces a large river raise after checking the turn, fold.
    """

    def __init__(self, inner_agent: Any, profile: dict | None = None):
        self.inner = inner_agent
        self.profile = profile or _load_exploit_profile()
        self.ep = self.profile.get("exploit_params", {})

    def reload_profile(self) -> None:
        self.profile = _load_exploit_profile()
        self.ep = self.profile.get("exploit_params", {})

    def act(self, observation: dict, reward: float, terminated: bool, truncated: bool, info: dict, action_history: list | None = None):
        # Get base decision from wrapped agent
        if isinstance(self.inner, CloneBotAdapter):
            result = self.inner.act(observation, reward, terminated, truncated, info, action_history)
        else:
            result = self.inner.act(observation, reward, terminated, truncated, info)

        action_type, raise_amt, k1, k2 = result
        street = observation.get("street", 0)
        valid = observation.get("valid_actions", [1, 1, 1, 1, 1])
        pot = observation.get("pot_size", 4)
        my_bet = observation.get("my_bet", 0)
        opp_bet = observation.get("opp_bet", 0)
        min_raise = observation.get("min_raise", 2)
        max_raise = observation.get("max_raise", 98)
        opp_last = observation.get("opp_last_action", "")
        history = action_history or []

        # ── 1. River: always bet (convert CHECK → RAISE) ─────────────────────
        if (street == 3 and action_type == 2  # CHECK on river
                and valid[1]                   # RAISE is valid
                and self.ep.get("river_bet_always", False)):
            ratio = self.ep.get("river_bet_size_ratio", 0.75)
            target = max(min_raise, min(int(pot * ratio), max_raise))
            return (1, target, 0, 1)

        # ── 2. Preflop: always raise (convert CALL → RAISE) ──────────────────
        if (street == 0 and action_type == 3   # CALL on preflop
                and valid[1]                    # RAISE is valid
                and self.ep.get("preflop_always_raise", False)):
            bb_mult = self.ep.get("preflop_raise_size_bb", 4)
            target = max(min_raise, min(bb_mult * 2, max_raise))  # 2 = BIG_BLIND
            return (1, target, 0, 1)

        # ── 3. River trap: fold marginal hand when human check-raise traps ────
        if (street == 3 and action_type == 3   # bot would CALL
                and opp_last == "raise"         # human just raised
                and self.ep.get("detect_river_trap", False)):
            # Check if human checked the turn (trap pattern)
            human_checked_turn = any(
                e.get("player") == "human" and e.get("type") == "check" and e.get("street") == 2
                for e in history
            )
            if human_checked_turn:
                # Human raise after turn check = strong hand 77%+ → fold unless we have a monster
                # We use raise size as a signal: if opp raised large (>0.7x pot), fold
                opp_raise_amt = opp_bet - my_bet
                if pot > 0 and opp_raise_amt / pot > 0.5 and valid[0]:
                    return (0, 0, 0, 1)  # FOLD

        # ── 4. Flop thin-raise suppression: don't bluff if human never folds ─
        if (street == 1 and action_type == 1   # RAISE on flop
                and self.ep.get("suppress_flop_bluff", False)
                and valid[2]):                  # CHECK is available
            # Suppress only small raises relative to pot (likely bluffs)
            if pot > 0 and raise_amt / pot < 0.6:
                return (2, 0, 0, 1)  # CHECK instead

        # ── 5. C-bet suppression: don't continuation-bet if human never folds ─
        if (street == 1 and action_type == 1   # RAISE on flop (c-bet)
                and self.ep.get("suppress_cbet", False)
                and valid[2]):
            # Only suppress if we don't have a strong made hand (rely on hand strength feature)
            # For GenesisV2 inner agent: trust it knows when to c-bet for value
            pass  # ExploitLayer doesn't have hand strength here; leave to future enhancement

        return (action_type, raise_amt, k1, k2)


# ── Global server state ───────────────────────────────────────────────────────
_lock = threading.Lock()
_lineage = load_lineage()
_base_agent: Any = GenesisV2Agent(stream=False)
_agent: Any = ExploitLayer(_base_agent)
_exploit_enabled: bool = True
_current_model_label = "genesisV2+exploit"
_current_generation = 0
_action_history: list = []  # per-session action log for feature extraction


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Poker Bot Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ActRequest(BaseModel):
    observation: dict
    hand_number: int = 0
    last_human_action: str = ""
    reward: float = 0.0
    terminated: bool = False


class HotSwapRequest(BaseModel):
    model_path: str
    hidden_dim: int = 256


@app.get("/health")
def health():
    return {
        "ok": True,
        "generation": _current_generation,
        "model": _current_model_label,
    }


@app.post("/act")
def act(req: ActRequest):
    global _action_history

    obs = dict(req.observation)

    # Inject opp_last_action for genesisV2's opponent model
    obs["opp_last_action"] = req.last_human_action or ""

    info = {"hand_number": req.hand_number}

    # Detect hand boundary — reset action history per hand
    if not hasattr(act, "_last_hand") or act._last_hand != req.hand_number:
        act._last_hand = req.hand_number
        _action_history = []

    with _lock:
        if isinstance(_agent, CloneBotAdapter):
            result = _agent.act(obs, req.reward, req.terminated, False, info, _action_history)
        else:
            result = _agent.act(obs, req.reward, req.terminated, False, info)

    action_type, raise_amt, k1, k2 = result

    action_labels = {0: "fold", 1: "raise", 2: "check", 3: "call", 4: "discard"}
    label = action_labels.get(action_type, "fold")
    if action_type == 1:
        label = f"raise {raise_amt}"

    # Track bot's own actions for future feature extraction
    _action_history.append({"player": "bot", "type": action_labels.get(action_type, "fold"), "amount": raise_amt})

    return {
        "action": [action_type, raise_amt, k1, k2],
        "action_label": label.upper(),
    }


@app.post("/reset")
def reset():
    global _agent, _action_history
    with _lock:
        if isinstance(_agent, CloneBotAdapter):
            pass  # stateless, no reset needed
        else:
            _agent = GenesisV2Agent(stream=False)
        _action_history = []
    return {"ok": True}


@app.post("/hot_swap")
def hot_swap(req: HotSwapRequest):
    global _agent, _current_model_label, _current_generation, _lineage
    model_path = req.model_path
    if not Path(model_path).exists():
        # Try relative to training/
        model_path = str(_HERE / req.model_path)
    if not Path(model_path).exists():
        return {"ok": False, "error": f"Model not found: {req.model_path}"}
    try:
        new_agent = CloneBotAdapter(model_path, hidden_dim=req.hidden_dim)
        with _lock:
            _agent = new_agent
            _current_model_label = new_agent.label
            _lineage = load_lineage()
            _current_generation = _lineage.get("current_generation", 0)
        return {"ok": True, "model": _current_model_label, "generation": _current_generation}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/revert_genesis")
def revert_genesis():
    global _agent, _current_model_label, _current_generation
    with _lock:
        _agent = GenesisV2Agent(stream=False)
        _current_model_label = "genesisV2"
        _current_generation = 0
    return {"ok": True}


@app.post("/reload_exploit")
def reload_exploit():
    """
    Reload exploit_profile.json and update ExploitLayer rules.
    Called automatically by watch_and_train.py after each retrain.
    The new profile also updates the [98-107] opponent-tendency features
    used by any CloneBotAdapter for the next inference call.
    """
    global _agent
    with _lock:
        if isinstance(_agent, ExploitLayer):
            _agent.reload_profile()
            ep = _agent.ep
            return {
                "ok": True,
                "river_bet_always": ep.get("river_bet_always"),
                "preflop_always_raise": ep.get("preflop_always_raise"),
                "detect_river_trap": ep.get("detect_river_trap"),
                "suppress_flop_bluff": ep.get("suppress_flop_bluff"),
                "suppress_cbet": ep.get("suppress_cbet"),
            }
        return {"ok": False, "reason": "agent is not an ExploitLayer (CloneBot without exploit wrapper)"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Starting bot server on {args.host}:{args.port}")
    print(f"Initial bot: {_current_model_label} (gen {_current_generation})")
    print("Next.js will auto-detect this server and switch from heuristic mode.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
