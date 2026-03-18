"""
PPO Reinforcement Learning training for the poker bot.

The bot plays against a rotating pool of opponents (GenesisV2, ProbAgent, past clones)
and learns from win/loss chip outcomes. Warm-starts from the best behavioral cloning
checkpoint if one exists.

After each generation:
  - Saves full checkpoint:       models/rl_genN.pt
  - Saves hot-swap checkpoint:   models/rl_genN_swap.pt  (no value_head, CloneBotAdapter-compatible)
  - Hot-swaps the live bot_server if it is running

Usage:
    python rl_train.py                            # 5 gens, 2048 steps/gen
    python rl_train.py --generations 10
    python rl_train.py --hands-per-gen 200        # rollout_steps = hands * 10
    python rl_train.py --resume models/rl_gen3.pt
    python rl_train.py --eval-only models/rl_gen3.pt
    python rl_train.py --no-warmstart
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ── Path setup (mirrors selfplay.py) ─────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_TRAINER_ROOT = _HERE.parent
_REPO_ROOT = _TRAINER_ROOT.parent
for p in [str(_REPO_ROOT), str(_TRAINER_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

from gym_env import PokerEnv
from submission.genesisV2 import GenesisV2Agent
from agents.prob_agent import ProbabilityAgent
from model import PokerPolicyNet, raise_bucket_to_amount, KEEP_COMBOS, FEATURE_DIM
from bot_server import extract_features_from_gym_obs, CloneBotAdapter

MODELS_DIR = _HERE / "models"
LINEAGE_PATH = _HERE / "lineage.json"
MODELS_DIR.mkdir(exist_ok=True)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    rollout_steps: int = 2048       # transitions collected per generation
    n_epochs: int = 4               # PPO epochs per rollout
    mini_batch_size: int = 256
    lr: float = 3e-4
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01     # keeps policy from collapsing to always-fold
    vf_coeff: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    hidden_dim: int = 256
    value_warmup_steps: int = 200   # freeze policy heads, update only value head first
    elo_eval_hands: int = 100       # hands to play vs GenesisV2 for ELO measurement


# ── Rollout buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores transitions collected during a rollout."""

    def __init__(self, max_size: int, feat_dim: int = FEATURE_DIM):
        self.max_size = max_size
        self.obs = np.zeros((max_size, feat_dim), dtype=np.float32)
        # actions: [action_type (0-4), raise_bucket (0-9), discard_idx (0-9)]
        self.actions = np.zeros((max_size, 3), dtype=np.int32)
        self.log_probs = np.zeros(max_size, dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.valid_masks = np.zeros((max_size, 5), dtype=np.float32)  # valid_actions[5]
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, log_prob, value, reward, done, valid_mask):
        i = self.ptr % self.max_size
        self.obs[i] = obs
        self.actions[i] = action       # [action_type, raise_bucket, discard_idx]
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.valid_masks[i] = valid_mask
        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)

    def assign_terminal_reward(self, reward: float):
        """Assign reward to the most recently added transition."""
        if self.ptr == 0:
            return
        i = (self.ptr - 1) % self.max_size
        self.rewards[i] = reward

    def compute_gae(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns. Returns (advantages, returns)."""
        n = self.size
        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value * (1.0 - self.dones[t])
            else:
                next_value = self.values[t + 1] * (1.0 - self.dones[t])
            delta = self.rewards[t] + 0.99 * next_value - self.values[t]
            last_gae = delta + 0.99 * 0.95 * (1.0 - self.dones[t]) * last_gae
            adv[t] = last_gae

        returns = adv + self.values[:n]
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def get_minibatches(self, advantages: np.ndarray, returns: np.ndarray, batch_size: int, n_epochs: int):
        """Yield shuffled mini-batches for each PPO epoch."""
        n = self.size
        for _ in range(n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = indices[start:start + batch_size]
                yield (
                    torch.tensor(self.obs[idx]),
                    torch.tensor(self.actions[idx]),
                    torch.tensor(self.log_probs[idx]),
                    torch.tensor(advantages[idx]),
                    torch.tensor(returns[idx]),
                    torch.tensor(self.valid_masks[idx]),
                    torch.tensor(self.values[idx]),
                )

    def reset(self):
        self.ptr = 0
        self.size = 0


# ── Log-prob and entropy ───────────────────────────────────────────────────────

def compute_log_prob_and_entropy(
    a_logits: torch.Tensor,
    r_logits: torch.Tensor,
    d_logits: torch.Tensor,
    actions: torch.Tensor,
    valid_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log-prob and entropy for the compound multi-head action space.

    Discard step (valid_masks[:, 4] == 1):
        log_prob = log_prob(discard combo from discard_head)
        entropy  = H(discard_dist)

    Betting step (valid_masks[:, 4] == 0):
        log_prob = log_prob(action_type from masked action_head)
                 + (if action == raise) log_prob(raise bucket from raise_head)
        entropy  = H(action_dist) + (if raise) H(raise_dist)
    """
    NEG_INF = -1e9
    is_discard = valid_masks[:, 4].bool()

    # ── Discard branch ────────────────────────────────────────────────────────
    d_dist = Categorical(logits=d_logits)
    d_lp = d_dist.log_prob(actions[:, 2])
    d_ent = d_dist.entropy()

    # ── Betting branch — mask invalid action types ────────────────────────────
    a_logits_m = a_logits.clone()
    for i in range(4):
        a_logits_m[~valid_masks[:, i].bool(), i] = NEG_INF
    a_dist = Categorical(logits=a_logits_m)
    # Clamp action indices to [0,3] — discard steps (action==4) are masked out
    # by torch.where below, so their log_prob values are never used.
    safe_atype = actions[:, 0].clamp(0, 3)
    a_lp = a_dist.log_prob(safe_atype)
    a_ent = a_dist.entropy()

    # Raise sub-head (contributes only when action_type == 1)
    is_raise = (actions[:, 0] == 1) & ~is_discard
    r_dist = Categorical(logits=r_logits)
    r_lp = torch.where(is_raise, r_dist.log_prob(actions[:, 1]), torch.zeros_like(a_lp))
    r_ent = torch.where(is_raise, r_dist.entropy(), torch.zeros_like(a_ent))

    log_probs = torch.where(is_discard, d_lp, a_lp + r_lp)
    entropy = torch.where(is_discard, d_ent, a_ent + r_ent)
    return log_probs, entropy


# ── Action sampling ───────────────────────────────────────────────────────────

def sample_action(
    model: PokerPolicyNet,
    obs_tensor: torch.Tensor,
    valid_actions: list,
    obs_dict: dict,
) -> tuple[tuple, np.ndarray, float, float]:
    """
    Sample action from the policy network.

    Returns:
        action_env:    (action_type, raise_amt, k1, k2) — for env.step()
        stored_action: [action_type, raise_bucket, discard_idx] — for buffer
        log_prob:      float
        value:         float
    """
    with torch.no_grad():
        a_logits, r_logits, d_logits, value = model(obs_tensor)

    is_discard = bool(valid_actions[4])
    a_logits = a_logits[0]
    r_logits = r_logits[0]
    d_logits = d_logits[0]
    val = value[0].item()

    if is_discard:
        d_dist = Categorical(logits=d_logits)
        combo_idx = d_dist.sample()
        lp = d_dist.log_prob(combo_idx).item()
        k1, k2 = KEEP_COMBOS[combo_idx.item()]
        return (4, 0, k1, k2), np.array([4, 0, combo_idx.item()], dtype=np.int32), lp, val

    # Mask invalid betting actions
    NEG_INF = -1e9
    a_masked = a_logits.clone()
    for i in range(4):
        if not valid_actions[i]:
            a_masked[i] = NEG_INF

    a_dist = Categorical(logits=a_masked)
    atype = a_dist.sample()
    lp = a_dist.log_prob(atype).item()

    raise_bucket = 0
    raise_amt = 0
    if atype.item() == 1:
        r_dist = Categorical(logits=r_logits)
        raise_bucket_t = r_dist.sample()
        raise_bucket = raise_bucket_t.item()
        lp += r_dist.log_prob(raise_bucket_t).item()
        min_r = obs_dict.get("min_raise", 2)
        max_r = obs_dict.get("max_raise", 98)
        raise_amt = raise_bucket_to_amount(raise_bucket, int(min_r), int(max_r))

    stored = np.array([atype.item(), raise_bucket, 0], dtype=np.int32)
    return (atype.item(), raise_amt, 0, 1), stored, lp, val


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(
    model: PokerPolicyNet,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: np.ndarray,
    returns: np.ndarray,
    cfg: PPOConfig,
    freeze_policy: bool = False,
) -> dict:
    """Run PPO update epochs. Returns dict of mean losses."""
    total_pl, total_vl, total_ent, total_batches = 0.0, 0.0, 0.0, 0

    for (obs_b, act_b, old_lp_b, adv_b, ret_b, mask_b, old_val_b) in buffer.get_minibatches(
        advantages, returns, cfg.mini_batch_size, cfg.n_epochs
    ):
        a_logits, r_logits, d_logits, values = model(obs_b)

        new_lps, entropy = compute_log_prob_and_entropy(a_logits, r_logits, d_logits, act_b, mask_b)

        # PPO clipped surrogate
        ratio = (new_lps - old_lp_b).exp()
        surr1 = ratio * adv_b
        surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (clipped)
        val_clipped = old_val_b + (values - old_val_b).clamp(-cfg.clip_eps, cfg.clip_eps)
        vf_loss = torch.max((values - ret_b) ** 2, (val_clipped - ret_b) ** 2).mean()

        # Entropy bonus
        entropy_loss = -entropy.mean()

        if freeze_policy:
            loss = cfg.vf_coeff * vf_loss
        else:
            loss = policy_loss + cfg.vf_coeff * vf_loss + cfg.entropy_coeff * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        total_pl += policy_loss.item()
        total_vl += vf_loss.item()
        total_ent += (-entropy_loss).item()
        total_batches += 1

    n = max(total_batches, 1)
    return {"policy_loss": total_pl / n, "value_loss": total_vl / n, "entropy": total_ent / n}


# ── Opponent pool ─────────────────────────────────────────────────────────────

class OpponentPool:
    """
    Weighted random pool of opponents to train against.
    Fixed: GenesisV2 (50%), ProbAgent (30%), past RL clones (20% split).
    """

    def __init__(self):
        self._fixed: list[tuple[str, Any, float]] = []   # (name, factory_fn, weight)
        self._clones: list[tuple[str, str]] = []          # (name, path)
        self._max_clones = 5

        # Fixed opponents
        self._fixed.append(("genesisV2", lambda: GenesisV2Agent(stream=False), 3.0))
        self._fixed.append(("prob_agent", lambda: ProbabilityAgent(), 2.0))

        # Pre-instantiate fixed agents (expensive for GenesisV2)
        self._instances: dict[str, Any] = {}
        for name, factory, _ in self._fixed:
            try:
                self._instances[name] = factory()
            except Exception as e:
                print(f"Warning: could not instantiate {name}: {e}")

    def add_clone(self, path: str, gen: int, hidden_dim: int = 256):
        """Add a trained clone checkpoint to the pool."""
        name = f"rl_gen{gen}"
        self._clones.append((name, path))
        if len(self._clones) > self._max_clones:
            self._clones.pop(0)

    def sample(self) -> tuple[str, Any]:
        """Return (name, agent_instance) sampled by weight."""
        choices: list[tuple[str, Any, float]] = []

        for name, _, weight in self._fixed:
            agent = self._instances.get(name)
            if agent is not None:
                choices.append((name, agent, weight))

        # Clone weight split equally across 20% total
        if self._clones:
            clone_total_weight = 1.0
            per_clone = clone_total_weight / len(self._clones)
            for name, path in self._clones:
                try:
                    clone = CloneBotAdapter(path)
                    choices.append((name, clone, per_clone))
                except Exception:
                    pass

        if not choices:
            # Fallback — always have genesis
            return "genesisV2", GenesisV2Agent(stream=False)

        names = [c[0] for c in choices]
        agents = [c[1] for c in choices]
        weights = [c[2] for c in choices]
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = random.choices(range(len(choices)), weights=probs, k=1)[0]
        return names[idx], agents[idx]


# ── ELO tracking ──────────────────────────────────────────────────────────────

class EloTracker:
    """Tracks bot ELO vs a fixed reference (GenesisV2)."""
    K = 32
    INITIAL = 1000.0

    def __init__(self):
        self.bot_elo = self.INITIAL
        self.genesis_elo = self.INITIAL

    def measure(self, model: PokerPolicyNet, n_hands: int = 100, hidden_dim: int = 256) -> tuple[float, float, float]:
        """
        Play n_hands of model vs GenesisV2 with model as player 0.
        Returns (elo, win_rate, avg_reward_per_hand).
        """
        genesis = GenesisV2Agent(stream=False)
        env = PokerEnv()
        wins = 0
        total_reward = 0.0
        model.eval()

        for hand_num in range(n_hands):
            obs_pair, _ = env.reset()
            done = False
            action_history: list[dict] = []
            hand_reward = 0.0
            last_p0_action = None
            max_steps = 40
            steps = 0

            while not done and steps < max_steps:
                steps += 1
                actor = obs_pair[0]["acting_agent"]

                if actor == 0:
                    feat = extract_features_from_gym_obs(obs_pair[0], action_history)
                    obs_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                    valid = obs_pair[0]["valid_actions"]
                    action_env, _, _, _ = sample_action(model, obs_t, valid, obs_pair[0])
                    last_p0_action = action_env
                else:
                    try:
                        action_env = genesis.act(obs_pair[1], 0.0, False, False, {"hand_number": hand_num})
                    except Exception:
                        action_env = (0, 0, 0, 1)

                obs_pair, reward, done, truncated, info = env.step(action_env)
                if done:
                    hand_reward = reward[0]

            total_reward += hand_reward
            if hand_reward > 0:
                wins += 1

        model.train()
        win_rate = wins / max(n_hands, 1)
        avg_reward = total_reward / max(n_hands, 1)

        # ELO update
        expected = 1.0 / (1.0 + 10 ** ((self.genesis_elo - self.bot_elo) / 400))
        self.bot_elo += self.K * (win_rate - expected)

        return round(self.bot_elo, 1), win_rate, avg_reward


# ── Rollout collection ────────────────────────────────────────────────────────

def collect_rollout(
    model: PokerPolicyNet,
    opponent_pool: OpponentPool,
    cfg: PPOConfig,
    verbose: bool = False,
) -> RolloutBuffer:
    """Collect rollout_steps transitions from player-0 perspective."""
    buffer = RolloutBuffer(cfg.rollout_steps)
    env = PokerEnv()
    model.eval()

    steps = 0
    hand_count = 0
    opp_name = "?"

    while steps < cfg.rollout_steps:
        opp_name, opponent = opponent_pool.sample()
        obs_pair, _ = env.reset()
        done = False
        action_history: list[dict] = []
        last_p0_step_idx: int | None = None
        max_steps_per_hand = 40
        hand_steps = 0

        while not done and hand_steps < max_steps_per_hand:
            hand_steps += 1
            actor = obs_pair[0]["acting_agent"]

            if actor == 0:
                feat = extract_features_from_gym_obs(obs_pair[0], action_history)
                obs_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                valid = obs_pair[0]["valid_actions"]
                action_env, stored_action, log_prob, value = sample_action(
                    model, obs_t, valid, obs_pair[0]
                )
                valid_mask = np.array(valid, dtype=np.float32)
                buffer.add(feat, stored_action, log_prob, value, 0.0, False, valid_mask)
                last_p0_step_idx = buffer.ptr - 1
                steps += 1
                obs_pair, reward, done, truncated, info = env.step(action_env)
            else:
                try:
                    opp_action = opponent.act(obs_pair[1], 0.0, done, False, {"hand_number": hand_count})
                except Exception:
                    opp_action = (0, 0, 0, 1)
                obs_pair, reward, done, truncated, info = env.step(opp_action)

            # Track for feature extraction
            action_labels = {0: "fold", 1: "raise", 2: "check", 3: "call", 4: "discard"}
            atype = action_env[0] if actor == 0 else (opp_action[0] if actor == 1 else 0)
            action_history.append({
                "player": "bot" if actor == 0 else "opponent",
                "street": obs_pair[0].get("street", 0),
                "type": action_labels.get(atype, "fold"),
            })

            if done:
                # Assign terminal reward to last player-0 step
                if last_p0_step_idx is not None:
                    idx = last_p0_step_idx % buffer.max_size
                    buffer.rewards[idx] = float(reward[0])
                    buffer.dones[idx] = 1.0

        hand_count += 1

        if verbose and hand_count % 20 == 0:
            print(f"  Collected {steps}/{cfg.rollout_steps} steps ({hand_count} hands, last opp: {opp_name})")

        if steps >= cfg.rollout_steps:
            break

    model.train()
    return buffer


# ── Warm start ────────────────────────────────────────────────────────────────

def warm_start(model: PokerPolicyNet, models_dir: Path) -> bool:
    """Try to load the latest BC clone checkpoint into model. Returns True on success."""
    # Check RL checkpoints first (resume from latest RL gen)
    rl_candidates = sorted(models_dir.glob("rl_gen*.pt"),
                           key=lambda p: int(p.stem.replace("rl_gen", "").replace("_swap", "")))
    rl_full = [p for p in rl_candidates if "_swap" not in p.name]
    if rl_full:
        path = str(rl_full[-1])
        print(f"Warm-starting from RL checkpoint: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        return True

    # Fall back to best BC clone
    bc_candidates = sorted(models_dir.glob("clone_v*.pt"),
                           key=lambda p: int(p.stem.split("v")[1]))
    if bc_candidates:
        path = str(bc_candidates[-1])
        print(f"Warm-starting from BC checkpoint: {path}")
        try:
            PokerPolicyNet.from_clone_checkpoint.__func__(model.__class__, path, model.backbone[0].out_features)
            # Use the classmethod correctly
            loaded = PokerPolicyNet.from_clone_checkpoint(path)
            model.load_state_dict(loaded.state_dict(), strict=False)
            return True
        except Exception as e:
            # Simpler fallback
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=False)
                print(f"  (strict=False loaded, value_head randomly initialized)")
                return True
            except Exception as e2:
                print(f"  Warm-start failed: {e2}")

    print("No checkpoint found — starting from random initialization")
    return False


# ── Checkpoint save + hot-swap ─────────────────────────────────────────────────

def save_checkpoint(model: PokerPolicyNet, gen: int, elo: float) -> tuple[Path, Path]:
    """Save full and swap checkpoints. Returns (full_path, swap_path)."""
    full_path = MODELS_DIR / f"rl_gen{gen}.pt"
    swap_path = MODELS_DIR / f"rl_gen{gen}_swap.pt"

    torch.save(model.state_dict(), full_path)
    torch.save(model.to_clone_state_dict(), swap_path)

    print(f"Saved: {full_path.name}  (full)")
    print(f"Saved: {swap_path.name}  (hot-swap compatible)")
    return full_path, swap_path


def hot_swap(swap_path: Path, server_url: str, hidden_dim: int = 256) -> bool:
    if not _REQUESTS_OK:
        print("requests not installed — skipping hot-swap")
        return False
    try:
        resp = requests.post(
            f"{server_url}/hot_swap",
            json={"model_path": str(swap_path), "hidden_dim": hidden_dim},
            timeout=10,
        )
        if resp.ok and resp.json().get("ok"):
            print(f"Hot-swap OK: {resp.json().get('model')} (gen {resp.json().get('generation')})")
            return True
    except requests.exceptions.ConnectionError:
        print("Bot server not running — skipping hot-swap (model saved, loads on next server start)")
    except Exception as e:
        print(f"Hot-swap error: {e}")
    return False


# ── Lineage ───────────────────────────────────────────────────────────────────

def load_lineage() -> dict:
    if LINEAGE_PATH.exists():
        return json.loads(LINEAGE_PATH.read_text())
    return {"current_generation": 0, "config": {}, "models": []}


def update_lineage(lineage: dict, gen: int, full_path: Path, swap_path: Path, elo: float, win_rate: float):
    entry = {
        "gen": gen,
        "type": "rl_ppo",
        "path": str(full_path),
        "swap_path": str(swap_path),
        "elo": elo,
        "win_rate_vs_genesis": round(win_rate, 4),
        "trained_at": datetime.utcnow().isoformat(),
    }
    lineage["models"] = [m for m in lineage["models"] if m.get("gen") != gen]
    lineage["models"].append(entry)
    lineage["current_generation"] = gen
    LINEAGE_PATH.write_text(json.dumps(lineage, indent=2))


# ── Generation loop ───────────────────────────────────────────────────────────

def run_generation(
    gen: int,
    model: PokerPolicyNet,
    optimizer: torch.optim.Optimizer,
    opponent_pool: OpponentPool,
    elo_tracker: EloTracker,
    cfg: PPOConfig,
    server_url: str,
    verbose: bool,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Generation {gen}")
    print(f"{'='*60}")

    t0 = time.time()

    # Phase 1: Collect rollout
    print(f"Collecting {cfg.rollout_steps} steps...")
    buffer = collect_rollout(model, opponent_pool, cfg, verbose=verbose)
    print(f"  Collected {buffer.size} steps in {time.time()-t0:.1f}s")

    # Phase 2: Compute GAE
    # Get bootstrap value for last state
    last_feat = buffer.obs[(buffer.ptr - 1) % buffer.max_size]
    with torch.no_grad():
        *_, last_val = model(torch.tensor(last_feat).unsqueeze(0))
    bootstrap = 0.0 if buffer.dones[(buffer.ptr - 1) % buffer.max_size] else last_val[0].item()
    advantages, returns = buffer.compute_gae(bootstrap)

    # Phase 2a: Value head warm-up (freeze policy, update only critic)
    if gen == 1:
        print(f"Value head warm-up ({cfg.value_warmup_steps} steps)...")
        for p in list(model.action_head.parameters()) + list(model.raise_head.parameters()) + list(model.discard_head.parameters()) + list(model.backbone.parameters()):
            p.requires_grad_(False)
        warmup_buf = RolloutBuffer(min(cfg.value_warmup_steps, buffer.size))
        warmup_buf.obs[:warmup_buf.max_size] = buffer.obs[:warmup_buf.max_size]
        warmup_buf.actions[:warmup_buf.max_size] = buffer.actions[:warmup_buf.max_size]
        warmup_buf.log_probs[:warmup_buf.max_size] = buffer.log_probs[:warmup_buf.max_size]
        warmup_buf.values[:warmup_buf.max_size] = buffer.values[:warmup_buf.max_size]
        warmup_buf.rewards[:warmup_buf.max_size] = buffer.rewards[:warmup_buf.max_size]
        warmup_buf.dones[:warmup_buf.max_size] = buffer.dones[:warmup_buf.max_size]
        warmup_buf.valid_masks[:warmup_buf.max_size] = buffer.valid_masks[:warmup_buf.max_size]
        warmup_buf.ptr = warmup_buf.max_size
        warmup_buf.size = warmup_buf.max_size
        wu_adv, wu_ret = warmup_buf.compute_gae(0.0)
        ppo_update(model, optimizer, warmup_buf, wu_adv, wu_ret, cfg, freeze_policy=True)
        for p in model.parameters():
            p.requires_grad_(True)

    # Phase 3: PPO updates
    print(f"Running {cfg.n_epochs} PPO epochs...")
    t1 = time.time()
    losses = ppo_update(model, optimizer, buffer, advantages, returns, cfg)
    print(f"  Policy loss: {losses['policy_loss']:.4f}  Value loss: {losses['value_loss']:.4f}  Entropy: {losses['entropy']:.4f}  ({time.time()-t1:.1f}s)")

    # Phase 4: ELO measurement
    print(f"Measuring ELO vs GenesisV2 ({cfg.elo_eval_hands} hands)...")
    elo, win_rate, avg_reward = elo_tracker.measure(model, cfg.elo_eval_hands)
    print(f"  ELO: {elo}  Win rate: {win_rate:.1%}  Avg reward: {avg_reward:+.2f} chips/hand")

    # Phase 5: Save
    full_path, swap_path = save_checkpoint(model, gen, elo)

    # Phase 6: Hot-swap
    hot_swap(swap_path, server_url, cfg.hidden_dim)

    # Phase 7: Update lineage
    lineage = load_lineage()
    update_lineage(lineage, gen, full_path, swap_path, elo, win_rate)

    # Phase 8: Register in opponent pool for future generations
    opponent_pool.add_clone(str(swap_path), gen, cfg.hidden_dim)

    elapsed = time.time() - t0
    print(f"\nGeneration {gen} complete in {elapsed:.0f}s")
    return {"gen": gen, "elo": elo, "win_rate": win_rate, "avg_reward": avg_reward, **losses}


# ── Eval-only mode ────────────────────────────────────────────────────────────

def eval_only(checkpoint_path: str, cfg: PPOConfig):
    """Evaluate a checkpoint against GenesisV2 and print stats."""
    print(f"Evaluating {checkpoint_path}...")
    model = PokerPolicyNet(hidden_dim=cfg.hidden_dim)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    tracker = EloTracker()
    elo, win_rate, avg_reward = tracker.measure(model, n_hands=200, hidden_dim=cfg.hidden_dim)
    print(f"\nResults vs GenesisV2 (200 hands):")
    print(f"  Win rate:   {win_rate:.1%}")
    print(f"  Avg reward: {avg_reward:+.2f} chips/hand")
    print(f"  ELO:        {elo}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PPO RL training for poker bot")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--hands-per-gen", type=int, default=None,
                   help="Sets rollout_steps = hands * 10 (approx steps/hand)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--no-warmstart", action="store_true")
    p.add_argument("--server", default="http://127.0.0.1:8765")
    p.add_argument("--resume", metavar="PATH", help="Resume training from a full RL checkpoint")
    p.add_argument("--eval-only", metavar="PATH", help="Evaluate a checkpoint, no training")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = PPOConfig(
        rollout_steps=args.hands_per_gen * 10 if args.hands_per_gen else args.rollout_steps,
        n_epochs=args.n_epochs,
        mini_batch_size=args.batch_size,
        lr=args.lr,
        clip_eps=args.clip_eps,
        entropy_coeff=args.entropy_coeff,
        hidden_dim=args.hidden_dim,
    )

    if args.eval_only:
        eval_only(args.eval_only, cfg)
        return

    # Build model
    model = PokerPolicyNet(hidden_dim=cfg.hidden_dim)

    if args.resume:
        print(f"Resuming from {args.resume}")
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    elif not args.no_warmstart:
        warm_start(model, MODELS_DIR)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    opponent_pool = OpponentPool()
    elo_tracker = EloTracker()

    # Determine starting generation
    lineage = load_lineage()
    rl_gens = [m["gen"] for m in lineage.get("models", []) if m.get("type") == "rl_ppo"]
    start_gen = (max(rl_gens) + 1) if rl_gens else 1

    print(f"\nPPO RL Training")
    print(f"  Generations:   {args.generations}")
    print(f"  Rollout steps: {cfg.rollout_steps} (~{cfg.rollout_steps // 10} hands)")
    print(f"  Start gen:     {start_gen}")
    print(f"  Device:        CPU")

    history = []
    for i in range(args.generations):
        gen = start_gen + i
        result = run_generation(
            gen=gen,
            model=model,
            optimizer=optimizer,
            opponent_pool=opponent_pool,
            elo_tracker=elo_tracker,
            cfg=cfg,
            server_url=args.server,
            verbose=args.verbose,
        )
        history.append(result)

    print(f"\n{'='*60}")
    print(f"Training complete — {args.generations} generations")
    print(f"{'='*60}")
    print(f"{'Gen':>4}  {'ELO':>6}  {'Win%':>6}  {'Reward':>8}")
    for r in history:
        print(f"{r['gen']:>4}  {r['elo']:>6.0f}  {r['win_rate']:>6.1%}  {r['avg_reward']:>+8.2f}")


if __name__ == "__main__":
    main()
