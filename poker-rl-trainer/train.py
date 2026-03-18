"""
Main RL training entry point.

Usage:
    python train.py --phase all          # BC warm-up then full PPO training
    python train.py --phase precompute   # generate equity tables only
    python train.py --phase bc           # BC warm-up only → checkpoints/poker_clone.pt
    python train.py --phase rl           # PPO only (requires poker_clone.pt)
    python train.py --phase eval         # evaluate latest checkpoint vs all opponents
    python train.py --phase export       # export bot/ tournament submission

All hyperparameters are in config.py (TrainingConfig).
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import TrainingConfig
from model import PokerNetV2, KEEP_COMBOS
from features import extract_features, split_features, update_opp_stats, CARD_DIM, CONTEXT_DIM
from rewards import RewardShaper
from opponent_pool import load_opponent_pool, OpponentPool
from evaluation import run_evaluation
from precompute import load_tables
from env.poker_env import PokerTrainingEnv

cfg = TrainingConfig()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_dirs():
    for d in [cfg.tables_dir, cfg.checkpoints_dir, cfg.logs_dir, cfg.bot_dir]:
        os.makedirs(d, exist_ok=True)


def _load_genesis():
    path = cfg.genesis_knowledge_path
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"[train] WARNING: genesis_knowledge.json not found at {path}")
    return {}


def _latest_checkpoint() -> Optional[str]:
    latest = os.path.join(cfg.checkpoints_dir, "latest_model.pt")
    if os.path.exists(latest):
        return latest
    bc = os.path.join(cfg.checkpoints_dir, "poker_clone.pt")
    if os.path.exists(bc):
        return bc
    return None


# ── Phase: precompute ────────────────────────────────────────────────────────

def run_precompute():
    print("=== Phase: Precompute equity tables ===")
    import subprocess
    ret = subprocess.run(
        [sys.executable, os.path.join(_HERE, "precompute.py"),
         "--table", "all",
         "--workers", str(cfg.precompute_workers),
         "--tables-dir", cfg.tables_dir],
        check=False,
    )
    if ret.returncode != 0:
        print("[train] precompute.py returned non-zero exit code")
    else:
        print("[train] Precompute complete.")


# ── Phase: BC warm-up ────────────────────────────────────────────────────────

def _load_bc_data(tables: dict, genesis: dict):
    """
    Load accumulated_session.json and extract 277-dim features.
    Returns (X_card, X_ctx, y_action, y_raise_bucket, y_discard) as float tensors.
    """
    data_path = cfg.data_path
    if not os.path.exists(data_path):
        print(f"[BC] WARNING: training data not found at {data_path}")
        return None

    print(f"[BC] Loading training data from {data_path} …")
    with open(data_path) as f:
        records = json.load(f)

    print(f"[BC] {len(records)} records loaded. Extracting features …")

    feats, act_labels, raise_labels, discard_labels = [], [], [], []
    opp_stats: dict = {}

    for rec in records:
        obs = rec.get("observation", {})
        if not obs:
            continue

        action = rec.get("action")
        if action is None:
            continue

        action_type = action[0] if isinstance(action, (list, tuple)) else action
        raise_amt   = action[1] if isinstance(action, (list, tuple)) and len(action) > 1 else 0
        keep1       = action[2] if isinstance(action, (list, tuple)) and len(action) > 2 else 0
        keep2       = action[3] if isinstance(action, (list, tuple)) and len(action) > 3 else 1

        feat = extract_features(obs, opp_stats, genesis, tables)
        feats.append(feat)

        # Action label: 0=fold 1=raise 2=check 3=call (discard handled separately)
        if action_type == 4:  # discard → label from keep combo
            act_labels.append(2)  # placeholder for action head
            raise_labels.append(0)
            keep_pair = tuple(sorted([keep1, keep2]))
            try:
                discard_labels.append(KEEP_COMBOS.index(keep_pair))
            except ValueError:
                discard_labels.append(0)
        else:
            act_labels.append(min(action_type, 3))
            # Raise bucket: 0-9 proportional to amount
            if action_type == 1 and obs.get("max_raise", 100) > obs.get("min_raise", 2):
                rng = obs["max_raise"] - obs["min_raise"]
                bucket = round((raise_amt - obs.get("min_raise", 2)) / rng * 9)
                raise_labels.append(max(0, min(9, bucket)))
            else:
                raise_labels.append(0)
            discard_labels.append(0)

        update_opp_stats(opp_stats, obs, action)

    if not feats:
        print("[BC] No valid records found.")
        return None

    X = np.array(feats, dtype=np.float32)
    X_card = torch.from_numpy(X[:, :CARD_DIM])
    X_ctx  = torch.from_numpy(X[:, CARD_DIM:])
    y_act  = torch.tensor(act_labels, dtype=torch.long)
    y_raise = torch.tensor(raise_labels, dtype=torch.long)
    y_disc  = torch.tensor(discard_labels, dtype=torch.long)

    print(f"[BC] Feature matrix: {X.shape}  actions={y_act.shape[0]}")
    return X_card, X_ctx, y_act, y_raise, y_disc


def run_bc_phase():
    print("=== Phase: Behavioral Cloning warm-up ===")
    _ensure_dirs()
    tables   = load_tables(cfg.tables_dir)
    genesis  = _load_genesis()

    data = _load_bc_data(tables, genesis)
    if data is None:
        print("[BC] No training data — skipping BC phase. Model will start from random init.")
        # Save a randomly initialised model so RL phase can load it
        model = PokerNetV2(cfg.card_dim, cfg.context_dim, cfg.hidden_dim, cfg.num_residual_blocks)
        out = os.path.join(cfg.checkpoints_dir, "poker_clone.pt")
        model.save(out, {"bc_epochs": 0, "note": "random init (no BC data)"})
        return

    X_card, X_ctx, y_act, y_raise, y_disc = data
    N = X_card.shape[0]
    val_n = max(1, int(N * cfg.bc_val_split))
    idx = torch.randperm(N)
    train_idx, val_idx = idx[val_n:], idx[:val_n]

    model = PokerNetV2(cfg.card_dim, cfg.context_dim, cfg.hidden_dim, cfg.num_residual_blocks)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.bc_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_ctr = 0
    out_path = os.path.join(cfg.checkpoints_dir, "poker_clone.pt")

    print(f"[BC] Training on {len(train_idx)} samples, validating on {len(val_idx)} …")
    print(f"[BC] Device: {device}")

    for epoch in range(cfg.bc_epochs):
        model.train()
        # Shuffle training data
        perm = train_idx[torch.randperm(len(train_idx))]
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(perm), cfg.bc_batch_size):
            batch = perm[start:start + cfg.bc_batch_size]
            xc = X_card[batch].to(device)
            xctx = X_ctx[batch].to(device)
            ya = y_act[batch].to(device)
            yr = y_raise[batch].to(device)
            yd = y_disc[batch].to(device)

            a_logits, raise_amt, d_logits, _ = model(xc, xctx)

            # Separate discard and betting losses
            discard_mask = ya == 2  # treated as discard placeholder
            betting_mask = ~discard_mask

            loss = torch.tensor(0.0, device=device)
            if betting_mask.any():
                loss_act = F.cross_entropy(a_logits[betting_mask], ya[betting_mask])
                loss += loss_act

                raise_mask = (ya[betting_mask] == 1)
                if raise_mask.any():
                    raise_logits_fake = raise_amt[betting_mask][raise_mask].expand(-1, 10)
                    # Treat raise_amt sigmoid as bucket 0-9 prediction via MSE proxy
                    r_targets = yr[betting_mask][raise_mask].float() / 9.0
                    loss += 0.3 * F.mse_loss(raise_amt[betting_mask][raise_mask].squeeze(-1), r_targets)

            if discard_mask.any():
                loss += 0.2 * F.cross_entropy(d_logits[discard_mask], yd[discard_mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            xc = X_card[val_idx].to(device)
            xctx = X_ctx[val_idx].to(device)
            ya = y_act[val_idx].to(device)
            a_logits, _, d_logits, _ = model(xc, xctx)
            val_loss = F.cross_entropy(a_logits, ya).item()
            act_acc = (a_logits.argmax(dim=-1) == ya).float().mean().item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            model.save(out_path, {"bc_epoch": epoch, "val_loss": val_loss})
        else:
            patience_ctr += 1

        if epoch % 5 == 0 or epoch == cfg.bc_epochs - 1:
            print(f"  [BC] epoch {epoch:3d}  train_loss={avg_train:.4f}  "
                  f"val_loss={val_loss:.4f}  act_acc={act_acc:.2%}  patience={patience_ctr}")

        if patience_ctr >= cfg.bc_patience:
            print(f"  [BC] Early stopping at epoch {epoch}")
            break

    print(f"[BC] Done. Best val_loss={best_val_loss:.4f}. Saved to {out_path}")


# ── Rollout collection ────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one rollout batch for PPO."""

    def __init__(self):
        self.card_features: List[np.ndarray] = []
        self.ctx_features: List[np.ndarray]  = []
        self.actions: List[tuple]            = []
        self.rewards: List[float]            = []
        self.values: List[float]             = []
        self.log_probs: List[float]          = []
        self.dones: List[bool]               = []
        self.valid_actions: List[List[bool]] = []
        self.obs_list: List[dict]            = []

    def add(self, cf, ctx, action, reward, value, log_prob, done, valid, obs):
        self.card_features.append(cf)
        self.ctx_features.append(ctx)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.valid_actions.append(valid)
        self.obs_list.append(obs)

    def __len__(self):
        return len(self.rewards)


def _sample_action(
    model: PokerNetV2,
    card_feat: torch.Tensor,
    ctx_feat: torch.Tensor,
    valid_actions: List[bool],
    obs: dict,
    device: torch.device,
) -> Tuple[tuple, float, float]:
    """
    Sample an action from the current policy.
    Returns (action_tuple, log_prob, value).
    """
    cf = card_feat.unsqueeze(0).to(device)
    ctx = ctx_feat.unsqueeze(0).to(device)

    a_logits, raise_sigmoid, d_logits, value = model(cf, ctx)

    # ── Discard phase ───────────────────────────────────────────────────────
    if valid_actions[4]:
        masked = d_logits[0].clone()
        dist = torch.distributions.Categorical(logits=masked)
        idx = dist.sample()
        lp = dist.log_prob(idx).item()
        k1, k2 = KEEP_COMBOS[idx.item()]
        return (4, 0, k1, k2), lp, value[0].item()

    # ── Betting action ──────────────────────────────────────────────────────
    logits = a_logits[0].clone()
    for i, v in enumerate(valid_actions[:4]):
        if not v:
            logits[i] = -1e9

    dist_act = torch.distributions.Categorical(logits=logits)
    action_type = dist_act.sample()
    lp_act = dist_act.log_prob(action_type).item()

    raise_amt = 0
    lp_raise = 0.0
    if action_type.item() == 1:  # RAISE
        frac = raise_sigmoid[0].item()
        raise_amt = int(round(
            obs["min_raise"] + frac * (obs["max_raise"] - obs["min_raise"])
        ))
        raise_amt = max(obs["min_raise"], min(obs["max_raise"], raise_amt))
        # Log prob of raise amount (treat as Gaussian centred on sigmoid output)
        lp_raise = -0.5 * ((frac - 0.5) ** 2) / (0.25 ** 2)

    log_prob = lp_act + 0.3 * lp_raise
    return (action_type.item(), raise_amt, 0, 0), log_prob, value[0].item()


def collect_rollout(
    model: PokerNetV2,
    opponent: "InProcessOpponent",
    tables: dict,
    genesis: dict,
    shaper: RewardShaper,
    training_progress: float,
    device: torch.device,
    n_steps: int = 2048,
) -> RolloutBuffer:
    """
    Collect n_steps transitions by playing against opponent.
    Returns a RolloutBuffer.
    """
    buffer = RolloutBuffer()
    env = PokerTrainingEnv()
    opp_stats: dict = {}

    obs_p0, obs_p1 = env.reset()
    hand_num = 0
    steps = 0

    while steps < n_steps:
        acting = env.acting_player

        if acting == 0:
            # Our bot's turn
            feat = extract_features(obs_p0, opp_stats, genesis, tables)
            cf, ctx = split_features(feat)
            cf_t = torch.from_numpy(cf)
            ctx_t = torch.from_numpy(ctx)

            with torch.no_grad():
                action, log_prob, value = _sample_action(
                    model, cf_t, ctx_t, list(obs_p0["valid_actions"]), obs_p0, device
                )

            prev_obs = obs_p0.copy()
            obs_p0, obs_p1, r0, r1, done, info = env.step(action)

            # Compute reward with shaping
            hand_result = None
            if done:
                showdown = info.get("player_0_cards") is not None
                hand_result = {
                    "chips_won_lost": r0,
                    "won": r0 > 0,
                    "showdown": showdown,
                }
            reward = shaper.shape(prev_obs, action, hand_result, training_progress, tables)

            buffer.add(cf, ctx, action, reward, value, log_prob, done,
                       list(prev_obs["valid_actions"]), prev_obs)

            update_opp_stats(opp_stats, obs_p0, action)
            opp.observe(obs_p1, r1, done, False, {"hand_number": hand_num})
            steps += 1
        else:
            # Opponent's turn — step but don't add to buffer
            opp_action = opponent.act(obs_p1, 0.0, False, False, {"hand_number": hand_num})
            obs_p0, obs_p1, r0, r1, done, info = env.step(opp_action)
            opp.observe(obs_p1, r1, done, False, {"hand_number": hand_num})
            update_opp_stats(opp_stats, obs_p0, opp_action)

        if done:
            obs_p0, obs_p1 = env.reset()
            hand_num += 1

    return buffer


# ── GAE + PPO update ──────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma=1.0, lam=0.95):
    """Generalised Advantage Estimation."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(n)):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def ppo_update(
    model: PokerNetV2,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: np.ndarray,
    returns: np.ndarray,
    cfg: TrainingConfig,
    bc_reference: Optional[PokerNetV2],
    kl_coef: float,
    device: torch.device,
    entropy_coeff: float,
) -> dict:
    """Run PPO epochs over the rollout buffer. Returns loss dict."""
    n = len(buffer)
    idx = np.arange(n)

    X_card  = torch.from_numpy(np.stack(buffer.card_features)).to(device)
    X_ctx   = torch.from_numpy(np.stack(buffer.ctx_features)).to(device)
    old_lp  = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
    adv_t   = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret_t   = torch.tensor(returns, dtype=torch.float32, device=device)
    actions = buffer.actions

    # Normalise advantages
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    stats = defaultdict(list)

    for _ in range(cfg.n_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, cfg.batch_size):
            b = idx[start:start + cfg.batch_size]
            b_t = torch.tensor(b, dtype=torch.long, device=device)

            cf_b  = X_card[b_t]
            ctx_b = X_ctx[b_t]
            a_logits, raise_sigmoid, d_logits, values_pred = model(cf_b, ctx_b)

            # Recompute log probs for current policy
            new_lps = []
            for i, bi in enumerate(b):
                act = actions[bi]
                va  = buffer.valid_actions[bi]
                if act[0] == 4:  # discard
                    logits = d_logits[i]
                    combo_idx = KEEP_COMBOS.index(tuple(sorted([act[2], act[3]])))
                    lp = F.log_softmax(logits, dim=-1)[combo_idx]
                else:
                    logits = a_logits[i].clone()
                    for k, v in enumerate(va[:4]):
                        if not v:
                            logits[k] = -1e9
                    lp = F.log_softmax(logits, dim=-1)[act[0]]
                new_lps.append(lp)

            new_lp_t = torch.stack(new_lps)
            ratio = torch.exp(new_lp_t - old_lp[b_t])

            adv_b = adv_t[b_t]
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            vf_loss = F.mse_loss(values_pred, ret_t[b_t])

            # Entropy bonus
            probs = F.softmax(a_logits, dim=-1)
            entropy = -(probs * probs.log().clamp(min=-20)).sum(dim=-1).mean()

            # KL penalty vs BC reference
            kl_loss = torch.tensor(0.0, device=device)
            if bc_reference is not None and kl_coef > 0:
                with torch.no_grad():
                    ref_logits, _, _, _ = bc_reference(cf_b, ctx_b)
                p_curr = F.softmax(a_logits, dim=-1)
                p_ref  = F.softmax(ref_logits, dim=-1)
                kl_loss = (p_curr * (p_curr.log() - p_ref.log())).sum(dim=-1).mean()

            loss = (policy_loss
                    + cfg.vf_coeff * vf_loss
                    - entropy_coeff * entropy
                    + kl_coef * kl_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["vf_loss"].append(vf_loss.item())
            stats["entropy"].append(entropy.item())
            stats["kl"].append(kl_loss.item())

    return {k: float(np.mean(v)) for k, v in stats.items()}


# ── Phase: RL ─────────────────────────────────────────────────────────────────

def run_rl_phase():
    print("=== Phase: PPO Reinforcement Learning ===")
    _ensure_dirs()

    tables  = load_tables(cfg.tables_dir)
    genesis = _load_genesis()

    # Load model (from BC checkpoint or random init)
    bc_path = os.path.join(cfg.checkpoints_dir, "poker_clone.pt")
    latest  = _latest_checkpoint()

    if latest and os.path.exists(latest):
        print(f"[RL] Loading model from {latest}")
        model = PokerNetV2.from_bc_checkpoint(latest, cfg.card_dim, cfg.context_dim,
                                              cfg.hidden_dim, cfg.num_residual_blocks)
    else:
        print("[RL] No checkpoint found — starting from random initialisation")
        model = PokerNetV2(cfg.card_dim, cfg.context_dim, cfg.hidden_dim, cfg.num_residual_blocks)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Frozen BC reference for KL penalty
    bc_reference: Optional[PokerNetV2] = None
    if os.path.exists(bc_path):
        bc_reference = PokerNetV2.from_bc_checkpoint(
            bc_path, cfg.card_dim, cfg.context_dim, cfg.hidden_dim, cfg.num_residual_blocks
        )
        bc_reference = bc_reference.to(device)
        bc_reference.eval()
        for p in bc_reference.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    shaper    = RewardShaper()

    print("[RL] Loading opponent pool …")
    opponents = load_opponent_pool(cfg.other_bots_dir)
    pool      = OpponentPool(opponents)
    print(f"[RL] Opponents: {pool.names()}")

    best_win_rate = 0.0
    consecutive_above_target = 0
    total_cycles = cfg.total_cycles

    log_path = os.path.join(cfg.logs_dir, "training_log.csv")
    log_header_written = os.path.exists(log_path)

    for cycle in range(total_cycles):
        training_progress = cycle / total_cycles
        kl_coef     = cfg.get_kl_coef(cycle)
        lr          = cfg.get_lr(cycle)
        entropy_c   = cfg.get_entropy_coeff(cycle)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Rollout ────────────────────────────────────────────────────────
        opponent = pool.sample()
        model.train()
        t_collect = time.time()
        buffer = collect_rollout(
            model, opponent, tables, genesis, shaper,
            training_progress, device, cfg.n_steps
        )
        t_collect = time.time() - t_collect

        # ── GAE ────────────────────────────────────────────────────────────
        adv, ret = compute_gae(buffer.rewards, buffer.values, buffer.dones,
                               cfg.gamma, cfg.gae_lambda)

        # ── PPO update ─────────────────────────────────────────────────────
        t_update = time.time()
        loss_dict = ppo_update(
            model, optimizer, buffer, adv, ret, cfg,
            bc_reference, kl_coef, device, entropy_c
        )
        t_update = time.time() - t_update

        # ── Logging ────────────────────────────────────────────────────────
        mean_reward = float(np.mean(buffer.rewards))
        print(f"[RL] cycle={cycle:4d}  opp={opponent.name:<12}"
              f"  mean_r={mean_reward:+.2f}  "
              f"pol={loss_dict.get('policy_loss', 0):.4f}  "
              f"vf={loss_dict.get('vf_loss', 0):.4f}  "
              f"kl={loss_dict.get('kl', 0):.4f}  "
              f"ent={loss_dict.get('entropy', 0):.3f}  "
              f"lr={lr:.2e}  "
              f"t={t_collect:.1f}+{t_update:.1f}s")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not log_header_written:
                writer.writerow(["cycle", "opponent", "mean_reward",
                                 "policy_loss", "vf_loss", "entropy", "kl",
                                 "lr", "kl_coef"])
                log_header_written = True
            writer.writerow([cycle, opponent.name, f"{mean_reward:.4f}",
                              f"{loss_dict.get('policy_loss', 0):.4f}",
                              f"{loss_dict.get('vf_loss', 0):.4f}",
                              f"{loss_dict.get('entropy', 0):.4f}",
                              f"{loss_dict.get('kl', 0):.4f}",
                              f"{lr:.2e}", f"{kl_coef:.3f}"])

        # Save latest checkpoint every cycle
        model.save(
            os.path.join(cfg.checkpoints_dir, "latest_model.pt"),
            {"cycle": cycle, "win_rate": best_win_rate}
        )

        # ── Evaluation ─────────────────────────────────────────────────────
        if (cycle + 1) % cfg.eval_every == 0:
            model.eval()
            eval_result = run_evaluation(
                model, pool, tables, genesis, cycle,
                n_hands=cfg.eval_hands,
                device="cpu",
                bc_reference=bc_reference,
                logs_dir=cfg.logs_dir,
            )
            model.train()
            model.to(device)

            if eval_result.aggregate_win_rate > best_win_rate:
                best_win_rate = eval_result.aggregate_win_rate
                model.save(
                    os.path.join(cfg.checkpoints_dir, "best_model.pt"),
                    {"cycle": cycle, "win_rate": best_win_rate}
                )
                print(f"[RL] ★ New best: {best_win_rate:.1%}")

            if eval_result.aggregate_win_rate >= cfg.target_win_rate:
                consecutive_above_target += 1
            else:
                consecutive_above_target = 0

            if consecutive_above_target >= cfg.early_stop_consecutive:
                print(f"[RL] Target win rate {cfg.target_win_rate:.0%} reached for "
                      f"{cfg.early_stop_consecutive} consecutive evaluations. Stopping.")
                break

    print(f"[RL] Training complete. Best win rate: {best_win_rate:.1%}")


# ── Phase: eval ──────────────────────────────────────────────────────────────

def run_eval_phase():
    print("=== Phase: Evaluation ===")
    _ensure_dirs()

    latest = _latest_checkpoint() or os.path.join(cfg.checkpoints_dir, "best_model.pt")
    if not os.path.exists(latest):
        print(f"[eval] No checkpoint found at {latest}")
        return

    tables   = load_tables(cfg.tables_dir)
    genesis  = _load_genesis()

    model = PokerNetV2.from_bc_checkpoint(latest, cfg.card_dim, cfg.context_dim,
                                          cfg.hidden_dim, cfg.num_residual_blocks)
    opponents = load_opponent_pool(cfg.other_bots_dir)
    pool      = OpponentPool(opponents)

    run_evaluation(model, pool, tables, genesis, cycle=-1,
                   n_hands=cfg.eval_hands, device="cpu",
                   logs_dir=cfg.logs_dir)


# ── Phase: export ─────────────────────────────────────────────────────────────

def run_export_phase():
    print("=== Phase: Export tournament bot ===")
    _ensure_dirs()

    best = os.path.join(cfg.checkpoints_dir, "best_model.pt")
    if not os.path.exists(best):
        best = _latest_checkpoint()
    if best is None or not os.path.exists(best):
        print("[export] No model checkpoint found.")
        return

    import shutil

    bot_dir = cfg.bot_dir
    os.makedirs(bot_dir, exist_ok=True)

    # Save stripped model (no value_head)
    model = PokerNetV2.from_bc_checkpoint(best, cfg.card_dim, cfg.context_dim,
                                          cfg.hidden_dim, cfg.num_residual_blocks)
    model.save_for_tournament(os.path.join(bot_dir, "poker_final.pt"))

    # Copy supporting files
    for fname in ["model.py", "features.py"]:
        src = os.path.join(_HERE, fname)
        dst = os.path.join(bot_dir, fname)
        shutil.copy2(src, dst)
        print(f"  Copied {fname}")

    shutil.copy2(cfg.genesis_knowledge_path, os.path.join(bot_dir, "genesis_knowledge.json"))

    # Copy tables
    tables_dst = os.path.join(bot_dir, "tables")
    if os.path.isdir(cfg.tables_dir):
        if os.path.exists(tables_dst):
            shutil.rmtree(tables_dst)
        shutil.copytree(cfg.tables_dir, tables_dst)
        print(f"  Copied tables/")

    # Write bot.py
    _write_bot_py(bot_dir)

    # Verify inference time on CPU
    _check_inference_time(model)

    print(f"[export] Tournament bot written to {bot_dir}/")


def _write_bot_py(bot_dir: str):
    """Write the tournament submission bot.py."""
    bot_path = os.path.join(bot_dir, "bot.py")
    # (bot.py is written separately — see bot/bot.py)
    if os.path.exists(bot_path):
        print(f"  bot.py already exists at {bot_path}")
    else:
        print(f"  NOTE: write bot/bot.py separately (see template in repo)")


def _check_inference_time(model):
    """Verify model inference < 5ms on CPU."""
    model.eval().cpu()
    cf  = torch.randn(1, CARD_DIM)
    ctx = torch.randn(1, CONTEXT_DIM)
    times = []
    with torch.no_grad():
        for _ in range(500):
            t0 = time.perf_counter()
            model(cf, ctx)
            times.append((time.perf_counter() - t0) * 1000)
    median_ms = float(np.median(times))
    ok = "✓" if median_ms < cfg.inference_latency_budget_ms else "✗"
    print(f"  {ok} Inference latency (CPU): {median_ms:.2f}ms "
          f"(budget={cfg.inference_latency_budget_ms:.0f}ms)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Poker RL trainer")
    parser.add_argument(
        "--phase",
        choices=["all", "precompute", "bc", "rl", "eval", "export"],
        default="all",
    )
    args = parser.parse_args()

    if args.phase in ("precompute", "all"):
        run_precompute()

    if args.phase in ("bc", "all"):
        run_bc_phase()

    if args.phase in ("rl", "all"):
        run_rl_phase()

    if args.phase == "eval":
        run_eval_phase()

    if args.phase in ("export", "all"):
        run_export_phase()


if __name__ == "__main__":
    main()
