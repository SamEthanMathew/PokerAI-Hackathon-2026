"""
All hyperparameters for the RL training pipeline in one place.
"""
from dataclasses import dataclass, field
from typing import Dict
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class TrainingConfig:
    # ── Feature dimensions ─────────────────────────────────────────────────
    # Categories 1-6: raw cards, hand strength, draws, card removal, opp range, board texture
    card_dim: int = 214
    # Categories 7-11: betting context, action history, opp model, equity, genesis
    context_dim: int = 63
    total_feature_dim: int = 277   # must equal card_dim + context_dim

    # ── Model architecture ─────────────────────────────────────────────────
    hidden_dim: int = 256
    num_residual_blocks: int = 3
    attn_heads: int = 4
    dropout: float = 0.1

    # ── Behavioral Cloning warm-up ─────────────────────────────────────────
    bc_lr: float = 1e-3
    bc_epochs: int = 50
    bc_batch_size: int = 256
    bc_val_split: float = 0.1
    bc_patience: int = 10          # early stopping patience

    # ── PPO ────────────────────────────────────────────────────────────────
    n_envs: int = 8                # parallel environments for rollout
    n_steps: int = 2048            # steps per env per rollout (16K total)
    batch_size: int = 1024         # minibatch size for PPO update
    n_epochs: int = 10             # PPO epochs per rollout
    lr: float = 3e-4
    lr_final: float = 1e-5         # linear decay target
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    vf_coeff: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 1.0             # no temporal discounting in poker
    gae_lambda: float = 0.95

    # ── KL penalty schedule: {cycle_number: kl_coefficient} ───────────────
    kl_coef_schedule: Dict[int, float] = field(default_factory=lambda: {
        0:  0.30,   # Phase 1: strong constraint to human style
        20: 0.15,   # Phase 2 early: moderate
        50: 0.05,   # Phase 2 late: light
        80: 0.01,   # Phase 3+: nearly pure RL
    })

    # ── Training phases ────────────────────────────────────────────────────
    phase1_cycles: int = 50        # Warm-up: BC → RL transition
    phase2_cycles: int = 150       # Core training
    phase3_cycles: int = 200       # Hardening
    total_cycles: int = 400        # phase1 + phase2 + phase3
    eval_every: int = 10           # evaluate every N cycles
    target_win_rate: float = 1.0   # keep training (no early stop); set <1.0 to enable early stop
    early_stop_consecutive: int = 3  # stop if beats target for N evals in a row

    # ── Reward shaping ─────────────────────────────────────────────────────
    # Shaping fully decays by 80% through training
    shape_decay_end: float = 0.80
    fold_equity_bonus: float = 0.30
    bad_fold_penalty: float = 0.30
    discard_quality_bonus: float = 0.15
    value_extraction_bonus: float = 0.10
    big_pot_threshold: int = 20

    # ── Paths (relative to this file's directory) ──────────────────────────
    data_path: str = os.path.join(
        _HERE, "../poker-bot-trainer/training/data/accumulated_session.json"
    )
    other_bots_dir: str = os.path.join(_HERE, "../other bots")
    genesis_knowledge_path: str = os.path.join(_HERE, "genesis_knowledge.json")
    tables_dir: str = os.path.join(_HERE, "tables")
    checkpoints_dir: str = os.path.join(_HERE, "checkpoints")
    logs_dir: str = os.path.join(_HERE, "logs")
    bot_dir: str = os.path.join(_HERE, "bot")

    # ── Hardware ───────────────────────────────────────────────────────────
    device: str = "cuda"           # "cuda" for Jetson training, "cpu" for inference test
    num_workers: int = 8           # CPU cores for rollout collection (12 available on Jetson)
    precompute_workers: int = 12   # cores for table generation

    # ── Evaluation ─────────────────────────────────────────────────────────
    eval_hands: int = 1000
    inference_latency_budget_ms: float = 5.0

    def get_kl_coef(self, cycle: int) -> float:
        """Return KL coefficient for the given training cycle."""
        coef = 0.01
        for threshold, val in sorted(self.kl_coef_schedule.items()):
            if cycle >= threshold:
                coef = val
        return coef

    def get_lr(self, cycle: int) -> float:
        """Linear decay from lr to lr_final over total_cycles."""
        frac = min(cycle / max(self.total_cycles, 1), 1.0)
        return self.lr + frac * (self.lr_final - self.lr)

    def get_entropy_coeff(self, cycle: int) -> float:
        """Entropy coefficient schedule: 0.02 (phase1) → 0.01 (phase2) → 0.005 (phase3)."""
        if cycle < self.phase1_cycles:
            return 0.02
        elif cycle < self.phase1_cycles + self.phase2_cycles:
            return 0.01
        return 0.005
