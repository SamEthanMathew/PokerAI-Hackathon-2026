"""
PokerNetV2 — dual-stream architecture with cross-attention.

Stream 1 (Card stream):    raw cards, hand strength, draws, card removal,
                           opponent range, board texture  → 214 dims
Stream 2 (Context stream): betting context, action history, live opponent
                           model, equity, genesis knowledge → 63 dims

Streams merge → cross-attention → residual MLP → 4 output heads.
Total params: ~450K. Inference: ~2ms on ARM64 CPU (no GPU).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# Keep combos C(5,2): all (i,j) pairs with i < j where i,j in [0,4]
KEEP_COMBOS: List[Tuple[int, int]] = [
    (i, j) for i in range(5) for j in range(i + 1, 5)
]  # length 10


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class PokerNetV2(nn.Module):
    """
    Dual-stream architecture with cross-attention.

    Forward inputs:
        card_features:    (B, card_dim)     default card_dim=214
        context_features: (B, context_dim)  default context_dim=63

    Forward outputs:
        action_logits:  (B, 4)    — fold / raise / check / call
        raise_amount:   (B, 1)    — sigmoid ∈ [0,1]; scale to [min_raise, max_raise] at inference
        discard_logits: (B, 10)   — C(5,2)=10 keep-pair options
        value:          (B,)      — critic V(s) for PPO
    """

    def __init__(
        self,
        card_dim: int = 214,
        context_dim: int = 63,
        hidden_dim: int = 256,
        num_residual_blocks: int = 3,
        attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.card_dim = card_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        half = hidden_dim // 2

        # ── Stream 1: Card encoder ─────────────────────────────────────────
        self.card_encoder = nn.Sequential(
            nn.Linear(card_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, half),
            nn.ReLU(),
        )

        # ── Stream 2: Context encoder ──────────────────────────────────────
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, half),
            nn.ReLU(),
            nn.Linear(half, half),
            nn.ReLU(),
        )

        # ── Merge ──────────────────────────────────────────────────────────
        self.merge_proj = nn.Linear(hidden_dim, hidden_dim)

        # ── Cross-attention (single-token self-attention over merged rep) ──
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # ── Residual blocks ────────────────────────────────────────────────
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])

        # ── Shared trunk ───────────────────────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, half),
            nn.ReLU(),
        )

        # ── Output heads ───────────────────────────────────────────────────

        # Head 1: Action type — fold(0) / raise(1) / check(2) / call(3)
        self.action_head = nn.Sequential(
            nn.Linear(half, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        # Head 2: Raise amount (continuous sigmoid → scaled to [min, max] at inference)
        self.raise_head = nn.Sequential(
            nn.Linear(half, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Head 3: Discard — which 2 cards to keep from 5; C(5,2)=10 options
        self.discard_head = nn.Sequential(
            nn.Linear(half, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        # Head 4: Value function V(s) for PPO critic
        self.value_head = nn.Sequential(
            nn.Linear(half, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        card_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode both streams
        card_enc = self.card_encoder(card_features)          # (B, half)
        ctx_enc = self.context_encoder(context_features)     # (B, half)

        # Merge and project
        merged = torch.cat([card_enc, ctx_enc], dim=-1)      # (B, hidden)
        merged = self.merge_proj(merged)                      # (B, hidden)

        # Cross-attention: treat each sample as a single-token sequence
        q = merged.unsqueeze(1)                              # (B, 1, hidden)
        attn_out, _ = self.cross_attn(q, q, q)
        merged = self.attn_norm(merged + attn_out.squeeze(1))  # residual

        # Residual blocks
        for block in self.residual_blocks:
            merged = block(merged)

        # Shared trunk → half-width
        trunk = self.trunk(merged)                           # (B, half)

        # Heads
        action_logits = self.action_head(trunk)              # (B, 4)
        raise_amount = self.raise_head(trunk)                # (B, 1)
        discard_logits = self.discard_head(trunk)            # (B, 10)
        value = self.value_head(trunk).squeeze(-1)           # (B,)

        return action_logits, raise_amount, discard_logits, value

    # ── Inference helpers ──────────────────────────────────────────────────

    @torch.no_grad()
    def predict_action(
        self,
        card_features: torch.Tensor,
        context_features: torch.Tensor,
        valid_actions: List[bool],
        min_raise: int,
        max_raise: int,
    ) -> Tuple[int, int, int, int]:
        """
        Run inference and return a tournament-format action tuple.

        valid_actions: list of 5 bools [fold, raise, check, call, discard]
        Returns: (action_type, raise_amount, keep_card_1, keep_card_2)
          - action_type: 0=fold 1=raise 2=check 3=call 4=discard
          - raise_amount: chips (only meaningful if action_type==1)
          - keep_card_1, keep_card_2: card indices 0-4 (only for discard)
        """
        self.eval()
        if card_features.dim() == 1:
            card_features = card_features.unsqueeze(0)
        if context_features.dim() == 1:
            context_features = context_features.unsqueeze(0)

        action_logits, raise_amount, discard_logits, _ = self(
            card_features, context_features
        )

        # ── Handle discard phase ───────────────────────────────────────────
        if valid_actions[4]:  # DISCARD
            idx = discard_logits[0].argmax().item()
            k1, k2 = KEEP_COMBOS[idx]
            return (4, 0, k1, k2)

        # ── Normal betting action ──────────────────────────────────────────
        # Mask invalid actions: valid_actions[0..3] = [fold, raise, check, call]
        logits = action_logits[0].clone()
        for i, valid in enumerate(valid_actions[:4]):
            if not valid:
                logits[i] = -1e9

        action_type = logits.argmax().item()

        raise_amt = 0
        if action_type == 1:  # RAISE
            frac = raise_amount[0].item()
            raise_amt = int(round(min_raise + frac * (max_raise - min_raise)))
            raise_amt = max(min_raise, min(max_raise, raise_amt))

        return (action_type, raise_amt, 0, 0)

    @torch.no_grad()
    def predict_discard(
        self,
        card_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> Tuple[int, int]:
        """Return (keep_card_1, keep_card_2) indices into the 5-card hand."""
        self.eval()
        if card_features.dim() == 1:
            card_features = card_features.unsqueeze(0)
        if context_features.dim() == 1:
            context_features = context_features.unsqueeze(0)
        _, _, discard_logits, _ = self(card_features, context_features)
        idx = discard_logits[0].argmax().item()
        return KEEP_COMBOS[idx]

    # ── Checkpoint helpers ─────────────────────────────────────────────────

    def strip_value_head(self) -> dict:
        """Return state_dict with value_head removed (for tournament export)."""
        sd = self.state_dict()
        return {k: v for k, v in sd.items() if not k.startswith("value_head.")}

    @classmethod
    def from_bc_checkpoint(
        cls,
        path: str,
        card_dim: int = 214,
        context_dim: int = 63,
        hidden_dim: int = 256,
        num_residual_blocks: int = 3,
        strict: bool = False,
    ) -> "PokerNetV2":
        """
        Load a PokerNetV2 checkpoint trained in BC phase.
        The value_head will be randomly initialized if not present.
        """
        model = cls(
            card_dim=card_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_residual_blocks=num_residual_blocks,
        )
        state = torch.load(path, map_location="cpu")
        # Handle both raw state_dict and {"model_state_dict": ...} formats
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing:
            print(f"[PokerNetV2] Missing keys (will use random init): {missing}")
        if unexpected:
            print(f"[PokerNetV2] Unexpected keys (ignored): {unexpected}")
        return model

    def save(self, path: str, extra: Optional[dict] = None):
        """Save model weights + optional metadata dict."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {"model_state_dict": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[PokerNetV2] Saved → {path}")

    def save_for_tournament(self, path: str):
        """Save stripped weights (no value_head) for the bot/ submission."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"model_state_dict": self.strip_value_head()}, path)
        print(f"[PokerNetV2] Tournament export → {path}")
