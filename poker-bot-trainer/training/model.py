"""
PokerCloneNet — Multi-head neural network for behavioral cloning.

Architecture:
  Shared trunk: 3x Linear(hidden_dim) + LayerNorm + GELU
  action_head:  → 4 logits  (fold/check/call/raise)
  raise_head:   → 10 logits (raise amount bucket 0-9)
  discard_head: → 10 logits (keep-pair index, C(5,2)=10 combos)

Training loss:
  CE(action) + 0.3 * CE(raise_bucket) + 0.2 * CE(discard)
"""

import torch
import torch.nn as nn

FEATURE_DIM = 98
NUM_ACTIONS = 4      # fold, raise, check, call
NUM_RAISE_BUCKETS = 10
NUM_DISCARD_COMBOS = 10  # C(5,2)


class PokerCloneNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Action type head (fold/check/call/raise)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, NUM_ACTIONS),
        )

        # Raise amount head (10 buckets: 0=min, 9=all-in)
        self.raise_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, NUM_RAISE_BUCKETS),
        )

        # Discard head (which 2-card combo to keep from C(5,2)=10)
        self.discard_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, NUM_DISCARD_COMBOS),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, 98) float tensor

        Returns:
            action_logits: (batch_size, 4)
            raise_logits:  (batch_size, 10)
            discard_logits:(batch_size, 10)
        """
        shared = self.backbone(x)
        return (
            self.action_head(shared),
            self.raise_head(shared),
            self.discard_head(shared),
        )

    def predict_action(self, features: torch.Tensor, valid_actions: list[bool]) -> tuple[int, int]:
        """
        Inference-time action selection with valid action masking.

        Args:
            features: (1, 98) tensor
            valid_actions: [fold, raise, check, call, discard] booleans

        Returns:
            (action_type_int, raise_bucket_or_0)
        """
        self.eval()
        with torch.no_grad():
            action_logits, raise_logits, discard_logits = self(features)

        # Handle discard
        if valid_actions[4]:
            return 4, discard_logits.argmax().item()

        # Mask invalid actions: fold=0, raise=1, check=2, call=3
        probs = torch.softmax(action_logits[0], dim=-1).clone()
        for i in range(4):
            if not valid_actions[i]:
                probs[i] = 0.0

        if probs.sum() == 0:
            # Fallback: fold
            return 0, 0

        action = probs.argmax().item()
        raise_bucket = raise_logits[0].argmax().item() if action == 1 else 0
        return action, raise_bucket

    def predict_discard(self, features: torch.Tensor) -> int:
        """Returns the best keep-combo index (0-9)."""
        self.eval()
        with torch.no_grad():
            _, _, discard_logits = self(features)
        return discard_logits[0].argmax().item()


def raise_bucket_to_amount(bucket: int, min_raise: int, max_raise: int) -> int:
    """Convert a 0-9 bucket index to a chip amount in [min_raise, max_raise]."""
    if max_raise <= min_raise:
        return min_raise
    frac = bucket / 9
    amount = int(min_raise + frac * (max_raise - min_raise))
    return max(min_raise, min(amount, max_raise))


# Keep-pair combos: index → (card_idx_1, card_idx_2) into 5-card hand
KEEP_COMBOS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
