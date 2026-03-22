"""
Optional Equity Net for v14 Alpha — load and predict; trained on 2M npz.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

PAD_CARD_ID = 27
MAX_BOARD_CARDS = 5
MAX_DEAD_CARDS = 6
CARD_TOKEN_SLOTS = 2 + MAX_BOARD_CARDS + MAX_DEAD_CARDS

DEFAULT_EQUITY_NET_PATH = Path(__file__).resolve().parent / "data" / "equity_net.pth"


@dataclass(frozen=True)
class EquityNetConfig:
    card_embedding_dim: int = 32
    hidden_dims: tuple = (512, 256, 128)
    layer_norm_indices: tuple = (0, 1)

    @classmethod
    def from_dict(cls, raw: dict) -> "EquityNetConfig":
        return cls(
            card_embedding_dim=int(raw.get("card_embedding_dim", 32)),
            hidden_dims=tuple(int(d) for d in raw.get("hidden_dims", (512, 256, 128))),
            layer_norm_indices=tuple(int(i) for i in raw.get("layer_norm_indices", (0, 1))),
        )


LEGACY_CONFIG = EquityNetConfig(card_embedding_dim=8, hidden_dims=(128, 64), layer_norm_indices=(0,))


def _resolve_config(payload) -> EquityNetConfig:
    if isinstance(payload, dict):
        meta = payload.get("metadata") or {}
        cfg = meta.get("model_config")
        if isinstance(cfg, dict):
            return EquityNetConfig.from_dict(cfg)
    return LEGACY_CONFIG


def _canonical_cards(cards) -> tuple:
    return tuple(sorted(int(c) for c in cards))


if nn is not None:
    class EquityNet(nn.Module):
        def __init__(self, config: EquityNetConfig):
            super().__init__()
            self.config = config
            self.card_embedding = nn.Embedding(PAD_CARD_ID + 1, config.card_embedding_dim)
            input_dim = CARD_TOKEN_SLOTS * config.card_embedding_dim + 6
            layers = []
            d = input_dim
            for i, h in enumerate(config.hidden_dims):
                layers.append(nn.Linear(d, h))
                layers.append(nn.SiLU())
                if i in config.layer_norm_indices:
                    layers.append(nn.LayerNorm(h))
                d = h
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, card_tokens, numeric):
            embedded = self.card_embedding(card_tokens).reshape(card_tokens.shape[0], -1)
            return torch.sigmoid(self.net(torch.cat((embedded, numeric), dim=-1))).squeeze(-1)
else:
    EquityNet = None


class OptionalEquityNetPredictor:
    def __init__(self, weights_path: str | Path | None = None):
        self.available = False
        self.error = None
        self.model = None
        if torch is None or nn is None or np is None:
            self.error = "torch_or_numpy_unavailable"
            return
        path = self._resolve_path(weights_path)
        if path is None:
            self.error = "weights_not_found"
            return
        try:
            payload = torch.load(path, map_location="cpu")
            state = payload.get("model_state", payload)
            config = _resolve_config(payload)
            model = EquityNet(config)
            model.load_state_dict(state)
            model.eval()
            self.model = model
            self.available = True
            self.device = torch.device("cuda" if os.getenv("EQUITY_NET_USE_GPU") == "1" and torch.cuda.is_available() else "cpu")
            if self.device.type == "cuda":
                self.model.to(self.device)
        except Exception as e:
            self.error = str(e)

    def _resolve_path(self, requested) -> Path | None:
        for p in [requested, os.getenv("EQUITY_NET_PATH"), DEFAULT_EQUITY_NET_PATH]:
            if p is None:
                continue
            p = Path(p)
            if p.exists():
                return p
        return None

    def predict(self, hole_cards, board_cards, dead_cards=(), street: int | None = None) -> float | None:
        if not self.available or self.model is None:
            return None
        hole_cards = _canonical_cards(hole_cards)
        board_cards = _canonical_cards(board_cards)
        dead_cards = _canonical_cards(dead_cards)
        if len(hole_cards) != 2 or len(board_cards) < 3 or len(board_cards) > MAX_BOARD_CARDS or len(dead_cards) > MAX_DEAD_CARDS:
            return None
        derived_street = int(street) if street is not None else len(board_cards) - 2
        if derived_street not in (1, 2, 3):
            return None
        card_tokens = np.full(CARD_TOKEN_SLOTS, PAD_CARD_ID, dtype=np.int64)
        ordered = list(hole_cards) + list(board_cards) + list(dead_cards)
        card_tokens[:len(ordered)] = np.asarray(ordered, dtype=np.int64)
        numeric = np.array([
            1.0 if derived_street == 1 else 0.0,
            1.0 if derived_street == 2 else 0.0,
            1.0 if derived_street == 3 else 0.0,
            len(board_cards) / MAX_BOARD_CARDS,
            len(dead_cards) / MAX_DEAD_CARDS,
            len(ordered) / CARD_TOKEN_SLOTS,
        ], dtype=np.float32)
        with torch.no_grad():
            ct = torch.as_tensor(card_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            nt = torch.as_tensor(numeric, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.model(ct, nt)
        return float(out.cpu().item())
