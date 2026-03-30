"""
Background watcher — monitors accumulated_session.json for new hands
and automatically retrains + hot-swaps the bot every N new hands.

Run alongside bot_server.py while you play in the browser:

    Terminal 1:  python training/bot_server.py
    Terminal 2:  python training/watch_and_train.py
    Browser:     http://localhost:3000

Every time you finish N hands, the watcher:
  1. Trains a new PokerCloneNet on all accumulated data
  2. Saves the checkpoint to models/clone_vN.pt
  3. Hot-swaps the live bot_server to the new model
  4. The bot playing against you is now smarter

Options:
    --retrain-every   Retrain after this many new hands (default: 10)
    --mode            'bc' (behavioral cloning, fast) or 'rl' (PPO, slower)
    --min-hands       Minimum total hands before first training (default: 20)
    --server          bot_server URL (default: http://127.0.0.1:8765)
    --poll            Polling interval in seconds (default: 5)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_TRAINER_ROOT = _HERE.parent
_REPO_ROOT = _TRAINER_ROOT.parent.parent
for p in [str(_REPO_ROOT), str(_TRAINER_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

import torch

from extract_features import process_session_data, invalidate_opp_feature_cache
from model import PokerCloneNet
from exploit_profile import build_profile
from datetime import datetime

ACCUMULATED_PATH = _HERE / "data" / "accumulated_session.json"
MODELS_DIR = _HERE / "models"
LINEAGE_PATH = _HERE / "lineage.json"
EXPLOIT_PROFILE_PATH = _HERE / "exploit_profile.json"
MODELS_DIR.mkdir(exist_ok=True)


# ── Lineage helpers (inline to avoid circular imports) ────────────────────────

def _load_lineage() -> dict:
    if LINEAGE_PATH.exists():
        try:
            return json.loads(LINEAGE_PATH.read_text())
        except Exception:
            pass
    return {"current_generation": 0, "config": {}, "models": []}


def _save_lineage(lineage: dict) -> None:
    LINEAGE_PATH.write_text(json.dumps(lineage, indent=2))


def _next_gen(lineage: dict) -> int:
    gens = [m.get("gen", 0) for m in lineage.get("models", [])]
    return max(gens, default=0) + 1


# ── Accumulated data helpers ───────────────────────────────────────────────────

def _load_records() -> list[dict]:
    if not ACCUMULATED_PATH.exists():
        return []
    try:
        return json.loads(ACCUMULATED_PATH.read_text())
    except Exception:
        return []


def _count_unique_hands(records: list[dict]) -> int:
    """Count unique hand numbers in the records (deduplicated)."""
    return len(set(r.get("hand_number", 0) for r in records))


# ── Quick BC training ──────────────────────────────────────────────────────────

def train_bc(records: list[dict], gen: int, hidden_dim: int = 256, epochs: int = 30) -> str | None:
    """
    Train a PokerCloneNet on the given records using outcome-weighted loss.

    Samples are weighted by outcome_weight (set at save time):
      - Human winning actions: high weight (imitate good play)
      - Human losing actions: low weight (don't copy mistakes)
      - Bot winning actions:  high weight (what actually beats the human)
      - Bot losing actions:   low weight

    Discard labels use the objectively optimal keep-combo when final board is known.
    Returns path to saved checkpoint, or None if not enough data.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    b_feats, b_labels, r_buckets, d_feats, d_labels, b_weights, d_weights = process_session_data(records)

    if len(b_labels) < 30:
        print(f"  Not enough betting samples ({len(b_labels)}) — skipping training")
        return None

    # Log weight distribution so we can see how much is win vs loss data
    if len(b_weights) > 0:
        high = (b_weights > 1.0).sum()
        low  = (b_weights < 0.5).sum()
        print(f"  Samples: {len(b_labels)} betting ({high} high-weight winning, {low} low-weight losing), {len(d_labels)} discard")
    else:
        print(f"  Samples: {len(b_labels)} betting, {len(d_labels)} discard")

    device = torch.device("cpu")
    model = PokerCloneNet(hidden_dim=hidden_dim).to(device)

    # Warm-start from previous generation if available
    prev_models = sorted(MODELS_DIR.glob("clone_v*.pt"),
                         key=lambda p: int(p.stem.split("v")[1]))
    if prev_models:
        try:
            state = torch.load(str(prev_models[-1]), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            print(f"  Warm-started from {prev_models[-1].name}")
        except Exception:
            pass

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_none = nn.CrossEntropyLoss(reduction="none")  # per-sample loss for weighting

    B  = torch.tensor(b_feats, dtype=torch.float32)
    BL = torch.tensor(b_labels, dtype=torch.long)
    BR = torch.tensor(r_buckets, dtype=torch.long)
    BW = torch.tensor(b_weights if len(b_weights) == len(b_labels) else
                      [1.0] * len(b_labels), dtype=torch.float32)
    # Normalise weights so mean=1 (keeps LR scale stable)
    BW = BW / (BW.mean() + 1e-8)

    n_val = max(1, int(len(B) * 0.1))
    train_ds, val_ds = random_split(TensorDataset(B, BL, BR, BW), [len(B) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64)

    D_feats    = torch.tensor(d_feats,    dtype=torch.float32) if len(d_labels) > 0 else None
    D_labels_t = torch.tensor(d_labels,   dtype=torch.long)    if len(d_labels) > 0 else None
    D_weights  = torch.tensor(d_weights if len(d_weights) == len(d_labels) else
                              [1.0] * len(d_labels), dtype=torch.float32) if len(d_labels) > 0 else None
    if D_weights is not None:
        D_weights = D_weights / (D_weights.mean() + 1e-8)

    best_val = float("inf")
    patience, counter = 8, 0
    out_path = str(MODELS_DIR / f"clone_v{gen}.pt")

    for epoch in range(epochs):
        model.train()
        for bx, ba, br, bw in train_loader:
            al, rl, _ = model(bx)
            # Outcome-weighted action + raise loss
            loss = (ce_none(al, ba) * bw).mean() + 0.3 * (ce_none(rl, br) * bw).mean()
            if D_feats is not None:
                idx = torch.randint(len(D_labels_t), (min(64, len(D_labels_t)),))
                dw  = D_weights[idx]
                dl  = model(D_feats[idx])[2]
                loss += 0.2 * (ce_none(dl, D_labels_t[idx]) * dw).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, ba, br, bw in val_loader:
                al, rl, _ = model(bx)
                val_loss += ((ce_none(al, ba) + 0.3 * ce_none(rl, br)) * bw).sum().item()
        avg_val = val_loss / max(n_val, 1)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), out_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    print(f"  Trained {epoch+1} epochs, best val loss: {best_val:.4f} -> {out_path}")
    return out_path


# ── Exploit profile rebuild ────────────────────────────────────────────────────

def rebuild_exploit_profile(records: list[dict], server_url: str) -> None:
    """
    Rebuild exploit_profile.json from the latest accumulated records, then
    signal bot_server to reload its ExploitLayer rules and opponent features.
    Also invalidates the local feature cache so next training run picks up
    fresh opponent context features [98-107].
    """
    try:
        profile = build_profile(records)
        EXPLOIT_PROFILE_PATH.write_text(__import__("json").dumps(profile, indent=2))
        invalidate_opp_feature_cache()

        nn = profile.get("nn_opponent_features", {})
        adv = profile.get("advanced", {})
        ep  = profile.get("exploit_params", {})
        print(f"  Exploit profile rebuilt — "
              f"VPIP:{nn.get('vpip',0):.0%} "
              f"FoldRiver:{nn.get('fold_to_river_bet',0):.0%} "
              f"Showdown:{nn.get('showdown_pct',0):.0%} "
              f"CheckRaise:{adv.get('check_raise_pct',0):.0%}")
        print(f"  Rules — river_bet:{ep.get('river_bet_always')} "
              f"pf_raise:{ep.get('preflop_always_raise')} "
              f"trap:{ep.get('detect_river_trap')}")

        if _REQUESTS_OK:
            try:
                resp = requests.post(f"{server_url}/reload_exploit", timeout=5)
                if resp.ok:
                    print(f"  ExploitLayer reloaded on bot_server")
            except Exception:
                pass  # server may not be running; profile file is updated regardless
    except Exception as e:
        print(f"  Exploit profile rebuild failed: {e}")


# ── Hot-swap ───────────────────────────────────────────────────────────────────

def hot_swap(model_path: str, server_url: str, hidden_dim: int = 256) -> bool:
    if not _REQUESTS_OK:
        return False
    try:
        resp = requests.post(
            f"{server_url}/hot_swap",
            json={"model_path": model_path, "hidden_dim": hidden_dim},
            timeout=10,
        )
        if resp.ok and resp.json().get("ok"):
            label = resp.json().get("model", "?")
            print(f"  Hot-swap OK -> bot is now: {label}")
            return True
        print(f"  Hot-swap failed: {resp.text}")
    except requests.exceptions.ConnectionError:
        print("  Bot server not running — model saved but not hot-swapped")
    except Exception as e:
        print(f"  Hot-swap error: {e}")
    return False


def check_server(server_url: str) -> dict | None:
    """Returns server health dict or None if down."""
    if not _REQUESTS_OK:
        return None
    try:
        resp = requests.get(f"{server_url}/health", timeout=2)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return None


# ── Watcher loop ───────────────────────────────────────────────────────────────

def watch(args):
    retrain_every = args.retrain_every
    min_hands = args.min_hands
    poll = args.poll
    server_url = args.server
    mode = args.mode
    hidden_dim = 256

    print(f"Watch & Train started")
    print(f"  Mode:          {mode.upper()} ({'behavioral cloning' if mode == 'bc' else 'PPO reinforcement learning'})")
    print(f"  Retrain every: {retrain_every} new hands")
    print(f"  Min hands:     {min_hands}")
    print(f"  Poll interval: {poll}s")
    print(f"  Bot server:    {server_url}")
    print(f"  Data file:     {ACCUMULATED_PATH}")
    print(f"\nWaiting for hands... (play at http://localhost:3000)\n")

    last_hand_count = _count_unique_hands(_load_records())
    last_train_hand_count = last_hand_count
    gen = _next_gen(_load_lineage())

    # Show server status on startup
    health = check_server(server_url)
    if health:
        print(f"Bot server online: {health.get('model', '?')} (gen {health.get('generation', 0)})")
    else:
        print("Bot server offline — training will still run, hot-swap will happen when server starts")

    while True:
        try:
            time.sleep(poll)

            records = _load_records()
            current_hand_count = _count_unique_hands(records)
            new_hands = current_hand_count - last_train_hand_count

            if current_hand_count != last_hand_count:
                print(f"[{_now()}] Hands played: {current_hand_count} total (+{current_hand_count - last_hand_count} new)")
                last_hand_count = current_hand_count

            # Check if we should retrain
            if new_hands >= retrain_every and current_hand_count >= min_hands:
                print(f"\n[{_now()}] Triggering retrain (gen {gen}) — {new_hands} new hands, {current_hand_count} total, {len(records)} records")

                if mode == "bc":
                    checkpoint = train_bc(records, gen, hidden_dim=hidden_dim)
                else:
                    checkpoint = train_rl_quick(records, gen, server_url, hidden_dim=hidden_dim)

                if checkpoint:
                    hot_swap(checkpoint, server_url, hidden_dim=hidden_dim)

                    # Rebuild exploit profile with latest tendencies and reload bot rules
                    rebuild_exploit_profile(records, server_url)

                    # Update lineage
                    lineage = _load_lineage()
                    lineage["models"] = [m for m in lineage["models"] if m.get("gen") != gen]
                    lineage["models"].append({
                        "gen": gen,
                        "type": f"watch_{mode}",
                        "path": checkpoint,
                        "hands": current_hand_count,
                        "trained_at": datetime.utcnow().isoformat(),
                    })
                    lineage["current_generation"] = gen
                    _save_lineage(lineage)

                    last_train_hand_count = current_hand_count
                    gen += 1
                    print(f"[{_now()}] Done. Next retrain after {retrain_every} more hands.\n")
                else:
                    # Not enough data yet — try again next cycle
                    pass

        except KeyboardInterrupt:
            print("\nWatcher stopped.")
            break
        except Exception as e:
            print(f"[{_now()}] Error: {e}")
            time.sleep(poll)


def train_rl_quick(records: list[dict], gen: int, server_url: str, hidden_dim: int = 256) -> str | None:
    """
    Run a short PPO generation (200 rollout steps) on top of the current BC model.
    Falls back to BC if RL dependencies aren't ready.
    """
    try:
        from rl_train import PPOConfig, OpponentPool, EloTracker, collect_rollout, ppo_update, warm_start
        from model import PokerPolicyNet
        import numpy as np

        cfg = PPOConfig(rollout_steps=500, n_epochs=2, mini_batch_size=64, hidden_dim=hidden_dim)
        model = PokerPolicyNet(hidden_dim=hidden_dim)
        warm_start(model, MODELS_DIR)

        opponent_pool = OpponentPool()
        buffer = collect_rollout(model, opponent_pool, cfg, verbose=False)

        last_feat = buffer.obs[(buffer.ptr - 1) % buffer.max_size]
        with torch.no_grad():
            *_, last_val = model(torch.tensor(last_feat).unsqueeze(0))
        bootstrap = 0.0 if buffer.dones[(buffer.ptr - 1) % buffer.max_size] else last_val[0].item()
        adv, ret = buffer.compute_gae(bootstrap)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        losses = ppo_update(model, optimizer, buffer, adv, ret, cfg)
        print(f"  RL losses: policy={losses['policy_loss']:.4f} value={losses['value_loss']:.4f}")

        out_path = str(MODELS_DIR / f"clone_v{gen}.pt")
        torch.save(model.to_clone_state_dict(), out_path)
        return out_path

    except Exception as e:
        print(f"  RL training failed ({e}), falling back to BC")
        return train_bc(records, gen, hidden_dim=hidden_dim)


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Auto-retrain bot as you play")
    p.add_argument("--retrain-every", type=int, default=10,
                   help="Retrain after this many new hands (default: 10)")
    p.add_argument("--min-hands", type=int, default=20,
                   help="Minimum total hands before first training (default: 20)")
    p.add_argument("--mode", choices=["bc", "rl"], default="bc",
                   help="Training mode: bc=fast behavioral cloning, rl=PPO (default: bc)")
    p.add_argument("--server", default="http://127.0.0.1:8765",
                   help="Bot server URL (default: http://127.0.0.1:8765)")
    p.add_argument("--poll", type=float, default=5.0,
                   help="Poll interval in seconds (default: 5)")
    return p.parse_args()


if __name__ == "__main__":
    watch(parse_args())
