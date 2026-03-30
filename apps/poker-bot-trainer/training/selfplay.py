"""
Self-play training loop for the mutated bot.

Runs genesisV2 (or the current best clone) against itself for N hands,
collects all decisions as GameState dicts, merges with accumulated human
session data, trains a new PokerCloneNet, saves the checkpoint, and
hot-swaps the live bot_server to the new model.

Usage:
    # Single generation (from training/ directory):
    python selfplay.py

    # Multiple generations in sequence:
    python selfplay.py --generations 5

    # Specify hands per generation and custom server:
    python selfplay.py --hands 1000 --generations 3 --server http://127.0.0.1:8765

    # Merge a human session export and retrain immediately:
    python selfplay.py --import-human path/to/session_export.json

    # Dry-run: collect data but do not train or hot-swap:
    python selfplay.py --collect-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # apps/poker-bot-trainer/training/
_TRAINER_ROOT = _HERE.parent                      # apps/poker-bot-trainer/
_REPO_ROOT = _TRAINER_ROOT.parent.parent          # poker-engine-2026/

for p in [str(_REPO_ROOT), str(_TRAINER_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import requests
import torch

from extract_features import process_session_data
from model import PokerCloneNet
from submission.genesisV2 import GenesisV2Agent

# ── Paths ─────────────────────────────────────────────────────────────────────
LINEAGE_PATH = _HERE / "lineage.json"
MODELS_DIR = _HERE / "models"
DATA_DIR = _HERE / "data"
ACCUMULATED_PATH = DATA_DIR / "accumulated_session.json"

MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ── Card encoding (mirrors gym_env.py) ────────────────────────────────────────
NUM_RANKS = 9
NUM_SUITS = 3
DECK_SIZE = NUM_RANKS * NUM_SUITS  # 27

KEEP_COMBOS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
STRAIGHT_WINDOWS = [
    [8, 0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8],
]

SMALL_BLIND = 1
BIG_BLIND = 2
MAX_BET = 100
MAX_RAISES_PER_STREET = 4


# ── Minimal Python poker engine (mirrors TypeScript game.ts / gym_env.py) ─────

def _cr(c): return c % NUM_RANKS
def _cs(c): return c // NUM_RANKS


def _eval5(cards):
    from collections import Counter
    ranks = [_cr(c) for c in cards]
    suits = [_cs(c) for c in cards]
    rc = Counter(ranks)
    counts = sorted(rc.values(), reverse=True)
    is_f = len(set(suits)) == 1
    rs = set(ranks)
    is_s, si = False, -1
    for i in range(len(STRAIGHT_WINDOWS) - 1, -1, -1):
        if all(r in rs for r in STRAIGHT_WINDOWS[i]):
            is_s, si = True, i
            break
    if is_f and is_s:
        return 8, si / (len(STRAIGHT_WINDOWS) - 1)
    if counts[0] == 3 and len(counts) > 1 and counts[1] == 2:
        t = [r for r, c in rc.items() if c == 3][0]
        p = [r for r, c in rc.items() if c == 2][0]
        return 7, (t * NUM_RANKS + p) / (NUM_RANKS ** 2)
    if is_f:
        sr = sorted(ranks, reverse=True)
        return 6, sum(r / NUM_RANKS ** (i + 1) for i, r in enumerate(sr)) / NUM_RANKS
    if counts[0] == 3:
        return 5, [r for r, c in rc.items() if c == 3][0] / (NUM_RANKS - 1)
    if is_s:
        return 4, si / (len(STRAIGHT_WINDOWS) - 1)
    if counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        ps = sorted([r for r, c in rc.items() if c == 2], reverse=True)
        k = [r for r, c in rc.items() if c == 1][0]
        return 3, (ps[0] * NUM_RANKS ** 2 + ps[1] * NUM_RANKS + k) / (NUM_RANKS ** 3)
    if counts[0] == 2:
        p = [r for r, c in rc.items() if c == 2][0]
        ks = sorted([r for r, c in rc.items() if c == 1], reverse=True)
        return 2, (p * NUM_RANKS ** 2 + ks[0] * NUM_RANKS + (ks[1] if len(ks) > 1 else 0)) / (NUM_RANKS ** 3)
    sr = sorted(ranks, reverse=True)
    return 1, sum(r / NUM_RANKS ** (i + 1) for i, r in enumerate(sr)) / NUM_RANKS


def _eval_best(hole, comm):
    all_c = [c for c in (hole + comm) if c >= 0]
    if len(all_c) < 5:
        return 1, 0.0
    best = (0, 0.0)
    for combo in combinations(all_c, 5):
        r, tb = _eval5(list(combo))
        if r > best[0] or (r == best[0] and tb > best[1]):
            best = (r, tb)
    return best


class PokerHand:
    """Minimal stateful poker hand engine matching gym_env.py semantics."""

    def __init__(self, hand_number: int, small_blind_player: int = 0):
        # Deal
        deck = list(range(DECK_SIZE))
        random.shuffle(deck)
        self.hand_number = hand_number
        self.small_blind_player = small_blind_player
        self.big_blind_player = 1 - small_blind_player

        # 5 hole cards each + 5 community
        self.hole = [deck[0:5], deck[5:10]]
        self.community_all = deck[10:15]

        # Game state
        self.street = 0
        self.bets = [0, 0]
        self.discarded = [[], []]
        self.discard_done = [False, False]
        self.community_visible: list[int] = []
        self.terminated = False
        self.winner = None  # 0 or 1 or -1 (tie)
        self.raises_this_street = 0

        # Post blinds
        sb, bb = self.small_blind_player, self.big_blind_player
        self.bets[sb] = SMALL_BLIND
        self.bets[bb] = BIG_BLIND

        # Preflop: SB acts first
        self.acting_agent = self.small_blind_player

        # Track who acted last (for BB option on preflop)
        self._last_aggressor = self.big_blind_player
        self._street_actions = 0
        self._street_acted = [False, False]

    def _current_cards(self, player_idx: int) -> list[int]:
        if self.discard_done[player_idx]:
            kept = self.discarded[player_idx]  # actually kept indices
            # After discard: hole contains only kept cards (2 cards)
            return self._kept_cards[player_idx]
        return self.hole[player_idx]

    @property
    def _cards(self):
        """Return current hole cards for each player (post-discard if applicable)."""
        result = []
        for i in range(2):
            if self.discard_done[i] and hasattr(self, '_kept_cards'):
                result.append(self._kept_cards[i])
            else:
                result.append(self.hole[i])
        return result

    def _get_valid_actions(self, player_idx: int) -> dict:
        """Mirror gym_env._get_valid_actions()."""
        if self.street == 1 and not self.discard_done[player_idx]:
            return {"fold": False, "raise": False, "check": False, "call": False, "discard": True}

        opponent = 1 - player_idx
        can_raise = max(self.bets) < MAX_BET and self.raises_this_street < MAX_RAISES_PER_STREET
        can_check = self.bets[player_idx] == self.bets[opponent]
        can_call = self.bets[player_idx] < self.bets[opponent]
        return {
            "fold": True,
            "raise": can_raise,
            "check": can_check,
            "call": can_call,
            "discard": False,
        }

    def _min_raise(self) -> int:
        return max(1, max(self.bets) - min(self.bets) + 1) if max(self.bets) > 0 else BIG_BLIND

    def _max_raise(self) -> int:
        return MAX_BET - max(self.bets)

    def to_obs(self, player_idx: int) -> dict:
        """Build gym_env-style observation for a player."""
        opp_idx = 1 - player_idx
        cards = self._cards[player_idx] if hasattr(self, '_kept_cards') or not self.discard_done[player_idx] else self.hole[player_idx]
        # Properly handle post-discard cards
        if self.discard_done[player_idx] and hasattr(self, '_kept_cards'):
            my_cards = self._kept_cards[player_idx]
        else:
            my_cards = self.hole[player_idx]

        def pad(arr, n): return arr + [-1] * max(0, n - len(arr))

        valid = self._get_valid_actions(player_idx)
        min_r = self._min_raise()
        max_r = self._max_raise()

        opp_disc = self._kept_cards[opp_idx] if (self.discard_done[opp_idx] and hasattr(self, '_kept_cards')) else []
        # opp_discarded in gym_env = the 3 cards opponent threw away
        if self.discard_done[opp_idx] and hasattr(self, '_disc_cards'):
            opp_discarded = self._disc_cards[opp_idx]
        else:
            opp_discarded = []

        return {
            "street": self.street,
            "acting_agent": self.acting_agent,
            "my_cards": pad(my_cards, 5),
            "community_cards": pad(self.community_visible, 5),
            "my_bet": self.bets[player_idx],
            "my_discarded_cards": pad([], 3),  # my own discards not needed for obs
            "opp_bet": self.bets[opp_idx],
            "opp_discarded_cards": pad(opp_discarded, 3),
            "min_raise": min_r,
            "max_raise": max_r,
            "valid_actions": [
                1 if valid["fold"] else 0,
                1 if valid["raise"] else 0,
                1 if valid["check"] else 0,
                1 if valid["call"] else 0,
                1 if valid["discard"] else 0,
            ],
            "pot_size": self.bets[0] + self.bets[1],
            "blind_position": 0 if player_idx == self.small_blind_player else 1,
            "time_used": 0,
            "time_left": 1000,
            "opp_last_action": "",
        }

    def apply_action(self, player_idx: int, action_tuple: tuple) -> None:
        """Apply a (type, raise_amt, k1, k2) tuple."""
        atype, raise_amt, k1, k2 = action_tuple
        opp_idx = 1 - player_idx

        if atype == 0:  # FOLD
            self.terminated = True
            self.winner = opp_idx
            return

        if atype == 4:  # DISCARD
            # k1, k2 are indices 0-4 into hole cards to KEEP
            kept = sorted([k1, k2])
            all_idx = list(range(5))
            disc_idx = [i for i in all_idx if i not in kept]
            if not hasattr(self, '_kept_cards'):
                self._kept_cards = [None, None]
                self._disc_cards = [None, None]
            self._kept_cards[player_idx] = [self.hole[player_idx][i] for i in kept]
            self._disc_cards[player_idx] = [self.hole[player_idx][i] for i in disc_idx]
            self.discard_done[player_idx] = True

            # Advance to betting if both discarded
            if all(self.discard_done):
                self.street = 2  # turn
                self.community_visible = self.community_all[:4]
                self.raises_this_street = 0
                self._street_acted = [False, False]
                # After discard: BB just acted → SB acts first in post-flop
                self.acting_agent = self.small_blind_player
            else:
                # BB discards first, then SB
                self.acting_agent = self.small_blind_player
            return

        if atype == 1:  # RAISE
            additional = max(self._min_raise(), min(raise_amt, self._max_raise()))
            self.bets[player_idx] = max(self.bets) + additional
            self.raises_this_street += 1
            self._last_aggressor = player_idx
            self._street_acted[player_idx] = True
            self.acting_agent = opp_idx
            return

        if atype == 3:  # CALL
            self.bets[player_idx] = min(self.bets[opp_idx], MAX_BET)
            self._street_acted[player_idx] = True

        if atype == 2:  # CHECK
            self._street_acted[player_idx] = True

        # Check if street is over
        self._maybe_advance_street(player_idx)

    def _maybe_advance_street(self, last_actor: int) -> None:
        """Advance street if both players have had a chance to act."""
        opp = 1 - last_actor
        both_equal = self.bets[0] == self.bets[1]
        both_acted = self._street_acted[0] and self._street_acted[1]

        if self.street == 0:
            # Preflop: BB gets option. Street ends when both acted AND bets equal.
            if both_equal and both_acted:
                self._start_street1()
        else:
            if both_equal and both_acted:
                self._advance_postflop()

    def _start_street1(self):
        """Flop: deal 3 community cards, enter discard phase."""
        self.street = 1
        self.community_visible = self.community_all[:3]
        self.raises_this_street = 0
        self._street_acted = [False, False]
        # BB discards first on street 1
        self.acting_agent = self.big_blind_player

    def _advance_postflop(self):
        """Turn → River → Showdown."""
        if self.street == 2:
            self.street = 3
            self.community_visible = self.community_all[:5]
            self.raises_this_street = 0
            self._street_acted = [False, False]
            self.acting_agent = self.small_blind_player
        elif self.street == 3:
            self._showdown()

    def _showdown(self):
        self.terminated = True
        h0 = self._kept_cards[0] if (hasattr(self, '_kept_cards') and self._kept_cards[0]) else self.hole[0]
        h1 = self._kept_cards[1] if (hasattr(self, '_kept_cards') and self._kept_cards[1]) else self.hole[1]
        r0, tb0 = _eval_best(h0, self.community_all)
        r1, tb1 = _eval_best(h1, self.community_all)
        if r0 > r1 or (r0 == r1 and tb0 > tb1):
            self.winner = 0
        elif r1 > r0 or (r1 == r0 and tb1 > tb0):
            self.winner = 1
        else:
            self.winner = -1  # tie


# ── Observation → GameState dict ───────────────────────────────────────────────

def _obs_to_gamestate(
    obs: dict,
    player_idx: int,
    player_label: str,
    hand: PokerHand,
    action_history: list,
    action_tuple: tuple,
) -> dict:
    """Build a GameState dict (matches TypeScript schema) from a gym obs + context."""
    atype, raise_amt, k1, k2 = action_tuple
    action_labels = {0: "fold", 1: "raise", 2: "check", 3: "call", 4: "discard"}
    action_taken: dict[str, Any]
    if atype == 4:
        action_taken = {"type": "discard", "kept_cards": [k1, k2]}
    elif atype == 1:
        action_taken = {"type": "raise", "raise_amount": raise_amt}
    else:
        action_taken = {"type": action_labels.get(atype, "fold")}

    opp_idx = 1 - player_idx
    return {
        "player": player_label,
        "hand_number": hand.hand_number,
        "street": obs["street"],
        "position": "SB" if player_idx == hand.small_blind_player else "BB",
        "my_cards": obs["my_cards"],
        "community_cards": obs["community_cards"],
        "my_discarded": [],
        "opp_discarded": obs["opp_discarded_cards"],
        "pot_size": obs["pot_size"],
        "my_bet": obs["my_bet"],
        "opp_bet": obs["opp_bet"],
        "my_stack": MAX_BET - obs["my_bet"],
        "opp_stack": MAX_BET - obs["opp_bet"],
        "min_raise": obs["min_raise"],
        "max_raise": obs["max_raise"],
        "action_history": list(action_history),
        "action_taken": action_taken,
    }


# ── Self-play engine ───────────────────────────────────────────────────────────

def run_self_play(
    agent0,
    agent1,
    num_hands: int,
    label0: str = "bot_a",
    label1: str = "bot_b",
    verbose: bool = False,
) -> list[dict]:
    """Run `num_hands` hands between agent0 and agent1.

    Returns a list of GameState dicts (one per decision point, for both agents).
    """
    records: list[dict] = []
    agents = [agent0, agent1]
    labels = [label0, label1]

    for hand_num in range(1, num_hands + 1):
        # Alternate who is SB (index 0 is always SB structurally, but we swap agents)
        sb_player = (hand_num - 1) % 2
        hand = PokerHand(hand_number=hand_num, small_blind_player=sb_player)
        action_history: list[dict] = []
        hand_records: list[dict] = []

        max_actions = 40  # safety valve
        action_count = 0

        while not hand.terminated and action_count < max_actions:
            actor = hand.acting_agent
            obs = hand.to_obs(actor)
            agent = agents[actor]

            # Build action tuple from agent
            try:
                result = agent.act(obs, 0.0, False, False, {"hand_number": hand_num})
            except Exception as e:
                if verbose:
                    print(f"Agent {actor} error: {e}")
                result = (0, 0, 0, 1)  # fold on error

            action_tuple = result if isinstance(result, tuple) else tuple(result)
            if len(action_tuple) < 4:
                action_tuple = (action_tuple[0], 0, 0, 1)

            # Capture decision BEFORE applying
            snap = _obs_to_gamestate(obs, actor, labels[actor], hand, action_history, action_tuple)
            hand_records.append(snap)

            # Record action to history
            atype = action_tuple[0]
            action_labels = {0: "fold", 1: "raise", 2: "check", 3: "call", 4: "discard"}
            history_entry: dict = {
                "player": labels[actor],
                "street": hand.street,
                "type": action_labels.get(atype, "fold"),
            }
            if atype == 1:
                history_entry["amount"] = action_tuple[1]
            action_history.append(history_entry)

            hand.apply_action(actor, action_tuple)
            action_count += 1

        # Compute hand result
        winner = hand.winner
        bets = hand.bets
        # Chips won/lost from each player's perspective
        if winner == -1:
            results = [{}, {}]
            for i in range(2):
                results[i] = {"won": False, "chips_won_lost": 0, "showdown": True}
        else:
            results = [{}, {}]
            for i in range(2):
                if winner == i:
                    profit = bets[1 - i]  # won opponent's bet
                    results[i] = {"won": True, "chips_won_lost": profit, "showdown": hand.street >= 3}
                else:
                    results[i] = {"won": False, "chips_won_lost": -bets[i], "showdown": hand.street >= 3}

        # Attach result to each record
        actor_to_result = {labels[0]: results[0], labels[1]: results[1]}
        for rec in hand_records:
            rec["hand_result"] = actor_to_result.get(rec["player"], results[0])

        records.extend(hand_records)

        if verbose and hand_num % 50 == 0:
            print(f"  Hand {hand_num}/{num_hands} complete (total records: {len(records)})")

    return records


# ── Data accumulation ──────────────────────────────────────────────────────────

def load_accumulated() -> list[dict]:
    if ACCUMULATED_PATH.exists():
        try:
            return json.loads(ACCUMULATED_PATH.read_text())
        except Exception:
            return []
    return []


def save_accumulated(data: list[dict]) -> None:
    ACCUMULATED_PATH.write_text(json.dumps(data, indent=2))


def merge_human_session(human_path: str, weight: float = 3.0) -> int:
    """Load a human session export and add it to accumulated data (with repetition for weighting)."""
    human_path_p = Path(human_path)
    if not human_path_p.exists():
        print(f"ERROR: File not found: {human_path}")
        return 0

    raw = json.loads(human_path_p.read_text())
    records = raw if isinstance(raw, list) else raw.get("decisions", [])

    # Tag all records as human player
    for r in records:
        if "player" not in r:
            r["player"] = "human"

    # Weight human data by repeating records
    repeat = max(1, int(weight))
    weighted = records * repeat

    accumulated = load_accumulated()
    accumulated.extend(weighted)
    save_accumulated(accumulated)
    print(f"Imported {len(records)} human records → accumulated ({len(records)*repeat} weighted copies). Total: {len(accumulated)}")
    return len(records)


# ── Training ───────────────────────────────────────────────────────────────────

def train_on_accumulated(config: dict, out_path: str) -> float:
    """Train PokerCloneNet on accumulated_session.json. Returns best val loss."""
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    accumulated = load_accumulated()
    if len(accumulated) < config.get("min_samples_to_train", 200):
        print(f"Not enough data ({len(accumulated)} samples, need {config.get('min_samples_to_train', 200)}). Skipping training.")
        return float("inf")

    print(f"Training on {len(accumulated)} decision points...")
    b_feats, b_labels, r_buckets, d_feats, d_labels, *_ = process_session_data(accumulated)
    print(f"  Betting: {len(b_labels)}, Discard: {len(d_labels)}")

    if len(b_labels) < 10:
        print("Too few betting samples. Skipping.")
        return float("inf")

    epochs = config.get("epochs", 40)
    lr = config.get("lr", 0.001)
    batch_size = config.get("batch_size", 64)
    hidden_dim = config.get("hidden_dim", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PokerCloneNet(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    B = torch.tensor(b_feats, dtype=torch.float32)
    BL = torch.tensor(b_labels, dtype=torch.long)
    BR = torch.tensor(r_buckets, dtype=torch.long)

    n_val = max(1, int(len(B) * 0.1))
    n_train = len(B) - n_val
    train_ds, val_ds = random_split(TensorDataset(B, BL, BR), [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    D_feats = torch.tensor(d_feats, dtype=torch.float32) if len(d_labels) > 0 else None
    D_labels_t = torch.tensor(d_labels, dtype=torch.long) if len(d_labels) > 0 else None

    best_val = float("inf")
    patience = 10
    counter = 0

    for epoch in range(epochs):
        model.train()
        for bx, ba, br in train_loader:
            bx, ba, br = bx.to(device), ba.to(device), br.to(device)
            al, rl, _ = model(bx)
            loss = ce(al, ba) + 0.3 * ce(rl, br)
            if D_feats is not None:
                idx = torch.randint(len(D_labels_t), (min(batch_size, len(D_labels_t)),))
                dl = model(D_feats[idx].to(device))[2]
                loss += 0.2 * ce(dl, D_labels_t[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        vt = 0
        with torch.no_grad():
            for bx, ba, br in val_loader:
                bx, ba, br = bx.to(device), ba.to(device), br.to(device)
                al, rl, _ = model(bx)
                val_loss += (ce(al, ba) + 0.3 * ce(rl, br)).item() * len(bx)
                vt += len(bx)
        avg_val = val_loss / max(vt, 1)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Val loss: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), out_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  Best val loss: {best_val:.4f} → saved to {out_path}")
    return best_val


# ── Hot-swap ───────────────────────────────────────────────────────────────────

def hot_swap(model_path: str, server_url: str, hidden_dim: int = 256) -> bool:
    """Tell bot_server to swap to a new model checkpoint."""
    try:
        resp = requests.post(
            f"{server_url}/hot_swap",
            json={"model_path": model_path, "hidden_dim": hidden_dim},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            if data.get("ok"):
                print(f"Hot-swap OK: {data.get('model')} (gen {data.get('generation')})")
                return True
        print(f"Hot-swap failed: {resp.text}")
    except requests.exceptions.ConnectionError:
        print("Bot server not running — skipping hot-swap (model saved, will load on next server start)")
    return False


# ── Lineage management ─────────────────────────────────────────────────────────

def load_lineage() -> dict:
    if LINEAGE_PATH.exists():
        return json.loads(LINEAGE_PATH.read_text())
    return {
        "current_generation": 0,
        "config": {"hands_per_gen": 500, "epochs": 40, "lr": 0.001, "hidden_dim": 256, "min_samples_to_train": 200},
        "models": [{"gen": 0, "type": "genesis", "path": None, "hands": 0, "trained_at": None}],
    }


def save_lineage(lineage: dict) -> None:
    LINEAGE_PATH.write_text(json.dumps(lineage, indent=2))


def record_generation(lineage: dict, gen: int, model_path: str, hands: int, val_loss: float) -> None:
    entry = {
        "gen": gen,
        "type": "clone",
        "path": model_path,
        "hands": hands,
        "trained_at": datetime.utcnow().isoformat(),
        "val_loss": round(val_loss, 5),
    }
    # Remove existing entry for this gen if present
    lineage["models"] = [m for m in lineage["models"] if m["gen"] != gen]
    lineage["models"].append(entry)
    lineage["current_generation"] = gen
    save_lineage(lineage)


# ── Main generation loop ───────────────────────────────────────────────────────

def run_generation(
    gen: int,
    lineage: dict,
    server_url: str,
    collect_only: bool = False,
    verbose: bool = False,
) -> None:
    config = lineage.get("config", {})
    hands = config.get("hands_per_gen", 500)
    hidden_dim = config.get("hidden_dim", 256)

    print(f"\n{'='*60}")
    print(f"Generation {gen} — {hands} self-play hands")
    print(f"{'='*60}")

    # Load current best agent
    models = sorted([m for m in lineage["models"] if m.get("path")], key=lambda x: x["gen"], reverse=True)
    if models:
        latest = models[0]
        print(f"Loading agent from gen {latest['gen']}: {latest['path']}")
        from bot_server import CloneBotAdapter
        agent0 = CloneBotAdapter(latest["path"], hidden_dim=hidden_dim)
        agent1 = CloneBotAdapter(latest["path"], hidden_dim=hidden_dim)
        label = f"clone_v{latest['gen']}"
    else:
        print("Using genesisV2 baseline for self-play")
        agent0 = GenesisV2Agent(stream=False)
        agent1 = GenesisV2Agent(stream=False)
        label = "genesis"

    # Run self-play
    print(f"Running {hands} hands of self-play...")
    t0 = time.time()
    new_records = run_self_play(agent0, agent1, hands, label0=f"{label}_a", label1=f"{label}_b", verbose=verbose)
    elapsed = time.time() - t0
    print(f"Self-play done: {len(new_records)} records in {elapsed:.1f}s")

    # Merge into accumulated data
    accumulated = load_accumulated()
    accumulated.extend(new_records)
    save_accumulated(accumulated)
    print(f"Accumulated data: {len(accumulated)} total records")

    if collect_only:
        print("--collect-only: skipping training and hot-swap")
        return

    # Train
    out_path = str(MODELS_DIR / f"clone_v{gen}.pt")
    val_loss = train_on_accumulated(config, out_path)

    if val_loss == float("inf"):
        print("Training skipped — not enough data yet")
        return

    # Update lineage
    record_generation(lineage, gen, out_path, len(new_records), val_loss)

    # Hot-swap
    hot_swap(out_path, server_url, hidden_dim=hidden_dim)

    print(f"\nGeneration {gen} complete!")
    print(f"  Model: {out_path}")
    print(f"  Val loss: {val_loss:.4f}")
    print(f"  Total accumulated: {len(accumulated)} records")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Poker bot self-play training loop")
    p.add_argument("--generations", type=int, default=1, help="Number of generations to run")
    p.add_argument("--hands", type=int, default=None, help="Override hands per generation")
    p.add_argument("--server", default="http://127.0.0.1:8765", help="bot_server URL")
    p.add_argument("--collect-only", action="store_true", help="Collect data but skip training/hot-swap")
    p.add_argument("--import-human", metavar="PATH", help="Import a human session JSON and retrain")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    lineage = load_lineage()

    # Handle human data import first
    if args.import_human:
        weight = lineage.get("config", {}).get("human_weight", 3.0)
        count = merge_human_session(args.import_human, weight=weight)
        if count == 0:
            return

        # Retrain immediately on updated accumulated data
        config = lineage.get("config", {})
        gen = lineage.get("current_generation", 0) + 1
        out_path = str(MODELS_DIR / f"clone_v{gen}.pt")
        val_loss = train_on_accumulated(config, out_path)
        if val_loss < float("inf"):
            record_generation(lineage, gen, out_path, count, val_loss)
            hot_swap(out_path, args.server, hidden_dim=config.get("hidden_dim", 256))
        return

    # Override config if provided
    if args.hands:
        lineage.setdefault("config", {})["hands_per_gen"] = args.hands

    start_gen = lineage.get("current_generation", 0) + 1

    for i in range(args.generations):
        gen = start_gen + i
        run_generation(
            gen=gen,
            lineage=lineage,
            server_url=args.server,
            collect_only=args.collect_only,
            verbose=args.verbose,
        )
        # Reload lineage after each generation (in case of concurrent writes)
        lineage = load_lineage()

    print("\nAll generations complete.")
    print(f"Lineage: {lineage['current_generation']} generations, {len(lineage['models'])} models")


if __name__ == "__main__":
    main()
