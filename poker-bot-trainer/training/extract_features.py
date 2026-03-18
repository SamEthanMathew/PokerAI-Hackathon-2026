"""
Feature extraction for poker bot behavioral cloning.

CRITICAL: This must exactly match lib/data/features.ts

Feature index specification (98 dimensions, FROZEN):
  [0-26]   hole_cards_onehot       — 27 bits, 1 if card is in my hand (post-discard)
  [27-53]  community_cards_onehot  — 27 bits, 1 if card is visible on board
  [54]     street_preflop          — 1 if street==0
  [55]     street_flop_turn        — 1 if street==1 or 2
  [56]     street_river            — 1 if street==3
  [57]     position_sb             — 1 if SB, 0 if BB
  [58]     hand_rank_norm          — current best hand rank / 8 (0 if <3 community)
  [59]     has_flush_draw          — 1 if 4 suited cards in hand+board
  [60]     has_straight_draw       — 1 if 4 connected cards
  [61]     pair_count              — number of pairs in hole cards / 2
  [62]     has_ace                 — 1 if ace in my hole cards
  [63]     num_hole_cards_norm     — len(my_cards) / 5
  [64]     hand_rank_tiebreak      — tiebreaker [0,1] of best hand
  [65]     pot_norm                — pot_size / 200
  [66]     my_bet_norm             — my_bet / 100
  [67]     opp_bet_norm            — opp_bet / 100
  [68]     call_amount_norm        — (opp_bet - my_bet) / 100
  [69]     pot_odds                — call_amount / (pot + call_amount)
  [70]     min_raise_norm          — min_raise / 100
  [71]     max_raise_norm          — max_raise / 100
  [72]     stack_depth_norm        — my_stack / 100
  [73-82]  action_history          — last 10 actions (val/4), -0.25 for none
  [83-92]  opp_discard_onehot      — opponent's revealed discard cards (card_int/26, padded)
  [93]     hand_number_norm        — hand_number / 1000
  [94]     aggression_ratio        — raise_count / (action_count + 1)
  [95]     folded_before_ratio     — fold_count / (action_count + 1)
  [96]     stack_ratio             — my_stack / (my_stack + opp_stack + 1)
  [97]     went_to_showdown        — 0 at collection time
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

# ── Constants (matching deck.ts) ─────────────────────────────────────────────
NUM_RANKS = 9
NUM_SUITS = 3
DECK_SIZE = 27
RANKS = "23456789A"
SUITS = "dhs"

STRAIGHT_WINDOWS = [
    [8, 0, 1, 2, 3],  # A2345
    [0, 1, 2, 3, 4],  # 23456
    [1, 2, 3, 4, 5],  # 34567
    [2, 3, 4, 5, 6],  # 45678
    [3, 4, 5, 6, 7],  # 56789
    [4, 5, 6, 7, 8],  # 6789A
]

ACTION_TYPE_MAP = {"fold": 0, "raise": 1, "check": 2, "call": 3, "discard": 4}


def card_rank(c: int) -> int:
    return c % NUM_RANKS


def card_suit(c: int) -> int:
    return c // NUM_RANKS


# ── Hand evaluation helpers ───────────────────────────────────────────────────

def evaluate_5(cards: list[int]) -> tuple[int, float]:
    """Returns (hand_rank 1-8, tiebreaker 0-1)"""
    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    is_flush = len(set(suits)) == 1
    rank_set = set(ranks)
    is_straight = False
    straight_idx = -1
    for i in range(len(STRAIGHT_WINDOWS) - 1, -1, -1):
        if all(r in rank_set for r in STRAIGHT_WINDOWS[i]):
            is_straight = True
            straight_idx = i
            break

    if is_flush and is_straight:
        return 8, straight_idx / (len(STRAIGHT_WINDOWS) - 1)
    if counts[0] == 3 and counts[1] == 2:
        trip = [r for r, c in rank_counts.items() if c == 3][0]
        pair = [r for r, c in rank_counts.items() if c == 2][0]
        return 7, (trip * NUM_RANKS + pair) / (NUM_RANKS ** 2)
    if is_flush:
        sorted_r = sorted(ranks, reverse=True)
        tb = sum(r / NUM_RANKS ** (i + 1) for i, r in enumerate(sorted_r)) / NUM_RANKS
        return 6, tb
    if counts[0] == 3:
        trip = [r for r, c in rank_counts.items() if c == 3][0]
        return 5, trip / (NUM_RANKS - 1)
    if is_straight:
        return 4, straight_idx / (len(STRAIGHT_WINDOWS) - 1)
    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kicker = [r for r, c in rank_counts.items() if c == 1][0]
        return 3, (pairs[0] * NUM_RANKS ** 2 + pairs[1] * NUM_RANKS + kicker) / (NUM_RANKS ** 3)
    if counts[0] == 2:
        pair = [r for r, c in rank_counts.items() if c == 2][0]
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        k2 = kickers[1] if len(kickers) > 1 else 0
        return 2, (pair * NUM_RANKS ** 2 + kickers[0] * NUM_RANKS + k2) / (NUM_RANKS ** 3)
    sorted_r = sorted(ranks, reverse=True)
    tb = sum(r / NUM_RANKS ** (i + 1) for i, r in enumerate(sorted_r)) / NUM_RANKS
    return 1, tb


def evaluate_best(hole_cards: list[int], community: list[int]) -> tuple[int, float]:
    all_cards = [c for c in (hole_cards + community) if c >= 0]
    if len(all_cards) < 5:
        return 1, 0.0
    best_rank, best_tb = 0, 0.0
    for combo in combinations(all_cards, 5):
        r, tb = evaluate_5(list(combo))
        if r > best_rank or (r == best_rank and tb > best_tb):
            best_rank, best_tb = r, tb
    return best_rank, best_tb


def has_flush_draw(cards: list[int]) -> bool:
    suit_counts = Counter(card_suit(c) for c in cards if c >= 0)
    return max(suit_counts.values(), default=0) >= 4


def has_straight_draw(cards: list[int]) -> bool:
    rank_set = set(card_rank(c) for c in cards if c >= 0)
    for window in STRAIGHT_WINDOWS:
        if sum(r in rank_set for r in window) >= 4:
            return True
    return False


def count_pairs(cards: list[int]) -> int:
    rank_counts = Counter(card_rank(c) for c in cards if c >= 0)
    return sum(1 for cnt in rank_counts.values() if cnt >= 2)


# ── Main feature extraction ───────────────────────────────────────────────────

def extract_features(state: dict[str, Any]) -> np.ndarray:
    """
    Extract 98-dimensional feature vector from a GameState dict.
    Must mirror lib/data/features.ts exactly.
    """
    features = np.zeros(98, dtype=np.float32)

    my_cards = [c for c in state.get("my_cards", []) if c >= 0]
    comm_cards = [c for c in state.get("community_cards", []) if c >= 0]
    opp_discarded = [c for c in state.get("opp_discarded", []) if c >= 0]
    action_history = state.get("action_history", [])
    position = state.get("position", "BB")
    street = state.get("street", 0)
    hand_number = state.get("hand_number", 0)
    my_bet = state.get("my_bet", 0)
    opp_bet = state.get("opp_bet", 0)
    pot_size = state.get("pot_size", 0)
    my_stack = state.get("my_stack", 100)
    opp_stack = state.get("opp_stack", 100)
    min_raise = state.get("min_raise", 2)
    max_raise = state.get("max_raise", 100)

    # [0-26] hole cards one-hot
    for c in my_cards:
        if 0 <= c < 27:
            features[c] = 1

    # [27-53] community cards one-hot
    for c in comm_cards:
        if 0 <= c < 27:
            features[27 + c] = 1

    # [54-56] street one-hot
    if street == 0:
        features[54] = 1
    elif street <= 2:
        features[55] = 1
    else:
        features[56] = 1

    # [57] position
    features[57] = 1 if position == "SB" else 0

    # [58-64] hand strength
    all_known = my_cards + comm_cards
    if len(my_cards) >= 2 and len(comm_cards) >= 3:
        hr, tb = evaluate_best(my_cards, comm_cards)
        features[58] = (hr - 1) / 7
        features[64] = tb
    features[59] = 1 if has_flush_draw(all_known) else 0
    features[60] = 1 if has_straight_draw(all_known) else 0
    features[61] = min(count_pairs(my_cards), 2) / 2
    features[62] = 1 if any(card_rank(c) == 8 for c in my_cards) else 0
    features[63] = len(my_cards) / 5

    # [65-72] betting features
    to_call = max(0, opp_bet - my_bet)
    pot_plus_call = pot_size + to_call
    features[65] = pot_size / 200
    features[66] = my_bet / 100
    features[67] = opp_bet / 100
    features[68] = to_call / 100
    features[69] = to_call / pot_plus_call if pot_plus_call > 0 else 0
    features[70] = min_raise / 100
    features[71] = max_raise / 100
    features[72] = my_stack / 100

    # [73-82] action history (last 10, most recent first)
    history_slice = action_history[-10:]
    for i in range(10):
        if i < len(history_slice):
            entry = history_slice[-(i + 1)]
            features[73 + i] = ACTION_TYPE_MAP.get(entry.get("type", ""), 0) / 4
        else:
            features[73 + i] = -0.25

    # [83-92] opponent discard cards
    for i, c in enumerate(opp_discarded[:10]):
        features[83 + i] = c / 26 if c >= 0 else 0

    # [93-97] misc
    features[93] = hand_number / 1000
    my_actions = [a for a in action_history if a.get("player") == "human"]
    total = len(my_actions)
    raises = sum(1 for a in my_actions if a.get("type") == "raise")
    folds = sum(1 for a in my_actions if a.get("type") == "fold")
    features[94] = raises / (total + 1)
    features[95] = folds / (total + 1)
    features[96] = my_stack / (my_stack + opp_stack + 1)
    features[97] = 0  # went_to_showdown

    return features


# ── CLI: process session file → npy output ───────────────────────────────────

def process_session(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a session JSON, extract features and labels.

    Returns:
        features: (N, 98) float32
        action_labels: (N,) int64 — 0=fold, 1=raise, 2=check, 3=call
        raise_buckets: (N,) int64 — 0-9 bucket (only meaningful when action=raise)
        discard_labels: (N,) int64 — 0-9 index into C(5,2) combos (discard only)
    """
    KEEP_COMBOS = [(i, j) for i in range(5) for j in range(i + 1, 5)]

    with open(path) as f:
        session = json.load(f)

    hands = session.get("hands", [])
    betting_features, betting_labels, raise_buckets = [], [], []
    discard_features, discard_labels = [], []

    for hand in hands:
        feat = extract_features(hand)
        action = hand.get("action_taken", {})
        atype = action.get("type", "fold")

        if atype == "discard":
            kept = action.get("kept_cards")
            if kept and len(kept) == 2:
                pair = tuple(sorted(kept))
                if pair in [(i, j) for i, j in KEEP_COMBOS]:
                    label = KEEP_COMBOS.index(pair)
                    discard_features.append(feat)
                    discard_labels.append(label)
        else:
            label = {"fold": 0, "raise": 1, "check": 2, "call": 3}.get(atype, 2)
            betting_features.append(feat)
            betting_labels.append(label)
            # Raise bucket: normalize raise_amount to 0-9
            raise_amt = action.get("raise_amount", 0) or 0
            max_raise = hand.get("max_raise", 100) or 100
            bucket = min(int(raise_amt / max_raise * 10), 9) if max_raise > 0 else 0
            raise_buckets.append(bucket)

    return (
        np.array(betting_features, dtype=np.float32) if betting_features else np.zeros((0, 98), dtype=np.float32),
        np.array(betting_labels, dtype=np.int64),
        np.array(raise_buckets, dtype=np.int64),
        np.array(discard_features, dtype=np.float32) if discard_features else np.zeros((0, 98), dtype=np.float32),
        np.array(discard_labels, dtype=np.int64),
    )


def process_session_data(records: list) -> tuple:
    """
    Process a list of GameState dicts (not a file path).
    Same output format as process_session().
    Used by selfplay.py to train on accumulated in-memory data.
    """
    KEEP_COMBOS = [(i, j) for i in range(5) for j in range(i + 1, 5)]

    betting_features, betting_labels, raise_buckets = [], [], []
    discard_features, discard_labels = [], []

    for hand in records:
        feat = extract_features(hand)
        action = hand.get("action_taken", {})
        atype = action.get("type", "fold")

        if atype == "discard":
            kept = action.get("kept_cards")
            if kept and len(kept) == 2:
                pair = tuple(sorted(kept))
                if pair in [(i, j) for i, j in KEEP_COMBOS]:
                    label = KEEP_COMBOS.index(pair)
                    discard_features.append(feat)
                    discard_labels.append(label)
        else:
            label = {"fold": 0, "raise": 1, "check": 2, "call": 3}.get(atype, 2)
            betting_features.append(feat)
            betting_labels.append(label)
            raise_amt = action.get("raise_amount", 0) or 0
            max_raise = hand.get("max_raise", 100) or 100
            bucket = min(int(raise_amt / max_raise * 10), 9) if max_raise > 0 else 0
            raise_buckets.append(bucket)

    return (
        np.array(betting_features, dtype=np.float32) if betting_features else np.zeros((0, 98), dtype=np.float32),
        np.array(betting_labels, dtype=np.int64),
        np.array(raise_buckets, dtype=np.int64),
        np.array(discard_features, dtype=np.float32) if discard_features else np.zeros((0, 98), dtype=np.float32),
        np.array(discard_labels, dtype=np.int64),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py <session.json> [output_dir]")
        sys.exit(1)

    session_path = sys.argv[1]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    out_dir.mkdir(exist_ok=True)

    b_feats, b_labels, r_buckets, d_feats, d_labels = process_session(session_path)

    np.save(out_dir / "betting_features.npy", b_feats)
    np.save(out_dir / "betting_labels.npy", b_labels)
    np.save(out_dir / "raise_buckets.npy", r_buckets)
    np.save(out_dir / "discard_features.npy", d_feats)
    np.save(out_dir / "discard_labels.npy", d_labels)

    print(f"Saved {len(b_labels)} betting samples, {len(d_labels)} discard samples to {out_dir}/")
