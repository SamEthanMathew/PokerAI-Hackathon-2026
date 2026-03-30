"""
Feature extraction for PokerNetV2.

extract_features(obs, opp_stats, genesis_knowledge, tables) → (277,) float32

Feature layout (277 dims total):
  [  0: 214)  Card stream   — categories 1-6
  [214: 277)  Context stream — categories 7-11

Category 1:  Raw card encoding              156 dims  [  0:156)
Category 2:  Hand strength                   11 dims  [156:167)
Category 3:  Drawing potential                8 dims  [167:175)
Category 4:  Card removal effects            15 dims  [175:190)
Category 5:  Opponent range estimation       14 dims  [190:204)
Category 6:  Board texture                   10 dims  [204:214)
                                          ─────────
                                             214  (card_dim)

Category 7:  Betting context                 14 dims  [214:228)
Category 8:  Action history this hand        21 dims  [228:249)
Category 9:  Live opponent model             12 dims  [249:261)
Category 10: Precomputed equity               3 dims  [261:264)
Category 11: Genesis evolved knowledge       13 dims  [264:277)
                                          ─────────
                                              63  (context_dim)
"""

import math
import os
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Repo root for gym_env import ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gym_env import PokerEnv, WrappedEval

# ── Public constants ─────────────────────────────────────────────────────────
CARD_DIM: int = 214
CONTEXT_DIM: int = 63
TOTAL_DIM: int = CARD_DIM + CONTEXT_DIM  # 277

DECK_SIZE = 27
NUM_RANKS = 9
NUM_SUITS = 3

# KEEP_COMBOS[i] = (card_a, card_b) indices into the 5-card hand
KEEP_COMBOS: List[Tuple[int, int]] = [
    (i, j) for i in range(5) for j in range(i + 1, 5)
]

_evaluator: Optional[WrappedEval] = None


def _get_evaluator() -> WrappedEval:
    global _evaluator
    if _evaluator is None:
        _evaluator = WrappedEval()
    return _evaluator


# ── Card encoding helpers ────────────────────────────────────────────────────

def _encode_card(card_int: int) -> np.ndarray:
    """Encode a single card as 12-dim vector: rank_onehot(9) + suit_onehot(3)."""
    vec = np.zeros(12, dtype=np.float32)
    if card_int < 0:
        return vec  # unknown / empty slot
    rank_idx = card_int % NUM_RANKS
    suit_idx = card_int // NUM_RANKS
    vec[rank_idx] = 1.0
    vec[9 + suit_idx] = 1.0
    return vec


def _encode_card_slots(cards: list, n_slots: int) -> np.ndarray:
    """Encode up to n_slots cards, padding with zeros. Returns (n_slots*12,) array."""
    out = np.zeros(n_slots * 12, dtype=np.float32)
    for i, c in enumerate(cards[:n_slots]):
        out[i * 12:(i + 1) * 12] = _encode_card(c)
    return out


# ── Hand evaluation helpers ──────────────────────────────────────────────────

# Treys hand class constants (from treys source)
_TREYS_CLASS_STR_TO_IDX = {
    "Straight Flush": 7,
    "Full House":     6,
    "Flush":          5,
    "Straight":       4,
    "Three of a Kind": 3,
    "Two Pair":        2,
    "Pair":            1,
    "High Card":       0,
}

_MAX_RANK = 7462  # treys max rank (worst hand)


def _hand_class(rank: int) -> int:
    """Convert treys rank to 0-7 hand class (0=high card, 7=straight flush)."""
    ev = _get_evaluator()
    cls_str = ev.class_to_string(ev.get_rank_class(rank))
    return _TREYS_CLASS_STR_TO_IDX.get(cls_str, 0)


def _evaluate_hand(hole_cards: list, board_cards: list) -> Optional[int]:
    """
    Evaluate best 2-card hand + board. Returns treys rank or None if not enough cards.
    hole_cards: list of valid card ints (already filtered -1)
    board_cards: list of valid card ints
    """
    valid_hole = [c for c in hole_cards if c >= 0]
    valid_board = [c for c in board_cards if c >= 0]
    if len(valid_hole) < 2 or len(valid_board) < 3:
        return None
    try:
        treys_hole = [PokerEnv.int_to_card(c) for c in valid_hole[:2]]
        treys_board = [PokerEnv.int_to_card(c) for c in valid_board]
        ev = _get_evaluator()
        return ev.evaluate(treys_hole, treys_board)
    except Exception:
        return None


# ── Category 1: Raw card encoding (156 dims) ────────────────────────────────

def _cat1_cards(obs: dict) -> np.ndarray:
    """
    5 hole card slots × 12 + 5 community slots × 12 + 3 opp discard slots × 12
    = 60 + 60 + 36 = 156 dims
    Pre-discard: all 5 hole cards filled.
    Post-discard: keep 2 filled, discard 3 zeroed (obs["my_cards"] reflects this).
    """
    my_cards = list(obs["my_cards"])            # 5 slots, -1 if empty
    community = list(obs["community_cards"])    # 5 slots, -1 if not dealt
    opp_disc = list(obs["opp_discarded_cards"]) # 3 slots, -1 if not revealed

    out = np.zeros(156, dtype=np.float32)
    out[0:60]   = _encode_card_slots(my_cards, 5)
    out[60:120] = _encode_card_slots(community, 5)
    out[120:156] = _encode_card_slots(opp_disc, 3)
    return out


# ── Category 2: Hand strength (11 dims) ─────────────────────────────────────

def _cat2_hand_strength(obs: dict) -> np.ndarray:
    out = np.zeros(11, dtype=np.float32)

    my_cards = [c for c in obs["my_cards"] if c >= 0]
    board = [c for c in obs["community_cards"] if c >= 0]

    rank = _evaluate_hand(my_cards, board)
    if rank is None:
        return out  # zeros — not enough cards yet

    # [0] Current hand rank normalised: 1=best, 0=worst
    out[0] = 1.0 - rank / _MAX_RANK

    # [1-8] Hand category one-hot (8 classes: high_card ... straight_flush)
    hc = _hand_class(rank)
    out[1 + hc] = 1.0

    # [9] Kicker strength: best rank card normalised
    valid_ranks = sorted([c % NUM_RANKS for c in my_cards + board], reverse=True)
    out[9] = valid_ranks[0] / (NUM_RANKS - 1) if valid_ranks else 0.0

    # [10] Nut distance: approximate by rank / max_rank (0=nuts, 1=worst)
    out[10] = rank / _MAX_RANK

    return out


# ── Category 3: Drawing potential (8 dims) ──────────────────────────────────

def _cat3_draws(obs: dict) -> np.ndarray:
    out = np.zeros(8, dtype=np.float32)

    my_cards = [c for c in obs["my_cards"] if c >= 0]
    board = [c for c in obs["community_cards"] if c >= 0]
    all_cards = my_cards + board

    suits = [c // NUM_RANKS for c in all_cards]
    ranks = sorted([c % NUM_RANKS for c in all_cards])

    # [0] Flush draw: 4+ cards of same suit
    suit_counts = [suits.count(s) for s in range(NUM_SUITS)]
    max_suited = max(suit_counts) if suit_counts else 0
    out[0] = 1.0 if max_suited >= 4 else 0.0

    # [1] Flush draw strength: max suited / 5
    out[1] = max_suited / 5.0

    # [2] Open-ended straight draw: 4 consecutive ranks (wrapping at A-low)
    def _has_n_consecutive(r_sorted, n):
        for i in range(len(r_sorted) - n + 1):
            if r_sorted[i + n - 1] - r_sorted[i] == n - 1:
                # check no duplicates in window
                window = r_sorted[i:i + n]
                if len(set(window)) == n:
                    return True
        return False

    unique_ranks = sorted(set(ranks))
    out[2] = 1.0 if _has_n_consecutive(unique_ranks, 4) else 0.0

    # [3] Gutshot: 4 ranks within a span of 5 (one gap)
    def _has_gutshot(r_sorted):
        for i in range(len(r_sorted)):
            for j in range(i + 1, len(r_sorted)):
                span = r_sorted[j] - r_sorted[i]
                if span <= 4:
                    count_in_window = sum(1 for r in r_sorted if r_sorted[i] <= r <= r_sorted[j])
                    if count_in_window >= 4:
                        return True
        return False

    out[3] = 1.0 if _has_gutshot(unique_ranks) else 0.0

    # [4] Number of outs (approximate): cards that would make a flush or straight
    # Simplified: remaining cards of max suit / 15
    remaining_in_suit = max(0, 9 - max_suited)
    out[4] = min(remaining_in_suit, 15) / 15.0

    # [5] Draw completion probability: outs / remaining unknown cards
    known_cards = set(c for c in list(obs["my_cards"]) + list(obs["community_cards"])
                      + list(obs["opp_discarded_cards"]) if c >= 0)
    remaining_unknown = DECK_SIZE - len(known_cards)
    outs = remaining_in_suit
    out[5] = outs / max(remaining_unknown, 1)

    # [6] Backdoor flush draw (3 of same suit on flop)
    if len(board) == 3:
        flop_suits = [c // NUM_RANKS for c in board]
        max_flop_suited = max(flop_suits.count(s) for s in range(NUM_SUITS))
        out[6] = 1.0 if max_flop_suited >= 3 else 0.0

    # [7] Backdoor straight draw (3 connected on flop)
    if len(board) >= 3:
        flop_ranks = sorted(set(c % NUM_RANKS for c in board))
        out[7] = 1.0 if _has_n_consecutive(flop_ranks, 3) else 0.0

    return out


# ── Category 4: Card removal effects (15 dims) ──────────────────────────────

def _cat4_card_removal(obs: dict) -> np.ndarray:
    out = np.zeros(15, dtype=np.float32)

    known_cards = set(c for c in list(obs["my_cards"]) + list(obs["community_cards"])
                      + list(obs["my_discarded_cards"]) + list(obs["opp_discarded_cards"])
                      if c >= 0)

    # [0:3] Cards remaining per suit: remaining_in_suit / 9
    for s in range(NUM_SUITS):
        total_in_suit = NUM_RANKS
        used = sum(1 for c in known_cards if c // NUM_RANKS == s)
        out[s] = (total_in_suit - used) / total_in_suit

    # [3:12] Cards remaining per rank: remaining_of_rank / 3
    for r in range(NUM_RANKS):
        total_of_rank = NUM_SUITS
        used = sum(1 for c in known_cards if c % NUM_RANKS == r)
        out[3 + r] = (total_of_rank - used) / total_of_rank

    # [12:15] Flush still possible per suit: need ≥5 of suit remaining in unknown cards
    board = [c for c in obs["community_cards"] if c >= 0]
    for s in range(NUM_SUITS):
        already_on_board = sum(1 for c in board if c // NUM_RANKS == s)
        remaining_of_suit = sum(1 for c in range(DECK_SIZE)
                                if c // NUM_RANKS == s and c not in known_cards)
        total_available = already_on_board + remaining_of_suit
        out[12 + s] = 1.0 if total_available >= 5 else 0.0

    return out


# ── Category 5: Opponent range estimation (14 dims) ─────────────────────────

def _cat5_opp_range(obs: dict, tables: dict) -> np.ndarray:
    out = np.zeros(14, dtype=np.float32)

    opp_disc = [c for c in obs["opp_discarded_cards"] if c >= 0]
    if len(opp_disc) < 3:
        return out  # discards not yet revealed

    disc_set = set(opp_disc)

    # [0] Opp discarded an ace
    out[0] = 1.0 if any(c % NUM_RANKS == 8 for c in opp_disc) else 0.0

    # [1] Opp discarded a pair
    disc_ranks = [c % NUM_RANKS for c in opp_disc]
    out[1] = 1.0 if len(disc_ranks) != len(set(disc_ranks)) else 0.0

    # [2] Opp discarded suited cards (2+ same suit in discards)
    disc_suits = [c // NUM_RANKS for c in opp_disc]
    out[2] = 1.0 if len(disc_suits) != len(set(disc_suits)) else 0.0

    # [3] Opp discarded connected cards
    sorted_disc_ranks = sorted(set(disc_ranks))
    connected = any(sorted_disc_ranks[i + 1] - sorted_disc_ranks[i] == 1
                    for i in range(len(sorted_disc_ranks) - 1))
    out[3] = 1.0 if connected else 0.0

    # Range estimation from lookup table
    from precompute import lookup_opponent_range
    range_list = lookup_opponent_range(tables, opp_disc) if tables else None

    if range_list is None:
        # Fallback: all non-conflicting 2-card hands uniform
        known = set(c for c in list(obs["my_cards"]) + list(obs["community_cards"])
                    + list(obs["opp_discarded_cards"]) if c >= 0)
        possible = [(c1, c2) for c1, c2 in combinations(range(DECK_SIZE), 2)
                    if c1 not in known and c2 not in known]
        n = len(possible)
        range_list = [(h, 1.0 / n) for h in possible] if n > 0 else []

    # [4] Opp likely kept a pair: prob that any possible hand is a pair
    pair_prob = sum(p for (h, p) in range_list if h[0] % NUM_RANKS == h[1] % NUM_RANKS)
    out[4] = min(pair_prob, 1.0)

    # [5] Estimated range equity vs my hand (approximated as 0.5 if table absent)
    out[5] = 0.5  # placeholder; overwritten by category 10 equity

    # [6:14] Opp range hand category probabilities (8 categories)
    # Approximate using kept cards (opp_disc reveals 3; 2 remaining unknown)
    # Too expensive to evaluate exactly — use pair/suited heuristics
    known_all = set(c for c in list(obs["my_cards"]) + list(obs["community_cards"])
                    + list(obs["opp_discarded_cards"]) if c >= 0)
    board = [c for c in obs["community_cards"] if c >= 0]

    if len(board) >= 3 and range_list:
        # Sample top 20 most likely hands and evaluate each
        sample = range_list[:20]
        cat_probs = np.zeros(8, dtype=np.float32)
        total_w = 0.0
        for (h, p) in sample:
            if h[0] in known_all or h[1] in known_all:
                continue
            rank = _evaluate_hand(list(h), board)
            if rank is None:
                continue
            hc = _hand_class(rank)
            cat_probs[hc] += p
            total_w += p
        if total_w > 0:
            cat_probs /= total_w
        out[6:14] = cat_probs
    else:
        out[6] = 1.0  # default: high card

    return out


# ── Category 6: Board texture (10 dims) ─────────────────────────────────────

def _cat6_board_texture(obs: dict) -> np.ndarray:
    out = np.zeros(10, dtype=np.float32)

    board = [c for c in obs["community_cards"] if c >= 0]
    if not board:
        return out

    ranks = [c % NUM_RANKS for c in board]
    suits = [c // NUM_RANKS for c in board]
    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    suit_counts = {s: suits.count(s) for s in set(suits)}
    num_suits = len(suit_counts)

    # [0] Board is paired
    out[0] = 1.0 if any(v >= 2 for v in rank_counts.values()) else 0.0

    # [1] Board has two pair
    out[1] = 1.0 if sum(1 for v in rank_counts.values() if v >= 2) >= 2 else 0.0

    # [2] Board has trips
    out[2] = 1.0 if any(v >= 3 for v in rank_counts.values()) else 0.0

    # [3] Board monotone
    out[3] = 1.0 if num_suits == 1 else 0.0

    # [4] Board two-tone
    out[4] = 1.0 if num_suits == 2 else 0.0

    # [5] Board rainbow
    out[5] = 1.0 if num_suits == 3 else 0.0

    # [6] 3+ connected ranks
    def n_consec(sorted_ranks, n):
        for i in range(len(sorted_ranks) - n + 1):
            if sorted_ranks[i + n - 1] - sorted_ranks[i] == n - 1:
                return True
        return False

    unique_ranks = sorted(set(ranks))
    out[6] = 1.0 if n_consec(unique_ranks, 3) else 0.0

    # [7] 4+ connected ranks
    out[7] = 1.0 if n_consec(unique_ranks, 4) else 0.0

    # [8] Highest card on board (rank / 8)
    out[8] = max(ranks) / (NUM_RANKS - 1)

    # [9] Board wetness: (flush_possible + straight_possible + paired) / 3
    flush_possible = 1.0 if max(suit_counts.values()) >= 3 else 0.0
    straight_possible = out[6]
    wetness = (flush_possible + straight_possible + out[0]) / 3.0
    out[9] = wetness

    return out


# ── Category 7: Betting context (14 dims) ───────────────────────────────────

def _cat7_betting(obs: dict) -> np.ndarray:
    out = np.zeros(14, dtype=np.float32)
    pot = obs["pot_size"]
    my_bet = obs["my_bet"]
    opp_bet = obs["opp_bet"]
    min_raise = obs["min_raise"]
    max_raise = obs["max_raise"]
    blind_pos = obs.get("blind_position", 0)

    my_stack = PokerEnv.MAX_PLAYER_BET - my_bet
    opp_stack = PokerEnv.MAX_PLAYER_BET - opp_bet
    to_call = max(0, opp_bet - my_bet)

    out[0] = pot / 200.0
    out[1] = my_bet / 100.0
    out[2] = opp_bet / 100.0
    out[3] = my_stack / 100.0
    out[4] = opp_stack / 100.0
    out[5] = to_call / 100.0
    out[6] = min_raise / 100.0
    out[7] = max_raise / 100.0
    # Pot odds: call / (pot + call)
    out[8] = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
    # Stack-to-pot ratio (SPR), capped at 1.0
    out[9] = min(my_stack / max(pot, 1), 1.0)
    # Position: 1 = SB, 0 = BB
    out[10] = float(blind_pos == 0)  # blind_position=0 means SB in gym_env
    # Street one-hot [11:15)
    street = obs["street"]
    if 0 <= street < 4:
        out[11 + street] = 1.0

    return out


# ── Category 8: Action history (21 dims) ────────────────────────────────────

def _cat8_action_history(opp_stats: dict) -> np.ndarray:
    out = np.zeros(21, dtype=np.float32)

    raises_this_street = opp_stats.get("raises_this_street", 0)
    raises_this_hand   = opp_stats.get("raises_this_hand", 0)
    opp_raised_street  = opp_stats.get("opp_raised_street", 0)
    opp_raised_hand    = opp_stats.get("opp_raised_hand", 0)
    i_have_initiative  = opp_stats.get("i_have_initiative", 0)
    facing_bet         = opp_stats.get("facing_bet", 0)
    facing_raise       = opp_stats.get("facing_raise", 0)
    facing_3bet        = opp_stats.get("facing_3bet", 0)
    # Last 4 actions as (action_type / 4) for each player, 2 each = 8 dims,
    # plus padding to 13
    last_actions       = opp_stats.get("last_actions", [])

    out[0] = min(raises_this_street, 4) / 4.0
    out[1] = min(raises_this_hand, 8) / 8.0
    out[2] = float(opp_raised_street > 0)
    out[3] = float(opp_raised_hand > 0)
    out[4] = float(i_have_initiative)
    out[5] = float(facing_bet)
    out[6] = float(facing_raise)
    out[7] = float(facing_3bet)

    # Last 13 action values (action_type / 4, 0 = no action)
    for i, a in enumerate(last_actions[-13:]):
        out[8 + i] = a / 4.0

    return out


# ── Category 9: Live opponent model (12 dims, EMA) ──────────────────────────

def _cat9_opponent_model(opp_stats: dict) -> np.ndarray:
    out = np.zeros(12, dtype=np.float32)

    out[0]  = opp_stats.get("vpip",              0.0)
    out[1]  = opp_stats.get("pfr",               0.0)
    out[2]  = opp_stats.get("aggression_factor",  0.0)
    out[3]  = opp_stats.get("wtsd",              0.0)
    out[4]  = opp_stats.get("wsd",               0.0)
    out[5]  = opp_stats.get("fold_to_cbet",       0.0)
    out[6]  = opp_stats.get("fold_to_turn",       0.0)
    out[7]  = opp_stats.get("fold_to_river",      0.0)
    out[8]  = opp_stats.get("check_raise_freq",   0.0)
    out[9]  = opp_stats.get("avg_bet_size",       0.0)
    out[10] = opp_stats.get("bet_size_variance",  0.0)
    out[11] = min(opp_stats.get("hands_observed", 0), 100) / 100.0

    return out


def update_opp_stats(opp_stats: dict, obs: dict, action_taken: Optional[tuple] = None):
    """
    Update exponential moving average opponent stats from an observation.
    Call this in observe() for the opponent's actions.

    alpha=0.05 so recent hands are weighted more.
    """
    alpha = 0.05

    def ema(old, new):
        return old + alpha * (new - old)

    opp_stats.setdefault("hands_observed",       0)
    opp_stats.setdefault("vpip",                 0.5)
    opp_stats.setdefault("pfr",                  0.3)
    opp_stats.setdefault("aggression_factor",    1.0)
    opp_stats.setdefault("wtsd",                 0.5)
    opp_stats.setdefault("wsd",                  0.5)
    opp_stats.setdefault("fold_to_cbet",         0.5)
    opp_stats.setdefault("fold_to_turn",         0.5)
    opp_stats.setdefault("fold_to_river",        0.5)
    opp_stats.setdefault("check_raise_freq",     0.1)
    opp_stats.setdefault("avg_bet_size",         0.3)
    opp_stats.setdefault("bet_size_variance",    0.1)
    opp_stats.setdefault("_raise_sizes",         [])
    opp_stats.setdefault("raises_this_street",   0)
    opp_stats.setdefault("raises_this_hand",     0)
    opp_stats.setdefault("opp_raised_street",    0)
    opp_stats.setdefault("opp_raised_hand",      0)
    opp_stats.setdefault("i_have_initiative",    0)
    opp_stats.setdefault("facing_bet",           0)
    opp_stats.setdefault("facing_raise",         0)
    opp_stats.setdefault("facing_3bet",          0)
    opp_stats.setdefault("last_actions",         [])
    opp_stats.setdefault("_prev_street",         -1)

    opp_stats["hands_observed"] += 1

    if action_taken is None:
        return

    action_type = action_taken[0] if isinstance(action_taken, (tuple, list)) else action_taken

    # Track raises for current street/hand
    if action_type == 1:  # RAISE
        opp_stats["opp_raised_street"] += 1
        opp_stats["opp_raised_hand"] += 1
        raise_amt = action_taken[1] if isinstance(action_taken, (tuple, list)) else 0
        opp_stats["_raise_sizes"].append(raise_amt / 100.0)
        if len(opp_stats["_raise_sizes"]) > 100:
            opp_stats["_raise_sizes"] = opp_stats["_raise_sizes"][-100:]
        sizes = opp_stats["_raise_sizes"]
        opp_stats["avg_bet_size"] = float(np.mean(sizes))
        opp_stats["bet_size_variance"] = float(np.std(sizes))

        opp_stats["aggression_factor"] = ema(opp_stats["aggression_factor"], 1.5)
    elif action_type == 3:  # CALL
        opp_stats["aggression_factor"] = ema(opp_stats["aggression_factor"], 0.5)
    elif action_type == 0:  # FOLD
        street = obs.get("street", 0)
        if street == 3:
            opp_stats["fold_to_river"] = ema(opp_stats["fold_to_river"], 1.0)
        elif street == 2:
            opp_stats["fold_to_turn"] = ema(opp_stats["fold_to_turn"], 1.0)
        elif street == 1:
            opp_stats["fold_to_cbet"] = ema(opp_stats["fold_to_cbet"], 1.0)

    # Track last N actions
    opp_stats["last_actions"].append(action_type)
    if len(opp_stats["last_actions"]) > 20:
        opp_stats["last_actions"] = opp_stats["last_actions"][-20:]

    # Reset street-level counters on new street
    cur_street = obs.get("street", 0)
    if cur_street != opp_stats["_prev_street"]:
        opp_stats["raises_this_street"] = 0
        opp_stats["opp_raised_street"] = 0
        opp_stats["_prev_street"] = cur_street


# ── Category 10: Precomputed equity (3 dims) ─────────────────────────────────

def _cat10_equity(obs: dict, tables: dict) -> np.ndarray:
    out = np.zeros(3, dtype=np.float32)

    from precompute import lookup_equity, lookup_opponent_range

    my_cards = [c for c in obs["my_cards"] if c >= 0]
    if len(my_cards) < 2:
        return out

    # Use the first 2 valid hole cards (post-discard these are the kept pair)
    c1, c2 = my_cards[0], my_cards[1]

    # [0] Equity vs random hand
    out[0] = lookup_equity(tables, c1, c2)

    # [1] Equity vs opponent estimated range
    opp_disc = [c for c in obs["opp_discarded_cards"] if c >= 0]
    if len(opp_disc) == 3:
        range_list = lookup_opponent_range(tables, opp_disc)
        if range_list:
            board = [c for c in obs["community_cards"] if c >= 0]
            known = set(my_cards + board + opp_disc)
            total_w = 0.0
            weighted_eq = 0.0
            for (h, p) in range_list[:50]:  # sample top 50 for speed
                if h[0] in known or h[1] in known:
                    continue
                eq_h = lookup_equity(tables, h[0], h[1])
                weighted_eq += p * (1.0 - eq_h)  # my equity vs this hand
                total_w += p
            if total_w > 0:
                out[1] = weighted_eq / total_w
        else:
            out[1] = out[0]
    else:
        out[1] = out[0]

    # [2] Equity improvement potential (outs / remaining unknown)
    known_all = set(c for c in list(obs["my_cards"]) + list(obs["community_cards"])
                    + list(obs["opp_discarded_cards"]) if c >= 0)
    remaining = DECK_SIZE - len(known_all)
    board = [c for c in obs["community_cards"] if c >= 0]
    # Flush outs: cards of same suit as my best suited pair
    my_suits = [c // NUM_RANKS for c in my_cards]
    best_suit = max(set(my_suits), key=my_suits.count)
    suited_on_board = sum(1 for c in board if c // NUM_RANKS == best_suit)
    suited_total = my_suits.count(best_suit) + suited_on_board
    flush_outs = max(0, 5 - suited_total) if suited_total >= 2 else 0
    out[2] = flush_outs / max(remaining, 1)

    return out


# ── Category 11: Genesis evolved knowledge (13 dims) ────────────────────────

def _cat11_genesis(genesis_knowledge: dict, obs: dict) -> np.ndarray:
    out = np.zeros(13, dtype=np.float32)

    genomes = genesis_knowledge.get("top_genomes", [])
    if not genomes:
        return out

    # [0:3] Top 3 genome action suggestions (argmax action / 3)
    for gi, genome in enumerate(genomes[:3]):
        probs = genome.get("action_probs", [0.25, 0.25, 0.25, 0.25])
        action_idx = int(np.argmax(probs))
        out[gi] = action_idx / 3.0

    # [3:6] Top 3 genome raise sizing suggestions
    for gi, genome in enumerate(genomes[:3]):
        out[3 + gi] = genome.get("raise_size_fraction", 0.5)

    # [6] Ensemble agreement: 1 - entropy(action_votes) / log2(4)
    if len(genomes) >= 2:
        votes = [0, 0, 0, 0]
        for genome in genomes[:3]:
            probs = genome.get("action_probs", [0.25, 0.25, 0.25, 0.25])
            votes[int(np.argmax(probs))] += 1
        vote_arr = np.array(votes, dtype=float) / sum(votes)
        entropy = -sum(p * math.log(p + 1e-9) for p in vote_arr if p > 0)
        max_entropy = math.log(4)
        out[6] = 1.0 - entropy / max_entropy
    else:
        out[6] = 1.0

    # [7:13] Evolved average parameters (6 dims)
    ep = genesis_knowledge.get("ensemble_params", {})
    out[7]  = ep.get("bluff_freq",            0.14)
    out[8]  = ep.get("fold_to_raise",         0.65)
    out[9]  = ep.get("cbet_freq",             0.70)
    out[10] = ep.get("pot_odds_threshold",    0.35)
    out[11] = ep.get("discard_pair_weight",   0.80)
    out[12] = ep.get("discard_suited_weight", 0.60)

    return out


# ── Main extraction function ──────────────────────────────────────────────────

def extract_features(
    obs: dict,
    opp_stats: Optional[dict] = None,
    genesis_knowledge: Optional[dict] = None,
    tables: Optional[dict] = None,
) -> np.ndarray:
    """
    Extract the full 277-dim feature vector from a game observation.

    Args:
        obs:               Observation dict from gym_env / tournament API
        opp_stats:         Live opponent model dict (mutable, updated by update_opp_stats)
        genesis_knowledge: Loaded genesis_knowledge.json dict
        tables:            Loaded precomputed tables dict from precompute.load_tables()

    Returns:
        (277,) float32 ndarray, all values normalised to approximately [0, 1]
    """
    if opp_stats is None:
        opp_stats = {}
    if genesis_knowledge is None:
        genesis_knowledge = {}
    if tables is None:
        tables = {}

    feat = np.zeros(TOTAL_DIM, dtype=np.float32)

    # ── Card stream [0:214) ────────────────────────────────────────────────
    feat[0:156]   = _cat1_cards(obs)
    feat[156:167] = _cat2_hand_strength(obs)
    feat[167:175] = _cat3_draws(obs)
    feat[175:190] = _cat4_card_removal(obs)
    feat[190:204] = _cat5_opp_range(obs, tables)
    feat[204:214] = _cat6_board_texture(obs)

    # ── Context stream [214:277) ───────────────────────────────────────────
    feat[214:228] = _cat7_betting(obs)
    feat[228:249] = _cat8_action_history(opp_stats)
    feat[249:261] = _cat9_opponent_model(opp_stats)
    feat[261:264] = _cat10_equity(obs, tables)
    feat[264:277] = _cat11_genesis(genesis_knowledge, obs)

    return feat


def split_features(feat: np.ndarray):
    """
    Split 277-dim vector into (card_feat_214, ctx_feat_63).
    Returns numpy arrays; caller converts to tensors.
    """
    return feat[:CARD_DIM], feat[CARD_DIM:]
