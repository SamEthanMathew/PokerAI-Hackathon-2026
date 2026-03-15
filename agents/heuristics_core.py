# agents/heuristics_core.py
"""
Shared helpers for heuristic agents in the 27-card keep-2 poker variant.

Card encoding:
- cards are ints in [0..26]
- rank = card % 9  maps to "23456789A"
- suit = card // 9 maps to "dhs"  (diamonds/hearts/spades)

Action encoding:
- action is (action_type, raise_amount, keep1, keep2)
- action_type uses PokerEnv.ActionType values
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from gym_env import PokerEnv, WrappedEval

# -----------------------------
# Constants / precomputed lists
# -----------------------------
RANKS = "23456789A"
SUITS = "dhs"
DECK_SIZE = 27

action_types = PokerEnv.ActionType

# 10 ways to choose two indices out of 5 (for DISCARD)
KEEP_INDEX_PAIRS: List[Tuple[int, int]] = [(i, j) for i in range(5) for j in range(i + 1, 5)]

# All unordered 2-card combinations from 27 cards (351)
PAIR_CARDS = np.array([(a, b) for a in range(DECK_SIZE) for b in range(a + 1, DECK_SIZE)], dtype=np.uint8)  # (351,2)
PAIR_MASKS = np.array([(1 << int(a)) | (1 << int(b)) for (a, b) in PAIR_CARDS], dtype=np.uint32)  # (351,)

# Rank straight windows for this deck (Ace wraps only for A2345 and 6789A)
# rank ids: 0..8 == 2..9,A
STRAIGHT_WINDOWS = [
    {8, 0, 1, 2, 3},  # A2345
    {0, 1, 2, 3, 4},  # 23456
    {1, 2, 3, 4, 5},  # 34567
    {2, 3, 4, 5, 6},  # 45678
    {3, 4, 5, 6, 7},  # 56789
    {4, 5, 6, 7, 8},  # 6789A
]

# How many straight windows each rank participates in (centrality)
RANK_CENTRALITY = np.array(
    [sum(1 for w in STRAIGHT_WINDOWS if r in w) for r in range(9)],
    dtype=np.int8,
)

_EVAL = WrappedEval()  # matches engine evaluation semantics

# -----------------------------
# Low-level card utilities
# -----------------------------
def rank_of(card: int) -> int:
    return int(card) % 9

def suit_of(card: int) -> int:
    return int(card) // 9

def is_connector(r1: int, r2: int) -> bool:
    return abs(r1 - r2) == 1 or ({r1, r2} == {8, 0})

def is_gap1(r1: int, r2: int) -> bool:
    return abs(r1 - r2) == 2 or ({r1, r2} == {8, 1})

def mask_of(cards: List[int]) -> int:
    m = 0
    for c in cards:
        if c != -1:
            m |= (1 << int(c))
    return m

def count_ranks(cards: List[int]) -> np.ndarray:
    cnt = np.zeros(9, dtype=np.int8)
    for c in cards:
        if c == -1:
            continue
        cnt[rank_of(c)] += 1
    return cnt

def count_suits(cards: List[int]) -> np.ndarray:
    cnt = np.zeros(3, dtype=np.int8)
    for c in cards:
        if c == -1:
            continue
        cnt[suit_of(c)] += 1
    return cnt

def has_trips(cards5: List[int]) -> bool:
    rc = count_ranks(cards5)
    return bool((rc == 3).any())

# -----------------------------
# Board texture helpers
# -----------------------------
@dataclass
class BoardTexture:
    suit_counts: np.ndarray  # (3,)
    rank_counts: np.ndarray  # (9,)
    max_suit: int
    is_two_tone: bool
    is_monotone: bool
    pair_flag: bool
    trips_flag: bool
    double_pair_flag: bool
    straight_density: float  # [0..1]
    four_to_straight_flag: bool

def board_texture(board: List[int], street: int) -> BoardTexture:
    sc = count_suits(board)
    rc = count_ranks(board)
    max_s = int(sc.max()) if len(board) else 0
    is_two = (street == 1 and max_s == 2)
    is_mono = (street == 1 and max_s == 3)

    pair_flag = bool((rc == 2).any())
    trips_flag = bool((rc == 3).any())
    double_pair_flag = int((rc == 2).sum()) >= 2

    ranks_present = {r for r in range(9) if rc[r] > 0}
    if len(ranks_present) == 0:
        dens = 0.0
        four_flag = False
    else:
        best_in_window = max(len(ranks_present & w) for w in STRAIGHT_WINDOWS)
        dens = best_in_window / min(len(ranks_present), 5)
        four_flag = best_in_window >= 4 and street >= 2

    return BoardTexture(
        suit_counts=sc,
        rank_counts=rc,
        max_suit=max_s,
        is_two_tone=is_two,
        is_monotone=is_mono,
        pair_flag=pair_flag,
        trips_flag=trips_flag,
        double_pair_flag=double_pair_flag,
        straight_density=float(dens),
        four_to_straight_flag=four_flag,
    )

# -----------------------------
# Hand evaluation helpers
# -----------------------------
def treys_cards(cards: List[int]) -> List[int]:
    return [PokerEnv.int_to_card(c) for c in cards if c != -1]

def eval_score(hole2: List[int], boardN: List[int]) -> int:
    return int(_EVAL.evaluate(treys_cards(hole2), treys_cards(boardN)))

def eval_class(hole2: List[int], boardN: List[int]) -> int:
    # treys: 1=StraightFlush, 2=FourKind, 3=FullHouse, 4=Flush, 5=Straight,
    #        6=Trips, 7=TwoPair, 8=Pair, 9=HighCard
    score = eval_score(hole2, boardN)
    return int(_EVAL.get_rank_class(score))

def is_straight_plus(rank_class: int) -> bool:
    return rank_class <= 5

def is_two_pair_plus(rank_class: int) -> bool:
    return rank_class <= 7

def is_pair_plus(rank_class: int) -> bool:
    return rank_class <= 8

# -----------------------------
# Pot odds / bet sizing helpers
# -----------------------------
def continue_cost(obs: dict) -> int:
    return max(0, int(obs["opp_bet"]) - int(obs["my_bet"]))

def pot_size(obs: dict) -> int:
    return int(obs["my_bet"]) + int(obs["opp_bet"])

def pot_odds(obs: dict) -> float:
    c = continue_cost(obs)
    p = pot_size(obs)
    return (c / (c + p)) if c > 0 else 0.0

def clamp_raise(obs: dict, desired_raise_amount: int) -> int:
    mn = int(obs["min_raise"])
    mx = int(obs["max_raise"])
    if mx <= 0:
        return 0
    return int(max(mn, min(mx, desired_raise_amount)))

def pot_frac_raise(obs: dict, frac: float) -> int:
    desired = int(pot_size(obs) * frac)
    return clamp_raise(obs, desired)

# -----------------------------
# Safety: always return legal actions
# -----------------------------
def make_legal_action(obs: dict, action: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    va = obs["valid_actions"]
    at, ra, k1, k2 = action

    # DISCARD path
    if at == action_types.DISCARD.value:
        if va[action_types.DISCARD.value] != 1:
            at = action_types.CHECK.value
        else:
            if not (0 <= k1 <= 4 and 0 <= k2 <= 4 and k1 != k2):
                k1, k2 = 0, 1
            return (action_types.DISCARD.value, 0, int(k1), int(k2))

    # RAISE path
    if at == action_types.RAISE.value:
        if va[action_types.RAISE.value] != 1:
            at = action_types.CALL.value if va[action_types.CALL.value] else action_types.CHECK.value
            return make_legal_action(obs, (at, 0, 0, 0))
        ra2 = clamp_raise(obs, int(ra))
        if ra2 <= 0:
            at = action_types.CALL.value if va[action_types.CALL.value] else action_types.CHECK.value
            return make_legal_action(obs, (at, 0, 0, 0))
        return (action_types.RAISE.value, ra2, 0, 0)

    # CHECK / CALL / FOLD
    if at == action_types.CHECK.value and va[action_types.CHECK.value]:
        return (action_types.CHECK.value, 0, 0, 0)
    if at == action_types.CALL.value and va[action_types.CALL.value]:
        return (action_types.CALL.value, 0, 0, 0)
    return (action_types.FOLD.value, 0, 0, 0)

# -----------------------------
# User-priority discard scorer (core building block)
# -----------------------------
def user_priority_tier(hole2: List[int]) -> Tuple[int, int, int, int, int]:
    """
    Returns a sortable tuple where smaller is better.
    Encodes:
      tier (1 best .. 4 worst),
      suited bonus, connector/gap bonus, top rank, sum ranks
    """
    c1, c2 = hole2
    r1, r2 = rank_of(c1), rank_of(c2)
    s1, s2 = suit_of(c1), suit_of(c2)
    suited = int(s1 == s2)
    pair = int(r1 == r2)
    top = max(r1, r2)
    ssum = r1 + r2

    if suited and is_connector(r1, r2):
        tier = 1
        conn = 2
    elif suited and is_gap1(r1, r2):
        tier = 2
        conn = 1
    elif pair:
        tier = 3
        conn = 0
    else:
        tier = 4
        conn = 0

    return (tier, -suited, -conn, -top, -ssum)

def prefer_flush_leverage(hole2: List[int], flop3: List[int]) -> int:
    if suit_of(hole2[0]) != suit_of(hole2[1]):
        return 0
    sc = count_suits(flop3)
    return 1 if sc[suit_of(hole2[0])] >= 2 else 0

def straight_coverage(hole2: List[int]) -> int:
    r1, r2 = rank_of(hole2[0]), rank_of(hole2[1])
    return int(RANK_CENTRALITY[r1] + RANK_CENTRALITY[r2])

# -----------------------------
# Optional fast evaluation tables (for exact enumerators)
# -----------------------------
def load_rank7_table(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    arr = np.load(path, mmap_mode="r")
    return arr

def load_board_hand_table(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    arr = np.load(path, mmap_mode="r")
    return arr

# Precompute small nCr for combinadic indexes (n<=27, k<=7)
_NCR = [[0] * 8 for _ in range(28)]
for n in range(28):
    for k in range(0, min(7, n) + 1):
        _NCR[n][k] = math.comb(n, k)

def comb_index(sorted_cards: List[int], k: int) -> int:
    """Combinadic rank for combination of size k drawn from [0..26], cards must be sorted ascending."""
    idx = 0
    for i, c in enumerate(sorted_cards):
        idx += _NCR[int(c)][i + 1]
    return int(idx)

def hand2_id(c1: int, c2: int) -> int:
    a, b = (c1, c2) if c1 < c2 else (c2, c1)
    return comb_index([a, b], 2)

def board5_id(board5_cards: List[int]) -> int:
    s = sorted(board5_cards)
    return comb_index(s, 5)

def rank_board_hand(board5: List[int], hole2: List[int], table: Optional[np.ndarray]) -> int:
    if table is None:
        return eval_score(hole2, board5)
    bid = board5_id(board5)
    hid = hand2_id(hole2[0], hole2[1])
    return int(table[bid, hid])

# -----------------------------
# Exact enumerators (optimized patterns)
# -----------------------------
def enumerate_runouts(deadmask: int) -> np.ndarray:
    """Return array of (turn, river) pairs not overlapping deadmask."""
    ok = (PAIR_MASKS & deadmask) == 0
    return PAIR_CARDS[ok]  # shape (N,2)

def exact_strength_keep(
    my5: List[int],
    flop3: List[int],
    keep_idx: Tuple[int, int],
    opp_discards: List[int],
    board_hand_table: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Strength-only objective: mean score and variance over all turn/river runouts.
    Lower evaluator score is better, so we return mean_score and var_score.
    """
    i, j = keep_idx
    hole2 = [my5[i], my5[j]]

    dead = my5 + flop3 + [c for c in opp_discards if c != -1]
    dm = mask_of(dead)
    runouts = enumerate_runouts(dm)

    scores = []
    for t, r in runouts.tolist():
        board5 = flop3 + [int(t), int(r)]
        s = rank_board_hand(board5, hole2, board_hand_table)
        scores.append(s)

    arr = np.array(scores, dtype=np.float32)
    return float(arr.mean()), float(arr.var())

def exact_equity_keep(
    my5: List[int],
    flop3: List[int],
    keep_idx: Tuple[int, int],
    opp_discards: List[int],
    board_hand_table: Optional[np.ndarray] = None,
    tie_weight: float = 0.5,
) -> float:
    """
    Exact equity vs uniform random opponent hole2 from remaining unseen cards,
    averaged over all equally-likely turn/river runouts.
    """
    i, j = keep_idx
    hole2 = [my5[i], my5[j]]
    dead_base = my5 + flop3 + [c for c in opp_discards if c != -1]
    dm_base = mask_of(dead_base)
    runouts = enumerate_runouts(dm_base)

    total = 0
    wins = 0
    ties = 0

    for t, r in runouts.tolist():
        board5 = flop3 + [int(t), int(r)]
        runmask = (1 << int(t)) | (1 << int(r))
        dm2 = dm_base | runmask

        ok = (PAIR_MASKS & dm2) == 0
        opp_pairs = PAIR_CARDS[ok]
        if opp_pairs.shape[0] == 0:
            continue

        my_rank = rank_board_hand(board5, hole2, board_hand_table)

        opp_ranks = []
        for a, b in opp_pairs.tolist():
            opp_r = rank_board_hand(board5, [int(a), int(b)], board_hand_table)
            opp_ranks.append(opp_r)
        opp_ranks = np.array(opp_ranks, dtype=np.int32)

        total += int(opp_ranks.size)
        wins += int((my_rank < opp_ranks).sum())
        ties += int((my_rank == opp_ranks).sum())

    if total == 0:
        return 0.0
    return float((wins + tie_weight * ties) / total)

def exact_equity_discard(
    my5: List[int],
    flop3: List[int],
    opp_discards: List[int],
    board_hand_table: Optional[np.ndarray] = None,
    top_k: Optional[int] = None,
    pre_score_fn=None,
    tie_weight: float = 0.5,
) -> Tuple[Tuple[int, int], float]:
    """
    Find best keep indices among 10 choices by exact equity.
    Opponent ranks are reused across keep candidates per runout.
    """
    assert len(my5) == 5 and len(flop3) == 3

    cands = KEEP_INDEX_PAIRS[:]
    if top_k is not None and pre_score_fn is not None:
        scored = []
        for (i, j) in cands:
            scored.append((float(pre_score_fn(my5, flop3, (i, j), opp_discards)), (i, j)))
        scored.sort(reverse=True)
        cands = [ij for _, ij in scored[: int(top_k)]]

    dead_base = my5 + flop3 + [c for c in opp_discards if c != -1]
    dm_base = mask_of(dead_base)
    runouts = enumerate_runouts(dm_base)

    total = 0
    wins = {ij: 0 for ij in cands}
    ties = {ij: 0 for ij in cands}

    for t, r in runouts.tolist():
        board5 = flop3 + [int(t), int(r)]
        runmask = (1 << int(t)) | (1 << int(r))
        dm2 = dm_base | runmask

        ok = (PAIR_MASKS & dm2) == 0
        opp_pairs = PAIR_CARDS[ok]
        if opp_pairs.shape[0] == 0:
            continue

        opp_ranks = []
        for a, b in opp_pairs.tolist():
            opp_ranks.append(rank_board_hand(board5, [int(a), int(b)], board_hand_table))
        opp_ranks = np.array(opp_ranks, dtype=np.int32)

        total += int(opp_ranks.size)

        for (i, j) in cands:
            hole2 = [my5[i], my5[j]]
            my_rank = rank_board_hand(board5, hole2, board_hand_table)
            wins[(i, j)] += int((my_rank < opp_ranks).sum())
            ties[(i, j)] += int((my_rank == opp_ranks).sum())

    if total == 0:
        return (0, 1), 0.0

    best = None
    for ij in cands:
        eq = (wins[ij] + tie_weight * ties[ij]) / total
        if best is None or eq > best[1]:
            best = (ij, float(eq))
    return best[0], best[1]

# -----------------------------
# Simple opponent stats container
# -----------------------------
@dataclass
class OpponentStats:
    hands_seen: int = 0
    raised_seen: int = 0
    folded_after_our_raise: int = 0
    called_after_our_raise: int = 0
    raised_freq: float = 0.0
    fold_to_raise: float = 0.5
    call_down: float = 0.5

    def update_on_terminal(self, opp_last_action: str, we_raised_this_hand: bool):
        self.hands_seen += 1
        if we_raised_this_hand:
            if opp_last_action in ("FOLD", "INVALID"):
                self.folded_after_our_raise += 1
            elif opp_last_action in ("CALL", "CHECK", "RAISE"):
                self.called_after_our_raise += 1
            denom = max(1, self.folded_after_our_raise + self.called_after_our_raise)
            self.fold_to_raise = self.folded_after_our_raise / denom
            self.call_down = self.called_after_our_raise / denom

def default_rng(seed: Optional[int] = None) -> random.Random:
    return random.Random(0 if seed is None else seed)
