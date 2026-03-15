"""
Deck and card utilities for the 27-card poker variant.
Encoding: card_int in [0,26]. rank = card_int % 9, suit = card_int // 9.
Ranks: "23456789A" (indices 0-8). Suits: "dhs" (indices 0-2).
"""
import random

RANKS = "23456789A"
SUITS = "dhs"
NUM_RANKS = len(RANKS)
NUM_SUITS = len(SUITS)
DECK_SIZE = NUM_RANKS * NUM_SUITS  # 27
DECK = list(range(DECK_SIZE))

RANK_A = 8
RANK_9 = 7
RANK_8 = 6


def rank(c: int) -> int:
    return c % NUM_RANKS


def suit(c: int) -> int:
    return c // NUM_RANKS


def card_str(c: int) -> str:
    return RANKS[rank(c)] + SUITS[suit(c)]


def same_suit(c1: int, c2: int) -> bool:
    return suit(c1) == suit(c2)


def rank_gap(c1: int, c2: int) -> int:
    """Absolute rank difference between two cards."""
    return abs(rank(c1) - rank(c2))


def are_connected(c1: int, c2: int) -> bool:
    """Adjacent ranks (gap=1) or A wrapping to low (A-2 gap=8 counts)."""
    g = rank_gap(c1, c2)
    return g == 1 or g == 8  # A(8) and 2(0) are 8 apart but form A-2


def are_semi_connected(c1: int, c2: int) -> bool:
    g = rank_gap(c1, c2)
    return g == 2 or g == 7  # one-gapper, including A-3 wrap


def deal(n: int, exclude: set = None, rng: random.Random = None) -> list:
    """Deal n cards from the deck, excluding specified cards."""
    if exclude is None:
        exclude = set()
    pool = [c for c in DECK if c not in exclude]
    r = rng if rng else random
    return r.sample(pool, min(n, len(pool)))


def all_cards_of_suit(s: int) -> list:
    return [c for c in DECK if suit(c) == s]


def all_cards_of_rank(r: int) -> list:
    return [c for c in DECK if rank(c) == r]
