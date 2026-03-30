"""
Constants for the CMU AI Poker Tournament bot (v14 Alpha).
Avoids importing gym_env / PokerEnv at the module level so the submission
directory stays self-contained.
"""

# ── Action Types ──────────────────────────────────────────────────────────────
FOLD = 0
RAISE = 1
CHECK = 2
CALL = 3
DISCARD = 4
INVALID = 5

# ── Deck Constants ────────────────────────────────────────────────────────────
NUM_CARDS = 27
NUM_SUITS = 3
NUM_RANKS = 9

RANKS = "23456789A"  # index 0-8
SUITS = "dhs"        # index 0=diamonds, 1=hearts, 2=spades

# Rank indices for quick reference
RANK_2 = 0
RANK_3 = 1
RANK_4 = 2
RANK_5 = 3
RANK_6 = 4
RANK_7 = 5
RANK_8 = 6
RANK_9 = 7
RANK_A = 8

# ── Game Constants ────────────────────────────────────────────────────────────
SMALL_BLIND = 1
BIG_BLIND = 2
STARTING_STACK = 100
NUM_HANDS_PER_MATCH = 1000

# Street indices
STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3

# Player positions
POS_SB = 0  # Small blind
POS_BB = 1  # Big blind

# ── Time Budget ───────────────────────────────────────────────────────────────
PHASE2_TIME_BUDGET = 1000.0   # seconds for 1000 hands
PHASE3_TIME_BUDGET = 1500.0
