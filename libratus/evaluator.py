"""
Hand evaluator wrapper. Uses the engine's WrappedEval (treys-based).
Lower evaluate() score = stronger hand.
"""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gym_env import PokerEnv
from libratus.deck import DECK_SIZE, rank, suit, card_str

_ENV = PokerEnv()
_EVALUATOR = _ENV.evaluator
_int_to_card = PokerEnv.int_to_card


def evaluate_hand(hole2: list, board5: list) -> int:
    """Evaluate a 2-card hole hand against a 5-card board. Lower = better."""
    h = [_int_to_card(c) for c in hole2]
    b = [_int_to_card(c) for c in board5]
    return _EVALUATOR.evaluate(h, b)


def mc_equity(my2: list, community: list, dead: set = None, num_sims: int = 300,
              rng: random.Random = None) -> float:
    """
    Monte Carlo equity of my2 vs a random opponent 2-card hand.
    community can be 0-5 cards; missing board cards are sampled.
    dead cards (our discards, opp discards, etc.) are excluded from sampling.
    """
    if dead is None:
        dead = set()
    known = set(my2) | set(community) | dead
    remaining = [c for c in range(DECK_SIZE) if c not in known]
    board_needed = 5 - len(community)
    opp_needed = 2
    sample_needed = opp_needed + board_needed

    if sample_needed > len(remaining):
        return 0.5

    r = rng if rng else random
    wins = 0.0
    total = 0
    for _ in range(num_sims):
        sample = r.sample(remaining, sample_needed)
        opp = sample[:opp_needed]
        full_board = list(community) + sample[opp_needed:]
        my_rank = evaluate_hand(my2, full_board)
        opp_rank = evaluate_hand(opp, full_board)
        if my_rank < opp_rank:
            wins += 1.0
        elif my_rank == opp_rank:
            wins += 0.5
        total += 1
    return wins / total if total > 0 else 0.5
