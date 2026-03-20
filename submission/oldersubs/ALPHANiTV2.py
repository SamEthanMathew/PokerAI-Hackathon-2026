import random
from itertools import combinations

from agents.agent import Agent
from gym_env import PokerEnv

RANDOM_FACTOR_LO = 0.96
RANDOM_FACTOR_HI = 1.04
SLOW_PLAY_CHANCE = 0.20
STANDARD_OPEN = 6

RANKS = "23456789A"
NUM_RANKS = len(RANKS)
DECK_SIZE = 27

int_to_card = PokerEnv.int_to_card

FOLD = PokerEnv.ActionType.FOLD.value
RAISE = PokerEnv.ActionType.RAISE.value
CHECK = PokerEnv.ActionType.CHECK.value
CALL = PokerEnv.ActionType.CALL.value
DISCARD = PokerEnv.ActionType.DISCARD.value

PREMIUM_PAIRS = frozenset([
    frozenset([8, 8]),  # AA
    frozenset([7, 7]),  # 99
    frozenset([6, 6]),  # 88
])

PREMIUM_ANY_SUIT = frozenset([
    frozenset([8, 7]),  # A9
    frozenset([8, 6]),  # A8
])

PREMIUM_SUITED_ONLY = frozenset([
    frozenset([7, 6]),  # 98s
    frozenset([6, 5]),  # 87s
    frozenset([5, 4]),  # 76s
    frozenset([5, 7]),  # 79s
])


def _rank(card):
    return card % NUM_RANKS


def _suit(card):
    return card // NUM_RANKS


def _is_premium(c1, c2):
    r1, r2 = _rank(c1), _rank(c2)
    s1, s2 = _suit(c1), _suit(c2)
    ranks = frozenset([r1, r2])

    if r1 == r2 and frozenset([r1, r1]) in PREMIUM_PAIRS:
        return True
    if ranks in PREMIUM_ANY_SUIT:
        return True
    if s1 == s2 and ranks in PREMIUM_SUITED_ONLY:
        return True
    return False


def _has_any_premium(cards):
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _is_premium(cards[i], cards[j]):
                return True
    return False


def _is_premium_pair(c1, c2):
    r1, r2 = _rank(c1), _rank(c2)
    return r1 == r2 and frozenset([r1, r1]) in PREMIUM_PAIRS


def _has_premium_pair(cards):
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _is_premium_pair(cards[i], cards[j]):
                return True
    return False


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.evaluator = PokerEnv().evaluator

    def __name__(self):
        return "ALPHANIT_V2"

    def _compute_equity(self, my_cards, community, opp_discards, my_discards, num_sims=300):
        """
        Monte Carlo win probability. All known/dead cards are excluded from
        the sampling pool: our cards, visible community, both sets of discards.
        """
        dead = set(my_cards)
        for c in community:
            if c != -1:
                dead.add(c)
        for c in opp_discards:
            if c != -1:
                dead.add(c)
        for c in my_discards:
            if c != -1:
                dead.add(c)

        remaining = [i for i in range(DECK_SIZE) if i not in dead]
        board_needed = 5 - len(community)
        opp_needed = 2
        sample_size = opp_needed + board_needed

        if sample_size > len(remaining):
            return 0.5

        wins = 0
        total = 0
        for _ in range(num_sims):
            sample = random.sample(remaining, sample_size)
            opp_cards = sample[:opp_needed]
            full_board = list(community) + sample[opp_needed:]

            my_hand = list(map(int_to_card, my_cards))
            opp_hand = list(map(int_to_card, opp_cards))
            board = list(map(int_to_card, full_board))

            my_rank = self.evaluator.evaluate(my_hand, board)
            opp_rank = self.evaluator.evaluate(opp_hand, board)
            if my_rank < opp_rank:
                wins += 1
            elif my_rank == opp_rank:
                wins += 0.5
            total += 1

        return wins / total if total > 0 else 0.5

    def act(self, observation, reward, terminated, truncated, info):
        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        my_discards = [c for c in observation["my_discarded_cards"] if c != -1]
        valid = observation["valid_actions"]
        street = observation["street"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        pot_size = observation.get("pot_size", my_bet + opp_bet)

        # ---- Discard phase: MC eval all 10 keep combos ----
        if valid[DISCARD]:
            best_eq = -1.0
            best_ij = (0, 1)
            for i, j in combinations(range(len(my_cards)), 2):
                keep = [my_cards[i], my_cards[j]]
                toss = [my_cards[k] for k in range(len(my_cards)) if k != i and k != j]
                eq = self._compute_equity(keep, community, opp_discards, toss, num_sims=150)
                if eq > best_eq:
                    best_eq = eq
                    best_ij = (i, j)
            return (DISCARD, 0, best_ij[0], best_ij[1])

        # ---- Pre-flop (street 0): premium filter, standard open ----
        if street == 0:
            premium = _has_any_premium(my_cards)
            premium_pair = _has_premium_pair(my_cards)

            if premium:
                if premium_pair and random.random() < SLOW_PLAY_CHANCE:
                    if valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[CHECK]:
                        return (CHECK, 0, 0, 0)

                noise = random.uniform(0.85, 1.15)
                open_size = _clamp(int(STANDARD_OPEN * noise), min_raise, max_raise)
                if valid[RAISE]:
                    return (RAISE, open_size, 0, 0)
                if valid[CALL]:
                    return (CALL, 0, 0, 0)
                return (CHECK, 0, 0, 0)
            else:
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
                return (FOLD, 0, 0, 0)

        # ---- Post-flop (streets 1-3): equity-driven betting ----
        if len(my_cards) > 2:
            my_cards = my_cards[:2]

        equity = self._compute_equity(my_cards, community, opp_discards, my_discards, num_sims=300)
        noise = random.uniform(RANDOM_FACTOR_LO, RANDOM_FACTOR_HI)
        equity_adj = min(0.98, equity * noise)

        to_call = opp_bet - my_bet
        pot_odds = to_call / (pot_size + to_call) if to_call > 0 and (pot_size + to_call) > 0 else 0.0

        if equity_adj > 0.70:
            bet_frac = random.uniform(0.60, 0.85)
            raise_amt = _clamp(int(pot_size * bet_frac), min_raise, max_raise)
            if raise_amt < min_raise:
                raise_amt = min_raise
            if valid[RAISE]:
                return (RAISE, raise_amt, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            return (CHECK, 0, 0, 0)

        elif equity_adj > 0.50:
            bet_frac = random.uniform(0.30, 0.50)
            raise_amt = _clamp(int(pot_size * bet_frac), min_raise, max_raise)
            if valid[RAISE]:
                return (RAISE, raise_amt, 0, 0)
            if valid[CALL] and equity_adj >= pot_odds:
                return (CALL, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        elif equity_adj >= pot_odds and to_call <= pot_size * 0.3:
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        else:
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)
