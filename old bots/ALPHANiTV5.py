

import json
import os
import random
from itertools import combinations

from agents.agent import Agent
from gym_env import PokerEnv

# Load optional profile from logs/bot_profile.json (written by analyze_for_bot.py)
_PROFILE = {}
_profile_path = os.path.join(os.path.dirname(__file__), "..", "logs", "bot_profile.json")
if os.path.isfile(_profile_path):
    try:
        with open(_profile_path, "r", encoding="utf-8") as f:
            _PROFILE = json.load(f)
    except (json.JSONDecodeError, OSError):
        pass

RANDOM_FACTOR_LO = 0.96
RANDOM_FACTOR_HI = 1.04
SLOW_PLAY_CHANCE = 0.20
STANDARD_OPEN = _PROFILE.get("standard_open", 8)

MONSTER_THRESHOLD = 0.85
STRONG_THRESHOLD = 0.70
GOOD_THRESHOLD = 0.50
PREFLOP_COMMIT_THRESHOLD = 15
RIVER_CALL_POT_RATIO_MAX = _PROFILE.get("river_call_pot_ratio_max", 0.35)
BOARD_PAIRED_EQUITY_PENALTY = _PROFILE.get("board_paired_equity_penalty", 0.10)

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


def _board_paired_and_we_weak(my_cards, community):
    """
    Nit: on a paired board, we are weak if we only have one pair (or our pair
    is lower than the board pair). Returns True if we should reduce equity.
    """
    if len(my_cards) < 2 or len(community) < 2:
        return False
    board_ranks = [_rank(c) for c in community if c != -1]
    if len(board_ranks) < 2:
        return False
    board_pair_rank = None
    for r in set(board_ranks):
        if board_ranks.count(r) >= 2 and (board_pair_rank is None or r > board_pair_rank):
            board_pair_rank = r
    if board_pair_rank is None:
        return False
    r1, r2 = _rank(my_cards[0]), _rank(my_cards[1])
    if r1 == r2:
        our_pair_rank = r1
        if frozenset([r1, r1]) in PREMIUM_PAIRS:
            return False
        return our_pair_rank < board_pair_rank
    return True


def _infer_opp_threat(opp_discards, community):
    """
    Analyze opponent's 3 discarded cards to estimate how threatening
    their kept hand likely is. Returns a float 0.0-0.30.
    """
    if len(opp_discards) < 3:
        return 0.0

    threat = 0.0
    discard_suits = [_suit(c) for c in opp_discards]
    discard_ranks = sorted([_rank(c) for c in opp_discards])
    unique_discard_suits = set(discard_suits)

    if len(unique_discard_suits) == 3:
        threat += 0.08

    if all(r <= 5 for r in discard_ranks):
        threat += 0.07

    if len(set(discard_ranks)) == 3:
        threat += 0.05

    comm_suits = [_suit(c) for c in community]
    for s in range(3):
        if comm_suits.count(s) >= 2 and s not in unique_discard_suits:
            threat += 0.10
            break

    return min(0.30, threat)


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
            threat_suits = set()
            if opp_discards:
                unique_opp_suits = set(_suit(c) for c in opp_discards)
                for s in range(3):
                    if s not in unique_opp_suits:
                        threat_suits.add(s)

            best_eq = -1.0
            best_ij = (0, 1)
            for i, j in combinations(range(len(my_cards)), 2):
                keep = [my_cards[i], my_cards[j]]
                toss = [my_cards[k] for k in range(len(my_cards)) if k != i and k != j]
                eq = self._compute_equity(keep, community, opp_discards, toss, num_sims=150)
                if opp_discards and threat_suits:
                    blocking = sum(1 for c in keep if _suit(c) in threat_suits)
                    eq += 0.02 * blocking
                if eq > best_eq:
                    best_eq = eq
                    best_ij = (i, j)
            return (DISCARD, 0, best_ij[0], best_ij[1])

        # ---- Pre-flop (street 0): premium filter, standard open ----
        if street == 0:
            premium = _has_any_premium(my_cards)
            premium_pair = _has_premium_pair(my_cards)

            if premium:
                if premium_pair and opp_bet >= PREFLOP_COMMIT_THRESHOLD and valid[RAISE]:
                    return (RAISE, max_raise, 0, 0)

                if premium_pair and random.random() < SLOW_PLAY_CHANCE:
                    if valid[CALL]:
                        return (CALL, 0, 0, 0)
                    if valid[CHECK]:
                        return (CHECK, 0, 0, 0)

                noise = random.uniform(0.85, 1.15)
                open_size = _clamp(int(max(10, STANDARD_OPEN) * noise), min_raise, max_raise)
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

        opp_threat = _infer_opp_threat(opp_discards, community)
        equity_adj -= opp_threat
        if _board_paired_and_we_weak(my_cards, community):
            equity_adj -= BOARD_PAIRED_EQUITY_PENALTY

        to_call = opp_bet - my_bet
        pot_odds = to_call / (pot_size + to_call) if to_call > 0 and (pot_size + to_call) > 0 else 0.0

        if equity_adj > MONSTER_THRESHOLD:
            if street == 1:
                lo = _PROFILE.get("monster_bet_frac_flop_lo", 0.55)
                hi = _PROFILE.get("monster_bet_frac_flop_hi", 0.72)
                bet_frac = random.uniform(lo, hi)
            elif street == 2:
                bet_frac = random.uniform(0.70, 0.90)
            else:
                bet_frac = 1.0
            raise_amt = _clamp(int(pot_size * bet_frac), min_raise, max_raise)
            if street == 3 and valid[RAISE]:
                return (RAISE, max_raise, 0, 0)
            if valid[RAISE]:
                return (RAISE, raise_amt, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            return (CHECK, 0, 0, 0)

        elif equity_adj > STRONG_THRESHOLD:
            if street == 1:
                lo = _PROFILE.get("strong_bet_frac_flop_lo", 0.60)
                hi = _PROFILE.get("strong_bet_frac_flop_hi", 0.75)
                bet_frac = random.uniform(lo, hi)
            elif street == 2:
                bet_frac = random.uniform(0.65, 0.80)
            else:
                bet_frac = random.uniform(0.75, 0.90)
            raise_amt = _clamp(int(pot_size * bet_frac), min_raise, max_raise)
            if valid[RAISE]:
                return (RAISE, raise_amt, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            return (CHECK, 0, 0, 0)

        elif equity_adj > GOOD_THRESHOLD:
            bet_frac = random.uniform(0.30, 0.50)
            raise_amt = _clamp(int(pot_size * bet_frac), min_raise, max_raise)
            if valid[RAISE]:
                return (RAISE, raise_amt, 0, 0)
            if valid[CALL] and equity_adj >= pot_odds:
                return (CALL, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if len(my_cards) == 2 and _is_premium_pair(my_cards[0], my_cards[1]) and (to_call <= min_raise or (pot_size > 0 and to_call <= pot_size * 0.20)):
                if valid[CALL]:
                    return (CALL, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        elif equity_adj >= pot_odds and to_call <= pot_size * 0.3:
            if street == 3 and to_call > 0 and pot_size > 0 and to_call > pot_size * RIVER_CALL_POT_RATIO_MAX:
                if not (len(my_cards) == 2 and _is_premium_pair(my_cards[0], my_cards[1])):
                    if valid[CHECK]:
                        return (CHECK, 0, 0, 0)
                    return (FOLD, 0, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if len(my_cards) == 2 and _is_premium_pair(my_cards[0], my_cards[1]) and (to_call <= min_raise or (pot_size > 0 and to_call <= pot_size * 0.20)):
                if valid[CALL]:
                    return (CALL, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        else:
            if street == 3 and to_call > 0 and pot_size > 0 and to_call > pot_size * RIVER_CALL_POT_RATIO_MAX:
                if not (len(my_cards) == 2 and _is_premium_pair(my_cards[0], my_cards[1])):
                    if valid[CHECK]:
                        return (CHECK, 0, 0, 0)
                    return (FOLD, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if len(my_cards) == 2 and _is_premium_pair(my_cards[0], my_cards[1]) and (to_call <= min_raise or (pot_size > 0 and to_call <= pot_size * 0.20)):
                if valid[CALL]:
                    return (CALL, 0, 0, 0)
                if valid[CHECK]:
                    return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)
