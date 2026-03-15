import random
from itertools import combinations

from agents.agent import Agent
from gym_env import PokerEnv

# Small random factor range (multiply bet sizes and equity by this)
RANDOM_FACTOR_LO = 0.96
RANDOM_FACTOR_HI = 1.04
# Pre-flop: premium pairs (AA, 99, 88) raise this fraction of max to force folds
PREFLOP_PAIR_AGGRESSION_MIN = 0.88
PREFLOP_PAIR_AGGRESSION_MAX = 1.00

RANKS = "23456789A"
NUM_RANKS = len(RANKS)

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
    """True if the two cards form AA, 99, or 88."""
    r1, r2 = _rank(c1), _rank(c2)
    return r1 == r2 and frozenset([r1, r1]) in PREMIUM_PAIRS


def _has_premium_pair(cards):
    """True if any 2 of the cards form AA, 99, or 88."""
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _is_premium_pair(cards[i], cards[j]):
                return True
    return False


def _score_keep(c1, c2, community, dead_cards):
    """Score a 2-card keep pair. Higher is better."""
    r1, r2 = _rank(c1), _rank(c2)
    s1, s2 = _suit(c1), _suit(c2)
    score = 0.0

    if _is_premium(c1, c2):
        score += 10000

    if r1 == r2:
        score += 5000 + r1 * 100
    elif r1 == 8 or r2 == 8:
        score += 3000

    comm_ranks = [_rank(c) for c in community]
    comm_suits = [_suit(c) for c in community]

    for r in (r1, r2):
        if r in comm_ranks:
            score += 2000

    suit_counts = {}
    for s in [s1, s2] + comm_suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_flush = max(suit_counts.values()) if suit_counts else 0
    if max_flush >= 3:
        score += 800 * max_flush

    all_ranks = sorted(set([r1, r2] + comm_ranks))
    longest_run = 1
    current_run = 1
    for k in range(1, len(all_ranks)):
        if all_ranks[k] == all_ranks[k - 1] + 1:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    if 8 in all_ranks and 0 in all_ranks:
        low_run = 1
        for r in range(1, 8):
            if r in all_ranks:
                low_run += 1
            else:
                break
        longest_run = max(longest_run, low_run + 1)
    if longest_run >= 3:
        score += 400 * longest_run

    dead_ranks = set(_rank(c) for c in dead_cards)
    for r in (r1, r2):
        if r in dead_ranks:
            score += 150

    score += (r1 + r2) * 10
    if s1 == s2:
        score += 200

    return score


def _best_keep(my_cards, community, opp_discards):
    best_score = -1
    best_ij = (0, 1)
    for i, j in combinations(range(len(my_cards)), 2):
        s = _score_keep(my_cards[i], my_cards[j], community, opp_discards)
        if s > best_score:
            best_score = s
            best_ij = (i, j)
    return best_ij


def _estimate_equity_two_cards(c1, c2, community):
    """
    Consistent heuristic equity estimate for our 2-card hand vs a random hand.
    Returns float in [0, 1]. No Monte Carlo - fast and deterministic before random factor.
    """
    r1, r2 = _rank(c1), _rank(c2)
    s1, s2 = _suit(c1), _suit(c2)
    comm_ranks = [_rank(c) for c in community] if community else []
    comm_suits = [_suit(c) for c in community] if community else []

    if _is_premium_pair(c1, c2):
        base = 0.88 + (r1 - 6) * 0.02  # AA ~0.92, 99 ~0.90, 88 ~0.88
    elif _is_premium(c1, c2):
        base = 0.66
    elif r1 == r2:
        base = 0.52 + r1 * 0.02
    elif r1 == 8 or r2 == 8:
        base = 0.48
    else:
        base = 0.38 + (max(r1, r2) * 0.01)

    # Board help: pair with board
    for r in (r1, r2):
        if r in comm_ranks:
            base += 0.12
            break
    # Flush draw
    suit_counts = {s1: 1, s2: 1}
    for s in comm_suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    if max(suit_counts.values()) >= 3:
        base += 0.06
    return min(0.98, max(0.05, base))


def _estimate_equity_five_cards(cards):
    """Best equity among all 2-card combos from 5 (pre-flop estimate)."""
    best = 0.0
    for i, j in combinations(range(len(cards)), 2):
        eq = _estimate_equity_two_cards(cards[i], cards[j], [])
        if eq > best:
            best = eq
    return best


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType

    def __name__(self):
        return "ALPHANIT"

    def act(self, observation, reward, terminated, truncated, info):
        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        opp_discards = [c for c in observation["opp_discarded_cards"] if c != -1]
        valid = observation["valid_actions"]
        street = observation["street"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        pot_size = observation.get("pot_size", my_bet + opp_bet)

        if valid[DISCARD]:
            i, j = _best_keep(my_cards, community, opp_discards)
            return (DISCARD, 0, i, j)

        # Consistent equity estimate, then apply small random factor
        if len(my_cards) > 2:
            equity = _estimate_equity_five_cards(my_cards)
            premium = _has_any_premium(my_cards)
            premium_pair = _has_premium_pair(my_cards)
        else:
            equity = _estimate_equity_two_cards(my_cards[0], my_cards[1], community)
            premium = _is_premium(my_cards[0], my_cards[1])
            premium_pair = _is_premium_pair(my_cards[0], my_cards[1])

        noise = random.uniform(RANDOM_FACTOR_LO, RANDOM_FACTOR_HI)
        equity_used = min(0.98, equity * noise)

        # Pot odds: cost to call / (pot after we call)
        to_call = opp_bet - my_bet
        pot_after_call = pot_size + to_call if to_call > 0 else pot_size
        pot_odds = (to_call / pot_after_call) if to_call > 0 and pot_after_call > 0 else 0.0

        if not premium:
            # Nit: check if free, else fold
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        # Premium hand: use equity to size bets; premium pairs pre-flop are extra aggressive
        if valid[RAISE]:
            if street == 0 and premium_pair:
                # Force folds: bet 88–100% of max with random factor
                frac = random.uniform(PREFLOP_PAIR_AGGRESSION_MIN, PREFLOP_PAIR_AGGRESSION_MAX)
                frac *= random.uniform(RANDOM_FACTOR_LO, RANDOM_FACTOR_HI)
                raise_amount = min(max_raise, max(min_raise, int(max_raise * frac)))
            else:
                # Bet size scales with equity, with random factor
                frac = equity_used * random.uniform(RANDOM_FACTOR_LO, RANDOM_FACTOR_HI)
                frac = min(0.98, frac)
                raise_amount = min(max_raise, max(min_raise, int(min_raise + (max_raise - min_raise) * frac)))
            return (RAISE, raise_amount, 0, 0)

        # Can't raise: call if equity beats pot odds (with small random margin), else fold
        if valid[CALL]:
            # Premium hands: call unless pot odds are very bad
            if equity_used >= pot_odds - 0.08 * noise:
                return (CALL, 0, 0, 0)
            if to_call <= 2:
                return (CALL, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        return (CHECK, 0, 0, 0)
