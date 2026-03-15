#CURRENT: LibratusV1

"""
Libratus-Lite: A loose-aggressive, table-driven, mixed-strategy poker bot.
Runtime component -- lightweight policy lookup with MC discard evaluation.
"""
import random
from collections import Counter
from itertools import combinations

from agents.agent import Agent
from gym_env import PokerEnv

try:
    from submission.libratus_tables import POLICY, KEEP_EQUITY, POSTERIOR, MATCHUPS
except ImportError:
    from libratus_tables import POLICY, KEEP_EQUITY, POSTERIOR, MATCHUPS

# ── Constants ────────────────────────────────────────────────────────────────

RANKS = "23456789A"
SUITS = "dhs"
NUM_RANKS = 9
DECK_SIZE = 27
RANK_A = 8
RANK_9 = 7
RANK_8 = 6

FOLD = PokerEnv.ActionType.FOLD.value
RAISE = PokerEnv.ActionType.RAISE.value
CHECK = PokerEnv.ActionType.CHECK.value
CALL = PokerEnv.ActionType.CALL.value
DISCARD = PokerEnv.ActionType.DISCARD.value

_int_to_card = PokerEnv.int_to_card

# ── Card helpers ─────────────────────────────────────────────────────────────

def _rank(c):
    return c % NUM_RANKS

def _suit(c):
    return c // NUM_RANKS

def _same_suit(c1, c2):
    return _suit(c1) == _suit(c2)

def _rank_gap(c1, c2):
    return abs(_rank(c1) - _rank(c2))

def _effective_gap(c1, c2):
    g = _rank_gap(c1, c2)
    return g if g <= 4 else NUM_RANKS - g

def _is_connected(c1, c2):
    return _effective_gap(c1, c2) == 1

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ── Bucketing functions (inlined for runtime speed) ──────────────────────────

def _bucket_keep(keep2):
    r1, r2 = _rank(keep2[0]), _rank(keep2[1])
    s1, s2 = _suit(keep2[0]), _suit(keep2[1])
    is_pair = r1 == r2
    suited = s1 == s2
    gap = abs(r1 - r2)
    eg = gap if gap <= 4 else NUM_RANKS - gap

    if is_pair:
        if r1 in (RANK_A, RANK_9):
            return "premium_pair"
        if r1 >= RANK_8:
            return "medium_pair"
        return "low_pair"
    if suited:
        if eg <= 1:
            return "suited_connector"
        if eg <= 3:
            return "suited_semi"
        return "suited_gapper"
    if eg <= 1:
        return "offsuit_connector"
    return "offsuit_other"


def _bucket_flop_simple(community):
    if len(community) < 3:
        return "dry"
    suits = [_suit(c) for c in community[:3]]
    ranks = [_rank(c) for c in community[:3]]
    sc = Counter(suits)
    rc = Counter(ranks)
    max_sc = sc.most_common(1)[0][1]
    is_paired = rc.most_common(1)[0][1] >= 2

    sorted_r = sorted(set(ranks))
    conn = 0
    for i in range(len(sorted_r) - 1):
        if sorted_r[i + 1] - sorted_r[i] == 1:
            conn += 1
    if RANK_A in sorted_r and 0 in sorted_r:
        conn += 1

    score = 0
    if max_sc >= 3:
        score += 3
    elif max_sc >= 2:
        score += 1
    score += conn
    if is_paired:
        score += 1
    if score >= 3:
        return "wet"
    if score >= 1:
        return "medium"
    return "dry"


def _bucket_strength(equity):
    if equity > 0.80:
        return "monster"
    if equity > 0.65:
        return "strong"
    if equity > 0.50:
        return "good"
    if equity > 0.35:
        return "marginal"
    return "weak"


def _bucket_to_call(to_call, pot_size):
    if to_call <= 0:
        return "none"
    if pot_size <= 0:
        return "large"
    ratio = to_call / pot_size
    if ratio <= 0.15:
        return "small"
    if ratio <= 0.40:
        return "medium"
    return "large"


def _bucket_opp_discard(opp_discards):
    if len(opp_discards) < 3:
        return "unknown"
    ranks = [_rank(c) for c in opp_discards]
    suits = [_suit(c) for c in opp_discards]
    sc = Counter(suits)
    rc = Counter(ranks)
    has_pair = rc.most_common(1)[0][1] >= 2
    max_sc = sc.most_common(1)[0][1]
    has_ace = RANK_A in ranks

    sorted_r = sorted(ranks)
    conn = 0
    for i in range(len(sorted_r) - 1):
        if sorted_r[i + 1] - sorted_r[i] == 1:
            conn += 1
    if RANK_A in sorted_r and 0 in sorted_r:
        conn += 1

    if has_pair:
        return "discarded_pair"
    if max_sc >= 2:
        return "suited_cluster"
    if conn >= 2:
        return "connected_cluster"
    if has_ace:
        return "high_junk"
    if max(ranks) <= 5:
        return "low_junk"
    return "mixed_discard"


# ── Keep scoring (runtime version, lightweight MC) ───────────────────────────

def _structural_bonus(keep2):
    r1, r2 = _rank(keep2[0]), _rank(keep2[1])
    if r1 == r2:
        if r1 in (RANK_A, RANK_9):
            return 0.10
        if r1 >= RANK_8:
            return 0.05
        return 0.03
    eg = _effective_gap(keep2[0], keep2[1])
    if _same_suit(keep2[0], keep2[1]):
        if eg <= 1:
            return 0.12
        if eg <= 3:
            return 0.08
        return 0.06
    if eg <= 1:
        return 0.04
    return 0.0


def _board_interaction(keep2, community):
    if not community:
        return 0.0
    bonus = 0.0
    k_suits = [_suit(c) for c in keep2]
    k_ranks = [_rank(c) for c in keep2]
    b_suits = [_suit(c) for c in community]
    b_ranks = [_rank(c) for c in community]

    for s in set(k_suits):
        bm = sum(1 for bs in b_suits if bs == s)
        km = sum(1 for ks in k_suits if ks == s)
        if bm >= 2 and km >= 1:
            bonus += 0.08
            if km >= 2:
                bonus += 0.04
            break

    all_r = sorted(set(k_ranks + b_ranks))
    mc = 1
    best = 1
    for i in range(1, len(all_r)):
        if all_r[i] - all_r[i - 1] == 1:
            mc += 1
            best = max(best, mc)
        else:
            mc = 1
    if RANK_A in all_r and 0 in all_r:
        best = max(best, 2)
    if best >= 4:
        bonus += 0.06

    for kr in k_ranks:
        if kr in b_ranks:
            bonus += 0.04
            break
    return bonus


def _inference_bonus(keep2, opp_discards, community):
    if not opp_discards or len(opp_discards) < 3:
        return 0.0
    bonus = 0.0
    opp_b = _bucket_opp_discard(opp_discards)
    opp_d_suits = set(_suit(c) for c in opp_discards)
    k_suits = [_suit(c) for c in keep2]
    b_suits = [_suit(c) for c in community] if community else []

    if opp_b == "low_junk":
        if max(_rank(c) for c in keep2) <= 5:
            bonus -= 0.04

    if opp_b == "suited_cluster":
        threat_suits = set(range(3)) - opp_d_suits
        blocking = sum(1 for ks in k_suits if ks in threat_suits)
        bonus += 0.02 * blocking
        for ts in threat_suits:
            if sum(1 for bs in b_suits if bs == ts) >= 2:
                bonus += 0.03 * sum(1 for ks in k_suits if ks == ts)
                break

    if opp_b == "discarded_pair":
        bonus += 0.02
    return bonus


# ── PlayerAgent ──────────────────────────────────────────────────────────────

class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self._env = PokerEnv()
        self.evaluator = self._env.evaluator

        self.hand_number = 0
        self.opp_stats = {
            "fold_to_raise": [0, 0],  # [folds, total]
            "aggression": [0, 0],     # [raises, opportunities]
            "river_calldown": [0, 0],
        }

    def __name__(self):
        return "Libratus"

    # ── Seeded RNG ──────────────────────────────────────────────────────

    def _seed_rng(self, my_cards, street):
        seed_val = hash((self.hand_number, street, tuple(sorted(my_cards))))
        self._rng = random.Random(seed_val)

    # ── MC equity ───────────────────────────────────────────────────────

    def _mc_equity(self, my2, community, dead, num_sims=200):
        known = set(my2) | set(community) | dead
        remaining = [c for c in range(DECK_SIZE) if c not in known]
        board_needed = 5 - len(community)
        sample_needed = 2 + board_needed

        if sample_needed > len(remaining):
            return 0.5

        wins = 0.0
        total = 0
        for _ in range(num_sims):
            sample = self._rng.sample(remaining, sample_needed)
            opp = sample[:2]
            full_board = list(community) + sample[2:]
            my_hand = [_int_to_card(c) for c in my2]
            opp_hand = [_int_to_card(c) for c in opp]
            board = [_int_to_card(c) for c in full_board]
            mr = self.evaluator.evaluate(my_hand, board)
            orank = self.evaluator.evaluate(opp_hand, board)
            if mr < orank:
                wins += 1.0
            elif mr == orank:
                wins += 0.5
            total += 1
        return wins / total if total > 0 else 0.5

    # ── Discard: choose best keep ───────────────────────────────────────

    def _choose_keep(self, my_cards, community, opp_discards):
        best_score = -999.0
        best_ij = (0, 1)
        candidates = []

        for i, j in combinations(range(len(my_cards)), 2):
            keep = [my_cards[i], my_cards[j]]
            toss = [my_cards[k] for k in range(len(my_cards)) if k not in (i, j)]
            dead = set(toss)
            if opp_discards:
                dead |= set(opp_discards)

            eq = self._mc_equity(keep, community, dead, num_sims=150)
            struct = _structural_bonus(keep)
            board = _board_interaction(keep, community)
            infer = _inference_bonus(keep, opp_discards, community)

            score = 3.0 * eq + 1.5 * struct + 1.0 * board + 0.5 * infer
            candidates.append((i, j, score, eq))
            if score > best_score:
                best_score = score
                best_ij = (i, j)

        # Near-tie randomization
        ties = [(i, j) for i, j, sc, _ in candidates if best_score - sc < 0.06]
        if len(ties) > 1:
            best_ij = self._rng.choice(ties)

        return best_ij

    # ── Betting: policy table lookup + mixed strategy ───────────────────

    def _choose_bet(self, street, my_cards, community, opp_discards, my_discards,
                    valid, min_raise, max_raise, my_bet, opp_bet, pot_size, blind_pos):
        to_call = max(0, opp_bet - my_bet)
        position = "sb" if blind_pos == 0 else "bb"

        dead = set()
        if my_discards:
            dead |= set(my_discards)
        if opp_discards:
            dead |= set(opp_discards)

        # Compute equity
        if len(my_cards) == 2 and len(community) >= 3:
            equity = self._mc_equity(my_cards, community, dead, num_sims=200)
        elif len(my_cards) == 5 and street == 0:
            equity = self._mc_equity(my_cards[:2], [], dead, num_sims=100)
        else:
            equity = 0.45

        # Adjust equity based on opponent discard inference
        if opp_discards and len(opp_discards) >= 3:
            opp_b = _bucket_opp_discard(opp_discards)
            # Tighten slightly vs strong opponent discard signals
            if opp_b == "low_junk":
                equity -= 0.03  # they kept strong cards
            elif opp_b == "discarded_pair":
                equity += 0.02  # they're speculative

        strength = _bucket_strength(equity)
        board_b = _bucket_flop_simple(community) if street >= 1 else "any"
        tc_b = _bucket_to_call(to_call, pot_size)

        # Look up policy
        key = str((street, position, strength, board_b, tc_b))
        policy = POLICY.get(key)
        if not policy:
            # Fallback: try with "medium" board and "none" to_call
            key2 = str((street, position, strength, "medium", "none"))
            policy = POLICY.get(key2, {
                "fold": 0.20, "check_call": 0.40, "small_bet": 0.25,
                "medium_bet": 0.10, "large_bet": 0.05, "jam": 0.0
            })

        # Mild opponent-model adjustment
        opp_fold_rate = self._opp_fold_rate()
        if opp_fold_rate > 0.5:
            # Opponent folds a lot: bluff more
            policy = dict(policy)
            policy["small_bet"] = policy.get("small_bet", 0) + 0.05
            policy["fold"] = max(0, policy.get("fold", 0) - 0.05)

        # Sample action from mixed policy
        action = self._sample_action(policy)

        # Convert to game action
        return self._action_to_tuple(
            action, valid, min_raise, max_raise, pot_size, to_call, equity
        )

    def _sample_action(self, policy):
        r = self._rng.random()
        cumul = 0.0
        for act, prob in policy.items():
            cumul += prob
            if r < cumul:
                return act
        return "check_call"

    def _action_to_tuple(self, action, valid, min_raise, max_raise, pot_size, to_call, equity):
        if action == "fold":
            if valid[FOLD]:
                return (FOLD, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        if action == "check_call":
            if to_call > 0 and valid[CALL]:
                return (CALL, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            return (CHECK, 0, 0, 0)

        if action == "jam":
            if valid[RAISE] and max_raise > 0:
                return (RAISE, max_raise, 0, 0)
            if valid[CALL]:
                return (CALL, 0, 0, 0)
            if valid[CHECK]:
                return (CHECK, 0, 0, 0)
            return (FOLD, 0, 0, 0)

        # Bet sizing
        pot_ref = max(pot_size, 1)
        if action == "small_bet":
            frac = self._rng.uniform(0.30, 0.45)
        elif action == "medium_bet":
            frac = self._rng.uniform(0.55, 0.75)
        elif action == "large_bet":
            frac = self._rng.uniform(0.85, 1.10)
        else:
            frac = 0.50

        raw_amount = int(pot_ref * frac)
        amount = _clamp(raw_amount, min_raise, max_raise)

        if valid[RAISE] and max_raise >= min_raise:
            return (RAISE, amount, 0, 0)
        if valid[CALL]:
            return (CALL, 0, 0, 0)
        if valid[CHECK]:
            return (CHECK, 0, 0, 0)
        return (FOLD, 0, 0, 0)

    # ── Opponent model ──────────────────────────────────────────────────

    def _opp_fold_rate(self):
        folds, total = self.opp_stats["fold_to_raise"]
        if total < 10:
            return 0.35  # default
        return folds / total

    def _update_opp_stats(self, observation):
        pass  # tracked via external log analysis for now

    # ── Main act function ───────────────────────────────────────────────

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
        blind_pos = observation.get("blind_position", 0)

        # Track hand number for seeded RNG
        if street == 0 and my_bet <= 2 and opp_bet <= 2:
            self.hand_number += 1

        self._seed_rng(my_cards, street)

        # ── Discard phase ───────────────────────────────────────────
        if valid[DISCARD]:
            i, j = self._choose_keep(my_cards, community, opp_discards)
            return (DISCARD, 0, i, j)

        # ── Betting phase ───────────────────────────────────────────
        return self._choose_bet(
            street, my_cards, community, opp_discards, my_discards,
            valid, min_raise, max_raise, my_bet, opp_bet, pot_size, blind_pos
        )
