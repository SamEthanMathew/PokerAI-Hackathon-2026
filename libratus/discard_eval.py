"""
Keep-selection scoring engine. The center of the bot.
Evaluates all C(5,2)=10 ways to keep 2 of 5 cards at flop discard time.
"""
import random
from itertools import combinations

from libratus.deck import rank, suit, same_suit, are_connected, are_semi_connected, rank_gap, RANK_A, RANK_9, RANK_8, NUM_RANKS
from libratus.evaluator import mc_equity
from libratus.abstractions import bucket_opp_discard, keep_features

# Scoring weights (tunable)
W_EQUITY = 3.0
W_STRUCTURAL = 1.5
W_BOARD = 1.0
W_INFERENCE = 0.5


def _structural_bonus(keep2):
    """Bonus for structurally strong keeps (suited connectors, premium pairs, etc.)."""
    f = keep_features(keep2)
    if f["is_pair"]:
        if f["pair_rank"] in (RANK_A, RANK_9):
            return 0.10
        if f["pair_rank"] >= RANK_8:
            return 0.05
        return 0.03
    if f["suited"]:
        if f["effective_gap"] <= 1:
            return 0.12  # suited connector
        if f["effective_gap"] <= 3:
            return 0.08  # suited semi-connector
        return 0.06  # suited gapper
    if f["effective_gap"] <= 1:
        return 0.04  # offsuit connector
    return 0.0


def _board_interaction_bonus(keep2, community):
    """Bonus for keep2 interacting with the flop."""
    if not community:
        return 0.0
    bonus = 0.0
    k_suits = [suit(c) for c in keep2]
    k_ranks = [rank(c) for c in keep2]
    b_suits = [suit(c) for c in community]
    b_ranks = [rank(c) for c in community]

    # Flush draw: keep suit matches 2+ board cards
    for s in set(k_suits):
        board_match = sum(1 for bs in b_suits if bs == s)
        keep_match = sum(1 for ks in k_suits if ks == s)
        if board_match >= 2 and keep_match >= 1:
            bonus += 0.08
            if board_match >= 2 and keep_match >= 2:
                bonus += 0.04  # both cards flush draw
            break

    # Straight draw: keep ranks connect with board ranks
    all_ranks = sorted(set(k_ranks + b_ranks))
    consec = 1
    max_consec = 1
    for i in range(1, len(all_ranks)):
        if all_ranks[i] - all_ranks[i - 1] == 1:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 1
    # A-low wrap check
    if RANK_A in all_ranks and 0 in all_ranks:
        max_consec = max(max_consec, 2)
    if max_consec >= 4:
        bonus += 0.06

    # Trips draw: one of our cards matches a board rank
    for kr in k_ranks:
        if kr in b_ranks:
            bonus += 0.04
            break

    return bonus


def _inference_bonus(keep2, opp_discards, community):
    """
    SB-only bonus. Infer what opponent likely kept from their discards
    and award blocking/exploitation bonuses.
    """
    if not opp_discards or len(opp_discards) < 3:
        return 0.0

    bonus = 0.0
    opp_bucket = bucket_opp_discard(opp_discards)
    opp_d_suits = set(suit(c) for c in opp_discards)
    k_suits = [suit(c) for c in keep2]
    b_suits = [suit(c) for c in community] if community else []

    # If opponent discarded low junk, they kept high cards. Penalty for our low keeps.
    if opp_bucket == "low_junk":
        k_ranks = [rank(c) for c in keep2]
        if max(k_ranks) <= 5:
            bonus -= 0.04  # our keep is weak against their likely strong hand

    # If opponent discarded a suited cluster, they likely kept the other suit.
    # Block their flush by keeping cards of the suit they probably kept.
    if opp_bucket == "suited_cluster":
        threat_suits = set(range(3)) - opp_d_suits
        blocking = sum(1 for ks in k_suits if ks in threat_suits)
        bonus += 0.02 * blocking

        # Extra bonus if board also supports their flush suit
        for ts in threat_suits:
            if sum(1 for bs in b_suits if bs == ts) >= 2:
                blocking_board = sum(1 for ks in k_suits if ks == ts)
                bonus += 0.03 * blocking_board
                break

    # If opponent discarded a pair, they broke a pair to go suited/connected.
    # They are speculative, so we can be more aggressive.
    if opp_bucket == "discarded_pair":
        bonus += 0.02

    return bonus


def score_keep(keep2, toss3, community, opp_discards, num_sims=200, rng=None):
    """
    Score a single keep choice.
    Returns (total_score, equity, structural, board, inference).
    """
    dead = set(toss3)
    if opp_discards:
        dead |= set(opp_discards)

    equity = mc_equity(keep2, community, dead, num_sims=num_sims, rng=rng)
    structural = _structural_bonus(keep2)
    board = _board_interaction_bonus(keep2, community)
    inference = _inference_bonus(keep2, opp_discards, community)

    total = W_EQUITY * equity + W_STRUCTURAL * structural + W_BOARD * board + W_INFERENCE * inference
    return total, equity, structural, board, inference


def score_all_keeps(my5, community, opp_discards=None, num_sims=200, rng=None):
    """
    Score all 10 possible 2-card keeps from 5 hole cards.
    Returns list of (i, j, total_score, equity) sorted best-first.
    """
    results = []
    for i, j in combinations(range(len(my5)), 2):
        keep = [my5[i], my5[j]]
        toss = [my5[k] for k in range(len(my5)) if k != i and k != j]
        total, eq, st, bd, inf = score_keep(keep, toss, community, opp_discards, num_sims, rng)
        results.append((i, j, total, eq))
    results.sort(key=lambda x: -x[2])  # best total score first
    return results


def choose_best_keep(my5, community, opp_discards=None, num_sims=200, rng=None):
    """
    Choose the best 2-card keep. Returns (keep_idx_1, keep_idx_2).
    For near-ties (within 0.02 total score), use rng to pick randomly.
    """
    scored = score_all_keeps(my5, community, opp_discards, num_sims, rng)
    if not scored:
        return (0, 1)

    best_score = scored[0][2]
    candidates = [(i, j) for i, j, sc, _ in scored if best_score - sc < 0.06]

    if len(candidates) <= 1:
        return (scored[0][0], scored[0][1])

    r = rng if rng else random
    pick = r.choice(candidates)
    return pick
