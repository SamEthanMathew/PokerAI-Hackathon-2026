"""
Abstraction / bucketing system for the 27-card poker variant.
Four subsystems: preflop 5-card, flop texture, opponent discard, keep archetype.
"""
from collections import Counter
from libratus.deck import rank, suit, same_suit, rank_gap, are_connected, are_semi_connected, RANK_A, RANK_9, RANK_8, NUM_RANKS


# ============================================================
# A. Preflop 5-card bucket
# ============================================================

def _made_hand_class_5(cards):
    """Classify a 5-card hand. Returns string label and tier (lower=better)."""
    ranks = sorted([rank(c) for c in cards])
    suits = [suit(c) for c in cards]
    rc = Counter(ranks)
    sc = Counter(suits)
    most_common_rank = rc.most_common(1)[0][1]
    most_common_suit = sc.most_common(1)[0][1]

    is_flush = most_common_suit >= 5
    unique_ranks = sorted(rc.keys())

    is_straight = False
    if len(unique_ranks) == 5:
        if unique_ranks[-1] - unique_ranks[0] == 4:
            is_straight = True
        # A-low: A,2,3,4,5 = ranks 8,0,1,2,3
        if set(unique_ranks) == {0, 1, 2, 3, 8}:
            is_straight = True
        # A-high: 6,7,8,9,A = ranks 4,5,6,7,8
        if set(unique_ranks) == {4, 5, 6, 7, 8}:
            is_straight = True

    if is_flush and is_straight:
        return "straight_flush", 0
    if most_common_rank == 3 and len(rc) == 2:
        return "full_house", 1
    if is_flush:
        return "flush", 2
    if is_straight:
        return "straight", 3
    if most_common_rank == 3:
        return "trips", 4
    if most_common_rank == 2 and len(rc) == 3:
        return "two_pair", 5
    if most_common_rank == 2:
        return "pair", 6
    return "high_card", 7


def preflop_features(cards):
    """Extract features from 5 hole cards."""
    ranks_list = [rank(c) for c in cards]
    suits_list = [suit(c) for c in cards]
    rc = Counter(ranks_list)
    sc = Counter(suits_list)

    hand_class, tier = _made_hand_class_5(cards)
    max_suit_count = sc.most_common(1)[0][1]
    best_pair_rank = -1
    for r, cnt in rc.items():
        if cnt >= 2 and r > best_pair_rank:
            best_pair_rank = r

    connectivity = 0
    sorted_ranks = sorted(set(ranks_list))
    for i in range(len(sorted_ranks) - 1):
        if sorted_ranks[i + 1] - sorted_ranks[i] == 1:
            connectivity += 1
    # A-2 wrap
    if RANK_A in sorted_ranks and 0 in sorted_ranks:
        connectivity += 1

    contains_ace = RANK_A in ranks_list
    contains_nine = RANK_9 in ranks_list

    suited_connectors = 0
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if same_suit(cards[i], cards[j]) and are_connected(cards[i], cards[j]):
                suited_connectors += 1

    return {
        "hand_class": hand_class,
        "tier": tier,
        "max_suit_count": max_suit_count,
        "best_pair_rank": best_pair_rank,
        "connectivity": connectivity,
        "contains_ace": contains_ace,
        "contains_nine": contains_nine,
        "suited_connectors": suited_connectors,
    }


def bucket_preflop(cards):
    """Assign a 5-card preflop hand to a bucket label."""
    f = preflop_features(cards)
    if f["tier"] <= 4:  # trips+, full house, flush, straight, SF
        return "trips_plus"
    if f["hand_class"] == "two_pair":
        if f["contains_ace"] or f["contains_nine"]:
            return "premium_two_pair"
        return "two_pair"
    if f["hand_class"] == "pair":
        if f["best_pair_rank"] in (RANK_A, RANK_9):
            return "premium_pair"
        if f["best_pair_rank"] >= RANK_8:
            return "high_pair"
        return "medium_pair"
    # High card hands
    if f["suited_connectors"] >= 1 and f["max_suit_count"] >= 3:
        return "suited_connected_draw"
    if f["contains_ace"] and (f["connectivity"] >= 2 or f["max_suit_count"] >= 3):
        return "high_card_structured"
    if f["connectivity"] >= 2 or f["max_suit_count"] >= 3:
        return "marginal"
    return "trash"


# ============================================================
# B. Flop texture bucket
# ============================================================

def flop_features(community):
    """Extract features from 3 flop cards."""
    ranks_list = [rank(c) for c in community]
    suits_list = [suit(c) for c in community]
    sc = Counter(suits_list)
    rc = Counter(ranks_list)

    if sc.most_common(1)[0][1] == 3:
        suit_pattern = "monotone"
    elif sc.most_common(1)[0][1] == 2:
        suit_pattern = "two_tone"
    else:
        suit_pattern = "rainbow"

    is_paired = rc.most_common(1)[0][1] >= 2

    sorted_ranks = sorted(set(ranks_list))
    connectivity = 0
    for i in range(len(sorted_ranks) - 1):
        if sorted_ranks[i + 1] - sorted_ranks[i] == 1:
            connectivity += 1
    if RANK_A in sorted_ranks and 0 in sorted_ranks:
        connectivity += 1

    return {
        "suit_pattern": suit_pattern,
        "is_paired": is_paired,
        "connectivity": connectivity,
        "high_card": max(ranks_list),
        "low_card": min(ranks_list),
    }


def bucket_flop(community):
    """Assign 3 flop cards to a texture bucket."""
    if len(community) < 3:
        return "unknown"
    f = flop_features(community)
    connected = f["connectivity"] >= 1
    if f["suit_pattern"] == "monotone":
        return "monotone_connected" if connected else "monotone_scattered"
    if f["suit_pattern"] == "two_tone":
        return "two_tone_connected" if connected else "two_tone_scattered"
    return "rainbow_connected" if connected else "rainbow_dry"


def bucket_flop_simple(community):
    """Simplified 3-bucket flop classification: wet / medium / dry."""
    if len(community) < 3:
        return "dry"
    f = flop_features(community)
    score = 0
    if f["suit_pattern"] == "monotone":
        score += 3
    elif f["suit_pattern"] == "two_tone":
        score += 1
    score += f["connectivity"]
    if f["is_paired"]:
        score += 1
    if score >= 3:
        return "wet"
    if score >= 1:
        return "medium"
    return "dry"


# ============================================================
# C. Opponent discard bucket
# ============================================================

def opp_discard_features(opp_discards):
    """Extract features from 3 opponent discarded cards."""
    ranks_list = [rank(c) for c in opp_discards]
    suits_list = [suit(c) for c in opp_discards]
    sc = Counter(suits_list)
    rc = Counter(ranks_list)

    sorted_ranks = sorted(ranks_list)
    connectivity = 0
    for i in range(len(sorted_ranks) - 1):
        if sorted_ranks[i + 1] - sorted_ranks[i] == 1:
            connectivity += 1
    if RANK_A in sorted_ranks and 0 in sorted_ranks:
        connectivity += 1

    return {
        "unique_suits": len(sc),
        "unique_ranks": len(rc),
        "max_rank": max(ranks_list),
        "min_rank": min(ranks_list),
        "has_ace": RANK_A in ranks_list,
        "has_pair": rc.most_common(1)[0][1] >= 2,
        "connectivity": connectivity,
        "max_suit_count": sc.most_common(1)[0][1],
    }


def bucket_opp_discard(opp_discards):
    """Assign 3 opponent discards to a bucket."""
    if len(opp_discards) < 3:
        return "unknown"
    f = opp_discard_features(opp_discards)
    if f["has_pair"]:
        return "discarded_pair"
    if f["max_suit_count"] >= 2:
        return "suited_cluster"
    if f["connectivity"] >= 2:
        return "connected_cluster"
    if f["has_ace"]:
        return "high_junk"
    if f["max_rank"] <= 5:
        return "low_junk"
    return "mixed_discard"


# ============================================================
# D. Keep archetype bucket
# ============================================================

def keep_features(keep2):
    """Extract features from 2 kept cards."""
    r1, r2 = rank(keep2[0]), rank(keep2[1])
    s1, s2 = suit(keep2[0]), suit(keep2[1])
    gap = abs(r1 - r2)
    # A-2 wrap: if gap=8, effective gap=1 for connectivity
    effective_gap = gap if gap <= 4 else NUM_RANKS - gap

    return {
        "is_pair": r1 == r2,
        "pair_rank": r1 if r1 == r2 else -1,
        "suited": s1 == s2,
        "rank_gap": gap,
        "effective_gap": effective_gap,
        "high_rank": max(r1, r2),
        "low_rank": min(r1, r2),
    }


def bucket_keep(keep2):
    """Assign a 2-card keep to an archetype bucket."""
    f = keep_features(keep2)
    if f["is_pair"]:
        if f["pair_rank"] in (RANK_A, RANK_9):
            return "premium_pair"
        if f["pair_rank"] >= RANK_8:
            return "medium_pair"
        return "low_pair"
    if f["suited"]:
        if f["effective_gap"] <= 1:
            return "suited_connector"
        if f["effective_gap"] <= 3:
            return "suited_semi"
        return "suited_gapper"
    if f["effective_gap"] <= 1:
        return "offsuit_connector"
    return "offsuit_other"


# ============================================================
# Strength bucket (post-discard, based on MC equity)
# ============================================================

def bucket_strength(equity: float) -> str:
    if equity > 0.80:
        return "monster"
    if equity > 0.65:
        return "strong"
    if equity > 0.50:
        return "good"
    if equity > 0.35:
        return "marginal"
    return "weak"


def bucket_to_call(to_call: int, pot_size: int) -> str:
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
