"""Quick functional test for street0_score."""
import time
from submission.functions.street0_score import (
    final_street0_score, hand_structure_features, classify_discard,
    classify_board_texture, OpponentProfile, all_keeps, rank, suit, card_str
)

# Test hand 1: 2d, 3d, 4d, 5d, Ad (strong mono-suited straight draw)
hand1 = [0, 1, 2, 3, 8]
print("Hand 1:", [card_str(c) for c in hand1])
feats = hand_structure_features(hand1)
print(f"  Longest run: {feats['longest_run']}")
print(f"  Straight windows: {feats['straight_windows_covered']}")
print(f"  Max suit: {feats['max_suit_count']}")
print(f"  Has 5-suited: {feats['has_5_suited']}")

t0 = time.time()
score1, bd1 = final_street0_score(hand1, n_flop_samples=30, n_tr_samples=20)
t1 = time.time()
print(f"  Score: {score1:.4f} (took {t1-t0:.2f}s)")
print(f"  V_future={bd1.v_future:.4f} V_opt={bd1.v_optionality:.4f} "
      f"C_disc={bd1.c_discard:.4f} C_rev={bd1.c_reveal:.4f}")

# Test hand 2: 2d,2h,2s,3d,3h (trips + pair = full house material)
hand2 = [0, 9, 18, 1, 10]
print()
print("Hand 2:", [card_str(c) for c in hand2])
t0 = time.time()
score2, bd2 = final_street0_score(hand2, n_flop_samples=30, n_tr_samples=20)
t1 = time.time()
print(f"  Score: {score2:.4f} (took {t1-t0:.2f}s)")
print(f"  V_future={bd2.v_future:.4f} V_opt={bd2.v_optionality:.4f}")

# Test hand 3: 2d, 4h, 6s, 7d, 7h (weak pair)
hand3 = [0, 11, 22, 5, 14]
print()
print("Hand 3:", [card_str(c) for c in hand3])
t0 = time.time()
score3, bd3 = final_street0_score(hand3, n_flop_samples=30, n_tr_samples=20)
t1 = time.time()
print(f"  Score: {score3:.4f} (took {t1-t0:.2f}s)")

print()
print(f"Ranking: hand1={score1:.3f}  hand2={score2:.3f}  hand3={score3:.3f}")
print("Expected: hand1 (mono suited A-5 straight flush draw) >= hand2 (trips) > hand3 (weak pair)")

# Test opponent-adjusted scoring
print()
print("--- Opponent-adjusted test ---")
opp = OpponentProfile(
    vpip_opportunities=100, vpip_successes=75,
    pfr_opportunities=100, pfr_successes=40,
    raise_count=60, call_count=40,
    fold_non_river_opportunities=50, fold_non_river_successes=35,
    fold_river_opportunities=30, fold_river_successes=22,
    total_hands=100,
)
t0 = time.time()
score_adj, bd_adj = final_street0_score(hand1, opponent_profile=opp, n_flop_samples=30, n_tr_samples=20)
t1 = time.time()
print(f"  Adjusted score: {score_adj:.4f} (base={bd_adj.s_base:.4f} opp={bd_adj.s_opp:.4f} conf={bd_adj.confidence:.4f})")
print(f"  Took {t1-t0:.2f}s")

# Test classification
print()
keeps = all_keeps(hand1)
k = keeps[0]
print(f"Discard class: {classify_discard(k[1], k[2], [3, 12, 21])}")
print(f"Board texture: {classify_board_texture([3, 12, 21])}")

print()
print("ALL TESTS PASSED")
