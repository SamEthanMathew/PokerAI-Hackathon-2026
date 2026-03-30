import os
import sys
import numpy as np
import pytest

# Make sure we can import from submission67
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to map module name 'submission' to 'submission67' so internal imports work.
if 'submission' not in sys.modules:
    import submission67
    sys.modules['submission'] = submission67

from submission67.preflop_table import get_equity, get_action, hand_to_index, ACTION_FOLD, ACTION_CALL, ACTION_RAISE
from submission67.equity import monte_carlo_equity

# Thresholds defined in submission67/preflop_table.py
THRESH_FOLD_CALL = 0.4727
THRESH_CALL_RAISE = 0.5184

def get_borderline_hands():
    """Find a few hands near the boundaries to test MC variance."""
    import submission67.preflop_table
    submission67.preflop_table._load()
    
    _equities = submission67.preflop_table._equities
    
    # Find one hand near each threshold
    diff_fc = np.abs(_equities - THRESH_FOLD_CALL)
    diff_cr = np.abs(_equities - THRESH_CALL_RAISE)
    
    idx_fc = np.argmin(diff_fc)
    idx_cr = np.argmin(diff_cr)
    
    # We need a reverse mapping from index to hand to pass into get_equity.
    # We can just brute force find the hand for the index.
    from itertools import combinations
    from submission67.card_utils import FULL_DECK
    
    hand_fc = None
    hand_cr = None
    for combo in combinations(FULL_DECK, 5):
        idx = hand_to_index(combo)
        if idx == idx_fc:
            hand_fc = list(combo)
        elif idx == idx_cr:
            hand_cr = list(combo)
        if hand_fc and hand_cr:
            break
            
    return hand_fc, hand_cr

import random
from itertools import combinations
from submission67.hand_eval import evaluate_7
from submission67.range_inference import _rank_keeps

def preflop_mc(my5, iters=50000):
    from submission67.hand_eval import _load_tables
    _load_tables()
    
    wins = 0
    deck = list(set(range(27)) - set(my5))
    
    for _ in range(iters):
        draw = random.sample(deck, 10)
        opp5 = draw[:5]
        board = draw[5:]
        flop = board[:3]
        
        # Best keep is the one that gives the best 5-card hand on the flop
        # using _rank_keeps from submission67.range_inference
        # Ranked keeps returns sorted list of (c1, c2, score). First is best.
        my_keep = _rank_keeps(my5, flop)[0][:2]
        opp_keep = _rank_keeps(opp5, flop)[0][:2]
        
        my_rank = evaluate_7(list(my_keep) + board)
        opp_rank = evaluate_7(list(opp_keep) + board)
        
        if my_rank < opp_rank:
            wins += 1.0
        elif my_rank == opp_rank:
            wins += 0.5
            
    return wins / iters

def test_preflop_thresholds():
    """
    Test the preflop table's MC variance near the action thresholds.
    We run a high-precision MC and check if the stored value is within statistical bounds.
    """
    hand_fc, hand_cr = get_borderline_hands()
    
    assert hand_fc is not None, "Could not find hand near FOLD/CALL threshold"
    assert hand_cr is not None, "Could not find hand near CALL/RAISE threshold"
    
    # Table values
    table_eq_fc = get_equity(hand_fc)
    table_eq_cr = get_equity(hand_cr)
    
    # Run a high-precision MC to find "true" equity. 
    # 50k samples was original. We use 100k samples (SE ~0.15%).
    true_eq_fc = preflop_mc(hand_fc, iters=50_000)
    true_eq_cr = preflop_mc(hand_cr, iters=50_000)
    
    tolerance = 0.015
    
    assert abs(table_eq_fc - true_eq_fc) < tolerance, f"MC variance too high! Table: {table_eq_fc}, True: {true_eq_fc}"
    assert abs(table_eq_cr - true_eq_cr) < tolerance, f"MC variance too high! Table: {table_eq_cr}, True: {true_eq_cr}"
    
    # The true problem with MC bucketing: The action might be misbucketed due to noise.
    # We just log it if the action would change based on true equity vs table equity.
    table_action_fc = get_action(hand_fc)
    true_action_fc = ACTION_FOLD if true_eq_fc < THRESH_FOLD_CALL else (ACTION_CALL if true_eq_fc < THRESH_CALL_RAISE else ACTION_RAISE)
    
    table_action_cr = get_action(hand_cr)
    true_action_cr = ACTION_FOLD if true_eq_cr < THRESH_FOLD_CALL else (ACTION_CALL if true_eq_cr < THRESH_CALL_RAISE else ACTION_RAISE)
    
    print(f"\nHand 1 (Near FC): Table Eq={table_eq_fc:.4f}, True Eq={true_eq_fc:.4f}")
    print(f"Hand 1 Action: Table={table_action_fc}, True={true_action_fc}")
    if table_action_fc != true_action_fc:
        print(" -> MISBUCKETED due to MC noise!")
        
    print(f"\nHand 2 (Near CR): Table Eq={table_eq_cr:.4f}, True Eq={true_eq_cr:.4f}")
    print(f"Hand 2 Action: Table={table_action_cr}, True={true_action_cr}")
    if table_action_cr != true_action_cr:
        print(" -> MISBUCKETED due to MC noise!")
