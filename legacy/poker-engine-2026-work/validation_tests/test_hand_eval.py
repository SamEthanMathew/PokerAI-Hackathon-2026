import os
import sys

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if 'submission' not in sys.modules:
    import submission67
    sys.modules['submission'] = submission67

from submission67.hand_eval import evaluate_5, _classify_hand
from submission67.card_utils import string_to_card

def test_ace_low_straight():
    """
    Test that A-2-3-4-5 is correctly recognized as an Ace-low straight.
    Ace = rank 8, 2,3,4,5 = ranks 0,1,2,3.
    """
    cards = [string_to_card(c) for c in ['Ad', '2h', '3s', '4d', '5h']]
    rank = evaluate_5(cards)
    
    # Let's compare to another straight: 2-3-4-5-6
    higher_straight = [string_to_card(c) for c in ['2d', '3h', '4s', '5d', '6h']]
    higher_rank = evaluate_5(higher_straight)
    
    # A-2-3-4-5 is the lowest straight, so it should have a HIGHER rank number than 2-3-4-5-6 (lower rank = better).
    assert rank > higher_rank, f"A-2-3-4-5 rank ({rank}) should be worse than 2-3-4-5-6 rank ({higher_rank})"

    # Now verify it is actually classified as a straight (in the straight block)
    # The range for straight is ~300000 - 300005
    assert 300000 <= rank <= 300005, f"Expected straight rank range, got {rank}"
    
def test_full_house_order():
    """
    Test Full House ranking logic. 
    A full house of 3s over 2s should be worse than 4s over 2s.
    """
    fh_3_over_2 = [string_to_card(c) for c in ['3d', '3h', '3s', '2d', '2h']]
    fh_4_over_2 = [string_to_card(c) for c in ['4d', '4h', '4s', '2d', '2h']]
    
    r1 = evaluate_5(fh_3_over_2)
    r2 = evaluate_5(fh_4_over_2)
    
    assert r2 < r1, f"4s over 2s ({r2}) should be better than 3s over 2s ({r1})"
    
    # Also verify range is in Full House bracket (~100000 to 100079)
    assert 100000 <= r1 <= 100079
    assert 100000 <= r2 <= 100079

def test_tied_flushes():
    """
    Test that two flushes with the exact same ranks but different suits have identical hand_eval values.
    """
    flush1 = [string_to_card(c) for c in ['2d', '4d', '6d', '8d', '9d']]
    flush2 = [string_to_card(c) for c in ['2h', '4h', '6h', '8h', '9h']]
    
    r1 = evaluate_5(flush1)
    r2 = evaluate_5(flush2)
    
    assert r1 == r2, f"Identical rank flushes should have the same eval score: {r1} vs {r2}"

def test_four_of_a_kind_is_impossible():
    """
    There are only 3 suits. Let's make sure our _classify_hand (if called manually) doesn't have bugs handling edge cases.
    We just verify the base `evaluate_5` function works as expected for a high card hand.
    """
    high_card = [string_to_card(c) for c in ['Ad', '2h', '4s', '6d', '8h']]
    rank = evaluate_5(high_card)
    
    # Should be > 700000
    assert rank > 700000, f"Expected high card rank > 700000, got {rank}"

if __name__ == '__main__':
    test_ace_low_straight()
    test_full_house_order()
    test_tied_flushes()
    test_four_of_a_kind_is_impossible()
    print("Hand evaluation tests passed.")
