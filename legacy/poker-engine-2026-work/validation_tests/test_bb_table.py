import os
import sys
import numpy as np

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if 'submission' not in sys.modules:
    import submission67
    sys.modules['submission'] = submission67

from submission67.player import _load_bb_table, _canonicalize_py, KEEP2
from submission67.equity import exact_equity

def test_bb_table_quantization():
    """
    Test BB table equity quantization limits (stored as uint8).
    Resolution limit = 1/255 ≈ 0.39%.
    We pick a random canonical entry, compute EXACT equity for one of its keep-pairs
    (enumerating all runouts), and compare it to the table's uint8 scaled equity.
    """
    table = _load_bb_table()
    assert table is not None, "bb_discard_table.bin failed to load. Did you run fix_data.py?"

    # We need to find a VALID canonical entry (not a noise entry from flop_table.bin)
    # A valid entry has:
    # 1. Hole is sorted
    # 2. Flop is sorted
    # 3. No duplicate cards
    
    canon_tuple = None
    for test_idx in range(100_000, len(table._keys)):
        packed = table._keys[test_idx]
        canon_cards = []
        temp = int(packed)
        for _ in range(8):
            canon_cards.append(temp & 0xFF)
            temp >>= 8
        
        hole = canon_cards[:5]
        flop = canon_cards[5:8]
        
        if sorted(hole) == hole and sorted(flop) == flop and len(set(canon_cards)) == 8:
            canon_tuple = tuple(canon_cards)
            break
            
    assert canon_tuple is not None, "Could not find a valid canonical entry"
    
    hole = list(canon_tuple[:5])
    flop = list(canon_tuple[5:8])
    
    # Get equities from table (uint8 array of 10 values)
    eq_uint8 = table.lookup(canon_tuple)
    assert eq_uint8 is not None, "Lookup failed for an existing key!"
    
    # Let's verify the first keep pair: KEEP2[0] = (0, 1)
    keep_idx = 0
    i, j = KEEP2[keep_idx]
    keep = [hole[i], hole[j]]
    
    table_equity = float(eq_uint8[keep_idx]) / 255.0
    
    # Compute exact equity over all 43,890 evaluations.
    # Exact equity requires all 5 hole cards and 3 flop cards to be "known"
    # because they are dead cards (we discarded the other 3 hole cards, so opponent can't have them).
    all_known = hole + flop
    
    print(f"Testing Canonical Entry: hole={hole}, flop={flop}")
    print(f"Keep pair {keep}")
    print(f"Table Uint8 Eq: {eq_uint8[keep_idx]} -> {table_equity:.4f}")
    
    true_equity = exact_equity(keep, flop, all_known)
    
    print(f"True Exact Eq: {true_equity:.4f}")
    
    quantization_bound = 1.0 / 255.0  # 0.00392
    
    # Assert true equity is within 1 quantization step (maybe slightly more for rounding)
    diff = abs(table_equity - true_equity)
    print(f"Difference: {diff:.4f} (Max Expected ~0.0039)")
    
    # Give a bit of margin for float math, but strictly enforce the uint8 quantization bound 
    assert diff <= quantization_bound + 0.0001, f"BB table equity deviates significantly from exact equity! Diff: {diff}"
