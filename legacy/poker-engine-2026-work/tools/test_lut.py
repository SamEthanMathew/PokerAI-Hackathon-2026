import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

_RANKS_STR = '23456789A'
_SUITS_STR = 'dhs'
NUM_RANKS = 9

def _str_to_int(card_str):
    rank_ch = card_str[0]
    suit_ch = card_str[1]
    return _RANKS_STR.index(rank_ch) + _SUITS_STR.index(suit_ch) * NUM_RANKS

# Load LUT and _C
_EVAL_LUT = np.load(os.path.join(os.path.dirname(__file__), '..', 'submission', 'eval_table.npy'))
_C = [[0] * 28 for _ in range(28)]
for n in range(28):
    _C[n][0] = 1
    _C[n][n] = 1
for n in range(1, 28):
    for k in range(1, min(n, 8)):
        _C[n][k] = _C[n - 1][k - 1] + _C[n - 1][k]

def _lut_eval_7(cards_7):
    s = sorted(cards_7)
    return int(_EVAL_LUT[_C[s[0]][1] + _C[s[1]][2] + _C[s[2]][3] + _C[s[3]][4] + _C[s[4]][5] + _C[s[5]][6] + _C[s[6]][7]])

def compare(we_strs, opp_strs, comm_strs):
    we = [_str_to_int(c) for c in we_strs]
    opp = [_str_to_int(c) for c in opp_strs]
    comm = [_str_to_int(c) for c in comm_strs]
    
    we_rank = _lut_eval_7(we + comm)
    opp_rank = _lut_eval_7(opp + comm)
    
    print(f"We: {we_strs} | Opp: {opp_strs} | Comm: {comm_strs}")
    print(f"We rank: {we_rank} | Opp rank: {opp_rank}")
    if we_rank < opp_rank:
        print("Our bot thinks WE WON (smaller rank is better).")
    elif we_rank > opp_rank:
        print("Our bot thinks OPP WON.")
    else:
        print("Our bot thinks TIE.")

# Test Hand 1 from bad beats:
compare(['9s', '9d'], ['8s', '7h'], ['6d', 'Ad', '9h', '5s', '4s'])

# Test Hand 65 from bad beats:
compare(['8d', '9s'], ['8s', '4s'], ['7s', '5s', '6h', '2s', '6s'])

# Test hand where we definitely win e.g. Quad 9s vs Pair
compare(['9s', '9d'], ['2s', '2d'], ['9h', '3d', 'Ad', 'As', 'Ah'])
