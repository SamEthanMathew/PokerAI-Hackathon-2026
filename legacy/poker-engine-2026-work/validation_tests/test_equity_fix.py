import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from submission.player import PlayerAgent, _str_to_int

def test_bad_beat_equity_reduction():
    """
    Test that the equity calculation properly reacts to aggressive bet sizing
    and rejects hands accurately, leading to a much lower equity estimation
    when facing a huge bet (action_aggr).
    """
    agent = PlayerAgent()
    
    # Match 83829, Hand 1
    # We: ['9s', '9d'] | Comm: ['6d', 'Ad', '9h', '5s', '4s']
    # Opp shove all-in on river.
    my_cards = [_str_to_int(c) for c in ['9s', '9d']]
    comm = [_str_to_int(c) for c in ['6d', 'Ad', '9h', '5s', '4s']]
    dead = set(my_cards) | set(comm)
    
    # Old behavior (aggr = 1.0)
    old_eq = agent._compute_equity_ranged(my_cards, comm, dead, [], 1.0, 1.0, num_sims=500)
    
    # New behavior (aggr = 4.5, representing a massive river shove)
    new_eq = agent._compute_equity_ranged(my_cards, comm, dead, [], 1.0, 4.5, num_sims=500)
    
    # The old equity was around 0.86+, the new should be much lower (around 0.1-0.2)
    assert old_eq > 0.80, f"Old equity was {old_eq}, expected > 0.80"
    assert new_eq < 0.40, f"New equity was {new_eq}, expected < 0.40 due to range rejection"

def test_action_aggr_calculation():
    """
    Test the dynamic action_aggr calculation inside the act() method.
    """
    agent = PlayerAgent()
    
    # Match 83829, Hand 1
    obs = {
        "my_cards": [_str_to_int(c) if c != -1 else -1 for c in ['9s', '9d', -1, -1, -1]],
        "community_cards": [_str_to_int(c) for c in ['6d', 'Ad', '9h', '5s', '4s']],
        "opp_discarded_cards": [],
        "my_discarded_cards": [],
        "valid_actions": [1, 1, 0, 1, 0], # FOLD, CALL, RAISE
        "street": 3,
        "min_raise": 0,
        "max_raise": 100,
        "my_bet": 10,
        "opp_bet": 110, # Massive bet (100 to call)
        "pot_size": 120,
        "hands_completed": 1,
        "time_left": 1000.0,
        "blind_position": 1
    }
    
    # Run the act method
    # Since we have a Set but the opponent shoved a massive amount on a straight board,
    # the new logic should fold.
    action = agent.act(obs, 0, False, False, {})
    
    # 0 = FOLD
    assert action[0] == 0, f"Expected FOLD (0), got {action[0]}"
