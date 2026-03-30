import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from submission.player import PlayerAgent, _str_to_int


agent = PlayerAgent()

# Mock observation for Hand 176 from Match 83829
# We: ['6d', 'Ah'] | Comm: ['8d', '6h', '8s', '6s', '5d']
# Pot size, bets, etc.
observation = {
    'my_cards': [_str_to_int(c) for c in ['6d', 'Ah']],
    'community_cards': [_str_to_int(c) for c in ['8d', '6h', '8s', '6s', '5d']],
    'opp_discarded_cards': [],
    'my_discarded_cards': [],
    'valid_actions': [1, 1, 1, 1, 0],
    'street': 3,
    'min_raise': 2,
    'max_raise': 100,
    'my_bet': 10,
    'opp_bet': 40, # Opponent bet 30 into a pot of ~20, very aggressive
    'pot_size': 50,
    'hands_completed': 176,
    'time_left': 1000.0,
}

# Test the old logic behavior manually by overriding parameters:
dead = set(observation['my_cards']) | set(observation['community_cards'])

# Old baseline signals
signal_passive = 1.0
signal_aggr = 1.0
old_equity = agent._compute_equity_ranged(
    observation['my_cards'], observation['community_cards'], dead, [], 
    signal_passive, signal_aggr, num_sims=500
)

# New logic signals:
to_call = 30
pot_before_call = 20
bet_fraction = to_call / 20.0 # 1.5

action_aggr = 2.5 + 1.0 # 3.5 (river > 0.6)
new_signal_aggr = max(1.0, 1.0 + action_aggr)

new_equity = agent._compute_equity_ranged(
    observation['my_cards'], observation['community_cards'], dead, [], 
    signal_passive, new_signal_aggr, num_sims=500
)

print(f"Old Equity (no aggr reaction): {old_equity:.3f}")
print(f"New Equity (with aggr reaction): {new_equity:.3f}")

# Hand 1 from Match 83829
# We: ['9s', '9d'] | Comm: ['6d', 'Ad', '9h', '5s', '4s']
# Opp bet big.
my_cards3 = [_str_to_int(c) for c in ['9s', '9d']]
comm3 = [_str_to_int(c) for c in ['6d', 'Ad', '9h', '5s', '4s']]
dead3 = set(my_cards3) | set(comm3)

old_eq3 = agent._compute_equity_ranged(my_cards3, comm3, dead3, [], 1.0, 1.0, num_sims=500)
new_eq3 = agent._compute_equity_ranged(my_cards3, comm3, dead3, [], 1.0, 4.5, num_sims=500)

print(f"Hand 1 Old Eq: {old_eq3:.3f}")
print(f"Hand 1 New Eq: {new_eq3:.3f}")

