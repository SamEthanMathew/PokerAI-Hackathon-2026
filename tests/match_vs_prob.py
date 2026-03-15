#!/usr/bin/env python3
"""Match each heuristic agent against ProbabilityAgent (30 hands each)."""
import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gym_env import PokerEnv
from agents.prob_agent import ProbabilityAgent
from agents.heuristic_agents import (
    PriorityConservativeAgent,
    PriorityAggressiveAgent,
    BoardMadeFirstAgent,
    OutsMaximizerAgent,
    ExactStrengthEnumerateAgent,
    ExactEquityEnumerateAgent,
    OppDiscardAwareAgent,
    BlockerValueAgent,
    MinRiskPressureAgent,
    InfoHiderMixedAgent,
    TripsAutoFoldStrictAgent,
    AdaptiveOpponentModelAgent,
)

AGENTS = [
    PriorityConservativeAgent,
    PriorityAggressiveAgent,
    BoardMadeFirstAgent,
    OutsMaximizerAgent,
    ExactStrengthEnumerateAgent,
    ExactEquityEnumerateAgent,
    OppDiscardAwareAgent,
    BlockerValueAgent,
    MinRiskPressureAgent,
    InfoHiderMixedAgent,
    TripsAutoFoldStrictAgent,
    AdaptiveOpponentModelAgent,
]

def _augment(obs, opp_last="None"):
    obs["time_left"] = 100.0
    obs["time_used"] = 0.0
    obs["opp_last_action"] = opp_last
    return obs

def play_hand(env, a0, a1):
    (o0, o1), info = env.reset()
    o0 = _augment(o0)
    o1 = _augment(o1)
    terminated = False
    reward = (0, 0)
    last = "None"
    for _ in range(200):
        act_p = env.acting_agent
        obs = o0 if act_p == 0 else o1
        agent = a0 if act_p == 0 else a1
        other = a1 if act_p == 0 else a0
        obs["opp_last_action"] = last

        action = agent.act(obs, reward[act_p], terminated, False, info)
        if action is None:
            action = (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
        action = tuple(int(x) for x in action)
        last = PokerEnv.ActionType(action[0]).name

        (o0, o1), reward, terminated, truncated, info = env.step(action)
        o0 = _augment(o0, last)
        o1 = _augment(o1, last)

        if terminated:
            agent.observe(obs, reward[act_p], True, False, info)
            o_other = o1 if act_p == 0 else o0
            other.observe(o_other, reward[1 - act_p], True, False, info)
            break
    return reward[0], reward[1]

N_HANDS = 30

print(f"Testing {len(AGENTS)} heuristic agents vs ProbabilityAgent ({N_HANDS} hands each)")
print("=" * 70)

all_ok = True
for acls in AGENTS:
    env = PokerEnv()
    ha = acls()
    pa = ProbabilityAgent(stream=False)
    t0, t1, errs = 0, 0, 0
    for h in range(N_HANDS):
        try:
            r0, r1 = play_hand(env, ha, pa)
            t0 += r0
            t1 += r1
        except Exception:
            traceback.print_exc()
            errs += 1
    status = "OK" if errs == 0 else f"ERRORS={errs}"
    print(f"  {acls.__name__:40s} => heuristic={t0:+5d}  prob={t1:+5d}  {status}")
    if errs > 0:
        all_ok = False

print("=" * 70)
print("ALL MATCH TESTS PASSED" if all_ok else "SOME MATCH TESTS FAILED")
