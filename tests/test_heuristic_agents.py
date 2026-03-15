#!/usr/bin/env python3
"""
Tests for all 12 heuristic poker agents.
Runs via direct PokerEnv.step() calls (no HTTP).
"""

import sys
import os
import random
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gym_env import PokerEnv
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

ALL_AGENTS = [
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


def _augment_obs(obs: dict, opp_last_action: str = "None") -> dict:
    """Add fields the match runner would normally inject."""
    obs["time_left"] = 100.0
    obs["time_used"] = 0.0
    obs["opp_last_action"] = opp_last_action
    return obs


def _play_hand(env, agent0, agent1, verbose=False):
    """Play one full hand, returning (reward0, reward1, error_flag)."""
    (obs0, obs1), info = env.reset()
    obs0 = _augment_obs(obs0)
    obs1 = _augment_obs(obs1)
    terminated = False
    reward = (0, 0)
    last_action_name = "None"

    for step_i in range(200):
        acting = env.acting_agent
        obs_act = obs0 if acting == 0 else obs1
        agent_act = agent0 if acting == 0 else agent1
        obs_other = obs1 if acting == 0 else obs0
        agent_other = agent1 if acting == 0 else agent0

        obs_act["opp_last_action"] = last_action_name

        action = agent_act.act(obs_act, reward[acting], terminated, False, info)
        if action is None:
            action = (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
        action = tuple(int(x) for x in action)

        last_action_name = PokerEnv.ActionType(action[0]).name

        (obs0, obs1), reward, terminated, truncated, info = env.step(action)
        obs0 = _augment_obs(obs0, last_action_name)
        obs1 = _augment_obs(obs1, last_action_name)

        if terminated:
            agent_act.observe(obs_act, reward[acting], True, False, info)
            agent_other.observe(obs_other, reward[1 - acting], True, False, info)
            break

    return reward[0], reward[1], False


def _play_match(agent0_cls, agent1_cls, n_hands=20, verbose=False, **agent_kwargs):
    """Play n_hands between two agent classes; returns (total_r0, total_r1, errors)."""
    env = PokerEnv()
    a0 = agent0_cls(**agent_kwargs)
    a1 = agent1_cls(**agent_kwargs)
    t0, t1, errs = 0, 0, 0
    for h in range(n_hands):
        try:
            r0, r1, err = _play_hand(env, a0, a1, verbose=verbose)
            t0 += r0
            t1 += r1
            if err:
                errs += 1
        except Exception as e:
            traceback.print_exc()
            errs += 1
    return t0, t1, errs


# =====================
# Test 1: Discard legality (all 12)
# =====================
def test_discard_legality():
    """Ensure every agent's discard action returns valid keep indices."""
    print("Test 1: Discard legality for all agents...")
    env = PokerEnv()
    for agent_cls in ALL_AGENTS:
        agent = agent_cls()
        for trial in range(10):
            (obs0, obs1), _ = env.reset()
            obs0 = _augment_obs(obs0)
            obs1 = _augment_obs(obs1)

            # Force to street 1 (flop) to trigger discard
            env.street = 1
            env.discard_completed = [False, False]
            obs_test, _ = env._get_single_player_obs(0)
            obs_test = _augment_obs(obs_test)

            if obs_test["valid_actions"][PokerEnv.ActionType.DISCARD.value]:
                action = agent.act(obs_test, 0, False, False, {})
                assert action is not None, f"{agent_cls.__name__} returned None"
                at, ra, k1, k2 = action
                assert at == PokerEnv.ActionType.DISCARD.value, f"{agent_cls.__name__} didn't DISCARD when required"
                assert 0 <= k1 <= 4 and 0 <= k2 <= 4 and k1 != k2, \
                    f"{agent_cls.__name__} invalid keep indices: ({k1}, {k2})"

        print(f"  {agent_cls.__name__}: OK")
    print("  PASSED\n")


# =====================
# Test 2: Raise legality
# =====================
def test_raise_legality():
    """For agents that raise, verify min_raise <= amount <= max_raise."""
    print("Test 2: Raise legality...")
    env = PokerEnv()

    raisers = [
        PriorityConservativeAgent,
        PriorityAggressiveAgent,
        BoardMadeFirstAgent,
        MinRiskPressureAgent,
        BlockerValueAgent,
        AdaptiveOpponentModelAgent,
    ]

    for agent_cls in raisers:
        agent = agent_cls()
        raise_seen = 0
        for trial in range(50):
            (obs0, obs1), _ = env.reset()
            obs0 = _augment_obs(obs0)

            action = agent.act(obs0, 0, False, False, {})
            if action and action[0] == PokerEnv.ActionType.RAISE.value:
                ra = action[1]
                assert obs0["min_raise"] <= ra <= obs0["max_raise"], \
                    f"{agent_cls.__name__} raise {ra} out of [{obs0['min_raise']}, {obs0['max_raise']}]"
                raise_seen += 1

        print(f"  {agent_cls.__name__}: OK ({raise_seen} raises seen)")
    print("  PASSED\n")


# =====================
# Test 3: TripsAutoFoldStrict behavior
# =====================
def test_trips_auto_fold():
    """When dealt trips, TripsAutoFoldStrictAgent should fold at first bet."""
    print("Test 3: TripsAutoFoldStrict trips detection...")
    agent = TripsAutoFoldStrictAgent()

    trips_tested = 0
    for _ in range(200):
        env = PokerEnv()
        (obs0, obs1), _ = env.reset()
        obs0 = _augment_obs(obs0)
        obs1 = _augment_obs(obs1)

        my5 = [c for c in obs0["my_cards"] if c != -1]
        from agents.heuristics_core import has_trips
        if not has_trips(my5):
            continue

        env.street = 1
        env.discard_completed = [False, False]
        obs_disc, _ = env._get_single_player_obs(0)
        obs_disc = _augment_obs(obs_disc)

        agent._reset_hand_state()
        action = agent.act(obs_disc, 0, False, False, {})
        assert action[0] == PokerEnv.ActionType.DISCARD.value, "should still discard"
        assert agent.force_fold, "force_fold should be True after trips"

        env.discard_completed[0] = True
        env.player_cards[0] = [my5[action[2]], my5[action[3]]]
        obs_bet, _ = env._get_single_player_obs(0)
        obs_bet = _augment_obs(obs_bet)

        action2 = agent.act(obs_bet, 0, False, False, {})
        assert action2[0] == PokerEnv.ActionType.FOLD.value, "should FOLD after trips"
        trips_tested += 1
        if trips_tested >= 5:
            break

    print(f"  Trips hands tested: {trips_tested}")
    assert trips_tested > 0, "No trips hands found in 200 random deals"
    print("  PASSED\n")


# =====================
# Test 4: InfoHiderMixed reproducibility
# =====================
def test_info_hider_reproducibility():
    """Same seed should produce same actions."""
    print("Test 4: InfoHiderMixed reproducibility...")
    results = []
    for _ in range(3):
        agent = InfoHiderMixedAgent(rng_seed=42)
        env = PokerEnv()
        np.random.seed(999)
        (obs0, obs1), _ = env.reset()
        obs0 = _augment_obs(obs0)

        env.street = 1
        env.discard_completed = [False, False]
        obs_disc, _ = env._get_single_player_obs(0)
        obs_disc = _augment_obs(obs_disc)

        agent._reset_hand_state()
        action = agent.act(obs_disc, 0, False, False, {})
        results.append(action)

    assert all(r == results[0] for r in results), f"Non-deterministic: {results}"
    print(f"  All 3 runs produced {results[0]}")
    print("  PASSED\n")


# =====================
# Test 5: Smoke tests -- each agent vs calling station (20 hands)
# =====================
class CallingStation:
    """Minimal agent that always calls or checks."""
    def act(self, obs, reward, terminated, truncated, info):
        va = obs["valid_actions"]
        if va[PokerEnv.ActionType.DISCARD.value]:
            return (PokerEnv.ActionType.DISCARD.value, 0, 0, 1)
        if va[PokerEnv.ActionType.CALL.value]:
            return (PokerEnv.ActionType.CALL.value, 0, 0, 0)
        if va[PokerEnv.ActionType.CHECK.value]:
            return (PokerEnv.ActionType.CHECK.value, 0, 0, 0)
        return (PokerEnv.ActionType.FOLD.value, 0, 0, 0)

    def observe(self, obs, reward, terminated, truncated, info):
        pass


def test_smoke_vs_calling_station():
    """Each agent plays 20 hands vs CallingStation with no crashes."""
    print("Test 5: Smoke test (20 hands vs CallingStation)...")
    env = PokerEnv()

    for agent_cls in ALL_AGENTS:
        agent = agent_cls()
        station = CallingStation()
        errors = 0
        for h in range(20):
            try:
                _play_hand(env, agent, station, verbose=False)
            except Exception as e:
                traceback.print_exc()
                errors += 1
        status = "OK" if errors == 0 else f"ERRORS={errors}"
        print(f"  {agent_cls.__name__}: {status}")
        assert errors == 0, f"{agent_cls.__name__} crashed in smoke test"
    print("  PASSED\n")


# =====================
# Test 6: Agent vs itself (sanity)
# =====================
def test_self_play():
    """Each agent plays 10 hands against itself with no crashes."""
    print("Test 6: Self-play (10 hands)...")
    for agent_cls in ALL_AGENTS:
        env = PokerEnv()
        a0 = agent_cls()
        a1 = agent_cls()
        errors = 0
        for h in range(10):
            try:
                _play_hand(env, a0, a1)
            except Exception as e:
                traceback.print_exc()
                errors += 1
        status = "OK" if errors == 0 else f"ERRORS={errors}"
        print(f"  {agent_cls.__name__}: {status}")
        assert errors == 0, f"{agent_cls.__name__} crashed in self-play"
    print("  PASSED\n")


if __name__ == "__main__":
    test_discard_legality()
    test_raise_legality()
    test_trips_auto_fold()
    test_info_hider_reproducibility()
    test_smoke_vs_calling_station()
    test_self_play()
    print("=" * 50)
    print("ALL TESTS PASSED")
