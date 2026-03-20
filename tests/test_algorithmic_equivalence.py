#!/usr/bin/env python3
"""
Rigorous algorithmic equivalence test: OMICRoN V1.2 (optimized) vs V1.2 copy (original).

Tests three levels:
  1. LUT correctness: _lut_eval_7 matches _fast_evaluate for all eval paths
  2. Solver equivalence: exact solvers produce identical equity values
  3. Decision equivalence: both bots make identical act() decisions on the same
     observations with the same random seed
"""

import sys, os, random, copy, importlib, importlib.util
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv

# ── Load both modules ────────────────────────────────────────────────────────

_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")

spec_new = importlib.util.spec_from_file_location(
    "omicron_new", os.path.join(_base, "OMICRoN_V1.py"))
mod_new = importlib.util.module_from_spec(spec_new)
sys.modules[spec_new.name] = mod_new
spec_new.loader.exec_module(mod_new)

spec_old = importlib.util.spec_from_file_location(
    "omicron_old", os.path.join(_base, "OMICRoN_V1.2 copy.py"))
mod_old = importlib.util.module_from_spec(spec_old)
sys.modules[spec_old.name] = mod_old
spec_old.loader.exec_module(mod_old)


# ── Test 1: LUT vs _fast_evaluate ────────────────────────────────────────────

def test_lut_correctness():
    """Verify _lut_eval_7 produces the same rank as _fast_evaluate for many combos."""
    print("=" * 70)
    print("TEST 1: LUT correctness — _lut_eval_7 vs _fast_evaluate")
    print("=" * 70)

    DECK = 27
    INT_TO_TREYS = mod_old._INT_TO_TREYS
    INT_TO_TREYS_ALT = mod_old._INT_TO_TREYS_ALT
    fast_evaluate = mod_old._fast_evaluate
    lut_eval = mod_new._lut_eval_7

    mismatches = 0
    tested = 0

    # Test every single C(27,7) = 888,030 combination
    for combo in combinations(range(DECK), 7):
        hand = [INT_TO_TREYS[c] for c in combo[:2]]
        board = [INT_TO_TREYS[c] for c in combo[2:]]
        hand_alt = [INT_TO_TREYS_ALT[c] for c in combo[:2]]
        board_alt = [INT_TO_TREYS_ALT[c] for c in combo[2:]]

        old_rank = fast_evaluate(hand, board, hand_alt, board_alt)
        new_rank = lut_eval(list(combo))

        if old_rank != new_rank:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH at combo {combo}: old={old_rank} new={new_rank}")

        tested += 1
        if tested % 100000 == 0:
            print(f"  Checked {tested:>7d}/888030 — mismatches so far: {mismatches}")

    print(f"\n  Result: {tested} combos tested, {mismatches} mismatches")
    if mismatches == 0:
        print("  >>> PASS: LUT is a perfect replica of _fast_evaluate <<<")
    else:
        print("  >>> FAIL: LUT has mismatches <<<")
    print()
    return mismatches == 0


# ── Test 2: Exact solver equivalence ─────────────────────────────────────────

def test_solver_equivalence():
    """Compare exact solver outputs for specific test cases."""
    print("=" * 70)
    print("TEST 2: Exact solver equivalence")
    print("=" * 70)

    old_bot = mod_old.PlayerAgent(stream=False)
    new_bot = mod_new.PlayerAgent(stream=False)

    # Sync model weights so _opp_keep_weight produces the same output
    new_bot._opp_model.weights = list(old_bot._opp_model.weights)

    test_cases = [
        # (my_keep, community, dead_cards, description)
        ([0, 1], [9, 10, 11], set([0, 1, 2, 3, 4, 9, 10, 11]), "basic flop, 8 dead"),
        ([8, 17], [0, 9, 18], set([8, 17, 2, 3, 4, 0, 9, 18]), "suited aces flop"),
        ([6, 15], [1, 10, 19], set([6, 15, 7, 8, 16, 1, 10, 19]), "mid cards flop"),
    ]

    all_pass = True
    for my_keep, community, dead, desc in test_cases:
        old_eq = old_bot._exact_discard_equity(my_keep, community, dead)
        new_eq = new_bot._exact_discard_equity(my_keep, community, dead)

        match = abs(old_eq - new_eq) < 1e-10
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] _exact_discard_equity ({desc}): old={old_eq:.8f} new={new_eq:.8f}")
        if not match:
            all_pass = False

    # Test weighted solver
    opp_discards_cases = [
        ([0, 1], [9, 10, 11], set([0, 1, 2, 3, 4, 9, 10, 11, 5, 6, 7]),
         [5, 6, 7], "weighted with 3 opp discards"),
        ([8, 17], [0, 9, 18], set([8, 17, 2, 3, 4, 0, 9, 18, 5, 14, 23]),
         [5, 14, 23], "weighted aces with opp discards"),
    ]

    for my_keep, community, dead, opp_disc, desc in opp_discards_cases:
        old_eq = old_bot._exact_discard_equity_weighted(my_keep, community, dead, opp_disc)
        new_eq = new_bot._exact_discard_equity_weighted(my_keep, community, dead, opp_disc)

        match = abs(old_eq - new_eq) < 1e-10
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] _exact_discard_equity_weighted ({desc}): old={old_eq:.8f} new={new_eq:.8f}")
        if not match:
            all_pass = False

    if all_pass:
        print("  >>> PASS: All solver outputs match <<<")
    else:
        print("  >>> FAIL: Solver outputs diverge <<<")
    print()
    return all_pass


# ── Test 3: Full game decision equivalence ───────────────────────────────────

def test_decision_equivalence():
    """Run both bots through identical game states, compare every decision."""
    print("=" * 70)
    print("TEST 3: Decision equivalence — 100 hands, same seed, compare all acts")
    print("=" * 70)

    from agents.heuristic_agents import PriorityAggressiveAgent

    HANDS = 100
    mismatches = 0
    total_decisions = 0

    env = PokerEnv()
    old_bot = mod_old.PlayerAgent(stream=False)
    new_bot = mod_new.PlayerAgent(stream=False)
    opp = PriorityAggressiveAgent(stream=False)

    for hand_num in range(1, HANDS + 1):
        # Use a deterministic seed for env reset
        random.seed(1000 + hand_num)
        (o0, o1), info = env.reset()

        # High time_left forces "full" sim_mode in new bot
        o0["time_left"] = 999.0
        o1["time_left"] = 999.0
        o0["opp_last_action"] = "None"
        o1["opp_last_action"] = "None"

        terminated = False
        reward = (0, 0)
        last = "None"

        for step in range(200):
            act_p = env.acting_agent
            obs = o0 if act_p == 0 else o1
            obs["opp_last_action"] = last
            obs["time_left"] = 999.0

            if act_p == 0:
                # Both bots act on the exact same observation with the same seed
                obs_copy = copy.deepcopy(obs)

                seed = hand_num * 10000 + step
                random.seed(seed)
                old_action = old_bot.act(obs_copy, reward[0], terminated, False, info)
                old_action = tuple(int(x) for x in old_action)

                random.seed(seed)
                new_action = new_bot.act(obs, reward[0], terminated, False, info)
                new_action = tuple(int(x) for x in new_action)

                total_decisions += 1

                if old_action != new_action:
                    mismatches += 1
                    if mismatches <= 10:
                        street = obs.get("street", -1)
                        print(f"  MISMATCH hand={hand_num} step={step} street={street}: "
                              f"old={old_action} new={new_action}")

                action = new_action
            else:
                action = opp.act(obs, reward[1], terminated, False, info)
                if action is None:
                    action = (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
                action = tuple(int(x) for x in action)

            last = PokerEnv.ActionType(action[0]).name
            (o0, o1), reward, terminated, truncated, info = env.step(action)
            o0["time_left"] = 999.0
            o1["time_left"] = 999.0
            o0["opp_last_action"] = last
            o1["opp_last_action"] = last

            if terminated:
                # Both bots observe the same terminal state
                if act_p == 0:
                    obs_t = copy.deepcopy(o0)
                    old_bot.observe(obs_t, reward[0], True, False, info)
                    new_bot.observe(o0, reward[0], True, False, info)
                    opp.observe(o1, reward[1], True, False, info)
                else:
                    opp.observe(obs, reward[1], True, False, info)
                    obs_t = copy.deepcopy(o0)
                    old_bot.observe(obs_t, reward[0], True, False, info)
                    new_bot.observe(o0, reward[0], True, False, info)
                break

        if hand_num % 25 == 0:
            print(f"  Hand {hand_num:3d}: {total_decisions} decisions, {mismatches} mismatches")

    print(f"\n  Result: {total_decisions} decisions compared, {mismatches} mismatches")
    if mismatches == 0:
        print("  >>> PASS: Every single decision is identical <<<")
    else:
        pct = mismatches / total_decisions * 100
        print(f"  >>> FAIL: {pct:.2f}% of decisions differ <<<")
    print()
    return mismatches == 0


# ── Test 4: MC equity equivalence ────────────────────────────────────────────

def test_mc_equity_equivalence():
    """Compare MC equity computations with same seed."""
    print("=" * 70)
    print("TEST 4: MC equity equivalence — _compute_equity and _compute_equity_ranged")
    print("=" * 70)

    old_bot = mod_old.PlayerAgent(stream=False)
    new_bot = mod_new.PlayerAgent(stream=False)
    new_bot._opp_model.weights = list(old_bot._opp_model.weights)

    all_pass = True

    # Test _compute_equity
    test_cases = [
        ([0, 1], [9, 10, 11], [], [2, 3, 4], 100),
        ([8, 17], [], [], [0, 1, 2], 100),
        ([6, 15], [1, 10, 19, 2], [], [7, 8, 16], 100),
    ]

    for my_cards, community, opp_disc, my_disc, nsims in test_cases:
        random.seed(42)
        old_eq = old_bot._compute_equity(my_cards, community, opp_disc, my_disc, num_sims=nsims)
        random.seed(42)
        new_eq = new_bot._compute_equity(my_cards, community, opp_disc, my_disc, num_sims=nsims)

        match = abs(old_eq - new_eq) < 1e-10
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] _compute_equity: old={old_eq:.8f} new={new_eq:.8f}")
        if not match:
            all_pass = False

    # Test _compute_equity_ranged
    ranged_cases = [
        ([0, 1], [9, 10, 11], set([0, 1, 9, 10, 11, 5, 6, 7]), [5, 6, 7], 1.5, 100),
        ([8, 17], [0, 9, 18], set([8, 17, 0, 9, 18, 2, 3, 4]), [2, 3, 4], 0.0, 100),
    ]

    for my2, community, dead, opp_disc, opp_signal, nsims in ranged_cases:
        random.seed(42)
        old_eq = old_bot._compute_equity_ranged(my2, community, dead, opp_disc, opp_signal, nsims)
        random.seed(42)
        new_eq = new_bot._compute_equity_ranged(my2, community, dead, opp_disc, opp_signal, nsims)

        match = abs(old_eq - new_eq) < 1e-10
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] _compute_equity_ranged: old={old_eq:.8f} new={new_eq:.8f}")
        if not match:
            all_pass = False

    if all_pass:
        print("  >>> PASS: All MC equity computations match <<<")
    else:
        print("  >>> FAIL: MC equity computations diverge <<<")
    print()
    return all_pass


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    # Test 1 takes ~30s (all 888K combos). Run it first.
    results["LUT correctness"] = test_lut_correctness()
    results["MC equity"] = test_mc_equity_equivalence()
    results["Solver equivalence"] = test_solver_equivalence()
    results["Decision equivalence"] = test_decision_equivalence()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  *** ALL TESTS PASSED — bots are algorithmically identical ***")
        print("  (in 'full' sim_mode; emergency/conservative modes intentionally")
        print("   degrade gracefully under time pressure)")
    else:
        print("\n  *** SOME TESTS FAILED — see details above ***")

    sys.exit(0 if all_pass else 1)
