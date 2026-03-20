#!/usr/bin/env python3
"""Validate OMICRoN V1.2 (LUT + parallel) — 200 hands, measure timing."""

import sys, os, time, importlib, importlib.util, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv
from agents.heuristic_agents import PriorityAggressiveAgent

_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")
spec = importlib.util.spec_from_file_location("omicron", os.path.join(_base, "OMICRoN_V1.py"))
_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _mod
spec.loader.exec_module(_mod)
OMICRoN = _mod.PlayerAgent

HANDS = 200

ACTION_NAMES = {
    PokerEnv.ActionType.FOLD.value:    "FOLD",
    PokerEnv.ActionType.RAISE.value:   "RAISE",
    PokerEnv.ActionType.CHECK.value:   "CHECK",
    PokerEnv.ActionType.CALL.value:    "CALL",
    PokerEnv.ActionType.DISCARD.value: "DISCARD",
}


def play_hand(env, a0, a1):
    (o0, o1), info = env.reset()
    o0["time_left"] = 400.0
    o1["time_left"] = 400.0
    o0["opp_last_action"] = "None"
    o1["opp_last_action"] = "None"

    terminated = False
    reward = (0, 0)
    last = "None"
    t_omicron = 0.0

    for _ in range(200):
        act_p = env.acting_agent
        obs = o0 if act_p == 0 else o1
        agent = a0 if act_p == 0 else a1
        other = a1 if act_p == 0 else a0
        obs["opp_last_action"] = last

        t0 = time.perf_counter()
        action = agent.act(obs, reward[act_p], terminated, False, info)
        dt = time.perf_counter() - t0
        if act_p == 0:
            t_omicron += dt

        if action is None:
            action = (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
        action = tuple(int(x) for x in action)

        last = PokerEnv.ActionType(action[0]).name
        (o0, o1), reward, terminated, truncated, info = env.step(action)
        o0["time_left"] = 400.0
        o1["time_left"] = 400.0
        o0["opp_last_action"] = last
        o1["opp_last_action"] = last

        if terminated:
            try:
                agent.observe(obs, reward[act_p], True, False, info)
            except Exception:
                pass
            o_other = o1 if act_p == 0 else o0
            try:
                other.observe(o_other, reward[1 - act_p], True, False, info)
            except Exception:
                pass
            break

    return reward[0], reward[1], t_omicron


def main():
    print(f"OMICRoN V1.2 Validation — {HANDS} hands vs PriorityAggressiveAgent")
    print("=" * 70)

    env = PokerEnv()
    try:
        bot = OMICRoN(stream=False)
    except TypeError:
        bot = OMICRoN()
    try:
        opp = PriorityAggressiveAgent(stream=False)
    except TypeError:
        opp = PriorityAggressiveAgent()

    total_pnl = 0
    errors = 0
    hand_times = []

    t_total = time.time()

    for h in range(1, HANDS + 1):
        try:
            r0, r1, t_hand = play_hand(env, bot, opp)
            total_pnl += r0
            hand_times.append(t_hand)
        except Exception as e:
            traceback.print_exc()
            errors += 1

        if h % 50 == 0:
            avg_ms = sum(hand_times[-50:]) / len(hand_times[-50:]) * 1000
            print(f"  Hand {h:4d}  PnL={total_pnl:+6d}  "
                  f"avg_last50={avg_ms:.1f}ms/hand  errors={errors}")

    elapsed = time.time() - t_total
    avg_ms = sum(hand_times) / len(hand_times) * 1000 if hand_times else 0
    max_ms = max(hand_times) * 1000 if hand_times else 0
    min_ms = min(hand_times) * 1000 if hand_times else 0

    print("\n" + "=" * 70)
    print(f"Results: {HANDS} hands in {elapsed:.1f}s")
    print(f"  PnL:    {total_pnl:+d}")
    print(f"  Errors: {errors}")
    print(f"  Timing: avg={avg_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")
    print(f"  Total OMICRoN time: {sum(hand_times):.2f}s")

    if errors > 0:
        print("\n*** VALIDATION FAILED — errors detected ***")
        sys.exit(1)
    else:
        print("\n*** VALIDATION PASSED ***")


if __name__ == "__main__":
    main()
