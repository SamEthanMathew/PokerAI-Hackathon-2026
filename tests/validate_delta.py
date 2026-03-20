#!/usr/bin/env python3
"""Validate DELTA V1 — 200 hands each vs OMICRoN and PriorityAggressive."""

import sys, os, time, importlib, importlib.util, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv
from agents.heuristic_agents import PriorityAggressiveAgent

_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")


def load_bot(filename, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_base, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.PlayerAgent


DELTA = load_bot("DELTA_V1.py", "delta_mod")
OMICRON = load_bot("OMICRoN_V1.py", "omicron_mod")

HANDS = 200


def play_hand(env, a0, a1, idx0=0):
    (o0, o1), info = env.reset()
    o0["time_left"] = 400.0
    o1["time_left"] = 400.0
    o0["opp_last_action"] = "None"
    o1["opp_last_action"] = "None"

    terminated = False
    reward = (0, 0)
    last = "None"
    t_target = 0.0

    for _ in range(200):
        act_p = env.acting_agent
        obs = o0 if act_p == 0 else o1
        agent = a0 if act_p == 0 else a1
        other = a1 if act_p == 0 else a0
        obs["opp_last_action"] = last

        t0 = time.perf_counter()
        action = agent.act(obs, reward[act_p], terminated, False, info)
        dt = time.perf_counter() - t0
        if act_p == idx0:
            t_target += dt

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

    return reward[0], reward[1], t_target


def run_match(name_a, cls_a, name_b, cls_b, num_hands):
    env = PokerEnv()
    a0 = cls_a(stream=False)
    a1 = cls_b(stream=False)

    total_a = 0
    total_b = 0
    times = []
    errors = 0

    print(f"\n{'='*60}")
    print(f"  {name_a} vs {name_b}  ({num_hands} hands)")
    print(f"{'='*60}")

    for h in range(num_hands):
        try:
            r0, r1, dt = play_hand(env, a0, a1, idx0=0)
            total_a += r0
            total_b += r1
            times.append(dt)
        except Exception as e:
            errors += 1
            if errors <= 3:
                traceback.print_exc()
            continue

        if (h + 1) % 50 == 0:
            avg_t = sum(times[-50:]) / len(times[-50:]) * 1000
            print(f"  Hand {h+1:4d}: {name_a}={total_a:+5d}  {name_b}={total_b:+5d}  "
                  f"avg_ms={avg_t:.1f}")

    avg_ms = sum(times) / len(times) * 1000 if times else 0
    max_ms = max(times) * 1000 if times else 0
    min_ms = min(times) * 1000 if times else 0

    print(f"\n  RESULT: {name_a} {total_a:+d}  vs  {name_b} {total_b:+d}")
    print(f"  Winner: {name_a if total_a > total_b else name_b}")
    print(f"  {name_a} timing: avg={avg_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")
    print(f"  Errors: {errors}")

    return total_a, total_b, avg_ms, errors


def main():
    results = []

    pnl_a, pnl_b, avg_ms, err = run_match("DELTA", DELTA, "PriorityAggro", PriorityAggressiveAgent, HANDS)
    results.append(("DELTA vs PriorityAggro", pnl_a, pnl_b, avg_ms, err))

    pnl_a, pnl_b, avg_ms, err = run_match("DELTA", DELTA, "OMICRoN", OMICRON, HANDS)
    results.append(("DELTA vs OMICRoN", pnl_a, pnl_b, avg_ms, err))

    print(f"\n{'='*60}")
    print(f"  TOURNAMENT SUMMARY")
    print(f"{'='*60}")
    for name, pa, pb, ms, err in results:
        opp_name = name.split(" vs ")[1]
        winner = "DELTA" if pa > pb else opp_name if pb > pa else "TIE"
        print(f"  {name:30s}: DELTA={pa:+5d}  {opp_name}={pb:+5d}  winner={winner}  avg_ms={ms:.1f}  errors={err}")

    total_errors = sum(r[4] for r in results)
    if total_errors == 0:
        print(f"\n  ALL MATCHES COMPLETED WITHOUT ERRORS")
    else:
        print(f"\n  WARNING: {total_errors} total errors across matches")


if __name__ == "__main__":
    main()
