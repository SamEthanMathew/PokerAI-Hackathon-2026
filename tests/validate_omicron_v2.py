#!/usr/bin/env python3
"""Head-to-head: OMICRoN V2 (new) vs OMICRoN V1 (backup).
3 matches x 1000 hands, swapping seats each match."""

import sys, os, time, importlib, importlib.util, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv

_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")


def _load(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_base, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.PlayerAgent


V2Agent = _load("OMICRoN_V1.py", "omicron_v2")
V1Agent = _load("OMICRoN_V1_backup.py", "omicron_v1_backup")

HANDS_PER_MATCH = 1000
NUM_MATCHES = 3


def play_hand(env, a0, a1, time_budget=400.0):
    (o0, o1), info = env.reset()
    o0["time_left"] = time_budget
    o1["time_left"] = time_budget
    o0["opp_last_action"] = "None"
    o1["opp_last_action"] = "None"

    terminated = False
    reward = (0, 0)
    last = "None"
    t0_total = 0.0
    t1_total = 0.0

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
            t0_total += dt
        else:
            t1_total += dt

        if action is None:
            action = (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
        action = tuple(int(x) for x in action)

        last = PokerEnv.ActionType(action[0]).name
        (o0, o1), reward, terminated, truncated, info = env.step(action)
        o0["time_left"] = time_budget
        o1["time_left"] = time_budget
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

    return reward[0], reward[1], t0_total, t1_total


def run_match(match_num, v2_seat):
    env = PokerEnv()
    try:
        v2 = V2Agent(stream=False)
    except TypeError:
        v2 = V2Agent()
    try:
        v1 = V1Agent(stream=False)
    except TypeError:
        v1 = V1Agent()

    if v2_seat == 0:
        a0, a1 = v2, v1
        label = "V2=seat0, V1=seat1"
    else:
        a0, a1 = v1, v2
        label = "V1=seat0, V2=seat1"

    print(f"\n{'='*70}")
    print(f"Match {match_num}: {label} — {HANDS_PER_MATCH} hands")
    print(f"{'='*70}")

    v2_pnl = 0
    errors = 0
    v2_time = 0.0
    v1_time = 0.0

    t_start = time.time()
    for h in range(1, HANDS_PER_MATCH + 1):
        try:
            r0, r1, t0, t1 = play_hand(env, a0, a1)
            if v2_seat == 0:
                v2_pnl += r0
                v2_time += t0
                v1_time += t1
            else:
                v2_pnl += r1
                v2_time += t1
                v1_time += t0
        except Exception:
            traceback.print_exc()
            errors += 1

        if h % 250 == 0:
            print(f"  Hand {h:4d}  V2 PnL={v2_pnl:+6d}  errors={errors}")

    elapsed = time.time() - t_start
    print(f"  Finished in {elapsed:.1f}s  |  V2 PnL={v2_pnl:+d}  "
          f"|  V2 time={v2_time:.1f}s  V1 time={v1_time:.1f}s  "
          f"|  errors={errors}")
    return v2_pnl, v2_time, v1_time, errors


def main():
    print("OMICRoN V2 vs V1 Head-to-Head Validation")
    print(f"{NUM_MATCHES} matches x {HANDS_PER_MATCH} hands")

    results = []
    total_errors = 0
    seats = [0, 1, 0]

    for m in range(NUM_MATCHES):
        pnl, t_v2, t_v1, errs = run_match(m + 1, seats[m])
        results.append((pnl, t_v2, t_v1))
        total_errors += errs

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total_pnl = 0
    for i, (pnl, t_v2, t_v1) in enumerate(results):
        total_pnl += pnl
        print(f"  Match {i+1}: V2 PnL={pnl:+6d}  (V2 {t_v2:.1f}s / V1 {t_v1:.1f}s)")
    print(f"  ──────────────────────────────────")
    print(f"  Total V2 PnL: {total_pnl:+d}")
    print(f"  Total errors: {total_errors}")

    if total_pnl > 0:
        print(f"\n  *** V2 WINS by {total_pnl} chips ***")
    elif total_pnl < 0:
        print(f"\n  *** V1 WINS by {-total_pnl} chips ***")
    else:
        print(f"\n  *** TIE ***")

    if total_errors > 0:
        print("  *** ERRORS DETECTED — review output ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
