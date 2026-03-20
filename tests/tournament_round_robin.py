#!/usr/bin/env python3
"""
Round-robin tournament: ALPHANiTV8, ALPHANiTV7, METAV6, METAV5, LambdaV1.
Each pair plays 10 matches of 1000 hands. Results logged; final table printed.
"""

import sys
import os
import time
import importlib.util
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")

BOT_SPECS = [
    ("ALPHANiTV8", "ALPHANiTV8.py"),
    ("ALPHANiTV7", "ALPHANiTV7.py"),
    ("METAV6", "METAV6.py"),
    ("METAV5", "METAV5.py"),
    ("LambdaV1", "lambdaV1.py"),
]

HANDS_PER_MATCH = 1000
MATCHES_PER_PAIR = 10

# Optional: override via env for quick test, e.g. QUICK_TEST=1 python3 tournament_round_robin.py
if os.environ.get("QUICK_TEST"):
    HANDS_PER_MATCH = 5
    MATCHES_PER_PAIR = 1


def load_bot(name, filename):
    path = os.path.join(BASE, filename)
    spec = importlib.util.spec_from_file_location(name.lower().replace(" ", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PlayerAgent


def _augment(obs, opp_last="None"):
    obs["time_left"] = 400.0
    obs["opp_last_action"] = opp_last
    return obs


def play_hand(env, a0, a1, opp_last_by_side):
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

        try:
            action = agent.act(obs, reward[act_p], terminated, False, info)
        except Exception:
            action = None
        if action is None:
            action = (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
        action = tuple(int(x) for x in action)

        last = PokerEnv.ActionType(action[0]).name
        (o0, o1), reward, terminated, truncated, info = env.step(action)
        o0 = _augment(o0, last)
        o1 = _augment(o1, last)

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

    return reward[0], reward[1]


def run_match(env, agent_a_class, agent_b_class, name_a, name_b, match_num, logf):
    try:
        a0 = agent_a_class(stream=False)
    except TypeError:
        a0 = agent_a_class()
    try:
        a1 = agent_b_class(stream=False)
    except TypeError:
        a1 = agent_b_class()

    tot_a, tot_b = 0, 0
    for h in range(1, HANDS_PER_MATCH + 1):
        try:
            r0, r1 = play_hand(env, a0, a1, None)
            tot_a += r0
            tot_b += r1
        except Exception as e:
            logf.write(f"  {name_a} vs {name_b} match {match_num} hand {h} CRASH: {e}\n")
    return tot_a, tot_b


def main():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "tournament_round_robin.log")
    csv_path = os.path.join(log_dir, "tournament_round_robin_results.csv")

    bots = {}
    for name, filename in BOT_SPECS:
        bots[name] = load_bot(name, filename)

    pairs = list(combinations(BOT_SPECS, 2))
    results = []
    totals = {name: 0 for name, _ in BOT_SPECS}
    match_wins = {name: 0 for name, _ in BOT_SPECS}

    with open(log_path, "w") as logf:
        def flog(msg):
            logf.write(msg)
            logf.flush()

        flog(f"Round-robin: {[n for n, _ in BOT_SPECS]}\n")
        flog(f"Each pair: {MATCHES_PER_PAIR} matches x {HANDS_PER_MATCH} hands\n\n")

        for (name_a, file_a), (name_b, file_b) in pairs:
            flog(f"=== {name_a} vs {name_b} ({MATCHES_PER_PAIR} matches) ===\n")
            env = PokerEnv()
            pair_tot_a, pair_tot_b = 0, 0
            wins_a, wins_b = 0, 0

            for m in range(1, MATCHES_PER_PAIR + 1):
                t0 = time.time()
                tot_a, tot_b = run_match(env, bots[name_a], bots[name_b], name_a, name_b, m, logf)
                elapsed = time.time() - t0
                pair_tot_a += tot_a
                pair_tot_b += tot_b
                if tot_a > tot_b:
                    wins_a += 1
                else:
                    wins_b += 1
                results.append((name_a, name_b, m, tot_a, tot_b))
                flog(f"  Match {m}: {name_a} {tot_a:+d}  {name_b} {tot_b:+d}  ({elapsed:.0f}s)\n")
                print(f"  {name_a} vs {name_b}  Match {m}/{MATCHES_PER_PAIR}: {tot_a:+d} vs {tot_b:+d}  ({elapsed:.0f}s)")

            totals[name_a] += pair_tot_a
            totals[name_b] += pair_tot_b
            match_wins[name_a] += wins_a
            match_wins[name_b] += wins_b
            flog(f"  Total: {name_a} {pair_tot_a:+d}  {name_b} {pair_tot_b:+d}  (Wins: {wins_a}-{wins_b})\n\n")

        flog("\n" + "=" * 60 + "\nFINAL STANDINGS\n" + "=" * 60 + "\n\n")

        sorted_bots = sorted(totals.keys(), key=lambda x: -totals[x])
        col_w = max(len(n) for n in totals) + 2
        flog(f"{'Bot':<{col_w}} {'Total Chips':>12} {'Match Wins':>12}\n")
        flog("-" * (col_w + 26) + "\n")
        for name in sorted_bots:
            flog(f"{name:<{col_w}} {totals[name]:>+12d} {match_wins[name]:>12}\n")
        flog("\n")

        flog("Head-to-head (total chip diff in 10 matches):\n")
        flog("-" * 50 + "\n")
        for (name_a, _), (name_b, _) in pairs:
            pair_tot_a = sum(r[3] for r in results if r[0] == name_a and r[1] == name_b)
            pair_tot_b = sum(r[4] for r in results if r[0] == name_a and r[1] == name_b)
            flog(f"  {name_a} vs {name_b}:  {name_a} {pair_tot_a:+d}  {name_b} {pair_tot_b:+d}\n")

    with open(csv_path, "w") as f:
        f.write("bot_a,bot_b,match_num,chips_a,chips_b\n")
        for name_a, name_b, m, tot_a, tot_b in results:
            f.write(f"{name_a},{name_b},{m},{tot_a},{tot_b}\n")

    print("\n" + "=" * 60)
    print("FINAL STANDINGS")
    print("=" * 60)
    print(f"\n{'Bot':<14} {'Total Chips':>14} {'Match Wins':>12}")
    print("-" * 42)
    for name in sorted_bots:
        print(f"{name:<14} {totals[name]:>+14d} {match_wins[name]:>12}")
    print("\nHead-to-head (total chip diff over 10 matches):")
    print("-" * 50)
    for (name_a, _), (name_b, _) in pairs:
        pair_tot_a = sum(r[3] for r in results if r[0] == name_a and r[1] == name_b)
        pair_tot_b = sum(r[4] for r in results if r[0] == name_a and r[1] == name_b)
        print(f"  {name_a} vs {name_b}:  {name_a} {pair_tot_a:+d}  {name_b} {pair_tot_b:+d}")

    print(f"\nLog: {log_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
