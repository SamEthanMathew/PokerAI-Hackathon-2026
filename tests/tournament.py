#!/usr/bin/env python3
"""
Round-robin tournament: all 12 heuristic agents + ALPHANiTV5 + Libratus.
Each pair plays N_HANDS hands. Prints leaderboard sorted by total EV.
"""

import sys
import os
import importlib
import traceback
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gym_env import PokerEnv

# ── Import all agents with unique names ──────────────────────────────────────

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

# Import teammates' bots via importlib (they all use "PlayerAgent")
_alphanit_mod = importlib.import_module("submission.ALPHANiTV5")
ALPHANiTV5Agent = _alphanit_mod.PlayerAgent

_libratus_mod = importlib.import_module("submission.Libratus")
LibratusAgent = _libratus_mod.PlayerAgent

# ── Bot registry ─────────────────────────────────────────────────────────────

BOTS = {
    "PriorityConserv": PriorityConservativeAgent,
    "PriorityAggrss":  PriorityAggressiveAgent,
    "BoardMadeFirst":  BoardMadeFirstAgent,
    "OutsMaximizer":   OutsMaximizerAgent,
    "ExactStrength":   ExactStrengthEnumerateAgent,
    "ExactEquity":     ExactEquityEnumerateAgent,
    "OppDiscardAware": OppDiscardAwareAgent,
    "BlockerValue":    BlockerValueAgent,
    "MinRiskPressure": MinRiskPressureAgent,
    "InfoHiderMixed":  InfoHiderMixedAgent,
    "TripsAutoFold":   TripsAutoFoldStrictAgent,
    "AdaptiveOppMdl":  AdaptiveOpponentModelAgent,
    "ALPHANiTV5":      ALPHANiTV5Agent,
    "Libratus":        LibratusAgent,
}

N_HANDS = 1000  # per matchup (tournament spec)

# ── Match runner ─────────────────────────────────────────────────────────────

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


def run_match(name_a, cls_a, name_b, cls_b, n_hands):
    env = PokerEnv()
    try:
        a0 = cls_a(stream=False)
    except TypeError:
        a0 = cls_a()
    try:
        a1 = cls_b(stream=False)
    except TypeError:
        a1 = cls_b()

    total_a, total_b, errors = 0, 0, 0
    for _ in range(n_hands):
        try:
            r0, r1 = play_hand(env, a0, a1)
            total_a += r0
            total_b += r1
        except Exception:
            errors += 1

    return total_a, total_b, errors


# ── Tournament ───────────────────────────────────────────────────────────────

def main():
    names = list(BOTS.keys())
    n = len(names)
    print(f"Round-robin tournament: {n} bots, {N_HANDS} hands per matchup")
    print(f"Total matchups: {n * (n - 1) // 2}")
    print("=" * 90)

    # Track cumulative EV, wins, matchup details
    ev = {name: 0 for name in names}
    match_wins = {name: 0 for name in names}
    match_losses = {name: 0 for name in names}
    match_draws = {name: 0 for name in names}
    results_grid = {}

    t_start = time.time()
    matchup_num = 0
    total_matchups = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            matchup_num += 1
            na, nb = names[i], names[j]
            cls_a, cls_b = BOTS[na], BOTS[nb]

            ta, tb, errs = run_match(na, cls_a, nb, cls_b, N_HANDS)

            ev[na] += ta
            ev[nb] += tb

            if ta > tb:
                match_wins[na] += 1
                match_losses[nb] += 1
            elif tb > ta:
                match_wins[nb] += 1
                match_losses[na] += 1
            else:
                match_draws[na] += 1
                match_draws[nb] += 1

            results_grid[(na, nb)] = (ta, tb)
            results_grid[(nb, na)] = (tb, ta)

            err_s = f" [{errs} errs]" if errs else ""
            elapsed = time.time() - t_start
            print(f"  [{matchup_num:3d}/{total_matchups}] {na:18s} vs {nb:18s} => {ta:+5d} / {tb:+5d}{err_s}  ({elapsed:.0f}s)")

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s")

    # ── Leaderboard ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'RANK':<5} {'BOT':<22} {'TOTAL EV':>10} {'W':>4} {'L':>4} {'D':>4} {'EV/MATCH':>10}")
    print("-" * 90)

    ranked = sorted(names, key=lambda x: ev[x], reverse=True)
    for rank, name in enumerate(ranked, 1):
        total_matches = match_wins[name] + match_losses[name] + match_draws[name]
        ev_per = ev[name] / max(1, total_matches)
        print(f"{rank:<5} {name:<22} {ev[name]:>+10d} {match_wins[name]:>4} {match_losses[name]:>4} {match_draws[name]:>4} {ev_per:>+10.1f}")

    # ── Head-to-head matrix ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD MATRIX (row's EV vs column)")
    print()

    short = {name: name[:8] for name in names}
    header = f"{'':>18} " + " ".join(f"{short[n]:>8}" for n in ranked)
    print(header)
    print("-" * len(header))

    for na in ranked:
        row = f"{na:>18} "
        for nb in ranked:
            if na == nb:
                row += f"{'---':>8} "
            elif (na, nb) in results_grid:
                val = results_grid[(na, nb)][0]
                row += f"{val:>+8d} "
            else:
                row += f"{'':>8} "
        print(row)

    print()


if __name__ == "__main__":
    main()
