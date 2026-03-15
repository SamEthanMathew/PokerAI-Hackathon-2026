#!/usr/bin/env python3
"""
Round-robin tournament with statistical significance.
  - ALPHANiTV5       (original)
  - ALPHANiTV5 copy  (new discard engine)
  - METAV4           (original)
  - METAV4 copy      (new discard engine)

Each pair plays REPS matches of HANDS_PER_MATCH hands.
Reports mean EV, std dev, and 95% confidence intervals.
"""

import sys
import os
import math
import importlib
import importlib.util
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv

# ── Import bots ──────────────────────────────────────────────────────────────

_a5_orig = importlib.import_module("submission.ALPHANiTV5")
ALPHAv5_Orig = _a5_orig.PlayerAgent

_m4_orig = importlib.import_module("submission.METAV4")
METAV4_Orig = _m4_orig.PlayerAgent

def _load_copy(filepath, modname):
    spec = importlib.util.spec_from_file_location(modname, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PlayerAgent

_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "submission")

ALPHAv5_Copy = _load_copy(os.path.join(_base, "ALPHANiTV5 copy.py"), "a5copy")
METAV4_Copy  = _load_copy(os.path.join(_base, "METAV4 copy.py"),     "m4copy")

# ── Bot registry ─────────────────────────────────────────────────────────────

BOTS = {
    "ALPHAv5-orig":  ALPHAv5_Orig,
    "ALPHAv5-copy":  ALPHAv5_Copy,
    "METAV4-orig":   METAV4_Orig,
    "METAV4-copy":   METAV4_Copy,
}

HANDS_PER_MATCH = 1000
REPS = 20   # each pair plays 20 independent matches

# ── Match infrastructure ─────────────────────────────────────────────────────

def _augment(obs, opp_last="None"):
    obs["time_left"] = 400.0
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


def run_single_match(cls_a, cls_b, n_hands):
    """Play one fresh match of n_hands. Returns (ev_a, ev_b, errors)."""
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


# ── Stats helpers ────────────────────────────────────────────────────────────

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def _ci95(xs):
    """95% confidence interval half-width (t ≈ 2.093 for df=19)."""
    if len(xs) < 2:
        return 0.0
    t_crit = 2.093
    return t_crit * _std(xs) / math.sqrt(len(xs))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    names = list(BOTS.keys())
    n = len(names)
    total_pairings = n * (n - 1) // 2
    total_matches = total_pairings * REPS
    total_hands = total_matches * HANDS_PER_MATCH

    print(f"Round-robin tournament: {n} bots")
    print(f"  {REPS} matches per pairing, {HANDS_PER_MATCH} hands per match")
    print(f"  {total_pairings} pairings x {REPS} reps = {total_matches} matches "
          f"({total_hands:,} total hands)")
    print("=" * 100)

    # Cumulative EV across all reps
    cum_ev = {name: 0 for name in names}
    cum_wins = {name: 0 for name in names}
    cum_losses = {name: 0 for name in names}
    cum_draws = {name: 0 for name in names}

    # Per-pairing: list of (ev_a, ev_b) across reps
    pairing_evs = {}

    t_start = time.time()
    pairing_num = 0

    for i in range(n):
        for j in range(i + 1, n):
            pairing_num += 1
            na, nb = names[i], names[j]
            cls_a, cls_b = BOTS[na], BOTS[nb]

            print(f"\n[Pairing {pairing_num}/{total_pairings}] {na} vs {nb}  "
                  f"({REPS} x {HANDS_PER_MATCH} hands)")

            evs_a = []
            evs_b = []
            wins_a, wins_b, draws = 0, 0, 0

            for rep in range(REPS):
                rep_start = time.time()
                ta, tb, errs = run_single_match(cls_a, cls_b, HANDS_PER_MATCH)
                rep_time = time.time() - rep_start

                evs_a.append(ta)
                evs_b.append(tb)
                cum_ev[na] += ta
                cum_ev[nb] += tb

                if ta > tb:
                    wins_a += 1
                    cum_wins[na] += 1
                    cum_losses[nb] += 1
                elif tb > ta:
                    wins_b += 1
                    cum_wins[nb] += 1
                    cum_losses[na] += 1
                else:
                    draws += 1
                    cum_draws[na] += 1
                    cum_draws[nb] += 1

                err_s = f" [{errs}err]" if errs else ""
                elapsed = time.time() - t_start
                print(f"  rep {rep+1:2d}/{REPS}: {na} {ta:+5d} / {nb} {tb:+5d}"
                      f"{err_s}  ({rep_time:.0f}s, total {elapsed:.0f}s)")

            mean_a = _mean(evs_a)
            mean_b = _mean(evs_b)
            ci_a = _ci95(evs_a)
            std_a = _std(evs_a)

            pairing_evs[(na, nb)] = evs_a
            pairing_evs[(nb, na)] = evs_b

            print(f"  --- {na} vs {nb} summary ---")
            print(f"  {na}: mean={mean_a:+.0f}  std={std_a:.0f}  "
                  f"95%CI=[{mean_a - ci_a:+.0f}, {mean_a + ci_a:+.0f}]  "
                  f"record {wins_a}W-{wins_b}L-{draws}D")
            sig = "YES" if abs(mean_a) > ci_a and ci_a > 0 else "no"
            print(f"  Statistically significant at 95%? {sig}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 100}")
    print(f"Completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # ── Leaderboard ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"LEADERBOARD  (across all {REPS} reps per pairing)")
    print(f"{'RANK':<5} {'BOT':<18} {'TOTAL EV':>10} {'W':>5} {'L':>5} {'D':>5} "
          f"{'EV/MATCH':>10} {'WIN%':>7}")
    print("-" * 80)

    ranked = sorted(names, key=lambda x: cum_ev[x], reverse=True)
    for rank, name in enumerate(ranked, 1):
        total_m = cum_wins[name] + cum_losses[name] + cum_draws[name]
        ev_per = cum_ev[name] / max(1, total_m)
        win_pct = 100 * cum_wins[name] / max(1, total_m)
        print(f"{rank:<5} {name:<18} {cum_ev[name]:>+10d} "
              f"{cum_wins[name]:>5} {cum_losses[name]:>5} {cum_draws[name]:>5} "
              f"{ev_per:>+10.1f} {win_pct:>6.1f}%")

    # ── Head-to-head summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("HEAD-TO-HEAD  (mean EV per 1000-hand match, +/- 95% CI)")
    print()

    header = f"{'':>18} " + " ".join(f"{n[:14]:>20}" for n in ranked)
    print(header)
    print("-" * len(header))

    for na in ranked:
        row = f"{na:>18} "
        for nb in ranked:
            if na == nb:
                row += f"{'---':>20} "
            elif (na, nb) in pairing_evs:
                evs = pairing_evs[(na, nb)]
                m = _mean(evs)
                ci = _ci95(evs)
                sig = "*" if abs(m) > ci and ci > 0 else " "
                row += f"{m:>+7.0f} +/-{ci:>4.0f}{sig}   "
            else:
                row += f"{'':>20} "
        print(row)

    print()
    print("  * = statistically significant at 95% confidence")
    print()


if __name__ == "__main__":
    main()
