#!/usr/bin/env python3
"""
METAV5 vs LambdaV1
10 matches x 1000 hands, with per-hand action logging.
"""

import sys, os, math, time, importlib, importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv

# ── Import bots ──────────────────────────────────────────────────────────────

_opp = importlib.import_module("submission.lambdaV1")
OPP = _opp.PlayerAgent

_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")
spec = importlib.util.spec_from_file_location("metav5", os.path.join(_base, "METAV5.py"))
_m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_m5)
METAV5 = _m5.PlayerAgent

HANDS_PER_MATCH = 1000
MATCHES = 10
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "logs", "metav5_postflop_v2_vs_lambdav1_10x1000.log")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

ACTION_NAMES = {
    PokerEnv.ActionType.FOLD.value:    "FOLD",
    PokerEnv.ActionType.RAISE.value:   "RAISE",
    PokerEnv.ActionType.CHECK.value:   "CHECK",
    PokerEnv.ActionType.CALL.value:    "CALL",
    PokerEnv.ActionType.DISCARD.value: "DISCARD",
}


def _augment(obs, opp_last="None"):
    obs["time_left"] = 400.0
    obs["opp_last_action"] = opp_last
    return obs


def _cards_str(card_ids):
    return " ".join(PokerEnv.int_card_to_str(int(c)) for c in card_ids if c != -1)


def play_hand_logged(env, a0, a1, logf, match_num, hand_num):
    (o0, o1), info = env.reset()
    o0 = _augment(o0)
    o1 = _augment(o1)

    m5_cards = [c for c in o0["my_cards"] if c != -1]
    opp_cards = [c for c in o1["my_cards"] if c != -1]
    m5_pos = "SB" if o0.get("blind_pos", o0.get("position", 0)) == 0 else "BB"
    opp_pos = "SB" if m5_pos == "BB" else "BB"

    logf.write(f"=== Match {match_num} Hand {hand_num} ===\n")
    logf.write(f"  METAV5 pos={m5_pos} dealt=[{_cards_str(m5_cards)}]\n")
    logf.write(f"  LambdaV1 pos={opp_pos} dealt=[{_cards_str(opp_cards)}]\n")

    terminated = False
    reward = (0, 0)
    last = "None"
    prev_street = -1
    street_actions = {0: [], 1: [], 2: [], 3: []}

    for _ in range(200):
        act_p = env.acting_agent
        obs = o0 if act_p == 0 else o1
        agent = a0 if act_p == 0 else a1
        other = a1 if act_p == 0 else a0
        obs["opp_last_action"] = last
        street = obs.get("street", 0)

        if street != prev_street:
            community = [c for c in obs.get("community_cards", []) if c != -1]
            if community:
                logf.write(f"  -- Street {street} board=[{_cards_str(community)}]\n")
            prev_street = street

        try:
            action = agent.act(obs, reward[act_p], terminated, False, info)
        except Exception as e:
            logf.write(f"  ERROR {e}\n")
            action = None
        if action is None:
            action = (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
        action = tuple(int(x) for x in action)

        act_name = ACTION_NAMES.get(action[0], f"?{action[0]}")
        who = "METAV5" if act_p == 0 else "LambdaV1"

        if action[0] == PokerEnv.ActionType.DISCARD.value:
            my_c = [c for c in obs["my_cards"] if c != -1]
            i, j = action[2], action[3]
            kept = [my_c[i], my_c[j]] if i < len(my_c) and j < len(my_c) else []
            thrown = [my_c[k] for k in range(len(my_c)) if k not in (i, j)] if kept else []
            detail = f"keep=[{_cards_str(kept)}] throw=[{_cards_str(thrown)}]"
        elif action[0] == PokerEnv.ActionType.RAISE.value:
            detail = f"amt={action[1]}"
        else:
            detail = ""

        line = f"  {who} {act_name} {detail}".rstrip()
        logf.write(line + "\n")
        street_actions[street].append(line.strip())

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

    community = [c for c in o0.get("community_cards", []) if c != -1]
    m5_held = [c for c in o0.get("my_cards", []) if c != -1]
    opp_held = [c for c in o1.get("my_cards", []) if c != -1]

    logf.write(f"  RESULT: METAV5 ev={reward[0]:+d} LambdaV1 ev={reward[1]:+d}"
               f" board=[{_cards_str(community)}]"
               f" m5_held=[{_cards_str(m5_held)}] opp_held=[{_cards_str(opp_held)}]\n")
    logf.write("\n")

    return reward[0], reward[1]


def run_match_logged(match_num, logf):
    env = PokerEnv()
    try:
        a0 = METAV5(stream=False)
    except TypeError:
        a0 = METAV5()
    try:
        a1 = OPP(stream=False)
    except TypeError:
        a1 = OPP()

    total_m5, total_a5, errors = 0, 0, 0
    for h in range(1, HANDS_PER_MATCH + 1):
        try:
            r0, r1 = play_hand_logged(env, a0, a1, logf, match_num, h)
            total_m5 += r0
            total_a5 += r1
        except Exception as e:
            logf.write(f"=== Match {match_num} Hand {h} ERROR: {e} ===\n\n")
            errors += 1

    return total_m5, total_a5, errors


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    print(f"METAV5 vs LambdaV1")
    print(f"  {MATCHES} matches x {HANDS_PER_MATCH} hands = {MATCHES * HANDS_PER_MATCH:,} total hands")
    print(f"  Logging to {LOG_FILE}")
    print("=" * 80)

    m5_evs = []
    opp_evs = []
    m5_wins = 0
    opp_wins = 0

    t_start = time.time()

    with open(LOG_FILE, "w") as logf:
        logf.write(f"METAV5 vs LambdaV1 — {MATCHES} matches x {HANDS_PER_MATCH} hands\n")
        logf.write("=" * 80 + "\n\n")

        for m in range(1, MATCHES + 1):
            mt = time.time()
            print(f"\n[Match {m}/{MATCHES}] starting...")

            r_m5, r_opp, errs = run_match_logged(m, logf)

            m5_evs.append(r_m5)
            opp_evs.append(r_opp)
            if r_m5 > r_opp:
                m5_wins += 1
            elif r_opp > r_m5:
                opp_wins += 1

            elapsed = time.time() - mt
            total_elapsed = time.time() - t_start
            err_s = f" [{errs} errors]" if errs else ""
            print(f"  Match {m}: METAV5 {r_m5:+d}  LambdaV1 {r_opp:+d}{err_s}"
                  f"  ({elapsed:.0f}s, total {total_elapsed:.0f}s)")

            logf.write(f"{'='*80}\n")
            logf.write(f"MATCH {m} SUMMARY: METAV5 {r_m5:+d}  LambdaV1 {r_opp:+d}{err_s}\n")
            logf.write(f"{'='*80}\n\n")

    total_time = time.time() - t_start

    print(f"\n{'=' * 80}")
    print(f"TOURNAMENT RESULTS  ({total_time:.0f}s / {total_time/60:.1f} min)")
    print(f"{'=' * 80}")
    print(f"  METAV5  total EV: {sum(m5_evs):+d}  mean: {_mean(m5_evs):+.1f}  "
          f"std: {_std(m5_evs):.1f}  wins: {m5_wins}/{MATCHES}")
    print(f"  LambdaV1 total EV: {sum(opp_evs):+d}  mean: {_mean(opp_evs):+.1f}  "
          f"std: {_std(opp_evs):.1f}  wins: {opp_wins}/{MATCHES}")

    draws = MATCHES - m5_wins - opp_wins
    print(f"\n  Record: METAV5 {m5_wins}W-{opp_wins}L-{draws}D")
    print(f"  EV/hand: METAV5 {sum(m5_evs)/(MATCHES*HANDS_PER_MATCH):+.3f}  "
          f"LambdaV1 {sum(opp_evs)/(MATCHES*HANDS_PER_MATCH):+.3f}")

    for i, (em, ea) in enumerate(zip(m5_evs, opp_evs), 1):
        print(f"    Match {i}: METAV5 {em:+d}  LambdaV1 {ea:+d}")

    print(f"\n  Detailed log: {LOG_FILE}")


if __name__ == "__main__":
    main()
