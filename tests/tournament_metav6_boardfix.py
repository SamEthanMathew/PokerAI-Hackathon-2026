#!/usr/bin/env python3
"""
METAV6 (board-texture awareness) vs LambdaV1 and ALPHANiTV7
10 matches x 1000 hands each, with per-hand action logging.
"""

import sys, os, math, time, importlib, importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv

_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")
spec = importlib.util.spec_from_file_location("metav6", os.path.join(_base, "METAV6.py"))
_m6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_m6)
METAV6 = _m6.PlayerAgent

HANDS_PER_MATCH = 1000
MATCHES = 10

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


def play_hand_logged(env, a0, a1, logf, match_num, hand_num, opp_label):
    (o0, o1), info = env.reset()
    o0 = _augment(o0)
    o1 = _augment(o1)

    m6_cards = [c for c in o0["my_cards"] if c != -1]
    opp_cards = [c for c in o1["my_cards"] if c != -1]
    m6_pos = "SB" if o0.get("blind_pos", o0.get("position", 0)) == 0 else "BB"
    opp_pos = "SB" if m6_pos == "BB" else "BB"

    logf.write(f"=== Match {match_num} Hand {hand_num} ===\n")
    logf.write(f"  METAV6 pos={m6_pos} dealt=[{_cards_str(m6_cards)}]\n")
    logf.write(f"  {opp_label} pos={opp_pos} dealt=[{_cards_str(opp_cards)}]\n")

    terminated = False
    reward = (0, 0)
    last = "None"
    prev_street = -1

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
        who = "METAV6" if act_p == 0 else opp_label

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
    m6_held = [c for c in o0.get("my_cards", []) if c != -1]
    opp_held = [c for c in o1.get("my_cards", []) if c != -1]

    logf.write(f"  RESULT: METAV6 ev={reward[0]:+d} {opp_label} ev={reward[1]:+d}"
               f" board=[{_cards_str(community)}]"
               f" m6_held=[{_cards_str(m6_held)}] opp_held=[{_cards_str(opp_held)}]\n")
    logf.write("\n")

    return reward[0], reward[1]


def run_match_logged(match_num, logf, OPP, opp_label):
    env = PokerEnv()
    try:
        a0 = METAV6(stream=False)
    except TypeError:
        a0 = METAV6()
    try:
        a1 = OPP(stream=False)
    except TypeError:
        a1 = OPP()

    total_m6, total_opp, errors = 0, 0, 0
    for h in range(1, HANDS_PER_MATCH + 1):
        try:
            r0, r1 = play_hand_logged(env, a0, a1, logf, match_num, h, opp_label)
            total_m6 += r0
            total_opp += r1
        except Exception as e:
            logf.write(f"=== Match {match_num} Hand {h} ERROR: {e} ===\n\n")
            errors += 1

    return total_m6, total_opp, errors


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def run_tournament(OPP, opp_label, log_file):
    print(f"\nMETAV6 vs {opp_label}")
    print(f"  {MATCHES} matches x {HANDS_PER_MATCH} hands = {MATCHES * HANDS_PER_MATCH:,} total hands")
    print(f"  Logging to {log_file}")
    print("=" * 80)

    m6_evs = []
    opp_evs = []
    m6_wins = 0
    opp_wins = 0

    t_start = time.time()

    with open(log_file, "w") as logf:
        logf.write(f"METAV6 vs {opp_label} — {MATCHES} matches x {HANDS_PER_MATCH} hands\n")
        logf.write("=" * 80 + "\n\n")

        for m in range(1, MATCHES + 1):
            mt = time.time()
            print(f"\n[Match {m}/{MATCHES}] starting...")

            r_m6, r_opp, errs = run_match_logged(m, logf, OPP, opp_label)

            m6_evs.append(r_m6)
            opp_evs.append(r_opp)
            if r_m6 > r_opp:
                m6_wins += 1
            elif r_opp > r_m6:
                opp_wins += 1

            elapsed = time.time() - mt
            total_elapsed = time.time() - t_start
            err_s = f" [{errs} errors]" if errs else ""
            print(f"  Match {m}: METAV6 {r_m6:+d}  {opp_label} {r_opp:+d}{err_s}"
                  f"  ({elapsed:.0f}s, total {total_elapsed:.0f}s)")

            logf.write(f"{'='*80}\n")
            logf.write(f"MATCH {m} SUMMARY: METAV6 {r_m6:+d}  {opp_label} {r_opp:+d}{err_s}\n")
            logf.write(f"{'='*80}\n\n")

    total_time = time.time() - t_start

    print(f"\n{'=' * 80}")
    print(f"TOURNAMENT RESULTS vs {opp_label}  ({total_time:.0f}s / {total_time/60:.1f} min)")
    print(f"{'=' * 80}")
    print(f"  METAV6  total EV: {sum(m6_evs):+d}  mean: {_mean(m6_evs):+.1f}  "
          f"std: {_std(m6_evs):.1f}  wins: {m6_wins}/{MATCHES}")
    print(f"  {opp_label} total EV: {sum(opp_evs):+d}  mean: {_mean(opp_evs):+.1f}  "
          f"std: {_std(opp_evs):.1f}  wins: {opp_wins}/{MATCHES}")

    draws = MATCHES - m6_wins - opp_wins
    print(f"\n  Record: METAV6 {m6_wins}W-{opp_wins}L-{draws}D")
    print(f"  EV/hand: METAV6 {sum(m6_evs)/(MATCHES*HANDS_PER_MATCH):+.3f}  "
          f"{opp_label} {sum(opp_evs)/(MATCHES*HANDS_PER_MATCH):+.3f}")

    for i, (em, ea) in enumerate(zip(m6_evs, opp_evs), 1):
        print(f"    Match {i}: METAV6 {em:+d}  {opp_label} {ea:+d}")

    print(f"\n  Detailed log: {log_file}")


def main():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)

    _lambda = importlib.import_module("submission.lambdaV1")
    LambdaV1 = _lambda.PlayerAgent
    run_tournament(LambdaV1, "LambdaV1",
                   os.path.join(log_dir, "metav6_boardfix_vs_lambdav1_10x1000.log"))

    _alpha7 = importlib.import_module("submission.ALPHANiTV7")
    ALPHANiTV7 = _alpha7.PlayerAgent
    run_tournament(ALPHANiTV7, "ALPHANiTV7",
                   os.path.join(log_dir, "metav6_boardfix_vs_alphav7_10x1000.log"))


if __name__ == "__main__":
    main()
