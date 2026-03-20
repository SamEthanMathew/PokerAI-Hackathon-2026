#!/usr/bin/env python3
"""METAV6 vs ALPHANiTV8 -- 2-match validation tournament."""

import sys, os, math, time, importlib, importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv

_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission")

spec6 = importlib.util.spec_from_file_location("metav6", os.path.join(_base, "METAV6.py"))
_m6 = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(_m6)
METAV6 = _m6.PlayerAgent

spec8 = importlib.util.spec_from_file_location("alphav8", os.path.join(_base, "ALPHANiTV8.py"))
_a8 = importlib.util.module_from_spec(spec8)
spec8.loader.exec_module(_a8)
ALPHANiTV8 = _a8.PlayerAgent

HANDS = 1000
MATCHES = 2

ACTION_NAMES = {
    PokerEnv.ActionType.FOLD.value: "FOLD",
    PokerEnv.ActionType.RAISE.value: "RAISE",
    PokerEnv.ActionType.CHECK.value: "CHECK",
    PokerEnv.ActionType.CALL.value: "CALL",
    PokerEnv.ActionType.DISCARD.value: "DISCARD",
}


def _augment(obs, opp_last="None"):
    obs["time_left"] = 400.0
    obs["opp_last_action"] = opp_last
    return obs


def play_hand(env, a0, a1, logf, match_num, hand_num):
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
        except Exception as e:
            logf.write(f"  M{match_num}H{hand_num} ERROR act: {e}\n")
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


def main():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "metav6_vs_alphav8_validation.log")

    print(f"METAV6 vs ALPHANiTV8 -- {MATCHES} matches x {HANDS} hands")

    with open(log_file, "w") as logf:
        logf.write(f"METAV6 vs ALPHANiTV8 -- {MATCHES}x{HANDS}\n\n")

        for m in range(1, MATCHES + 1):
            env = PokerEnv()
            try:
                a0 = METAV6(stream=False)
            except TypeError:
                a0 = METAV6()
            try:
                a1 = ALPHANiTV8(stream=False)
            except TypeError:
                a1 = ALPHANiTV8()

            t0 = time.time()
            tot_m6, tot_a8, errs = 0, 0, 0
            for h in range(1, HANDS + 1):
                try:
                    r0, r1 = play_hand(env, a0, a1, logf, m, h)
                    tot_m6 += r0
                    tot_a8 += r1
                except Exception as e:
                    logf.write(f"M{m}H{h} CRASH: {e}\n")
                    errs += 1
            elapsed = time.time() - t0
            err_s = f" [{errs} errors]" if errs else ""
            print(f"  Match {m}: METAV6 {tot_m6:+d}  ALPHANiTV8 {tot_a8:+d}{err_s}  ({elapsed:.0f}s)")
            logf.write(f"MATCH {m}: METAV6 {tot_m6:+d}  ALPHANiTV8 {tot_a8:+d}{err_s}\n")

            opp_w = a1._opp_model.weights
            print(f"    Learned opp model weights: [{', '.join(f'{w:.1f}' for w in opp_w)}]")

    print(f"\nLog: {log_file}")


if __name__ == "__main__":
    main()
