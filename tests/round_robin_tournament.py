#!/usr/bin/env python3
"""Round-robin tournament: DELTA vs OMICRoN vs MetaV5 vs GenesisV2.
Each pair plays 5 matches of 1000 hands. Results logged to logs/tournament_results.log."""

import sys, os, time, importlib, importlib.util, traceback, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sub = os.path.join(_base, "submission")
_gen = os.path.join(_base, "genesis")

LOG_DIR = os.path.join(_base, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "tournament_results.log")


def _load(filepath, modname, classname="PlayerAgent"):
    spec = importlib.util.spec_from_file_location(modname, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, classname)


BOTS = {
    "DELTA":    (_load(os.path.join(_sub, "DELTA_V1.py"),    "delta_t"),                       "DELTA_V1"),
    "OMICRoN":  (_load(os.path.join(_sub, "OMICRoN_V1.py"), "omicron_t"),                      "OMICRoN_V1"),
    "MetaV5":   (_load(os.path.join(_sub, "METAV5.py"),      "metav5_t"),                      "METAV5"),
    "GenesisV2":(_load(os.path.join(_gen, "genesisV2.py"),   "genesis_t", "GenesisV2Agent"),   "GenesisV2"),
}

HANDS_PER_MATCH = 1000
MATCHES_PER_PAIR = 5
TIME_LIMIT = 1500.0


def log(msg, fh):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    fh.write(line + "\n")
    fh.flush()


def play_hand(env, a0, a1, time_left_0, time_left_1):
    (o0, o1), info = env.reset()
    o0["time_left"] = time_left_0
    o1["time_left"] = time_left_1
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

        t_start = time.perf_counter()
        action = agent.act(obs, reward[act_p], terminated, False, info)
        dt = time.perf_counter() - t_start
        if act_p == 0:
            t0_total += dt
        else:
            t1_total += dt

        if action is None:
            action = (PokerEnv.ActionType.FOLD.value, 0, 0, 0)
        action = tuple(int(x) for x in action)

        last = PokerEnv.ActionType(action[0]).name
        (o0, o1), reward, terminated, truncated, info = env.step(action)
        o0["time_left"] = time_left_0
        o1["time_left"] = time_left_1
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


def run_match(name_a, cls_a, name_b, cls_b, num_hands, fh):
    env = PokerEnv()
    a0 = cls_a(stream=False)
    a1 = cls_b(stream=False)

    total_a = 0
    total_b = 0
    time_a = 0.0
    time_b = 0.0
    errors = 0

    for h in range(num_hands):
        tl_a = max(10.0, TIME_LIMIT - time_a)
        tl_b = max(10.0, TIME_LIMIT - time_b)
        try:
            r0, r1, dt0, dt1 = play_hand(env, a0, a1, tl_a, tl_b)
            total_a += r0
            total_b += r1
            time_a += dt0
            time_b += dt1
        except Exception:
            errors += 1
            if errors <= 2:
                traceback.print_exc()
            continue

        if (h + 1) % 250 == 0:
            log(f"  hand {h+1:4d}: {name_a}={total_a:+5d}  {name_b}={total_b:+5d}  "
                f"t_a={time_a:.1f}s  t_b={time_b:.1f}s", fh)

    winner = name_a if total_a > total_b else (name_b if total_b > total_a else "TIE")
    margin = abs(total_a - total_b)
    avg_ms_a = time_a / max(1, num_hands) * 1000
    avg_ms_b = time_b / max(1, num_hands) * 1000
    return total_a, total_b, winner, margin, avg_ms_a, avg_ms_b, time_a, time_b, errors


def main():
    bot_names = list(BOTS.keys())
    pairs = []
    for i in range(len(bot_names)):
        for j in range(i + 1, len(bot_names)):
            pairs.append((bot_names[i], bot_names[j]))

    total_matches = len(pairs) * MATCHES_PER_PAIR
    start_time = datetime.datetime.now()

    with open(LOG_PATH, "w") as fh:
        log(f"ROUND-ROBIN TOURNAMENT", fh)
        log(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", fh)
        log(f"Bots: {', '.join(bot_names)}", fh)
        log(f"Pairs: {len(pairs)}, Matches/pair: {MATCHES_PER_PAIR}, "
            f"Hands/match: {HANDS_PER_MATCH}", fh)
        log(f"Total matches: {total_matches}", fh)
        log(f"{'='*72}", fh)

        standings = {name: {"wins": 0, "losses": 0, "ties": 0, "pnl": 0,
                            "matches": 0, "total_time": 0.0, "errors": 0}
                     for name in bot_names}
        match_results = []
        match_num = 0

        for name_a, name_b in pairs:
            cls_a = BOTS[name_a][0]
            cls_b = BOTS[name_b][0]

            for m in range(MATCHES_PER_PAIR):
                match_num += 1
                log(f"\nMATCH {match_num}/{total_matches}: {name_a} vs {name_b} "
                    f"(game {m+1}/{MATCHES_PER_PAIR})", fh)

                pnl_a, pnl_b, winner, margin, ms_a, ms_b, ta, tb, errs = \
                    run_match(name_a, cls_a, name_b, cls_b, HANDS_PER_MATCH, fh)

                log(f"  RESULT: {name_a}={pnl_a:+d}  {name_b}={pnl_b:+d}  "
                    f"winner={winner} (+{margin})  "
                    f"avg_ms: {name_a}={ms_a:.1f} {name_b}={ms_b:.1f}  "
                    f"errors={errs}", fh)

                match_results.append({
                    "a": name_a, "b": name_b, "game": m + 1,
                    "pnl_a": pnl_a, "pnl_b": pnl_b, "winner": winner,
                    "margin": margin, "ms_a": ms_a, "ms_b": ms_b, "errors": errs,
                })

                standings[name_a]["matches"] += 1
                standings[name_b]["matches"] += 1
                standings[name_a]["pnl"] += pnl_a
                standings[name_b]["pnl"] += pnl_b
                standings[name_a]["total_time"] += ta
                standings[name_b]["total_time"] += tb
                standings[name_a]["errors"] += errs
                standings[name_b]["errors"] += errs

                if winner == name_a:
                    standings[name_a]["wins"] += 1
                    standings[name_b]["losses"] += 1
                elif winner == name_b:
                    standings[name_b]["wins"] += 1
                    standings[name_a]["losses"] += 1
                else:
                    standings[name_a]["ties"] += 1
                    standings[name_b]["ties"] += 1

        elapsed = (datetime.datetime.now() - start_time).total_seconds()

        log(f"\n{'='*72}", fh)
        log(f"TOURNAMENT COMPLETE — elapsed {elapsed:.0f}s", fh)
        log(f"{'='*72}", fh)

        log(f"\n{'─'*72}", fh)
        log(f"  HEAD-TO-HEAD RESULTS", fh)
        log(f"{'─'*72}", fh)
        for na, nb in pairs:
            games = [r for r in match_results if r["a"] == na and r["b"] == nb]
            a_wins = sum(1 for g in games if g["winner"] == na)
            b_wins = sum(1 for g in games if g["winner"] == nb)
            ties = sum(1 for g in games if g["winner"] == "TIE")
            a_total = sum(g["pnl_a"] for g in games)
            b_total = sum(g["pnl_b"] for g in games)
            log(f"  {na:12s} vs {nb:12s}:  {na} {a_wins}W-{b_wins}L-{ties}T  "
                f"PnL: {na}={a_total:+6d}  {nb}={b_total:+6d}", fh)

        log(f"\n{'─'*72}", fh)
        log(f"  FINAL STANDINGS (sorted by wins, then PnL)", fh)
        log(f"{'─'*72}", fh)
        log(f"  {'Bot':12s}  {'W':>3s}  {'L':>3s}  {'T':>3s}  {'PnL':>7s}  "
            f"{'Avg ms':>7s}  {'Errors':>6s}", fh)
        log(f"  {'─'*12}  {'─'*3}  {'─'*3}  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*6}", fh)

        ranked = sorted(standings.items(),
                        key=lambda x: (x[1]["wins"], x[1]["pnl"]), reverse=True)
        for name, s in ranked:
            avg_ms = s["total_time"] / max(1, s["matches"]) / HANDS_PER_MATCH * 1000
            log(f"  {name:12s}  {s['wins']:3d}  {s['losses']:3d}  {s['ties']:3d}  "
                f"{s['pnl']:+7d}  {avg_ms:7.1f}  {s['errors']:6d}", fh)

        log(f"\n  Log saved to: {LOG_PATH}", fh)


if __name__ == "__main__":
    main()
