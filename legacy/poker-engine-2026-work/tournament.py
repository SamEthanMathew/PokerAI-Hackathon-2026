"""
Tournament runner: plays N independent 1000-hand matches between two bots.
Each match runs in a fresh subprocess so global state resets cleanly.
"""

import json
import subprocess
import sys
import os
import signal
import textwrap

BOT0_PATH = "submission.player.PlayerAgent"
BOT1_PATH = "HRT_submission.player1.PlayerAgent"
BOT0_NAME = "PlayerAgent"
BOT1_NAME = "HRT_Player1"
NUM_GAMES = 5
HANDS_PER_GAME = 1000
PORT0 = 8000
PORT1 = 8001


def kill_port_holders():
    """Kill any processes holding our ports."""
    for port in (PORT0, PORT1):
        try:
            out = subprocess.check_output(
                f"lsof -ti:{port}", shell=True, text=True, stderr=subprocess.DEVNULL
            ).strip()
            for pid in out.split():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass
        except subprocess.CalledProcessError:
            pass


def run_single_match(game_num):
    kill_port_holders()

    csv_path = f"./tournament_game_{game_num}.csv"
    config = {
        "bot0": {"file_path": BOT0_PATH, "port": PORT0, "player_id": "bot0"},
        "bot1": {"file_path": BOT1_PATH, "port": PORT1, "player_id": "bot1"},
        "match_settings": {"csv_output_path": csv_path},
    }
    config_path = f"_tournament_config_{game_num}.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    runner_code = textwrap.dedent(f"""\
        import logging, json, importlib, multiprocessing, time, os, signal, sys
        from match import run_api_match
        import match as _m

        _m.bankrolls = [0, 0]
        _m.time_used_0 = 0.0
        _m.time_used_1 = 0.0
        _m.failure_tracker = _m.AgentFailureTracker()

        with open({config_path!r}) as f:
            config = json.load(f)

        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
        logger = logging.getLogger("game{game_num}")

        mod0, cls0 = config["bot0"]["file_path"].rsplit(".", 1)
        mod1, cls1 = config["bot1"]["file_path"].rsplit(".", 1)
        Bot0 = getattr(importlib.import_module(mod0), cls0)
        Bot1 = getattr(importlib.import_module(mod1), cls1)

        p0 = multiprocessing.Process(target=Bot0.run, args=(False, config["bot0"]["port"]),
                                      kwargs={{"player_id": config["bot0"]["player_id"]}})
        p1 = multiprocessing.Process(target=Bot1.run, args=(False, config["bot1"]["port"]),
                                      kwargs={{"player_id": config["bot1"]["player_id"]}})
        p0.daemon = True
        p1.daemon = True
        p0.start(); p1.start()
        time.sleep(2)

        try:
            result = run_api_match(
                f"http://localhost:{{config['bot0']['port']}}",
                f"http://localhost:{{config['bot1']['port']}}",
                logger, num_hands={HANDS_PER_GAME},
                csv_path=config["match_settings"]["csv_output_path"],
                team_0_name="{BOT0_NAME}", team_1_name="{BOT1_NAME}")
        finally:
            for p in (p0, p1):
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)

        sys.stdout.write("RESULT_JSON:" + json.dumps(result) + "\\n")
        sys.stdout.flush()
    """)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            capture_output=True, text=True, timeout=1800,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "result": "invalid", "error": "subprocess timed out"}
    finally:
        try:
            os.remove(config_path)
        except OSError:
            pass
        kill_port_holders()

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])

    print(f"  [Game {game_num}] No result found. Last 20 stderr lines:")
    for ln in proc.stderr.splitlines()[-20:]:
        print(f"    {ln}")
    return {"status": "error", "result": "invalid", "error": "no result line found"}


def main():
    print(f"{'='*60}")
    print(f"  TOURNAMENT: {BOT0_NAME} vs {BOT1_NAME}")
    print(f"  {NUM_GAMES} games x {HANDS_PER_GAME} hands each")
    print(f"{'='*60}")
    print()

    results = []
    bot0_total = 0
    bot1_total = 0
    bot0_wins = 0
    bot1_wins = 0

    for g in range(1, NUM_GAMES + 1):
        print(f"--- Game {g}/{NUM_GAMES} ---", flush=True)
        res = run_single_match(g)
        results.append(res)

        b0 = res.get("bot0_reward", 0)
        b1 = res.get("bot1_reward", 0)
        bot0_total += b0
        bot1_total += b1
        status = res.get("status", "?")
        winner_str = res.get("result", "?")

        if winner_str == "win":
            bot0_wins += 1
            tag = f"{BOT0_NAME} WINS"
        elif winner_str == "loss":
            bot1_wins += 1
            tag = f"{BOT1_NAME} WINS"
        else:
            tag = "TIE / ERROR"

        print(f"  Status: {status} | {tag}")
        print(f"  {BOT0_NAME}: {b0:+}  |  {BOT1_NAME}: {b1:+}")
        t0 = res.get("bot0_time_used", 0)
        t1 = res.get("bot1_time_used", 0)
        print(f"  Time: {BOT0_NAME} {t0:.1f}s | {BOT1_NAME} {t1:.1f}s")
        print(flush=True)

    print(f"{'='*60}")
    print(f"  TOURNAMENT RESULTS")
    print(f"{'='*60}")
    print(f"  Games won:  {BOT0_NAME} {bot0_wins}  |  {BOT1_NAME} {bot1_wins}  |  Ties {NUM_GAMES - bot0_wins - bot1_wins}")
    print(f"  Total PnL:  {BOT0_NAME} {bot0_total:+}  |  {BOT1_NAME} {bot1_total:+}")
    print(f"  Avg PnL/game: {BOT0_NAME} {bot0_total/NUM_GAMES:+.1f}  |  {BOT1_NAME} {bot1_total/NUM_GAMES:+.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
