"""
Run a short match of 50 hands using agent_config.json.
Prints result and a compute-rules check (time per hand vs tournament limits).
"""
import json
import logging
import multiprocessing
import importlib
import time

import match as match_module
from match import run_api_match

NUM_HANDS = 50

PHASE1_TIME_LIMIT_SEC = 500
PHASE2_TIME_LIMIT_SEC = 1000
PHASE3_TIME_LIMIT_SEC = 1500
HANDS_PER_MATCH = 1000


def load_agent_class(file_path):
    module_path, class_name = file_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    # Reset match.py globals so previous runs don't contaminate
    match_module.bankrolls = [0, 0]
    match_module.time_used_0 = 0.0
    match_module.time_used_1 = 0.0

    with open("agent_config.json", "r") as f:
        config = json.load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    bot0_class = load_agent_class(config["bot0"]["file_path"])
    bot1_class = load_agent_class(config["bot1"]["file_path"])

    process0 = multiprocessing.Process(
        target=bot0_class.run,
        args=(True, config["bot0"]["port"]),
        kwargs={"player_id": config["bot0"]["player_id"]},
    )
    process1 = multiprocessing.Process(
        target=bot1_class.run,
        args=(True, config["bot1"]["port"]),
        kwargs={"player_id": config["bot1"]["player_id"]},
    )

    process0.start()
    process1.start()

    # Give servers time to start
    time.sleep(3)

    logger.info("Starting match: %d hands", NUM_HANDS)
    try:
        result = run_api_match(
            f"http://localhost:{config['bot0']['port']}",
            f"http://localhost:{config['bot1']['port']}",
            logger,
            num_hands=NUM_HANDS,
            csv_path="./match_50_hands.csv",
            team_0_name=bot0_class.__name__,
            team_1_name=bot1_class.__name__,
        )
    except Exception as e:
        logger.error("Match failed: %s", e)
        result = {"status": "error", "error": str(e)}
    finally:
        process0.terminate()
        process1.terminate()
        process0.join(timeout=5)
        process1.join(timeout=5)
        if process0.is_alive():
            process0.kill()
        if process1.is_alive():
            process1.kill()

    # --- Result summary ---
    print("\n" + "=" * 60)
    print("MATCH RESULT (%d hands)" % NUM_HANDS)
    print("=" * 60)
    print("Status: %s" % result.get("status", "?"))
    if result.get("bot0_reward") is not None:
        print("Bot0 (%s): %s chips" % (bot0_class.__name__, result["bot0_reward"]))
        print("Bot1 (%s): %s chips" % (bot1_class.__name__, result["bot1_reward"]))
        print("Result: %s" % result.get("result", "?"))

    # --- Compute rules check ---
    t0 = result.get("bot0_time_used")
    t1 = result.get("bot1_time_used")
    if t0 is not None and t1 is not None:
        sec_per_hand_0 = t0 / NUM_HANDS
        sec_per_hand_1 = t1 / NUM_HANDS
        phase1_max = PHASE1_TIME_LIMIT_SEC / HANDS_PER_MATCH
        phase2_max = PHASE2_TIME_LIMIT_SEC / HANDS_PER_MATCH
        phase3_max = PHASE3_TIME_LIMIT_SEC / HANDS_PER_MATCH

        print("\n" + "=" * 60)
        print("COMPUTE RULES CHECK (tournament time limits)")
        print("=" * 60)
        print("Time used this run (%d hands):" % NUM_HANDS)
        print("  Bot0: %.2f s  (%.4f s/hand)" % (t0, sec_per_hand_0))
        print("  Bot1: %.2f s  (%.4f s/hand)" % (t1, sec_per_hand_1))
        print("")
        print("Tournament: %d hands/match. Time limit per match:" % HANDS_PER_MATCH)
        print("  Phase 1: %ds -> max %.2f s/hand" % (PHASE1_TIME_LIMIT_SEC, phase1_max))
        print("  Phase 2: %ds -> max %.2f s/hand" % (PHASE2_TIME_LIMIT_SEC, phase2_max))
        print("  Phase 3: %ds -> max %.2f s/hand" % (PHASE3_TIME_LIMIT_SEC, phase3_max))
        print("")
        ok0 = sec_per_hand_0 <= phase1_max
        ok1 = sec_per_hand_1 <= phase1_max
        if ok0 and ok1:
            print("  -> Both bots PASS Phase 1 limit (%.2f s/hand)." % phase1_max)
        else:
            if not ok0:
                print("  -> WARNING: Bot0 EXCEEDS Phase 1 (%.4f > %.2f s/hand)." % (sec_per_hand_0, phase1_max))
            if not ok1:
                print("  -> WARNING: Bot1 EXCEEDS Phase 1 (%.4f > %.2f s/hand)." % (sec_per_hand_1, phase1_max))
        print("=" * 60)

    if result.get("error"):
        print("\nERROR: %s" % result["error"])

    return result


if __name__ == "__main__":
    main()
