"""Run a short match: ALPHANiT (bot0) vs default player (bot1)."""
import logging
import multiprocessing
import json
import importlib

from match import run_api_match

NUM_HANDS = 100


def load_agent_class(file_path):
    module_path, class_name = file_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    with open("agent_config.json", "r") as f:
        config = json.load(f)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    bot0_class = load_agent_class(config["bot0"]["file_path"])
    bot1_class = load_agent_class(config["bot1"]["file_path"])

    process0 = multiprocessing.Process(
        target=bot0_class.run,
        args=(False, config["bot0"]["port"]),
        kwargs={"player_id": config["bot0"]["player_id"]},
    )
    process1 = multiprocessing.Process(
        target=bot1_class.run,
        args=(False, config["bot1"]["port"]),
        kwargs={"player_id": config["bot1"]["player_id"]},
    )

    process0.start()
    process1.start()

    logger.info("Starting ALPHANiT vs PlayerAgent match (%d hands)", NUM_HANDS)
    result = run_api_match(
        f"http://localhost:{config['bot0']['port']}",
        f"http://localhost:{config['bot1']['port']}",
        logger,
        num_hands=NUM_HANDS,
        csv_path="./match_alphanit_vs_player.csv",
        team_0_name="ALPHANiT",
        team_1_name="PlayerAgent",
    )
    logger.info("Match result: %s", result)

    process0.terminate()
    process1.terminate()
    process0.join()
    process1.join()

    if result["status"] == "completed":
        r0, r1 = result["rewards"]
        print("\n" + "=" * 50)
        print("RESULT ({} hands)".format(NUM_HANDS))
        print("=" * 50)
        print("ALPHANiT (bot0):  {} chips".format(r0))
        print("PlayerAgent (bot1): {} chips".format(r1))
        if r0 > r1:
            print("Winner: ALPHANiT")
        elif r1 > r0:
            print("Winner: PlayerAgent")
        else:
            print("Tie")
        print("=" * 50)


if __name__ == "__main__":
    main()
