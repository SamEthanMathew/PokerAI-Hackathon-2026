import importlib
import json
import logging
import multiprocessing
import os
from pathlib import Path

from match import run_api_match

_REPO_ROOT = Path(__file__).resolve().parent

def load_agent_class(file_path):
    """
    Dynamically imports and returns an agent class from a string path.
    Example: 'agents.test_agents.AllInAgent' -> AllInAgent class
    """
    module_path, class_name = file_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def main():
    # Load configuration
    (_REPO_ROOT / "outputs").mkdir(exist_ok=True)
    cfg_path = _REPO_ROOT / "config" / "agent_config.json"
    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Load agent classes dynamically
    bot0_class = load_agent_class(config['bot0']['file_path'])
    bot1_class = load_agent_class(config['bot1']['file_path'])

    rollout = config.get("rollout", {})
    os.environ["OMICRON_LIVE_STAGE"] = str(rollout.get("live_stage", 1))
    os.environ["OMICRON_SHADOW_ONLY"] = "1" if rollout.get("shadow_only", False) else "0"
    os.environ["OMICRON_DEBUG_VERBOSE"] = "1" if rollout.get("debug_verbose", False) else "0"

    # Create processes using the configuration (stream=True so agent logs appear in console)
    # To disable agent logs, set stream=False
    process0 = multiprocessing.Process(
        target=bot0_class.run,
        args=(True, config['bot0']['port']),
        kwargs={"player_id": config['bot0']['player_id']}
    )
    process1 = multiprocessing.Process(
        target=bot1_class.run,
        args=(True, config['bot1']['port']),
        kwargs={"player_id": config['bot1']['player_id']}
    )

    process0.start()
    process1.start()

    logger.info("Starting API-based match")
    result = run_api_match(
        f"http://localhost:{config['bot0']['port']}",
        f"http://localhost:{config['bot1']['port']}",
        logger,
        csv_path=config['match_settings']['csv_output_path'],
        team_0_name=bot0_class.__name__,
        team_1_name=bot1_class.__name__,
        num_hands=int(config.get("match_settings", {}).get("num_hands", 1000)),
    )
    logger.info(f"Match result: {result}")

    # Clean up processes
    process0.terminate()
    process1.terminate()
    process0.join()
    process1.join()


if __name__ == "__main__":
    main()
