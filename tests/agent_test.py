"""
Basic Test Suite for PlayerAgent which checks that it never does an invalid action
"""

import importlib.util
import logging
import multiprocessing
import os
import socket
import sys
import time
from logging import getLogger
from pathlib import Path
from typing import Optional, Type

from agents.agent import Agent
from agents.test_agents import FoldAgent, CallingStationAgent, AllInAgent, RandomAgent
from match import run_api_match

NUM_HANDS = 5
TIME_PER_HAND = 25


def _repo_root() -> Path:
    """Dev layout (tests/agent_test.py) or release zip (agent_test.py next to match.py)."""
    here = Path(__file__).resolve().parent
    if (here / "match.py").is_file():
        return here
    if (here.parent / "match.py").is_file():
        return here.parent
    return here.parent


def _shutdown_process(proc: multiprocessing.Process, join_timeout: float = 10.0) -> None:
    """Terminate child agent servers so fixed ports (8000/8001) are released for the next match."""
    if proc is None:
        return
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=join_timeout)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=3.0)
    else:
        proc.join(timeout=0.2)


def _reserve_free_port() -> int:
    """
    Reserve and return an ephemeral local TCP port.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def verify_submission() -> Optional[str]:
    """
    Verify that the submission contains required files and can be imported

    Args:
        submission_dir: Path to the submission directory

    Returns:
        Optional[str]: Error message if verification fails, None if successful
    """
    if not os.path.isdir("submission"):
        return "Submission directory not found"

    if not os.path.isfile("submission/player.py"):
        return "Required file 'player.py' not found in submission directory"

    try:
        spec = importlib.util.spec_from_file_location("player", "submission/player.py")
        if spec is None or spec.loader is None:
            return "Could not load player.py"

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "PlayerAgent"):
            return "player.py does not contain PlayerAgent class"

        if not issubclass(module.PlayerAgent, Agent):
            return "PlayerAgent must inherit from Agent class"

    except Exception as e:
        return f"Error importing PlayerAgent: {str(e)}"

    return None


def get_player_agent() -> Optional[Type[Agent]]:
    """
    Import and return the PlayerAgent class

    Returns:
        Optional[Type[Agent]]: PlayerAgent class if successful, None if import fails
    """
    try:
        from submission.player import PlayerAgent

        return PlayerAgent
    except ImportError:
        return None


def run_test_match(test_agent_class: Agent, logger):
    """
    Run a match between PlayerAgent and a test agent using the API interface

    Args:
        test_agent_class (Agent): The test agent class to play against
        logger: Logger instance

    Returns:
        dict: Match results

    Raises:
        RuntimeError: If there are initialization or runtime errors
    """
    PlayerAgent = get_player_agent()
    if PlayerAgent is None:
        raise RuntimeError("Could not import PlayerAgent")

    port0 = _reserve_free_port()
    port1 = _reserve_free_port()
    while port1 == port0:
        port1 = _reserve_free_port()

    process0 = multiprocessing.Process(target=PlayerAgent.run, args=(False, port0))
    process1 = multiprocessing.Process(target=test_agent_class.run, args=(False, port1))

    try:
        process0.start()
        process1.start()

        time.sleep(2)

        result = run_api_match(
            f"http://127.0.0.1:{port0}",
            f"http://127.0.0.1:{port1}",
            logger,
            num_hands=NUM_HANDS,
            csv_path=f"outputs/match_{test_agent_class.__name__}.csv",
        )

        return result

    finally:
        _shutdown_process(process0)
        _shutdown_process(process1)
        # Brief pause so the OS releases listening ports before the next subprocess starts.
        time.sleep(0.5)


def main():
    """
    Runs a test suite of games between PlayerAgent and various test agents to verify:
    1. The submission contains required files
    2. The agent can be imported successfully
    3. The agent can be initialized and run as an API server
    4. The agent can play full games without crashing
    5. The agent responds within time limits
    
    Returns:
        dict: Test results containing:
            - verification_error: str or None
            - games_completed: int
            - runtime_errors: int
            - timeout_errors: int
            - passed: bool
    """
    root = _repo_root()
    os.chdir(root)
    (root / "outputs").mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = getLogger(__name__)
    # Keep telemetry overhead low during verification runs.
    os.environ.setdefault("OMICRON_LOG_EVENTS", "0")
    os.environ["OMICRON_LOG_NODE_UPDATES"] = "off"
    os.environ["OMICRON_LOG_DISCARD_CANDIDATES"] = "0"

    verification_error = verify_submission()
    if verification_error:
        print(f"Submission verification failed: {verification_error}")
        return {
            "verification_error": verification_error,
            "games_completed": 0,
            "runtime_errors": 0,
            "timeout_errors": 0,
            "passed": False
        }

    test_results = {"games_completed": 0, "runtime_errors": 0, "timeout_errors": 0}

    test_agents = [FoldAgent, CallingStationAgent, AllInAgent, RandomAgent]

    for test_agent_class in test_agents:
        print(f"\nTesting user bot against {test_agent_class.__name__}")
        print("-" * 50)

        start_time = time.time()

        try:
            result = run_test_match(test_agent_class, logger)

            if result["status"] == "completed":
                test_results["games_completed"] += NUM_HANDS
                print(f"[OK] Completed {NUM_HANDS} games successfully")
            elif result["status"] == "timeout":
                test_results["timeout_errors"] += 1
                print("[FAIL] Time limit exceeded")
            else:
                test_results["runtime_errors"] += 1
                print("[FAIL] Runtime error")
                print(f"  {result.get('error', 'Unknown error')}")

        except Exception as e:
            test_results["runtime_errors"] += 1
            print("[FAIL] Runtime error")
            print(f"  {str(e)}")
            continue

        end_time = time.time()
        time_per_hand = (end_time - start_time) / NUM_HANDS

        if time_per_hand > TIME_PER_HAND:
            test_results["timeout_errors"] += 1
            print(f"[FAIL] Time limit exceeded: {time_per_hand:.2f}s per hand (limit: {TIME_PER_HAND}s)")
        else:
            print(f"[OK] Time check passed: {time_per_hand:.2f}s per hand")

    print("\nTest Suite Summary")
    print("-" * 50)
    print(f"Games completed successfully: {test_results['games_completed']}")
    print(f"Runtime errors encountered: {test_results['runtime_errors']}")
    print(f"Time limit violations: {test_results['timeout_errors']}")

    test_results["verification_error"] = None
    test_results["passed"] = (
        test_results["games_completed"] > 0 and
        test_results["runtime_errors"] == 0 and
        test_results["timeout_errors"] == 0
    )
    
    return test_results


if __name__ == "__main__":
    results = main()
    if not results["passed"]:
        sys.exit(1)

