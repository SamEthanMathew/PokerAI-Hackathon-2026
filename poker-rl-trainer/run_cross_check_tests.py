#!/usr/bin/env python3
"""
Cross-check tests for poker-rl-trainer and bot. Run from repo root:
    python poker-rl-trainer/run_cross_check_tests.py

Or from poker-rl-trainer:
    python run_cross_check_tests.py
"""
import os
import sys

# Same path setup as train.py
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def test_imports():
    """Test 1: All train.py imports resolve."""
    from config import TrainingConfig
    from model import PokerNetV2, KEEP_COMBOS
    from features import extract_features, split_features, CARD_DIM, CONTEXT_DIM
    from evaluation import run_evaluation, EvalResult
    from opponent_pool import load_opponent_pool, OpponentPool
    from precompute import load_tables
    from env.poker_env import PokerTrainingEnv
    cfg = TrainingConfig()
    assert cfg.eval_every > 0 and cfg.total_cycles > 0
    return "imports OK"


def test_log_helper():
    """Test 2: _log_rl_genesis_eval exists and runs (from train module)."""
    import train
    result = train.EvalResult(
        cycle=99,
        aggregate_win_rate=0.55,
        aggregate_net_chips=10.0,
        timestamp="2026-01-01T00:00:00",
        win_rates={"genesis": 0.5, "blambot": 0.6},
        net_chips={"genesis": 5.0, "blambot": 10.0},
    )
    train._log_rl_genesis_eval(result, train.cfg.logs_dir)
    return "log_rl_genesis_eval OK"


def test_env_step():
    """Test 3: PokerTrainingEnv reset and one step."""
    from env.poker_env import PokerTrainingEnv
    env = PokerTrainingEnv()
    obs_p0, obs_p1 = env.reset()
    assert obs_p0 is not None and obs_p1 is not None
    assert "my_cards" in obs_p0 and "valid_actions" in obs_p0
    return "env OK"


def test_bot_import():
    """Test 4: Bot module loads (no HTTP server)."""
    bot_dir = os.path.join(_HERE, "bot")
    if bot_dir not in sys.path:
        sys.path.insert(0, bot_dir)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    # Bot dir on path: import model, features as in bot/bot.py
    import model as bot_model
    import features as bot_features
    assert hasattr(bot_model, "PokerNetV2") and hasattr(bot_features, "extract_features")
    return "bot imports OK"


def main():
    results = []
    for name, fn in [
        ("imports", test_imports),
        ("log_helper", test_log_helper),
        ("env_step", test_env_step),
        ("bot_import", test_bot_import),
    ]:
        try:
            msg = fn()
            results.append((name, True, msg))
            print(f"  [PASS] {name}: {msg}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    failed = [r for r in results if not r[1]]
    print()
    if failed:
        print(f"Cross-check failed: {len(failed)} test(s)")
        sys.exit(1)
    print("Cross-check passed: all tests OK")
    sys.exit(0)


if __name__ == "__main__":
    print("Running poker-rl-trainer cross-check tests...")
    main()
