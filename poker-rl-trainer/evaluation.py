"""
Evaluation module: run 1000-hand matches and report win rates.

Prints a formatted table like the spec and saves results to logs/eval_results.csv.
"""

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class EvalResult:
    cycle: int
    win_rates: Dict[str, float] = field(default_factory=dict)
    net_chips: Dict[str, float] = field(default_factory=dict)
    showdown_pcts: Dict[str, float] = field(default_factory=dict)
    aggregate_win_rate: float = 0.0
    aggregate_net_chips: float = 0.0
    avg_inference_ms: float = 0.0
    kl_divergence: float = 0.0
    policy_entropy: float = 0.0
    timestamp: str = ""


def run_evaluation(
    model,
    opponent_pool,
    tables: dict,
    genesis_knowledge: dict,
    cycle: int,
    n_hands: int = 1000,
    device: str = "cpu",
    bc_reference=None,
    logs_dir: str = os.path.join(_HERE, "logs"),
) -> EvalResult:
    """
    Run a 1000-hand evaluation match against every opponent in the pool.

    Args:
        model:             PokerNetV2 instance (will be moved to device for eval)
        opponent_pool:     OpponentPool instance
        tables:            Precomputed tables dict
        genesis_knowledge: Loaded genesis_knowledge.json dict
        cycle:             Current training cycle number (for logging)
        n_hands:           Hands per opponent (default 1000)
        device:            "cpu" for latency measurement
        bc_reference:      Optional frozen BC model for KL divergence measurement
        logs_dir:          Directory for eval_results.csv

    Returns:
        EvalResult
    """
    import json
    from datetime import datetime

    from env.poker_env import PokerTrainingEnv
    from features import extract_features, split_features, update_opp_stats

    model.eval()
    model_cpu = model.to("cpu")

    result = EvalResult(
        cycle=cycle,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )

    inference_times: List[float] = []
    all_win_rates: List[float] = []

    print()
    print(f"┌{'─'*55}┐")
    print(f"│ EVALUATION — Training Cycle {cycle:<26}│")
    print(f"│ Timestamp: {result.timestamp:<43}│")
    print(f"├{'─'*55}┤")
    print(f"│ {'Opponent':<18}│ {'Win Rate':>8} │ {'Net Chips':>9} │ {'Showdown%':>9} │")
    print(f"├{'─'*55}┤")

    for opp in opponent_pool.opponents:
        opp_name = opp.name
        opp_stats: dict = {}

        wins = 0
        net = 0.0
        showdowns = 0

        env = PokerTrainingEnv()

        for hand_num in range(n_hands):
            obs_p0, obs_p1 = env.reset()

            # Alternate who is SB — our bot is always "player 0" perspective
            hand_done = False
            hand_reward = 0.0
            went_to_showdown = False

            while not hand_done:
                acting = env.acting_player

                if acting == 0:
                    # Our bot acts
                    feat = extract_features(obs_p0, opp_stats, genesis_knowledge, tables)
                    cf, ctx = split_features(feat)
                    cf_t = torch.from_numpy(cf).unsqueeze(0)
                    ctx_t = torch.from_numpy(ctx).unsqueeze(0)

                    t0 = time.perf_counter()
                    with torch.no_grad():
                        action = model_cpu.predict_action(
                            cf_t, ctx_t,
                            list(obs_p0["valid_actions"]),
                            obs_p0["min_raise"],
                            obs_p0["max_raise"],
                        )
                    inference_times.append((time.perf_counter() - t0) * 1000)

                    obs_p0, obs_p1, r0, r1, hand_done, info = env.step(action)
                    # Let opponent observe our action
                    opp.observe(obs_p1, r1, hand_done, False, {"hand_number": hand_num})
                    update_opp_stats(opp_stats, obs_p0, action)

                else:
                    # Opponent acts
                    action = opp.act(obs_p1, 0.0, False, False, {"hand_number": hand_num})
                    obs_p0, obs_p1, r0, r1, hand_done, info = env.step(action)
                    # Let opponent observe result
                    opp.observe(obs_p1, r1, hand_done, False, {"hand_number": hand_num})
                    update_opp_stats(opp_stats, obs_p0, action)

                if hand_done:
                    hand_reward = r0
                    went_to_showdown = info.get("player_0_cards") is not None

            net += hand_reward
            if hand_reward > 0:
                wins += 1
            if went_to_showdown:
                showdowns += 1

        win_rate = wins / n_hands
        showdown_pct = showdowns / n_hands

        result.win_rates[opp_name] = win_rate
        result.net_chips[opp_name] = net
        result.showdown_pcts[opp_name] = showdown_pct
        all_win_rates.append(win_rate)

        print(
            f"│ {opp_name:<18}│ {win_rate:>7.1%} │ {net:>+9.0f} │ {showdown_pct:>8.1%} │"
        )

    result.aggregate_win_rate = float(np.mean(all_win_rates))
    result.aggregate_net_chips = float(np.mean(list(result.net_chips.values())))
    result.avg_inference_ms = float(np.median(inference_times)) if inference_times else 0.0

    # KL divergence from BC reference
    if bc_reference is not None:
        result.kl_divergence = _measure_kl(model_cpu, bc_reference, tables, genesis_knowledge)

    # Policy entropy
    result.policy_entropy = _measure_entropy(model_cpu)

    # Best so far (read from log)
    best_so_far = _read_best_win_rate(logs_dir)

    print(f"├{'─'*55}┤")
    print(f"│ {'AGGREGATE':<18}│ {result.aggregate_win_rate:>7.1%} │"
          f" {result.aggregate_net_chips:>+9.0f} │ {'':<8} │")
    print(f"│ Inference: {result.avg_inference_ms:.1f}ms  "
          f"│ KL: {result.kl_divergence:.3f}  "
          f"│ Entropy: {result.policy_entropy:.3f}{'':>5}│")
    if best_so_far is not None:
        print(f"│ Best so far: {best_so_far:.1%} (from log){'':>22}│")
    print(f"└{'─'*55}┘")
    print()

    _save_to_csv(result, logs_dir)

    return result


def _measure_kl(model, bc_reference, tables, genesis_knowledge) -> float:
    """Approximate KL divergence from BC reference on a small random batch."""
    try:
        import torch.nn.functional as F
        batch_size = 64
        cf = torch.randn(batch_size, 214)
        ctx = torch.randn(batch_size, 63)

        with torch.no_grad():
            logits_curr, _, _, _ = model(cf, ctx)
            logits_ref, _, _, _ = bc_reference(cf, ctx)

        p = F.softmax(logits_curr, dim=-1)
        q = F.softmax(logits_ref, dim=-1)
        kl = (p * (p.log() - q.log())).sum(dim=-1).mean().item()
        return float(kl)
    except Exception:
        return 0.0


def _measure_entropy(model) -> float:
    """Approximate policy entropy on a small random batch."""
    try:
        import torch.nn.functional as F
        batch_size = 64
        cf = torch.randn(batch_size, 214)
        ctx = torch.randn(batch_size, 63)
        with torch.no_grad():
            logits, _, _, _ = model(cf, ctx)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
        return float(entropy)
    except Exception:
        return 0.0


def _read_best_win_rate(logs_dir: str) -> Optional[float]:
    """Read the highest aggregate win rate seen so far from eval_results.csv."""
    csv_path = os.path.join(logs_dir, "eval_results.csv")
    if not os.path.exists(csv_path):
        return None
    best = None
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = float(row.get("aggregate_win_rate", 0))
                if best is None or val > best:
                    best = val
    except Exception:
        pass
    return best


def _save_to_csv(result: EvalResult, logs_dir: str):
    """Append evaluation result to logs/eval_results.csv."""
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, "eval_results.csv")

    fieldnames = [
        "cycle", "timestamp", "aggregate_win_rate", "aggregate_net_chips",
        "avg_inference_ms", "kl_divergence", "policy_entropy",
    ]
    # Add per-opponent columns
    for name in result.win_rates:
        fieldnames += [f"wr_{name}", f"chips_{name}", f"sd_{name}"]

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row: dict = {
            "cycle": result.cycle,
            "timestamp": result.timestamp,
            "aggregate_win_rate": f"{result.aggregate_win_rate:.4f}",
            "aggregate_net_chips": f"{result.aggregate_net_chips:.1f}",
            "avg_inference_ms": f"{result.avg_inference_ms:.3f}",
            "kl_divergence": f"{result.kl_divergence:.4f}",
            "policy_entropy": f"{result.policy_entropy:.4f}",
        }
        for name in result.win_rates:
            row[f"wr_{name}"]    = f"{result.win_rates[name]:.4f}"
            row[f"chips_{name}"] = f"{result.net_chips[name]:.1f}"
            row[f"sd_{name}"]    = f"{result.showdown_pcts.get(name, 0):.4f}"
        writer.writerow(row)
