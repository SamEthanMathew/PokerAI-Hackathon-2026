#!/usr/bin/env python3
"""
CMA-ES theta optimizer for the DecisionTreeAgent.

Uses a local backtest runner (no HTTP) to evaluate candidates quickly.
Each candidate theta is tested against a pool of opponents over N hands.

Usage:
    python scripts/tune_cmaes.py --generations 50 --pop_size 12 --hands 200
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import PokerEnv
from agents.libratus.decision_tree import (
    load_theta, theta_to_vector, vector_to_theta, DISCARD_FEATURE_DIM,
)
from agents.libratus.fast_eval import load_rank7_table
from agents.libratus.features import OpponentStats
from agents.libratus.exact_equity import equity_discard, equity_postflop
from agents.libratus.decision_tree import choose_discard, choose_bet_action
from agents.libratus.features import hole2_features, board_features

FOLD, RAISE, CHECK, CALL, DISCARD = 0, 1, 2, 3, 4


# ---- Lightweight agent wrappers (no FastAPI) ----

class LocalDTAgent:
    """DecisionTreeAgent without HTTP server overhead."""

    def __init__(self, theta: dict, rank7=None):
        self.logger = logging.getLogger("dt_local")
        self._theta = theta
        self._rank7 = rank7
        self._opp_stats = OpponentStats()
        self._last_my_action = None

    def act(self, obs, reward, terminated, truncated, info):
        va = obs["valid_actions"]
        if va[DISCARD]:
            eq_fn = None
            if self._rank7 is not None:
                eq_fn = lambda my5, f3, od: equity_discard(my5, f3, od, self._rank7)
            action = choose_discard(obs, self._theta, exact_equity_fn=eq_fn, rank7_table=self._rank7)
            self._last_my_action = "DISCARD"
            return action

        hole = [c for c in obs["my_cards"] if c != -1]
        board = [c for c in obs["community_cards"] if c != -1]
        opp_disc = [c for c in obs.get("opp_discarded_cards", []) if c >= 0]

        if self._rank7 is not None and len(hole) == 2 and len(board) >= 3:
            equity = equity_postflop(tuple(hole), board, opp_disc, self._rank7)
        else:
            equity = 0.5

        bf = board_features(board)
        hf = hole2_features(hole[0], hole[1]) if len(hole) == 2 else {}
        hand_feat = {**bf, **hf, **self._opp_stats.to_features()}

        action = choose_bet_action(obs, self._theta, equity, hand_feat)
        act_names = {FOLD: "FOLD", RAISE: "RAISE", CALL: "CALL", CHECK: "CHECK"}
        self._last_my_action = act_names.get(action[0], "CHECK")
        return action

    def observe(self, obs, reward, terminated, truncated, info):
        if terminated:
            return
        opp_action = obs.get("opp_last_action", "")
        if opp_action:
            self._opp_stats.update(opp_action, self._last_my_action == "RAISE")

    def reset_stats(self):
        self._opp_stats = OpponentStats()
        self._last_my_action = None


class LocalCallingStation:
    def act(self, obs, reward, terminated, truncated, info):
        va = obs["valid_actions"]
        if va[DISCARD]:
            return (DISCARD, 0, 0, 1)
        if va[CALL]:
            return (CALL, 0, 0, 0)
        return (CHECK, 0, 0, 0)

    def observe(self, *a):
        pass

    def reset_stats(self):
        pass


class LocalAllIn:
    def act(self, obs, reward, terminated, truncated, info):
        va = obs["valid_actions"]
        if va[DISCARD]:
            return (DISCARD, 0, 0, 1)
        if va[RAISE]:
            return (RAISE, obs["max_raise"], 0, 0)
        if va[CALL]:
            return (CALL, 0, 0, 0)
        return (CHECK, 0, 0, 0)

    def observe(self, *a):
        pass

    def reset_stats(self):
        pass


class LocalRandom:
    def __init__(self):
        self.rng = np.random.RandomState(42)

    def act(self, obs, reward, terminated, truncated, info):
        va = obs["valid_actions"]
        if va[DISCARD]:
            pair = self.rng.choice(5, 2, replace=False)
            return (DISCARD, 0, int(pair[0]), int(pair[1]))
        valid_idx = [i for i in range(5) if va[i]]
        act = int(self.rng.choice(valid_idx))
        ra = 0
        if act == RAISE:
            ra = int(self.rng.randint(obs["min_raise"], obs["max_raise"] + 1))
        return (act, ra, 0, 0)

    def observe(self, *a):
        pass

    def reset_stats(self):
        pass


# ---- Local match runner ----

def run_local_match(agent0, agent1, num_hands: int) -> float:
    """Run num_hands between two agents, return EV/hand for agent0."""
    env = PokerEnv()
    total_reward = 0.0

    for hand in range(num_hands):
        (obs0, obs1), info = env.reset(options={"small_blind_player": hand % 2})
        for o in [obs0, obs1]:
            o["time_left"] = 999.0
            o["time_used"] = 0.0
            o["opp_last_action"] = "None"

        terminated = False
        reward = (0, 0)
        last_act = [None, None]
        agents = [agent0, agent1]
        steps = 0

        while not terminated and steps < 100:
            steps += 1
            cur = obs0["acting_agent"]
            obs_list = [obs0, obs1]

            try:
                action = agents[cur].act(obs_list[cur], reward[cur], False, False, info)
            except Exception:
                action = (FOLD, 0, 0, 0)

            act_names = ["FOLD", "RAISE", "CHECK", "CALL", "DISCARD"]
            last_act[cur] = act_names[action[0]] if action[0] < 5 else "FOLD"

            (obs0, obs1), reward, terminated, truncated, info = env.step(action)
            for o in [obs0, obs1]:
                o["time_left"] = 999.0
                o["time_used"] = 0.0
            obs0["opp_last_action"] = last_act[1] or "None"
            obs1["opp_last_action"] = last_act[0] or "None"

        if terminated:
            total_reward += reward[0]
            agents[0].observe(obs0, reward[0], True, False, info)
            agents[1].observe(obs1, reward[1], True, False, info)

    return total_reward / max(num_hands, 1)


# ---- Simple CMA-ES ----

class SimpleCMAES:
    """Minimal (mu, lambda)-CMA-ES implementation."""

    def __init__(self, x0: np.ndarray, sigma0: float = 0.3, pop_size: int = 12):
        self.dim = len(x0)
        self.mean = x0.copy()
        self.sigma = sigma0
        self.pop_size = pop_size
        self.mu = pop_size // 2

        self.cov = np.eye(self.dim)
        self.gen = 0

    def ask(self) -> list:
        """Sample pop_size candidates."""
        candidates = []
        L = np.linalg.cholesky(self.cov)
        for _ in range(self.pop_size):
            z = np.random.randn(self.dim)
            x = self.mean + self.sigma * L @ z
            candidates.append(x)
        return candidates

    def tell(self, candidates: list, fitnesses: list):
        """Update distribution from top-mu candidates."""
        order = np.argsort(fitnesses)[::-1]  # descending (maximize)
        selected = [candidates[i] for i in order[:self.mu]]

        old_mean = self.mean.copy()
        self.mean = np.mean(selected, axis=0)

        diffs = np.array(selected) - old_mean
        self.cov = (diffs.T @ diffs) / self.mu
        self.cov += 1e-6 * np.eye(self.dim)

        self.gen += 1


def main():
    parser = argparse.ArgumentParser(description="CMA-ES theta tuner")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--hands", type=int, default=100, help="Hands per opponent per candidate")
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "libratus", "theta_tuned.json",
        )

    rank7 = None
    try:
        rank7 = load_rank7_table()
        print("Loaded rank7 table")
    except Exception:
        print("rank7 table not found; running without exact equity")

    theta0 = load_theta()
    x0 = theta_to_vector(theta0)
    print(f"Initial theta vector dim: {len(x0)}")

    opponents = [
        ("CallingStation", LocalCallingStation()),
        ("AllIn", LocalAllIn()),
        ("Random", LocalRandom()),
    ]

    cmaes = SimpleCMAES(x0, sigma0=args.sigma, pop_size=args.pop_size)

    best_theta = theta0
    best_fitness = -float("inf")

    for gen in range(args.generations):
        t0 = time.time()
        candidates = cmaes.ask()
        fitnesses = []

        for ci, cand in enumerate(candidates):
            theta_c = vector_to_theta(cand)
            total_ev = 0.0

            for opp_name, opp in opponents:
                agent = LocalDTAgent(theta_c, rank7)
                ev = run_local_match(agent, opp, args.hands)
                total_ev += ev

            avg_ev = total_ev / len(opponents)
            fitnesses.append(avg_ev)

            if avg_ev > best_fitness:
                best_fitness = avg_ev
                best_theta = theta_c

        cmaes.tell(candidates, fitnesses)
        elapsed = time.time() - t0
        mean_fit = np.mean(fitnesses)
        max_fit = np.max(fitnesses)

        print(f"Gen {gen+1}/{args.generations}: mean_ev={mean_fit:.3f} max_ev={max_fit:.3f} "
              f"best_ever={best_fitness:.3f} elapsed={elapsed:.1f}s")

    with open(args.output, "w") as f:
        json.dump(best_theta, f, indent=2)
    print(f"\nSaved best theta to {args.output}")
    print(f"Best fitness (EV/hand avg over opponents): {best_fitness:.3f}")


if __name__ == "__main__":
    main()
