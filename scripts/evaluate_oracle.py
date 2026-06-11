#!/usr/bin/env python3
"""Evaluate OracleBlueAgent against CybORG CAGE Challenge 4.

The oracle agent reads ground-truth simulator state (perfect information)
to establish an empirical upper bound on blue-team performance.

Usage:
    python scripts/evaluate_oracle.py [--episodes 100] [--steps 500] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_evaluation(n_episodes: int = 100, max_steps: int = 500, seed: int = 42) -> dict:
    from CybORG import CybORG
    from CybORG.Agents.Wrappers import BlueFlatWrapperV2
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent
    from CybORG.Agents.SimpleAgents.OracleBlueAgent import make_oracle_agents

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapperV2(env=cyborg)

    obs_dict, _ = env.reset()
    agents = make_oracle_agents(env)
    agent_names = env.possible_agents

    episode_rewards: list[float] = []
    phase_rewards = {0: [], 1: [], 2: []}
    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
        for name, ag in agents.items():
            ag.reset()
            ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)

        ep_reward = 0.0
        step_rewards = []

        for step in range(max_steps):
            actions: dict[str, int] = {}
            messages: dict[str, np.ndarray] = {}
            for name, ag in agents.items():
                raw_obs = obs_dict.get(name, np.zeros(1))
                mask = env.action_mask(name)
                action_idx, msg = ag.get_action(raw_obs, np.array(mask, dtype=bool))
                actions[name] = action_idx
                messages[name] = msg

            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions, messages=messages)
            step_rew = sum(rew_dict.values())
            ep_reward += step_rew
            step_rewards.append(step_rew)

            if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in agent_names):
                break

        episode_rewards.append(ep_reward)
        # Phase decomposition
        while len(step_rewards) < 500:
            step_rewards.append(0.0)
        phase_rewards[0].append(sum(step_rewards[0:167]))
        phase_rewards[1].append(sum(step_rewards[167:334]))
        phase_rewards[2].append(sum(step_rewards[334:]))

        print(f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:9.1f}  steps={step+1}")

    elapsed = time.perf_counter() - t0
    mean_r = float(np.mean(episode_rewards))
    std_r = float(np.std(episode_rewards))

    return {
        "mean_reward": mean_r,
        "std_reward": std_r,
        "ci95": 1.96 * std_r / np.sqrt(len(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "median_reward": float(np.median(episode_rewards)),
        "q25": float(np.percentile(episode_rewards, 25)),
        "q75": float(np.percentile(episode_rewards, 75)),
        "n_episodes": n_episodes,
        "elapsed_sec": elapsed,
        "phase_means": {p: float(np.mean(v)) for p, v in phase_rewards.items()},
        "phase_stds": {p: float(np.std(v)) for p, v in phase_rewards.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OracleBlueAgent on CC4")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nOracleBlueAgent — {args.episodes} episodes x {args.steps} steps (seed={args.seed})\n")
    results = run_evaluation(args.episodes, args.steps, args.seed)

    DUMMY_BASELINE = -30578.9  # SleepAgent from optimality experiments

    improvement = results["mean_reward"] - DUMMY_BASELINE

    print("\n" + "=" * 60)
    print("  OracleBlueAgent — Evaluation Results")
    print("=" * 60)
    print(f"  Mean reward    : {results['mean_reward']:10.1f} +/- {results['std_reward']:.1f}")
    print(f"  95% CI         : [{results['mean_reward']-results['ci95']:.1f}, {results['mean_reward']+results['ci95']:.1f}]")
    print(f"  Median         : {results['median_reward']:10.1f}")
    print(f"  Min / Max      : {results['min_reward']:10.1f} / {results['max_reward']:.1f}")
    print(f"  IQR            : [{results['q25']:.1f}, {results['q75']:.1f}]")
    print(f"  vs SleepAgent  : {improvement:+10.1f}  ({improvement/abs(DUMMY_BASELINE)*100:+.1f}%)")
    print(f"\n  Per-phase means:")
    for p in range(3):
        print(f"    Phase {p}: {results['phase_means'][p]:8.1f} +/- {results['phase_stds'][p]:.1f}")
    print(f"\n  Wall time      : {results['elapsed_sec']:10.1f} s")
    print("=" * 60)

    # Comparison with heuristic
    HEURISTIC_MEAN = -1039.3
    print(f"\n  Comparison:")
    print(f"    Heuristic v9.1:  {HEURISTIC_MEAN:8.1f}")
    print(f"    Oracle:          {results['mean_reward']:8.1f}")
    print(f"    Gap:             {results['mean_reward'] - HEURISTIC_MEAN:+8.1f}")
    print(f"    Oracle captures: {(results['mean_reward'] - DUMMY_BASELINE) / (-DUMMY_BASELINE) * 100:.1f}% of SleepAgent gap")
    print(f"    Heuristic captures: {(HEURISTIC_MEAN - DUMMY_BASELINE) / (-DUMMY_BASELINE) * 100:.1f}% of SleepAgent gap")


if __name__ == "__main__":
    main()
