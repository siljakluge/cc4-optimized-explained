#!/usr/bin/env python3
"""Evaluate EnterpriseHeuristicAgentV11a (Preemptive OZ Blocking) against CAGE Challenge 4.

Runs N episodes and reports mean/std total reward.
Compares against v10b baseline: -771.8 (30 eps, seed 42), -814.0 (100 eps, seed 42).

Usage:
    python scripts/evaluate_v11a.py [--episodes 30] [--steps 500] [--seed 42]
    python scripts/evaluate_v11a.py --multi-seed  # runs seeds 42 and 123
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

V10B_BASELINE_30 = -771.8   # v10b mean reward (30 eps, seed 42)
V10B_BASELINE_100 = -814.0  # v10b mean reward (100 eps, seed 42)
V10B_STD_100 = 247.7


def run_evaluation(n_episodes: int = 30, max_steps: int = 500, seed: int = 42) -> dict:
    from CybORG import CybORG
    from CybORG.Agents.Wrappers import BlueFlatWrapperV2
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapperV2(env=cyborg)

    obs_dict, _ = env.reset()
    from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11a import make_heuristic_agents_v11a
    agents = make_heuristic_agents_v11a(env)
    agent_names = env.possible_agents

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
        for name, ag in agents.items():
            ag.reset()
            ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)

        ep_reward = 0.0
        step = 0

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
            ep_reward += sum(rew_dict.values())

            if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in agent_names):
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(step + 1)
        print(f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:9.1f}  steps={step+1}", flush=True)

    elapsed = time.perf_counter() - t0
    mean_r = float(np.mean(episode_rewards))
    std_r = float(np.std(episode_rewards))
    mean_len = float(np.mean(episode_lengths))
    steps_per_sec = sum(episode_lengths) / elapsed

    return {
        "mean_reward": mean_r,
        "std_reward": std_r,
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": mean_len,
        "n_episodes": n_episodes,
        "steps_per_sec": steps_per_sec,
        "elapsed_sec": elapsed,
        "seed": seed,
        "episode_rewards": episode_rewards,
    }


def print_results(results: dict, label: str = "V11a Preemptive OZ Blocking") -> None:
    delta_30 = results["mean_reward"] - V10B_BASELINE_30
    delta_100 = results["mean_reward"] - V10B_BASELINE_100
    print(f"\n{'=' * 60}")
    print(f"  {label} -- seed={results['seed']}")
    print(f"{'=' * 60}")
    print(f"  Mean reward    : {results['mean_reward']:10.1f} +/- {results['std_reward']:.1f}")
    print(f"  Min / Max      : {results['min_reward']:10.1f} / {results['max_reward']:.1f}")
    print(f"  Mean ep length : {results['mean_length']:10.1f} steps")
    print(f"  vs v10b (30ep) : {delta_30:+10.1f}  ({delta_30/abs(V10B_BASELINE_30)*100:+.1f}%)")
    print(f"  vs v10b (100ep): {delta_100:+10.1f}  ({delta_100/abs(V10B_BASELINE_100)*100:+.1f}%)")
    print(f"  v10b baselines : {V10B_BASELINE_30:.1f} (30ep), {V10B_BASELINE_100:.1f} +/- {V10B_STD_100:.1f} (100ep)")
    print(f"  Throughput     : {results['steps_per_sec']:10.1f} steps/sec")
    print(f"  Wall time      : {results['elapsed_sec']:10.1f} s")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate V11a Preemptive OZ Blocking on CC4")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run with seeds 42 and 123 for cross-validation")
    args = parser.parse_args()

    if args.multi_seed:
        seeds = [42, 123]
        all_rewards = []
        all_results = []
        for seed in seeds:
            print(f"\n--- Running {args.episodes} episodes x {args.steps} steps (seed={seed}) ---\n")
            results = run_evaluation(args.episodes, args.steps, seed)
            print_results(results, f"V11a seed={seed}")
            all_rewards.extend(results["episode_rewards"])
            all_results.append(results)

        total_eps = len(all_rewards)
        agg_mean = float(np.mean(all_rewards))
        agg_std = float(np.std(all_rewards))
        delta = agg_mean - V10B_BASELINE_100
        print(f"\n{'=' * 60}")
        print(f"  AGGREGATE -- {total_eps} episodes across seeds {seeds}")
        print(f"{'=' * 60}")
        print(f"  Mean reward    : {agg_mean:10.1f} +/- {agg_std:.1f}")
        print(f"  vs v10b (100ep): {delta:+10.1f}  ({delta/abs(V10B_BASELINE_100)*100:+.1f}%)")
        print(f"  Per-seed means : {', '.join(f'{r['mean_reward']:.1f}' for r in all_results)}")
        print(f"  Per-seed stds  : {', '.join(f'{r['std_reward']:.1f}' for r in all_results)}")
        print(f"{'=' * 60}")
    else:
        print(f"\nRunning {args.episodes} episodes x {args.steps} steps (seed={args.seed})\n")
        results = run_evaluation(args.episodes, args.steps, args.seed)
        print_results(results)


if __name__ == "__main__":
    main()
