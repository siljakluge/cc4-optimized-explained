#!/usr/bin/env python3
"""Evaluate EnterpriseHeuristicAgentV11b (Messaging Redesign) against CAGE Challenge 4.

Runs N episodes and reports mean/std total reward.
Compares against V10b baseline: -814.0 +/- 247.7 (100 eps, seed 42).

Also runs V10b side-by-side for direct comparison on the same seeds.

Usage:
    python scripts/evaluate_v11b.py [--episodes 30] [--steps 500] [--seed 42]
    python scripts/evaluate_v11b.py --seed 42 --seed2 123  # run both seeds
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

V10B_BASELINE = -814.0  # V10b mean reward (100 eps, seed 42)
V10B_STD = 247.7


def run_evaluation(
    agent_module: str,
    factory_func: str,
    n_episodes: int = 30,
    max_steps: int = 500,
    seed: int = 42,
    label: str = "agent",
) -> dict:
    import importlib
    from CybORG import CybORG
    from CybORG.Agents.Wrappers import BlueFlatWrapperV2
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent

    mod = importlib.import_module(agent_module)
    make_agents = getattr(mod, factory_func)

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapperV2(env=cyborg)

    obs_dict, _ = env.reset()
    agents = make_agents(env)
    agent_names = env.possible_agents

    episode_rewards: list[float] = []
    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
        for name, ag in agents.items():
            ag.reset()
            ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)

        ep_reward = 0.0

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
        print(f"  [{label}] ep {ep+1:3d}/{n_episodes}  reward={ep_reward:9.1f}", flush=True)

    elapsed = time.perf_counter() - t0
    mean_r = float(np.mean(episode_rewards))
    std_r = float(np.std(episode_rewards))

    return {
        "mean_reward": mean_r,
        "std_reward": std_r,
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "n_episodes": n_episodes,
        "elapsed_sec": elapsed,
        "seed": seed,
        "episode_rewards": episode_rewards,
    }


def print_comparison(v10b_results: dict, v11b_results: dict, seed: int) -> None:
    from scipy import stats

    v10b_r = v10b_results["episode_rewards"]
    v11b_r = v11b_results["episode_rewards"]

    t_stat, p_value = stats.ttest_ind(v10b_r, v11b_r)
    delta = v11b_results["mean_reward"] - v10b_results["mean_reward"]
    pooled_std = np.sqrt((np.std(v10b_r)**2 + np.std(v11b_r)**2) / 2)
    cohens_d = delta / pooled_std if pooled_std > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  COMPARISON — seed={seed}, {v10b_results['n_episodes']} episodes")
    print(f"{'=' * 70}")
    print(f"  V10b (baseline)    : {v10b_results['mean_reward']:8.1f} +/- {v10b_results['std_reward']:.1f}")
    print(f"  V11b (msg redesign): {v11b_results['mean_reward']:8.1f} +/- {v11b_results['std_reward']:.1f}")
    print(f"  Delta              : {delta:+8.1f}")
    print(f"  p-value            : {p_value:.4f}")
    print(f"  Cohen's d          : {cohens_d:+.3f}")
    print(f"  Significant?       : {'YES' if p_value < 0.05 else 'NO'} (alpha=0.05)")
    print(f"{'=' * 70}")

    # Per-episode comparison
    print(f"\n  Per-episode rewards (seed={seed}):")
    print(f"  {'Ep':>4s} {'V10b':>10s} {'V11b':>10s} {'Delta':>10s}")
    print(f"  {'----':>4s} {'----------':>10s} {'----------':>10s} {'----------':>10s}")
    for i in range(len(v10b_r)):
        d = v11b_r[i] - v10b_r[i]
        marker = " ***" if abs(d) > 100 else ""
        print(f"  {i+1:4d} {v10b_r[i]:10.1f} {v11b_r[i]:10.1f} {d:+10.1f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate V11b messaging redesign on CC4")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed2", type=int, default=None, help="Optional second seed")
    parser.add_argument("--v11b-only", action="store_true", help="Skip V10b comparison")
    args = parser.parse_args()

    seeds = [args.seed]
    if args.seed2 is not None:
        seeds.append(args.seed2)

    all_v10b_rewards = []
    all_v11b_rewards = []

    for seed in seeds:
        print(f"\n{'#' * 70}")
        print(f"  SEED {seed} — {args.episodes} episodes x {args.steps} steps")
        print(f"{'#' * 70}")

        # Run V10b baseline
        if not args.v11b_only:
            print(f"\n--- V10b (baseline) ---")
            v10b = run_evaluation(
                "CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV10b",
                "make_heuristic_agents_v10b",
                args.episodes, args.steps, seed, "V10b",
            )
            all_v10b_rewards.extend(v10b["episode_rewards"])

        # Run V11b
        print(f"\n--- V11b (messaging redesign) ---")
        v11b = run_evaluation(
            "CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11b",
            "make_heuristic_agents_v11b",
            args.episodes, args.steps, seed, "V11b",
        )
        all_v11b_rewards.extend(v11b["episode_rewards"])

        if not args.v11b_only:
            print_comparison(v10b, v11b, seed)

    # Aggregate across seeds
    if len(seeds) > 1 and not args.v11b_only:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(all_v10b_rewards, all_v11b_rewards)
        delta = np.mean(all_v11b_rewards) - np.mean(all_v10b_rewards)
        print(f"\n{'=' * 70}")
        print(f"  AGGREGATE — {len(all_v10b_rewards)} episodes across seeds {seeds}")
        print(f"{'=' * 70}")
        print(f"  V10b: {np.mean(all_v10b_rewards):.1f} +/- {np.std(all_v10b_rewards):.1f}")
        print(f"  V11b: {np.mean(all_v11b_rewards):.1f} +/- {np.std(all_v11b_rewards):.1f}")
        print(f"  Delta: {delta:+.1f}, p={p_value:.4f}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
