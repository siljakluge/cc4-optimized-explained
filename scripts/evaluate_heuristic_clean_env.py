#!/usr/bin/env python3
"""Evaluate EnterpriseHeuristicAgent in a clean environment.

Phishing-email spread and green-agent false-positive detections are both
disabled (0%) so results isolate the heuristic's response to *genuine* red
activity rather than green-agent noise.

Usage:
    python scripts/evaluate_heuristic_clean_env.py [--episodes 20] [--steps 500] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_evaluation(n_episodes: int = 20, max_steps: int = 500, seed: int = 42) -> dict:
    from CybORG import CybORG
    from CybORG.Agents.Wrappers import BlueFlatWrapperV2
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import ZeroNoiseGreenAgent

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=ZeroNoiseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapperV2(env=cyborg)

    obs_dict, _ = env.reset()
    from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import make_heuristic_agents
    agents = make_heuristic_agents(env)
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
        print(f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:9.1f}  steps={step+1}")

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
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EnterpriseHeuristicAgent — clean env (phishing=0%, FP=0%)"
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes (default 20)")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode (default 500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nClean-env evaluation: phishing_error_rate=0%, fp_detection_rate=0%")
    print(f"Running {args.episodes} episodes × {args.steps} steps  (seed={args.seed})\n")
    results = run_evaluation(args.episodes, args.steps, args.seed)

    # Dummy baseline from prior testing: -18,386 per episode
    DUMMY_BASELINE = -18386.0
    improvement = results["mean_reward"] - DUMMY_BASELINE

    print("\n" + "=" * 55)
    print("  EnterpriseHeuristicAgent — Clean Env Evaluation Results")
    print("  (phishing=0%, false positives=0%)")
    print("=" * 55)
    print(f"  Mean reward    : {results['mean_reward']:10.1f} ± {results['std_reward']:.1f}")
    print(f"  Min / Max      : {results['min_reward']:10.1f} / {results['max_reward']:.1f}")
    print(f"  Mean ep length : {results['mean_length']:10.1f} steps")
    print(f"  vs dummy base  : {improvement:+10.1f}  ({improvement/abs(DUMMY_BASELINE)*100:+.1f}%)")
    print(f"  Throughput     : {results['steps_per_sec']:10.1f} steps/sec")
    print(f"  Wall time      : {results['elapsed_sec']:10.1f} s")
    print("=" * 55)


if __name__ == "__main__":
    main()
