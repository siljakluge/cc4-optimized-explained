#!/usr/bin/env python3
"""Evaluate OracleBlueAgentV3 against CybORG CAGE Challenge 4.

Tests Oracle V3 and compares against all baselines.

Usage:
    python scripts/evaluate_oracle_v3.py [--episodes 100] [--steps 500] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

BASELINES = {
    "SleepAgent":      -30578.9,
    "Heuristic v9.1":   -1039.3,
    "Oracle V1":        -1558.0,
    "Oracle V2":        -1938.0,
}


def run_evaluation(
    n_episodes: int = 100,
    max_steps: int = 500,
    seed: int = 42,
) -> dict:
    from CybORG import CybORG
    from CybORG.Agents.Wrappers import BlueFlatWrapperV2
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent
    from CybORG.Agents.SimpleAgents.OracleBlueAgentV3 import make_oracle_v3_agents

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapperV2(env=cyborg)

    obs_dict, _ = env.reset()
    agents = make_oracle_v3_agents(env)
    agent_names = env.possible_agents

    episode_rewards: list[float] = []
    phase_rewards = {0: [], 1: [], 2: []}

    action_stats = {
        "restores": [], "removes": [], "blocks": [],
        "allows": [], "sleeps": [],
    }
    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
        for name, ag in agents.items():
            ag.reset()
            ag.set_action_info(
                env.action_labels(name), env.action_mask(name), subnet_hosts
            )

        ep_reward = 0.0
        step_rewards = []
        ep_counts = {k: 0 for k in action_stats}

        for step in range(max_steps):
            actions: dict[str, int] = {}
            messages: dict[str, np.ndarray] = {}
            for name, ag in agents.items():
                raw_obs = obs_dict.get(name, np.zeros(1))
                amask = env.action_mask(name)
                action_idx, msg = ag.get_action(
                    raw_obs, np.array(amask, dtype=bool)
                )
                actions[name] = action_idx
                messages[name] = msg

                labels = env.action_labels(name)
                lbl = labels[action_idx] if action_idx < len(labels) else "Sleep"
                if lbl.startswith("Restore"):
                    ep_counts["restores"] += 1
                elif lbl.startswith("Remove"):
                    ep_counts["removes"] += 1
                elif lbl.startswith("Block"):
                    ep_counts["blocks"] += 1
                elif lbl.startswith("Allow"):
                    ep_counts["allows"] += 1
                elif lbl == "Sleep":
                    ep_counts["sleeps"] += 1

            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(
                actions, messages=messages
            )
            step_rew = sum(rew_dict.values())
            ep_reward += step_rew
            step_rewards.append(step_rew)

            if all(
                term_dict.get(n, False) or trunc_dict.get(n, False)
                for n in agent_names
            ):
                break

        episode_rewards.append(ep_reward)
        for k in action_stats:
            action_stats[k].append(ep_counts[k])

        while len(step_rewards) < 500:
            step_rewards.append(0.0)
        phase_rewards[0].append(sum(step_rewards[0:167]))
        phase_rewards[1].append(sum(step_rewards[167:334]))
        phase_rewards[2].append(sum(step_rewards[334:]))

        print(
            f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:9.1f}  "
            f"restores={ep_counts['restores']:3d}  "
            f"removes={ep_counts['removes']:3d}  "
            f"blocks={ep_counts['blocks']:2d}  "
            f"steps={step+1}"
        )

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
        "phase_means": {
            p: float(np.mean(v)) for p, v in phase_rewards.items()
        },
        "phase_stds": {
            p: float(np.std(v)) for p, v in phase_rewards.items()
        },
        "action_means": {
            k: float(np.mean(v)) for k, v in action_stats.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OracleBlueAgentV3 on CC4"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(
        f"\nOracleBlueAgentV3 -- {args.episodes} episodes x {args.steps} "
        f"steps (seed={args.seed})\n"
    )
    results = run_evaluation(args.episodes, args.steps, args.seed)

    print(f"\n{'='*65}")
    print("  OracleBlueAgentV3 -- Evaluation Results")
    print(f"{'='*65}")
    print(
        f"  Mean reward    : {results['mean_reward']:10.1f} "
        f"+/- {results['std_reward']:.1f}"
    )
    print(
        f"  95% CI         : [{results['mean_reward']-results['ci95']:.1f}, "
        f"{results['mean_reward']+results['ci95']:.1f}]"
    )
    print(f"  Median         : {results['median_reward']:10.1f}")
    print(
        f"  Min / Max      : {results['min_reward']:10.1f} / "
        f"{results['max_reward']:.1f}"
    )
    print(f"  IQR            : [{results['q25']:.1f}, {results['q75']:.1f}]")

    print(f"\n  Per-phase means:")
    for p in range(3):
        print(
            f"    Phase {p}: {results['phase_means'][p]:8.1f} "
            f"+/- {results['phase_stds'][p]:.1f}"
        )

    print(f"\n  Avg actions/ep:")
    for k, v in results["action_means"].items():
        print(f"    {k:12s}: {v:6.1f}")

    print(f"\n  Wall time      : {results['elapsed_sec']:10.1f} s")

    # Comparison table
    print(f"\n{'='*65}")
    print("  Comparison Table")
    print(f"{'='*65}")
    print(f"  {'Agent':<20s} {'Mean':>10s} {'vs Sleep':>12s} {'Capture%':>10s}")
    print(f"  {'-'*52}")

    sleep_r = BASELINES["SleepAgent"]
    total_gap = abs(sleep_r)

    all_agents = {**BASELINES, "Oracle V3": results["mean_reward"]}
    for name, mean_r in sorted(all_agents.items(), key=lambda x: x[1]):
        improvement = mean_r - sleep_r
        capture_pct = improvement / total_gap * 100
        print(
            f"  {name:<20s} {mean_r:10.1f} {improvement:+12.1f} "
            f"{capture_pct:9.1f}%"
        )

    v3_r = results["mean_reward"]
    heur_r = BASELINES["Heuristic v9.1"]
    v1_r = BASELINES["Oracle V1"]
    v2_r = BASELINES["Oracle V2"]
    print(f"\n  Oracle V3 vs Heuristic: {v3_r - heur_r:+.1f}")
    print(f"  Oracle V3 vs Oracle V1: {v3_r - v1_r:+.1f}")
    print(f"  Oracle V3 vs Oracle V2: {v3_r - v2_r:+.1f}")

    if v3_r > heur_r:
        print(
            f"\n  SUCCESS: Oracle V3 ({v3_r:.1f}) beats Heuristic ({heur_r:.1f})"
        )
    else:
        print(
            f"\n  Oracle V3 ({v3_r:.1f}) vs Heuristic ({heur_r:.1f}): "
            f"delta = {v3_r - heur_r:+.1f}"
        )

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
