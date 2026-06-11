#!/usr/bin/env python3
"""Scientific optimality analysis of EnterpriseHeuristicAgent v9.1.

Runs multiple experiment suites to establish:
1. Baseline performance with confidence intervals
2. Theoretical bounds (oracle agent, do-nothing agent)
3. Ablation studies (disable each feature, measure impact)
4. Per-phase reward decomposition
5. Action budget analysis
6. Detection timing analysis
7. Decoy effectiveness measurement

Usage:
    python scripts/evaluate_optimality.py [--seed 42] [--output docs/optimality_analysis_data.txt]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_env(seed, max_steps=500):
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
    return env, cyborg


def _make_agents(env):
    from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import make_heuristic_agents
    return make_heuristic_agents(env)


def _reset_agents(env, agents):
    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
    for name, ag in agents.items():
        ag.reset()
        ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)


def _run_episode(env, agents, max_steps=500, collect_actions=False):
    """Run one episode, return (total_reward, per_step_rewards, action_log)."""
    obs_dict, _ = env.reset()
    _reset_agents(env, agents)

    ep_reward = 0.0
    step_rewards = []
    action_log = []

    for step in range(max_steps):
        actions = {}
        messages = {}
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

        if collect_actions:
            action_labels = {}
            for name, ag in agents.items():
                try:
                    action_labels[name] = ag._labels[actions[name]]
                except (IndexError, AttributeError):
                    action_labels[name] = str(actions[name])
            action_log.append(action_labels)

        if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in env.possible_agents):
            break

    return ep_reward, step_rewards, action_log


# =============================================================================
# Experiment 1: Baseline with confidence intervals
# =============================================================================
def experiment_baseline(seed=42, n_episodes=100):
    """Run full baseline evaluation with statistical analysis."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Baseline Performance (100 episodes)")
    print("=" * 70)

    env, cyborg = _make_env(seed)
    agents = _make_agents(env)

    rewards = []
    phase_rewards = {0: [], 1: [], 2: []}  # per-phase reward accumulation

    for ep in range(n_episodes):
        ep_reward, step_rewards, _ = _run_episode(env, agents)
        rewards.append(ep_reward)

        # Phase decomposition (167/167/166 steps)
        phase_rewards[0].append(sum(step_rewards[0:167]))
        phase_rewards[1].append(sum(step_rewards[167:334]))
        phase_rewards[2].append(sum(step_rewards[334:]))

        if (ep + 1) % 25 == 0:
            print(f"  ep {ep+1:3d}/{n_episodes}  running_mean={np.mean(rewards):8.1f}")

    rewards = np.array(rewards)
    ci95 = 1.96 * np.std(rewards) / np.sqrt(len(rewards))

    results = {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "ci95": float(ci95),
        "median": float(np.median(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "q25": float(np.percentile(rewards, 25)),
        "q75": float(np.percentile(rewards, 75)),
        "phase_means": {p: float(np.mean(v)) for p, v in phase_rewards.items()},
        "phase_stds": {p: float(np.std(v)) for p, v in phase_rewards.items()},
    }

    print(f"\n  Mean:   {results['mean']:8.1f} +/- {results['std']:.1f}")
    print(f"  95% CI: [{results['mean']-ci95:.1f}, {results['mean']+ci95:.1f}]")
    print(f"  Median: {results['median']:8.1f}")
    print(f"  Range:  [{results['min']:.1f}, {results['max']:.1f}]")
    print(f"  IQR:    [{results['q25']:.1f}, {results['q75']:.1f}]")
    print(f"\n  Per-phase means:")
    for p in range(3):
        print(f"    Phase {p}: {results['phase_means'][p]:8.1f} +/- {results['phase_stds'][p]:.1f}")

    return results


# =============================================================================
# Experiment 2: SleepAgent baseline (theoretical worst-case blue)
# =============================================================================
def experiment_sleep_baseline(seed=42, n_episodes=50):
    """Run with SleepAgent (blue does nothing) for lower bound."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: SleepAgent Baseline (50 episodes)")
    print("=" * 70)

    from CybORG import CybORG
    from CybORG.Agents import SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapperV2
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent

    sg = EnterpriseScenarioGenerator(
        steps=500,
        blue_agent_class=SleepAgent,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapperV2(env=cyborg)

    rewards = []
    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        ep_reward = 0.0
        for step in range(500):
            actions = {name: 0 for name in env.possible_agents}  # Sleep
            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions)
            ep_reward += sum(rew_dict.values())
            if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in env.possible_agents):
                break
        rewards.append(ep_reward)
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d}/{n_episodes}  running_mean={np.mean(rewards):8.1f}")

    rewards = np.array(rewards)
    results = {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
    }
    print(f"\n  SleepAgent mean: {results['mean']:8.1f} +/- {results['std']:.1f}")
    return results


# =============================================================================
# Experiment 3: Ablation studies
# =============================================================================
def experiment_ablations(seed=42, n_episodes=30):
    """Systematically disable features and measure impact."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Ablation Studies (30 episodes each)")
    print("=" * 70)

    from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import EnterpriseHeuristicAgent

    ablation_results = {}

    # --- Ablation A: No decoys (skip P6, P7) ---
    print("\n  [A] No decoys...")
    env, _ = _make_env(seed)
    agents = _make_agents(env)
    def disable_decoys(ags):
        for ag in ags.values():
            ag._decoy.clear()
            ag._deploy_hosts = []
    rewards_a = _run_ablation_episodes(env, agents, n_episodes, modify_fn=disable_decoys)
    ablation_results["no_decoys"] = {
        "mean": float(np.mean(rewards_a)), "std": float(np.std(rewards_a))
    }
    print(f"    Mean: {ablation_results['no_decoys']['mean']:8.1f} +/- {ablation_results['no_decoys']['std']:.1f}")

    # --- Ablation B: No messaging (zero out all messages) ---
    print("\n  [B] No messaging...")
    env, _ = _make_env(seed)
    agents = _make_agents(env)
    rewards_b = []
    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        _reset_agents(env, agents)
        ep_reward = 0.0
        for step in range(500):
            actions = {}
            for name, ag in agents.items():
                raw_obs = obs_dict.get(name, np.zeros(1))
                mask = env.action_mask(name)
                action_idx, msg = ag.get_action(raw_obs, np.array(mask, dtype=bool))
                actions[name] = action_idx
            # No messages passed
            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions)
            ep_reward += sum(rew_dict.values())
            if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in env.possible_agents):
                break
        rewards_b.append(ep_reward)
    ablation_results["no_messaging"] = {
        "mean": float(np.mean(rewards_b)), "std": float(np.std(rewards_b))
    }
    print(f"    Mean: {ablation_results['no_messaging']['mean']:8.1f} +/- {ablation_results['no_messaging']['std']:.1f}")

    # --- Ablation C: No blocking (skip P2, P3) ---
    print("\n  [C] No blocking...")
    env, _ = _make_env(seed)
    agents = _make_agents(env)
    def disable_blocking(ags):
        for ag in ags.values():
            ag._block.clear()
            ag._allow.clear()
    rewards_c = _run_ablation_episodes(env, agents, n_episodes, modify_fn=disable_blocking)
    ablation_results["no_blocking"] = {
        "mean": float(np.mean(rewards_c)), "std": float(np.std(rewards_c))
    }
    print(f"    Mean: {ablation_results['no_blocking']['mean']:8.1f} +/- {ablation_results['no_blocking']['std']:.1f}")

    # --- Ablation D: No malfile detection (use BlueFlatWrapper) ---
    print("\n  [D] No malfile detection (plain BlueFlatWrapper)...")
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG import CybORG
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent
    sg = EnterpriseScenarioGenerator(
        steps=500, red_agent_class=FiniteStateRedAgent, green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env_v1 = BlueFlatWrapper(env=cyborg)
    agents_v1 = _make_agents(env_v1)
    rewards_d = _run_ablation_episodes(env_v1, agents_v1, n_episodes)
    ablation_results["no_malfile"] = {
        "mean": float(np.mean(rewards_d)), "std": float(np.std(rewards_d))
    }
    print(f"    Mean: {ablation_results['no_malfile']['mean']:8.1f} +/- {ablation_results['no_malfile']['std']:.1f}")

    # --- Ablation E: Remove-only (no Restore) ---
    print("\n  [E] Remove-only (no Restore)...")
    env, _ = _make_env(seed)
    agents = _make_agents(env)
    def disable_restore(ags):
        for ag in ags.values():
            ag._restore.clear()
    rewards_e = _run_ablation_episodes(env, agents, n_episodes, modify_fn=disable_restore)
    ablation_results["no_restore"] = {
        "mean": float(np.mean(rewards_e)), "std": float(np.std(rewards_e))
    }
    print(f"    Mean: {ablation_results['no_restore']['mean']:8.1f} +/- {ablation_results['no_restore']['std']:.1f}")

    # --- Ablation F: Restore-only (no Remove) ---
    print("\n  [F] Restore-only (no Remove, always Restore)...")
    env, _ = _make_env(seed)
    agents = _make_agents(env)
    def disable_remove(ags):
        for ag in ags.values():
            ag._remove.clear()
    rewards_f = _run_ablation_episodes(env, agents, n_episodes, modify_fn=disable_remove)
    ablation_results["no_remove"] = {
        "mean": float(np.mean(rewards_f)), "std": float(np.std(rewards_f))
    }
    print(f"    Mean: {ablation_results['no_remove']['mean']:8.1f} +/- {ablation_results['no_remove']['std']:.1f}")

    return ablation_results


def _run_ablation_episodes(env, agents, n_episodes, max_steps=500, modify_fn=None):
    """Run n episodes with an optional per-episode agent modification function.

    modify_fn(agents) is called after reset+_reset_agents each episode,
    allowing ablation modifications before the episode runs.
    """
    rewards = []
    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        _reset_agents(env, agents)
        if modify_fn:
            modify_fn(agents)

        ep_reward = 0.0
        for step in range(max_steps):
            actions = {}
            messages = {}
            for name, ag in agents.items():
                raw_obs = obs_dict.get(name, np.zeros(1))
                mask = env.action_mask(name)
                action_idx, msg = ag.get_action(raw_obs, np.array(mask, dtype=bool))
                actions[name] = action_idx
                messages[name] = msg
            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions, messages=messages)
            ep_reward += sum(rew_dict.values())
            if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in env.possible_agents):
                break
        rewards.append(ep_reward)
    return rewards


# =============================================================================
# Experiment 4: Action budget analysis
# =============================================================================
def experiment_action_budget(seed=42, n_episodes=30):
    """Track what proportion of actions go to each category."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Action Budget Analysis (30 episodes)")
    print("=" * 70)

    env, _ = _make_env(seed)
    agents = _make_agents(env)

    action_counts = Counter()
    total_actions = 0

    for ep in range(n_episodes):
        _, _, action_log = _run_episode(env, agents, collect_actions=True)
        for step_actions in action_log:
            for name, label in step_actions.items():
                total_actions += 1
                if "Sleep" in label:
                    action_counts["Sleep"] += 1
                elif "Restore" in label:
                    action_counts["Restore"] += 1
                elif "Remove" in label:
                    action_counts["Remove"] += 1
                elif "BlockTraffic" in label:
                    action_counts["Block"] += 1
                elif "AllowTraffic" in label:
                    action_counts["Allow"] += 1
                elif "DeployDecoy" in label:
                    action_counts["DeployDecoy"] += 1
                else:
                    action_counts["Other"] += 1

    results = {
        "total_actions": total_actions,
        "counts": dict(action_counts),
        "percentages": {k: round(100 * v / total_actions, 1) for k, v in action_counts.items()},
    }

    print(f"\n  Total actions: {total_actions}")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_actions
        bar = "#" * int(pct / 2)
        print(f"    {action:15s} {count:6d} ({pct:5.1f}%) {bar}")

    return results


# =============================================================================
# Experiment 5: Per-step reward curves
# =============================================================================
def experiment_reward_curves(seed=42, n_episodes=30):
    """Collect per-step reward averaged across episodes."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: Per-Step Reward Curves (30 episodes)")
    print("=" * 70)

    env, _ = _make_env(seed)
    agents = _make_agents(env)

    all_step_rewards = []
    for ep in range(n_episodes):
        _, step_rewards, _ = _run_episode(env, agents)
        # Pad to 500 if needed
        while len(step_rewards) < 500:
            step_rewards.append(0.0)
        all_step_rewards.append(step_rewards[:500])

    arr = np.array(all_step_rewards)  # (n_episodes, 500)
    mean_curve = np.mean(arr, axis=0)
    std_curve = np.std(arr, axis=0)

    # Compute cumulative
    cum_mean = np.cumsum(mean_curve)

    # Phase boundaries
    p0_mean = float(np.mean(mean_curve[0:167]))
    p1_mean = float(np.mean(mean_curve[167:334]))
    p2_mean = float(np.mean(mean_curve[334:]))

    results = {
        "per_step_mean_by_phase": {"phase0": p0_mean, "phase1": p1_mean, "phase2": p2_mean},
        "worst_10_steps": sorted(
            [(int(i), float(mean_curve[i])) for i in range(500)],
            key=lambda x: x[1]
        )[:10],
        "cumulative_at_phase_boundaries": {
            "step_167": float(cum_mean[166]),
            "step_334": float(cum_mean[333]),
            "step_500": float(cum_mean[499]),
        },
    }

    print(f"\n  Mean per-step reward by phase:")
    print(f"    Phase 0 (steps 1-167):   {p0_mean:.3f}")
    print(f"    Phase 1 (steps 168-334): {p1_mean:.3f}")
    print(f"    Phase 2 (steps 335-500): {p2_mean:.3f}")
    print(f"\n  Cumulative reward at phase boundaries:")
    for k, v in results["cumulative_at_phase_boundaries"].items():
        print(f"    {k}: {v:.1f}")
    print(f"\n  Worst 10 steps (by mean reward):")
    for step, rew in results["worst_10_steps"]:
        print(f"    Step {step:3d}: {rew:.2f}")

    return results


# =============================================================================
# Experiment 6: Seed sensitivity
# =============================================================================
def experiment_seed_sensitivity(n_episodes_per_seed=30):
    """Test across multiple seeds to verify robustness."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 6: Seed Sensitivity (5 seeds x 30 episodes)")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 1337]
    seed_results = {}

    for s in seeds:
        env, _ = _make_env(s)
        agents = _make_agents(env)
        rewards = []
        for ep in range(n_episodes_per_seed):
            ep_rew, _, _ = _run_episode(env, agents)
            rewards.append(ep_rew)
        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        seed_results[s] = {"mean": mean_r, "std": std_r}
        print(f"  Seed {s:5d}: {mean_r:8.1f} +/- {std_r:.1f}")

    all_means = [v["mean"] for v in seed_results.values()]
    print(f"\n  Cross-seed mean: {np.mean(all_means):.1f} +/- {np.std(all_means):.1f}")
    print(f"  Range: [{min(all_means):.1f}, {max(all_means):.1f}]")

    return seed_results


# =============================================================================
# Experiment 7: Reward histogram analysis
# =============================================================================
def experiment_reward_distribution(seed=42, n_episodes=100):
    """Analyse the shape of the reward distribution."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 7: Reward Distribution Analysis (100 episodes)")
    print("=" * 70)

    env, _ = _make_env(seed)
    agents = _make_agents(env)

    rewards = []
    for ep in range(n_episodes):
        ep_rew, _, _ = _run_episode(env, agents)
        rewards.append(ep_rew)

    rewards = np.array(rewards)

    # Histogram buckets
    buckets = [0, -200, -400, -600, -800, -1000, -1200, -1400, -1600, -1800, -2000, -float('inf')]
    hist = {}
    for i in range(len(buckets) - 1):
        upper = buckets[i]
        lower = buckets[i + 1]
        count = int(np.sum((rewards <= upper) & (rewards > lower)))
        label = f"({lower:.0f}, {upper:.0f}]" if lower != -float('inf') else f"(<-2000, {upper:.0f}]"
        hist[label] = count

    # Zero-reward steps analysis
    results = {
        "histogram": hist,
        "skewness": float(((rewards - rewards.mean()) ** 3).mean() / (rewards.std() ** 3 + 1e-12)),
        "pct_above_neg500": float(100 * np.mean(rewards > -500)),
        "pct_above_neg1000": float(100 * np.mean(rewards > -1000)),
    }

    print(f"\n  Distribution:")
    for label, count in hist.items():
        bar = "#" * (count * 2)
        print(f"    {label:25s}: {count:3d} {bar}")
    print(f"\n  Skewness: {results['skewness']:.2f}")
    print(f"  Episodes > -500:  {results['pct_above_neg500']:.0f}%")
    print(f"  Episodes > -1000: {results['pct_above_neg1000']:.0f}%")

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Optimality Analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="docs/optimality_analysis_data.json")
    args = parser.parse_args()

    t0 = time.perf_counter()
    all_results = {}

    all_results["baseline"] = experiment_baseline(args.seed, n_episodes=100)
    all_results["sleep_baseline"] = experiment_sleep_baseline(args.seed, n_episodes=50)
    all_results["ablations"] = experiment_ablations(args.seed, n_episodes=30)
    all_results["action_budget"] = experiment_action_budget(args.seed, n_episodes=30)
    all_results["reward_curves"] = experiment_reward_curves(args.seed, n_episodes=30)
    all_results["seed_sensitivity"] = experiment_seed_sensitivity(n_episodes_per_seed=30)
    all_results["reward_distribution"] = experiment_reward_distribution(args.seed, n_episodes=100)

    elapsed = time.perf_counter() - t0

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  All experiments complete in {elapsed:.0f}s")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
