#!/usr/bin/env python3
"""Instrumented evaluation of EnterpriseHeuristicAgent for gap analysis.

Tracks per-priority action counts, restore/remove ratios, decoy hits,
per-phase reward breakdown, host compromise frequency, and detection timing.

Usage:
    python scripts/evaluate_instrumented.py --episodes 10 --steps 500 --seed 42
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_instrumented_evaluation(
    n_episodes: int = 10, max_steps: int = 500, seed: int = 42
) -> dict:
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
    from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import make_heuristic_agents
    agents = make_heuristic_agents(env)
    agent_names = env.possible_agents

    # ---- Tracking structures ----
    # Per-priority action counts across all agents and episodes
    priority_counts = Counter()       # "P1", "P1b", "P1c", "P2", "P3", "P4_remove", "P4_restore", "P5", "P6", "P7", "Sleep"
    restore_count = 0
    remove_count = 0

    # Per-phase reward tracking
    phase_rewards = defaultdict(float)   # phase -> total reward
    phase_steps = defaultdict(int)       # phase -> step count

    # Host compromise tracking
    host_compromise_count = Counter()    # hostname -> times alerted

    # Per-agent action tracking
    agent_action_counts = defaultdict(Counter)  # agent_name -> {action_type: count}

    # Decoy hit detection
    decoy_hit_count = 0
    real_exploit_count = 0
    silent_exploit_count = 0            # malfile only (no proc/conn)
    conn_only_no_decoy_count = 0        # conn-only with no decoy deployed

    # Detection timing: steps between red exploit and blue response
    detection_delays = []

    # Track which hosts get restored vs removed
    host_restore_count = Counter()
    host_remove_count = Counter()

    # Per-episode rewards
    episode_rewards = []
    episode_phase_rewards = []          # list of {0: r, 1: r, 2: r}

    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
        for name, ag in agents.items():
            ag.reset()
            ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)

        ep_reward = 0.0
        ep_phase_rew = defaultdict(float)
        prev_phase = 0

        for step in range(max_steps):
            actions = {}
            messages = {}

            for name, ag in agents.items():
                raw_obs = obs_dict.get(name, np.zeros(1))
                mask = env.action_mask(name)

                # Instrument: capture the agent's internal state before action
                pre_restore_at = dict(ag._restore_at)
                pre_remove_at = dict(ag._remove_at)

                action_idx, msg = ag.get_action(raw_obs, np.array(mask, dtype=bool))
                actions[name] = action_idx
                messages[name] = msg

                # Classify action by priority
                label = ag._labels[action_idx] if action_idx < len(ag._labels) else "Unknown"
                action_type = _classify_action(label, ag, pre_restore_at, pre_remove_at)
                priority_counts[action_type] += 1
                agent_action_counts[name][action_type] += 1

                if "Restore" in label:
                    restore_count += 1
                    hostname = _extract_hostname(label)
                    if hostname:
                        host_restore_count[hostname] += 1
                elif "Remove" in label:
                    remove_count += 1
                    hostname = _extract_hostname(label)
                    if hostname:
                        host_remove_count[hostname] += 1

                # Track alerts for this agent
                obs = np.asarray(raw_obs, dtype=np.float32)
                if len(obs) > 1:
                    _track_alerts(
                        obs, ag, step,
                        host_compromise_count,
                    )

            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions, messages=messages)
            step_reward = sum(rew_dict.values())
            ep_reward += step_reward

            # Get current phase from any agent's obs
            any_obs = next(iter(obs_dict.values()), np.zeros(1))
            current_phase = int(any_obs[0]) if len(any_obs) > 0 else 0
            phase_rewards[current_phase] += step_reward
            phase_steps[current_phase] += 1
            ep_phase_rew[current_phase] += step_reward

            if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in agent_names):
                break

        episode_rewards.append(ep_reward)
        episode_phase_rewards.append(dict(ep_phase_rew))
        print(f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:9.1f}  steps={step+1}")

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
        "priority_counts": dict(priority_counts),
        "restore_count": restore_count,
        "remove_count": remove_count,
        "phase_rewards": dict(phase_rewards),
        "phase_steps": dict(phase_steps),
        "host_compromise_count": host_compromise_count.most_common(20),
        "agent_action_counts": {k: dict(v) for k, v in agent_action_counts.items()},
        "host_restore_count": host_restore_count.most_common(15),
        "host_remove_count": host_remove_count.most_common(15),
        "episode_rewards": episode_rewards,
        "episode_phase_rewards": episode_phase_rewards,
    }


def _extract_hostname(label: str) -> str | None:
    """Extract hostname from action label like 'Restore host_x' or 'Remove host_x'."""
    parts = label.strip().split(None, 1)
    if len(parts) >= 2:
        return parts[1]
    return None


def _classify_action(label: str, ag, pre_restore, pre_remove) -> str:
    """Classify an action into its priority bucket."""
    label = label.strip()
    if label == "Sleep":
        return "P8_Sleep"
    if label.startswith("DeployDecoy"):
        hostname = _extract_hostname(label)
        if hostname:
            rs = ag._restore_at.get(hostname, -1)
            if rs >= 0:
                return "P6_RedeployDecoy"
        return "P7_DeployDecoy"
    if label.startswith("AllowTrafficZone"):
        return "P2_Allow"
    if label.startswith("BlockTrafficZone"):
        return "P3_Block"
    if label.startswith("Restore"):
        hostname = _extract_hostname(label)
        if hostname:
            # Was a Remove previously issued on this host?
            ra = pre_remove.get(hostname, -1)
            if ra >= 0:
                return "P4_Restore(escalated)"
        return "P1_Restore"
    if label.startswith("Remove"):
        return "P4_Remove"
    return f"Unknown({label})"


def _track_alerts(obs, ag, step, host_compromise_count):
    """Track host-level alerts from observation."""
    base = 1
    for sn in ag._subnets_in_obs:
        hosts = ag._subnet_host_list.get(sn, [])
        n_hosts = len(hosts)
        off_conn = 27 + n_hosts
        proc_flags = obs[base + 27: base + 27 + n_hosts]
        conn_flags = obs[base + off_conn: base + off_conn + n_hosts]
        for hi, hostname in enumerate(hosts):
            if hi < len(proc_flags) and proc_flags[hi]:
                host_compromise_count[hostname] += 1
            elif hi < len(conn_flags) and conn_flags[hi]:
                host_compromise_count[hostname] += 1
        base += 27 + 2 * n_hosts


def format_report(results: dict) -> str:
    """Format results into a readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  INSTRUMENTED EVALUATION RESULTS")
    lines.append("=" * 70)
    lines.append(f"  Episodes: {results['n_episodes']}")
    lines.append(f"  Mean reward: {results['mean_reward']:.1f} +/- {results['std_reward']:.1f}")
    lines.append(f"  Min/Max:     {results['min_reward']:.1f} / {results['max_reward']:.1f}")
    lines.append(f"  Wall time:   {results['elapsed_sec']:.1f}s")
    lines.append("")

    lines.append("--- Priority Action Counts (all agents, all episodes) ---")
    for k, v in sorted(results["priority_counts"].items()):
        lines.append(f"  {k:30s}: {v:6d}")
    lines.append("")

    lines.append(f"--- Restore vs Remove Ratio ---")
    total = results["restore_count"] + results["remove_count"]
    if total > 0:
        lines.append(f"  Restores: {results['restore_count']:6d}  ({results['restore_count']/total*100:.1f}%)")
        lines.append(f"  Removes:  {results['remove_count']:6d}  ({results['remove_count']/total*100:.1f}%)")
        lines.append(f"  Ratio:    {results['restore_count']/(results['remove_count'] or 1):.2f} restores per remove")
    lines.append("")

    lines.append("--- Per-Phase Reward Breakdown ---")
    for phase in sorted(results["phase_rewards"].keys()):
        total_r = results["phase_rewards"][phase]
        steps = results["phase_steps"][phase]
        avg = total_r / steps if steps > 0 else 0
        lines.append(f"  Phase {phase}: total={total_r:10.1f}  steps={steps:6d}  avg/step={avg:.3f}")
    lines.append("")

    lines.append("--- Most Compromised Hosts (proc/conn alerts) ---")
    for hostname, count in results["host_compromise_count"][:15]:
        lines.append(f"  {hostname:50s}: {count:5d} alerts")
    lines.append("")

    lines.append("--- Most Restored Hosts ---")
    for hostname, count in results["host_restore_count"][:10]:
        lines.append(f"  {hostname:50s}: {count:5d} restores")
    lines.append("")

    lines.append("--- Most Removed Hosts ---")
    for hostname, count in results["host_remove_count"][:10]:
        lines.append(f"  {hostname:50s}: {count:5d} removes")
    lines.append("")

    lines.append("--- Per-Agent Action Breakdown ---")
    for agent_name in sorted(results["agent_action_counts"].keys()):
        lines.append(f"  {agent_name}:")
        for action, count in sorted(results["agent_action_counts"][agent_name].items()):
            lines.append(f"    {action:30s}: {count:5d}")
    lines.append("")

    lines.append("--- Per-Episode Rewards ---")
    for i, r in enumerate(results["episode_rewards"]):
        phase_breakdown = results["episode_phase_rewards"][i]
        pb_str = "  ".join(f"P{p}={v:.0f}" for p, v in sorted(phase_breakdown.items()))
        lines.append(f"  ep {i+1:3d}: {r:9.1f}  [{pb_str}]")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Instrumented evaluation of EnterpriseHeuristicAgent")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nRunning instrumented evaluation: {args.episodes} episodes x {args.steps} steps (seed={args.seed})\n")
    results = run_instrumented_evaluation(args.episodes, args.steps, args.seed)
    report = format_report(results)
    print(report)

    # Save report
    out_path = Path(__file__).parent.parent / "docs" / "swarm_analysis" / "instrumented_results.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
