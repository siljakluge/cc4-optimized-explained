#!/usr/bin/env python3
"""Diagnose V11b coordinated blocking behavior."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from CybORG import CybORG
from CybORG.Agents.Wrappers import BlueFlatWrapperV2
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent
from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11b import (
    make_heuristic_agents_v11b, _AGENT_SUBNETS,
    _BIT_RED_DETECTED, _BIT_REQUEST_BLOCK, _BIT_ZONE_CLEAR,
    _BIT_BUSY_RESTORING, _BIT_THREAT_CNT_LO, _BIT_THREAT_CNT_HI,
)

sg = EnterpriseScenarioGenerator(
    steps=500,
    red_agent_class=FiniteStateRedAgent,
    green_agent_class=EnterpriseGreenAgent,
)
cyborg = CybORG(scenario_generator=sg, seed=42)
env = BlueFlatWrapperV2(env=cyborg)

obs_dict, _ = env.reset()
agents = make_heuristic_agents_v11b(env)
agent_names = env.possible_agents
subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
for name, ag in agents.items():
    ag.reset()
    ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)

# Print available block actions for each agent
print("Available BLOCK actions per agent:")
for name, ag in sorted(agents.items()):
    print(f"  {name} (subnets: {ag._controlled_subnets}):")
    for pair, idx in sorted(ag._block.items()):
        print(f"    Block {pair[0]} -> {pair[1]}: action {idx}")
    print()

# Run one episode and track messages + actions
print("\n--- Running 1 episode (500 steps) ---\n")
obs_dict, _ = env.reset()
for name, ag in agents.items():
    ag.reset()
    ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)

coord_block_count = 0
coord_allow_count = 0
msg_stats = {i: {"red_detected": 0, "request_block": 0, "zone_clear": 0, "busy": 0} for i in range(5)}
action_counts = {name: {} for name in agent_names}
ep_reward = 0.0

for step in range(500):
    actions = {}
    messages = {}
    for name, ag in agents.items():
        raw_obs = obs_dict.get(name, np.zeros(1))
        mask = env.action_mask(name)
        action_idx, msg = ag.get_action(raw_obs, np.array(mask, dtype=bool))
        actions[name] = action_idx
        messages[name] = msg

        # Track messages sent
        idx = int(name.rsplit("_", 1)[-1])
        if msg[_BIT_RED_DETECTED]:
            msg_stats[idx]["red_detected"] += 1
        if msg[_BIT_REQUEST_BLOCK]:
            msg_stats[idx]["request_block"] += 1
        if msg[_BIT_ZONE_CLEAR]:
            msg_stats[idx]["zone_clear"] += 1
        if msg[_BIT_BUSY_RESTORING]:
            msg_stats[idx]["busy"] += 1

        # Track action labels
        label = ag._labels[action_idx] if action_idx < len(ag._labels) else "Unknown"
        action_counts[name][label] = action_counts[name].get(label, 0) + 1

        # Track preemptive blocks
        if label.startswith("BlockTrafficZone"):
            # Check if this is a coordinated block
            if ag._preemptive_blocked:
                for pair, s in ag._preemptive_blocked.items():
                    if s == ag._step:
                        coord_block_count += 1
                        if step < 50:
                            print(f"  step {step}: {name} COORD BLOCK {pair[0]}->{pair[1]}")

        if label.startswith("AllowTrafficZone"):
            if step < 50:
                print(f"  step {step}: {name} ALLOW (label: {label})")

    obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions, messages=messages)
    step_reward = sum(rew_dict.values())
    ep_reward += step_reward

    if step < 50 and step_reward < -10:
        print(f"  step {step}: reward={step_reward:.1f}")

print(f"\nTotal reward: {ep_reward:.1f}")
print(f"\nCoordinated blocks issued: {coord_block_count}")
print(f"Coordinated allows issued: {coord_allow_count}")

print(f"\nMessage stats (per agent):")
for i in range(5):
    print(f"  agent_{i}: red_detected={msg_stats[i]['red_detected']}, "
          f"request_block={msg_stats[i]['request_block']}, "
          f"zone_clear={msg_stats[i]['zone_clear']}, "
          f"busy={msg_stats[i]['busy']}")

print(f"\nPreemptive blocks at end of episode:")
for name, ag in sorted(agents.items()):
    if ag._preemptive_blocked:
        print(f"  {name}: {ag._preemptive_blocked}")

print(f"\nAction counts per agent:")
for name in sorted(action_counts):
    print(f"\n  {name}:")
    for label, count in sorted(action_counts[name].items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}")
