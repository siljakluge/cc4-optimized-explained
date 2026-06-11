#!/usr/bin/env python3
"""Instrumented evaluation: per-episode stats for gap analysis."""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from CybORG import CybORG
from CybORG.Agents.Wrappers import BlueFlatWrapperV2
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent
from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import make_heuristic_agents

N_EPISODES = 50
MAX_STEPS = 500
SEED = 42

sg = EnterpriseScenarioGenerator(steps=MAX_STEPS, red_agent_class=FiniteStateRedAgent,
                                  green_agent_class=EnterpriseGreenAgent)
cyborg = CybORG(scenario_generator=sg, seed=SEED)
env = BlueFlatWrapperV2(env=cyborg)

restore_counts = []
remove_counts = []
block_counts = []
allow_counts = []
decoy_counts = []
sleep_counts = []
ep_rewards = []
per_agent_rewards = []

t0 = time.perf_counter()
for ep in range(N_EPISODES):
    obs_dict, _ = env.reset()
    agents = make_heuristic_agents(env)
    subnet_hosts = getattr(env, '_cached_subnet_hosts', {})
    for name, ag in agents.items():
        ag.reset()
        ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)

    ep_reward = 0.0
    ep_restores = 0
    ep_removes = 0
    ep_blocks = 0
    ep_allows = 0
    ep_decoys = 0
    ep_sleeps = 0
    agent_names = env.possible_agents
    agent_reward_sums = {n: 0.0 for n in agent_names}

    for step in range(MAX_STEPS):
        actions, messages = {}, {}
        for name, ag in agents.items():
            raw_obs = obs_dict.get(name, np.zeros(1))
            mask = env.action_mask(name)
            action_idx, msg = ag.get_action(raw_obs, np.array(mask, dtype=bool))
            actions[name] = action_idx
            messages[name] = msg
            label = ag._labels[action_idx] if ag._labels else ''
            if 'Restore' in label: ep_restores += 1
            elif 'Remove' in label: ep_removes += 1
            elif 'BlockTraffic' in label: ep_blocks += 1
            elif 'AllowTraffic' in label: ep_allows += 1
            elif 'DeployDecoy' in label: ep_decoys += 1
            elif label == 'Sleep': ep_sleeps += 1

        obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions, messages=messages)
        for n in agent_names:
            r = rew_dict.get(n, 0.0)
            ep_reward += r
            agent_reward_sums[n] += r
        if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in agent_names):
            break

    ep_rewards.append(ep_reward)
    restore_counts.append(ep_restores)
    remove_counts.append(ep_removes)
    block_counts.append(ep_blocks)
    allow_counts.append(ep_allows)
    decoy_counts.append(ep_decoys)
    sleep_counts.append(ep_sleeps)
    per_agent_rewards.append(agent_reward_sums)

    elapsed = time.perf_counter() - t0
    per_agent = ep_reward / len(agent_names)
    print(f'ep {ep+1:3d}: reward={ep_reward:9.1f} (per-agent={per_agent:7.1f})  '
          f'restores={ep_restores:3d}  removes={ep_removes:3d}  '
          f'blocks={ep_blocks:3d}  allows={ep_allows:3d}  '
          f'decoys={ep_decoys:3d}  sleeps={ep_sleeps:3d}  '
          f'[{elapsed:.1f}s]')

print()
print('=' * 80)
print(f'Mean total reward:     {np.mean(ep_rewards):9.1f} +/- {np.std(ep_rewards):7.1f}')
print(f'Mean per-agent reward: {np.mean(ep_rewards)/5:9.1f} +/- {np.std(ep_rewards)/5:7.1f}')
print(f'Mean restores/ep:  {np.mean(restore_counts):6.1f}  (std {np.std(restore_counts):.1f})')
print(f'Mean removes/ep:   {np.mean(remove_counts):6.1f}  (std {np.std(remove_counts):.1f})')
print(f'Mean blocks/ep:    {np.mean(block_counts):6.1f}  (std {np.std(block_counts):.1f})')
print(f'Mean allows/ep:    {np.mean(allow_counts):6.1f}  (std {np.std(allow_counts):.1f})')
print(f'Mean decoys/ep:    {np.mean(decoy_counts):6.1f}  (std {np.std(decoy_counts):.1f})')
print(f'Mean sleeps/ep:    {np.mean(sleep_counts):6.1f}  (std {np.std(sleep_counts):.1f})')
print()
print(f'Restore-remove ratio: {np.mean(restore_counts)/max(np.mean(remove_counts),1):.2f}')
print()
worst_ep = int(np.argmin(ep_rewards))
best_ep = int(np.argmax(ep_rewards))
print(f'Worst episode {worst_ep+1}: reward={ep_rewards[worst_ep]:.1f}, '
      f'restores={restore_counts[worst_ep]}, removes={remove_counts[worst_ep]}')
print(f'Best  episode {best_ep+1}: reward={ep_rewards[best_ep]:.1f}, '
      f'restores={restore_counts[best_ep]}, removes={remove_counts[best_ep]}')

# Per-agent breakdown
print()
print('Per-agent mean rewards:')
for name in sorted(per_agent_rewards[0].keys()):
    agent_rews = [par[name] for par in per_agent_rewards]
    print(f'  {name}: {np.mean(agent_rews):8.1f} +/- {np.std(agent_rews):6.1f}')

# Action budget analysis
total_actions = np.array(restore_counts) + np.array(remove_counts) + np.array(block_counts) + np.array(allow_counts) + np.array(decoy_counts) + np.array(sleep_counts)
print(f'\nMean total actions/ep: {np.mean(total_actions):.0f}')
print(f'Action budget breakdown:')
print(f'  Restores: {100*np.mean(restore_counts)/np.mean(total_actions):.1f}%')
print(f'  Removes:  {100*np.mean(remove_counts)/np.mean(total_actions):.1f}%')
print(f'  Blocks:   {100*np.mean(block_counts)/np.mean(total_actions):.1f}%')
print(f'  Allows:   {100*np.mean(allow_counts)/np.mean(total_actions):.1f}%')
print(f'  Decoys:   {100*np.mean(decoy_counts)/np.mean(total_actions):.1f}%')
print(f'  Sleeps:   {100*np.mean(sleep_counts)/np.mean(total_actions):.1f}%')
