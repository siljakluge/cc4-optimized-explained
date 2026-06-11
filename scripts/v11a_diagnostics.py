#!/usr/bin/env python3
"""V11a Diagnostics: Determine blocking capabilities and comms_policy per phase.

Questions answered:
1. What BlockTrafficZone actions does each blue agent have?
2. What does comms_policy block/allow at steps 0, 167, 334 (phase boundaries)?
3. Which paths TO OZA/OZB are OPEN in each active phase?
4. Can agent_1 block traffic FROM other subnets TO OZA?
   Can agent_3 block traffic FROM other subnets TO OZB?
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from CybORG import CybORG
from CybORG.Agents.Wrappers import BlueFlatWrapperV2
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent

_SORTED_SUBNETS = [
    "admin_network_subnet",
    "contractor_network_subnet",
    "internet_subnet",
    "office_network_subnet",
    "operational_zone_a_subnet",
    "operational_zone_b_subnet",
    "public_access_zone_subnet",
    "restricted_zone_a_subnet",
    "restricted_zone_b_subnet",
]
NUM_SUBNETS = 9
_OFF_BLOCKED = NUM_SUBNETS        # 9
_OFF_COMMS   = NUM_SUBNETS * 2    # 18
_OFF_PROC    = NUM_SUBNETS * 3    # 27


def main():
    sg = EnterpriseScenarioGenerator(
        steps=500,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=42)
    env = BlueFlatWrapperV2(env=cyborg)

    obs_dict, _ = env.reset()
    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})

    print("=" * 80)
    print("DIAGNOSTIC 1: BlockTrafficZone actions per agent")
    print("=" * 80)

    agent_block_actions = {}
    agent_allow_actions = {}

    for agent_name in sorted(env.possible_agents):
        labels = env.action_labels(agent_name)
        blocks = {}
        allows = {}
        for idx, label in enumerate(labels):
            label = label.strip()
            if label.startswith("BlockTrafficZone"):
                import re
                m = re.match(r"BlockTrafficZone\s+(\S+)\s+\S+\s+<-\s+(\S+)", label)
                if m:
                    to_sn, fr_sn = m.group(1), m.group(2)
                    blocks[(fr_sn, to_sn)] = idx
            elif label.startswith("AllowTrafficZone"):
                import re
                m = re.match(r"AllowTrafficZone\s+(\S+)\s+\S+\s+<-\s+(\S+)", label)
                if m:
                    to_sn, fr_sn = m.group(1), m.group(2)
                    allows[(fr_sn, to_sn)] = idx

        agent_block_actions[agent_name] = blocks
        agent_allow_actions[agent_name] = allows

        print(f"\n{agent_name}:")
        print(f"  Total actions: {len(labels)}")
        print(f"  Block actions ({len(blocks)}):")
        for (fr, to), idx in sorted(blocks.items()):
            print(f"    [{idx:3d}] {fr} -> {to}")
        print(f"  Allow actions ({len(allows)}):")
        for (fr, to), idx in sorted(allows.items()):
            print(f"    [{idx:3d}] {fr} -> {to}")

    # Specific question: can agent_1 block paths TO OZA?
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 1b: Agent_1 (OZA) blocking capabilities")
    print("=" * 80)
    oza_blocks = {k: v for k, v in agent_block_actions.get("blue_agent_1", {}).items()
                  if "operational_zone_a" in k[1]}
    print(f"  Agent_1 can block {len(oza_blocks)} paths TO operational_zone_a:")
    for (fr, to), idx in sorted(oza_blocks.items()):
        print(f"    {fr} -> {to}")

    ozb_blocks = {k: v for k, v in agent_block_actions.get("blue_agent_3", {}).items()
                  if "operational_zone_b" in k[1]}
    print(f"\n  Agent_3 can block {len(ozb_blocks)} paths TO operational_zone_b:")
    for (fr, to), idx in sorted(ozb_blocks.items()):
        print(f"    {fr} -> {to}")

    # DIAGNOSTIC 2: comms_policy per phase
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 2: comms_policy at phase boundaries")
    print("=" * 80)

    # Need to step through environment and observe comms_policy changes
    obs_dict, _ = env.reset()

    # Parse comms_policy from observation for each agent at specific steps
    phase_steps = [0, 1, 167, 168, 334, 335]  # around phase transitions

    # First, build agent info for parsing
    agent_subnets = {}
    for agent_name in sorted(env.possible_agents):
        labels = env.action_labels(agent_name)
        controlled = set()
        for label in labels:
            label = label.strip()
            if label.startswith("BlockTrafficZone"):
                import re
                m = re.match(r"BlockTrafficZone\s+(\S+)\s+\S+\s+<-\s+(\S+)", label)
                if m:
                    controlled.add(m.group(1))
        agent_subnets[agent_name] = sorted(controlled)

    # Run through one episode collecting comms_policy data
    from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV10b import make_heuristic_agents_v10b
    agents = make_heuristic_agents_v10b(env)

    phase_comms = defaultdict(lambda: defaultdict(dict))
    # {phase: {agent: {(from, to): should_block}}}

    obs_dict, _ = env.reset()
    for name, ag in agents.items():
        ag.reset()
        ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)

    last_phase = None
    for step in range(500):
        # Parse observations for comms_policy BEFORE stepping
        for agent_name in sorted(env.possible_agents):
            raw_obs = obs_dict.get(agent_name)
            if raw_obs is None:
                continue
            obs = np.asarray(raw_obs, dtype=np.float32)
            phase = int(obs[0])

            # Only log at phase transitions and specific steps
            if step in phase_steps or phase != last_phase:
                subnets_in_obs = agent_subnets[agent_name]
                base = 1
                for sn in subnets_in_obs:
                    hosts = subnet_hosts.get(sn, [])
                    n_hosts = len(hosts)
                    comms_policy_vec = obs[base + _OFF_COMMS : base + _OFF_PROC]
                    blocked_vec = obs[base + _OFF_BLOCKED : base + _OFF_COMMS]

                    for i, src in enumerate(_SORTED_SUBNETS):
                        if src == sn:
                            continue
                        pair = (src, sn)
                        should_block = bool(comms_policy_vec[i])
                        is_blocked = bool(blocked_vec[i])
                        phase_comms[phase][agent_name][pair] = (should_block, is_blocked)

                    base += 27 + 2 * n_hosts

        last_phase = int(list(obs_dict.values())[0][0])

        # Step with sleep actions
        actions = {}
        messages = {}
        for name, ag in agents.items():
            raw_obs = obs_dict.get(name, np.zeros(1))
            mask = env.action_mask(name)
            action_idx, msg = ag.get_action(raw_obs, np.array(mask, dtype=bool))
            actions[name] = action_idx
            messages[name] = msg
        obs_dict, _, term_dict, trunc_dict, _ = env.step(actions, messages=messages)

        if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in env.possible_agents):
            break

    # Print comms_policy summary per phase
    for phase in sorted(phase_comms.keys()):
        print(f"\n--- Phase {phase} ---")
        for agent_name in sorted(phase_comms[phase].keys()):
            print(f"\n  {agent_name}:")
            for pair in sorted(phase_comms[phase][agent_name].keys()):
                should_block, is_blocked = phase_comms[phase][agent_name][pair]
                status = "BLOCK" if should_block else "allow"
                actual = "BLOCKED" if is_blocked else "open"
                # Highlight mismatches
                mismatch = " *** MISMATCH ***" if should_block != is_blocked else ""
                print(f"    {pair[0]:>35s} -> {pair[1]:<35s}  policy={status:5s}  actual={actual:7s}{mismatch}")

    # DIAGNOSTIC 3: Focus on OZA/OZB paths
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 3: Paths TO OZA/OZB per phase")
    print("=" * 80)

    for phase in sorted(phase_comms.keys()):
        print(f"\n--- Phase {phase} ---")

        # OZA paths (agent_1)
        agent_1_data = phase_comms[phase].get("blue_agent_1", {})
        oza_pairs = {k: v for k, v in agent_1_data.items()
                     if "operational_zone_a" in k[1]}
        if oza_pairs:
            print(f"\n  Paths TO OZA (agent_1 observes):")
            for pair, (should_block, is_blocked) in sorted(oza_pairs.items()):
                can_block = pair in agent_block_actions.get("blue_agent_1", {})
                status = "BLOCK" if should_block else "allow"
                actual = "BLOCKED" if is_blocked else "open"
                actionable = "HAS_BLOCK_ACTION" if can_block else "NO_ACTION"
                opportunity = ""
                if not should_block and not is_blocked and can_block:
                    opportunity = " <-- BLOCKING OPPORTUNITY (open, not mandated)"
                elif should_block and not is_blocked and can_block:
                    opportunity = " <-- NEEDS BLOCK (policy says block, but open)"
                print(f"    {pair[0]:>35s}  policy={status:5s}  actual={actual:7s}  {actionable}{opportunity}")

        # OZB paths (agent_3)
        agent_3_data = phase_comms[phase].get("blue_agent_3", {})
        ozb_pairs = {k: v for k, v in agent_3_data.items()
                     if "operational_zone_b" in k[1]}
        if ozb_pairs:
            print(f"\n  Paths TO OZB (agent_3 observes):")
            for pair, (should_block, is_blocked) in sorted(ozb_pairs.items()):
                can_block = pair in agent_block_actions.get("blue_agent_3", {})
                status = "BLOCK" if should_block else "allow"
                actual = "BLOCKED" if is_blocked else "open"
                actionable = "HAS_BLOCK_ACTION" if can_block else "NO_ACTION"
                opportunity = ""
                if not should_block and not is_blocked and can_block:
                    opportunity = " <-- BLOCKING OPPORTUNITY (open, not mandated)"
                elif should_block and not is_blocked and can_block:
                    opportunity = " <-- NEEDS BLOCK (policy says block, but open)"
                print(f"    {pair[0]:>35s}  policy={status:5s}  actual={actual:7s}  {actionable}{opportunity}")

    # DIAGNOSTIC 4: Summary of blocking opportunities
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 4: Summary of preemptive blocking opportunities")
    print("=" * 80)

    for phase in sorted(phase_comms.keys()):
        print(f"\n--- Phase {phase} ---")

        # Phase 1: OZA is target
        if phase == 1:
            agent_1_data = phase_comms[phase].get("blue_agent_1", {})
            oza_pairs = {k: v for k, v in agent_1_data.items()
                         if "operational_zone_a" in k[1]}
            open_no_policy = []
            for pair, (should_block, is_blocked) in sorted(oza_pairs.items()):
                can_block = pair in agent_block_actions.get("blue_agent_1", {})
                # Opportunity: path is open AND comms_policy does NOT require block
                # AND agent has block action for this pair
                if not is_blocked and can_block:
                    open_no_policy.append((pair, should_block))
            print(f"  OZA open paths agent_1 CAN block: {len(open_no_policy)}")
            for pair, policy_says_block in open_no_policy:
                note = "(policy already says BLOCK)" if policy_says_block else "(policy says ALLOW - preemptive)"
                print(f"    {pair[0]} -> {pair[1]}  {note}")

        # Phase 2: OZB is target
        if phase == 2:
            agent_3_data = phase_comms[phase].get("blue_agent_3", {})
            ozb_pairs = {k: v for k, v in agent_3_data.items()
                         if "operational_zone_b" in k[1]}
            open_no_policy = []
            for pair, (should_block, is_blocked) in sorted(ozb_pairs.items()):
                can_block = pair in agent_block_actions.get("blue_agent_3", {})
                if not is_blocked and can_block:
                    open_no_policy.append((pair, should_block))
            print(f"  OZB open paths agent_3 CAN block: {len(open_no_policy)}")
            for pair, policy_says_block in open_no_policy:
                note = "(policy already says BLOCK)" if policy_says_block else "(policy says ALLOW - preemptive)"
                print(f"    {pair[0]} -> {pair[1]}  {note}")


if __name__ == "__main__":
    main()
