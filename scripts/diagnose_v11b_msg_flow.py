#!/usr/bin/env python3
"""Trace message flow between agents during Phase 1/2."""
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
    make_heuristic_agents_v11b, _PHASE_BLOCKING_MAP,
    _BIT_RED_DETECTED, _BIT_REQUEST_BLOCK, _BIT_ZONE_CLEAR,
    _BIT_BUSY_RESTORING, NUM_MSG_BITS, _MSG_LEN,
)

sg = EnterpriseScenarioGenerator(steps=500, red_agent_class=FiniteStateRedAgent, green_agent_class=EnterpriseGreenAgent)
cyborg = CybORG(scenario_generator=sg, seed=42)
env = BlueFlatWrapperV2(env=cyborg)
obs_dict, _ = env.reset()
agents = make_heuristic_agents_v11b(env)
subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
for name, ag in agents.items():
    ag.reset()
    ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)

# Run until we hit Phase 1
found_phase1 = False
for step in range(500):
    phase = int(list(obs_dict.values())[0][0])

    actions, messages = {}, {}
    for name, ag in agents.items():
        raw_obs = obs_dict.get(name, np.zeros(1))
        mask = env.action_mask(name)
        action_idx, msg = ag.get_action(raw_obs, np.array(mask, dtype=bool))
        actions[name] = action_idx
        messages[name] = msg

    if phase == 1:
        if not found_phase1:
            print(f"--- Phase 1 starts at step {step} ---")
            found_phase1 = True

        # Check agent_0's outbound message
        msg0 = messages.get("blue_agent_0", np.zeros(8))
        ag0_red = bool(msg0[_BIT_RED_DETECTED])
        ag0_req = bool(msg0[_BIT_REQUEST_BLOCK])

        # Check what agent_1 receives
        ag1 = agents["blue_agent_1"]
        raw_obs1 = obs_dict.get("blue_agent_1", np.zeros(1))

        # Parse agent_1's observation to find message section
        n_malfile = sum(len(ag1._subnet_host_list.get(sn, [])) for sn in ag1._subnets_in_obs)
        base_sn_len = sum(27 + 2 * len(ag1._subnet_host_list.get(sn, [])) for sn in ag1._subnets_in_obs)
        expected_base = 1 + base_sn_len + NUM_MSG_BITS
        msg_start = 1 + base_sn_len

        # Read raw message section
        raw_msg_section = raw_obs1[msg_start:msg_start + NUM_MSG_BITS]

        # Parse what agent_1 sees in slot 0 (which should be agent_0)
        slot0 = raw_msg_section[0:_MSG_LEN]
        slot0_red = bool(slot0[_BIT_RED_DETECTED])
        slot0_req = bool(slot0[_BIT_REQUEST_BLOCK])

        # Always check what agent_1 receives from agent_0
        peer_msgs = ag1._read_peer_messages_v11b(raw_obs1, msg_start)
        agent0_msg = peer_msgs.get(0, {})

        if ag0_red or ag0_req or agent0_msg.get("red_detected") or agent0_msg.get("request_block"):
            print(f"  step {step}: agent_0 SENDS red={ag0_red} req={ag0_req} | "
                  f"agent_1 RECEIVES from_agent0={agent0_msg}")

        if step > 200 and phase == 1:
            break

    obs_dict, rew_dict, _, _, _ = env.step(actions, messages=messages)
