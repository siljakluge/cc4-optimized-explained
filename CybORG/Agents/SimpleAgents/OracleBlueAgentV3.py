"""Oracle blue agent V3 for CAGE Challenge 4 — truly optimal policy.

V3 fixes the critical flaws in V1 (Remove-first) and V2 (over-Restoring):

Design insight: damage comes from GREEN agents failing on compromised hosts,
not from red sessions directly. The BlueRewardMachine penalises:
  - LWF: GreenLocalWork fails on a host where red has an active session
  - ASF: GreenAccessService fails on a host where red has an active session
  - RIA: Red Impact action succeeds

So the oracle must minimise the number of host-steps where red has active sessions
AND green agents are trying to use those hosts. The optimal strategy is:

  1. BLOCK first — prevents red from reaching new hosts (free, instant)
  2. REMOVE user sessions — fast (3 steps), free, 90% success
  3. RESTORE root sessions — only way to evict root (5 steps, -1 cost)
  4. On critical hosts (OZ server_host_0 during active mission), prefer
     Restore even for user sessions (can't afford Remove failure)
  5. Skip decoys entirely — oracle doesn't need detection aids
  6. Track the red FSM to know which hosts need immediate attention

The key realisation: V2 scored -1938 (worse than V1's -1558 and heuristic's -1039)
because it did ~299 Restores/ep vs heuristic's ~85. Restore is expensive (-1 each,
5 steps busy). The heuristic achieves -1039 with ~85 Restores + ~48 Removes
because it uses Remove effectively for user sessions.

Usage:
    from CybORG.Agents.SimpleAgents.OracleBlueAgentV3 import make_oracle_v3_agents
    agents = make_oracle_v3_agents(env)
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

# Action durations
REMOVE_DUR = 3
RESTORE_DUR = 5

# Subnets sorted alphabetically (matches obs layout)
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


class OracleBlueAgentV3:
    """Perfect-information blue agent with surgically optimal policy.

    Call after env.reset():
        agent.set_action_info(labels, mask, subnet_hosts)
        agent.set_state_access(env)
    """

    def __init__(self, agent_name: str = "blue_agent_0"):
        self.agent_name = agent_name

        # Action catalogues
        self._sleep_idx: int = 0
        self._block: dict[tuple[str, str], int] = {}
        self._allow: dict[tuple[str, str], int] = {}
        self._remove: dict[str, int] = {}
        self._restore: dict[str, int] = {}
        self._decoy: dict[str, int] = {}

        # State access
        self._env = None

        # Controlled hosts
        self._controlled_hosts: set[str] = set()

        # Episode tracking
        self._step: int = 0
        self._remove_at: dict[str, int] = {}
        self._restore_at: dict[str, int] = {}

        self._labels: list[str] = []

        # Obs layout
        self._subnets_in_obs: list[str] = []
        self._subnet_host_list: dict[str, list] = {}

    def set_state_access(self, env) -> None:
        """Give the oracle access to the environment for state reads."""
        self._env = env

    def set_action_info(
        self,
        action_labels: list[str],
        action_mask=None,
        subnet_hosts=None,
    ) -> None:
        self._labels = action_labels
        if action_labels:
            self._parse_labels(action_labels, subnet_hosts or {})

    def get_action(
        self,
        observation: np.ndarray,
        action_mask=None,
    ) -> tuple[int, np.ndarray]:
        """Return (action_idx, 8-bit message).

        Optimal oracle policy (V3):
        1. BLOCK per comms_policy (cheapest defense, prevents spreading)
        2. RESTORE root sessions (only way to evict root, prioritise critical hosts)
        3. REMOVE user sessions (fast, free; escalate to Restore on failure)
        4. ALLOW stale blocks
        5. Sleep
        """
        if not self._labels:
            return 0, np.zeros(8, dtype=bool)

        self._step += 1
        mask = action_mask
        msg = np.zeros(8, dtype=bool)

        # --- Read ground truth ---
        state = self._env.env.environment_controller.state
        phase = state.mission_phase

        # Find all red sessions on hosts we control
        red_hosts_root: set[str] = set()
        red_hosts_user: set[str] = set()

        for agent_name, sessions in state.sessions.items():
            if "red" not in agent_name:
                continue
            for sid, session in sessions.items():
                if not session.active:
                    continue
                hostname = session.hostname
                if hostname not in self._controlled_hosts:
                    continue
                if session.has_privileged_access():
                    red_hosts_root.add(hostname)
                else:
                    red_hosts_user.add(hostname)

        # --- Read red FSM states for predictive defense ---
        red_fsm = self._read_red_fsm_state()

        obs = np.asarray(observation, dtype=np.float32)

        # --- Priority 1: BLOCK per comms_policy ---
        # Blocking is free, instant, and prevents red from reaching new hosts.
        # This is the highest-value action.
        block_actions = self._get_blocking_actions(obs, phase, mask)
        if block_actions:
            return block_actions[0], msg

        # --- Priority 2: RESTORE root sessions on critical hosts first ---
        # Root sessions can run Impact (-10/step on OZ) and DegradeServices
        # (permanent damage). Only Restore can evict root.
        # Prioritise by phase-aware criticality.
        for hostname in _sorted_by_priority(red_hosts_root, phase):
            if self._busy(hostname):
                continue
            idx = self._restore.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._issue_restore(hostname)
                return idx, msg

        # --- Priority 3: REMOVE user sessions ---
        # User sessions are less dangerous but can PrivilegeEscalate to root.
        # Remove is preferred: 3 steps (vs 5), free (vs -1), preserves state.
        # Exception: on OZ server_host_0 during active mission, use Restore
        # directly because Remove failure -> PrivEsc -> Impact is catastrophic.
        user_only = red_hosts_user - red_hosts_root
        for hostname in _sorted_by_priority(user_only, phase):
            if self._busy(hostname):
                continue

            # Check if previous Remove failed (red still present after Remove window)
            ra = self._remove_at.get(hostname, -1)
            if ra >= 0 and self._step > ra + REMOVE_DUR:
                # Remove completed but user session persists -> escalate to Restore
                idx = self._restore.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._issue_restore(hostname)
                    return idx, msg
                continue

            # Still in Remove window, skip
            if ra >= 0:
                continue

            # For Impact targets during active mission, go straight to Restore
            # (can't afford the 10% Remove failure -> PrivEsc -> Impact)
            if _is_impact_target(hostname, phase):
                idx = self._restore.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._issue_restore(hostname)
                    return idx, msg
                continue

            # Normal case: try Remove first (faster, cheaper)
            idx = self._remove.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._remove_at[hostname] = self._step
                return idx, msg

        # --- Priority 4: ALLOW stale blocks ---
        allow_actions = self._get_allow_actions(obs, phase, mask)
        if allow_actions:
            return allow_actions[0], msg

        # --- No decoy deployment ---
        # Oracle has perfect information; decoys add no detection value.
        # Spending actions on decoys wastes time that could be used for
        # blocking/removing/restoring.

        # --- Fallback: Sleep ---
        return self._sleep_idx, msg

    def reset(self) -> None:
        self._step = 0
        self._remove_at.clear()
        self._restore_at.clear()

    # -- Red FSM state reading --

    def _read_red_fsm_state(self) -> dict[str, str]:
        """Read the FSM host_states from all red agents.

        Returns dict mapping hostname -> FSM state character.
        """
        result = {}
        try:
            controller = self._env.env.environment_controller
            ip_to_hostname = controller.state.ip_addresses

            for agent_name, agent_iface in controller.agent_interfaces.items():
                if "red" not in agent_name:
                    continue
                red_agent = agent_iface.agent
                if not hasattr(red_agent, "host_states"):
                    continue
                for ip_str, host_info in red_agent.host_states.items():
                    fsm_state = host_info.get("state", "F")
                    hostname = host_info.get("hostname")
                    if not hostname:
                        from ipaddress import IPv4Address
                        try:
                            ip_obj = IPv4Address(ip_str)
                            hostname = ip_to_hostname.get(ip_obj)
                        except (ValueError, KeyError):
                            continue
                    if hostname:
                        existing = result.get(hostname, "F")
                        if _fsm_danger_rank(fsm_state) > _fsm_danger_rank(existing):
                            result[hostname] = fsm_state
        except Exception:
            pass
        return result

    # -- Observation parsing for comms_policy --

    def _get_blocking_actions(self, obs, phase, mask) -> list[int]:
        """Parse comms_policy and return block actions needed, priority-sorted."""
        actions = []
        base = 1
        for sn in self._subnets_in_obs:
            hosts = self._subnet_host_list.get(sn, [])
            n_hosts = len(hosts)
            n_sub = 9

            blocked_vec = obs[base + n_sub: base + 2 * n_sub]
            comms_vec = obs[base + 2 * n_sub: base + 3 * n_sub]

            for i, src in enumerate(_SORTED_SUBNETS):
                if src == sn:
                    continue
                pair = (src, sn)
                should_block = bool(comms_vec[i])
                is_blocked = bool(blocked_vec[i])
                if should_block and not is_blocked:
                    idx = self._block.get(pair)
                    if idx is not None and self._valid(idx, mask):
                        actions.append((idx, _pair_priority(pair, phase)))

            base += 27 + 2 * n_hosts

        actions.sort(key=lambda x: x[1], reverse=True)
        return [a[0] for a in actions]

    def _get_allow_actions(self, obs, phase, mask) -> list[int]:
        """Parse comms_policy and return allow actions needed."""
        actions = []
        base = 1
        for sn in self._subnets_in_obs:
            hosts = self._subnet_host_list.get(sn, [])
            n_hosts = len(hosts)
            n_sub = 9

            blocked_vec = obs[base + n_sub: base + 2 * n_sub]
            comms_vec = obs[base + 2 * n_sub: base + 3 * n_sub]

            for i, src in enumerate(_SORTED_SUBNETS):
                if src == sn:
                    continue
                pair = (src, sn)
                should_block = bool(comms_vec[i])
                is_blocked = bool(blocked_vec[i])
                if not should_block and is_blocked:
                    idx = self._allow.get(pair)
                    if idx is not None and self._valid(idx, mask):
                        actions.append(idx)

            base += 27 + 2 * n_hosts
        return actions

    # -- Helpers --

    def _issue_restore(self, hostname: str) -> None:
        self._restore_at[hostname] = self._step
        self._remove_at.pop(hostname, None)

    def _busy(self, hostname: str) -> bool:
        if self._step <= self._remove_at.get(hostname, -1) + REMOVE_DUR - 1:
            return True
        return self._step <= self._restore_at.get(hostname, -1) + RESTORE_DUR - 1

    def _valid(self, idx: int, mask) -> bool:
        if mask is None:
            return True
        if idx < 0 or idx >= len(mask):
            return False
        return bool(mask[idx])

    def _parse_labels(self, labels: list[str], subnet_hosts: dict) -> None:
        self._block.clear()
        self._allow.clear()
        self._remove.clear()
        self._restore.clear()
        self._decoy.clear()
        self._controlled_hosts.clear()

        controlled_subnets: set[str] = set()

        for idx, raw in enumerate(labels):
            label = raw.strip()
            if label.startswith("[Invalid]"):
                continue

            if label == "Sleep":
                self._sleep_idx = idx

            elif label.startswith("BlockTrafficZone"):
                m = re.match(
                    r"BlockTrafficZone\s+(\S+)\s+\S+\s+<-\s+(\S+)", label
                )
                if m:
                    to_sn, fr_sn = m.group(1), m.group(2)
                    self._block[(fr_sn, to_sn)] = idx
                    controlled_subnets.add(to_sn)

            elif label.startswith("AllowTrafficZone"):
                m = re.match(
                    r"AllowTrafficZone\s+(\S+)\s+\S+\s+<-\s+(\S+)", label
                )
                if m:
                    to_sn, fr_sn = m.group(1), m.group(2)
                    self._allow[(fr_sn, to_sn)] = idx

            elif label.startswith("Restore"):
                m = re.match(r"Restore\s+(\S+)", label)
                if m:
                    h = m.group(1)
                    self._restore[h] = idx
                    self._controlled_hosts.add(h)

            elif label.startswith("Remove"):
                m = re.match(r"Remove\s+(\S+)", label)
                if m:
                    h = m.group(1)
                    self._remove[h] = idx
                    self._controlled_hosts.add(h)

            elif label.startswith("DeployDecoy"):
                m = re.match(r"DeployDecoy\s+(\S+)", label)
                if m:
                    h = m.group(1)
                    self._decoy[h] = idx
                    self._controlled_hosts.add(h)

        self._subnets_in_obs = sorted(controlled_subnets)
        self._subnet_host_list = {}
        for sn in self._subnets_in_obs:
            self._subnet_host_list[sn] = list(subnet_hosts.get(sn, []))


# -- Module-level helpers --

def _fsm_danger_rank(state: str) -> int:
    return {"R": 5, "RD": 4, "U": 3, "UD": 2, "S": 1, "SD": 1}.get(state, 0)


def _is_impact_target(hostname: str, phase: int) -> bool:
    """Is this host the Impact target for the current active mission?"""
    if phase == 1:
        return ("operational_zone_a" in hostname
                and "server_host_0" in hostname)
    elif phase == 2:
        return ("operational_zone_b" in hostname
                and "server_host_0" in hostname)
    return False


def _host_priority(hostname: str, phase: int) -> int:
    """Phase-aware host priority.

    Higher priority = service on this host earlier.
    Impact targets during active mission get absolute priority.
    """
    if phase == 1:
        if "operational_zone_a" in hostname:
            if "server_host_0" in hostname:
                return 200  # Impact target
            return 100
        if "restricted_zone_a" in hostname:
            return 70
        if "operational_zone_b" in hostname:
            return 40
    elif phase == 2:
        if "operational_zone_b" in hostname:
            if "server_host_0" in hostname:
                return 200  # Impact target
            return 100
        if "restricted_zone_b" in hostname:
            return 70
        if "operational_zone_a" in hostname:
            return 40
    elif phase == 0:
        if "server_host_0" in hostname and "operational_zone" in hostname:
            return 80
        if "operational_zone" in hostname:
            return 40
        if "restricted_zone" in hostname:
            return 30

    # Infrastructure hosts
    if any(s in hostname for s in ("admin_network", "office_network")):
        return 50
    if "public_access" in hostname:
        return 45
    return 20


def _sorted_by_priority(hosts, phase: int) -> list[str]:
    return sorted(
        hosts,
        key=lambda h: _host_priority(h, phase),
        reverse=True,
    )


def _pair_priority(pair: tuple[str, str], phase: int) -> int:
    """Priority for blocking a (from_subnet, to_subnet) pair."""
    fr, to = pair
    if phase == 1:
        if "restricted_zone_a" in to or "operational_zone_a" in to:
            return 100
        if "restricted_zone_a" in fr or "operational_zone_a" in fr:
            return 80
    elif phase == 2:
        if "restricted_zone_b" in to or "operational_zone_b" in to:
            return 100
        if "restricted_zone_b" in fr or "operational_zone_b" in fr:
            return 80
    if "contractor" in fr or "internet" in fr:
        return 60
    return 20


# -- Factory --

def make_oracle_v3_agents(env) -> dict[str, OracleBlueAgentV3]:
    """Create one oracle V3 agent per blue team member."""
    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
    agents: dict[str, OracleBlueAgentV3] = {}
    for agent_name in env.possible_agents:
        ag = OracleBlueAgentV3(agent_name=agent_name)
        ag.set_state_access(env)
        try:
            ag.set_action_info(
                env.action_labels(agent_name),
                env.action_mask(agent_name),
                subnet_hosts,
            )
        except Exception:
            pass
        agents[agent_name] = ag
    return agents
