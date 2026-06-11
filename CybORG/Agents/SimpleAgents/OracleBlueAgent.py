"""Oracle blue agent for CAGE Challenge 4 — perfect-information upper bound.

This agent cheats: it reads the simulator's ground-truth State object to know
exactly which hosts have red sessions, whether they are user or root level,
and which red agents are on which hosts. It then takes the optimal action
given perfect information.

Purpose: establish an empirical upper bound on blue-team performance. The gap
between the oracle and the heuristic agent represents the information loss
from observations alone.

Usage:
    python scripts/evaluate_oracle.py --episodes 100 --steps 500 --seed 42
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np


# Action durations (must match environment)
REMOVE_DUR = 3
RESTORE_DUR = 5
MAX_DECOYS = 3

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


class OracleBlueAgent:
    """Blue agent with full access to simulator state.

    On each step, reads state.sessions to find all red sessions, then
    takes the optimal defensive action. Never wastes actions on green
    false positives.

    Call after env.reset():
        agent.set_action_info(labels, mask, subnet_hosts)
        agent.set_state_access(env)  # <-- gives oracle access to state
    """

    def __init__(self, agent_name: str = "blue_agent_0"):
        self.agent_name = agent_name

        # Action catalogues (populated by _parse_labels)
        self._sleep_idx: int = 0
        self._block: dict[tuple[str, str], int] = {}
        self._allow: dict[tuple[str, str], int] = {}
        self._remove: dict[str, int] = {}
        self._restore: dict[str, int] = {}
        self._decoy: dict[str, int] = {}

        # State access
        self._env = None  # BlueFlatWrapperV2 instance

        # Controlled hosts (derived from action labels)
        self._controlled_hosts: set[str] = set()
        self._deploy_hosts: list[str] = []
        self._decoy_deployed: dict[str, int] = {}

        # Episode tracking
        self._step: int = 0
        self._restore_at: dict[str, int] = {}
        self._remove_at: dict[str, int] = {}

        self._labels: list[str] = []

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

        Ignores observations entirely — reads ground truth from state.
        """
        if not self._labels:
            return 0, np.zeros(8, dtype=bool)

        self._step += 1
        mask = action_mask
        msg = np.zeros(8, dtype=bool)  # Oracle doesn't need messaging

        # --- Read ground truth from simulator state ---
        state = self._env.env.environment_controller.state
        phase = state.mission_phase

        # Find all red sessions on hosts we control
        red_hosts_user: set[str] = set()   # hosts with user-level red sessions
        red_hosts_root: set[str] = set()   # hosts with root-level red sessions
        red_hosts_any: set[str] = set()    # any red presence

        for agent_name, sessions in state.sessions.items():
            if "red" not in agent_name:
                continue
            for sid, session in sessions.items():
                if not session.active:
                    continue
                hostname = session.hostname
                if hostname not in self._controlled_hosts:
                    continue
                red_hosts_any.add(hostname)
                if session.has_privileged_access():
                    red_hosts_root.add(hostname)
                else:
                    red_hosts_user.add(hostname)

        # --- Priority 1: RESTORE hosts with root red sessions ---
        # Root sessions survive Remove; only Restore works.
        for hostname in _sorted_by_priority(red_hosts_root, phase):
            if self._busy(hostname):
                continue
            idx = self._restore.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._restore_at[hostname] = self._step
                self._remove_at.pop(hostname, None)
                self._decoy_deployed.pop(hostname, None)
                return idx, msg

        # --- Priority 2: REMOVE hosts with user-only red sessions ---
        # User sessions can be killed by Remove (90% success, free, 3 steps).
        # Cheaper than Restore (-1 cost, 5 steps, wipes decoys).
        for hostname in _sorted_by_priority(red_hosts_user - red_hosts_root, phase):
            if self._busy(hostname):
                continue
            # If we already tried Remove and red is back, escalate to Restore
            ra = self._remove_at.get(hostname, -1)
            if ra >= 0 and self._step > ra + REMOVE_DUR:
                idx = self._restore.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._restore_at[hostname] = self._step
                    self._remove_at.pop(hostname, None)
                    self._decoy_deployed.pop(hostname, None)
                    return idx, msg
                continue

            if ra < 0:  # Haven't tried Remove yet
                idx = self._remove.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._remove_at[hostname] = self._step
                    return idx, msg

        # --- Priority 3: Block per comms_policy ---
        # Read comms_policy from observation (oracle still uses obs for this
        # since policy is deterministic from phase, not hidden info)
        obs = np.asarray(observation, dtype=np.float32)
        block_actions = self._get_blocking_actions(obs, phase, mask)
        if block_actions:
            return block_actions[0], msg

        # --- Priority 4: Allow stale blocks per comms_policy ---
        allow_actions = self._get_allow_actions(obs, phase, mask)
        if allow_actions:
            return allow_actions[0], msg

        # --- Priority 5: Redeploy decoys after Restore ---
        for hostname in self._deploy_hosts:
            rs = self._restore_at.get(hostname, -1)
            if rs >= 0 and self._step >= rs + RESTORE_DUR:
                if self._decoy_deployed.get(hostname, 0) < MAX_DECOYS and hostname in self._decoy:
                    idx = self._decoy[hostname]
                    if self._valid(idx, mask):
                        self._decoy_deployed[hostname] = self._decoy_deployed.get(hostname, 0) + 1
                        return idx, msg

        # --- Priority 6: Deploy initial decoys ---
        for hostname in self._deploy_hosts:
            if self._busy(hostname):
                continue
            if self._decoy_deployed.get(hostname, 0) < MAX_DECOYS and hostname in self._decoy:
                idx = self._decoy[hostname]
                if self._valid(idx, mask):
                    self._decoy_deployed[hostname] = self._decoy_deployed.get(hostname, 0) + 1
                    return idx, msg

        # --- Fallback: Sleep ---
        return self._sleep_idx, msg

    def reset(self) -> None:
        self._step = 0
        self._restore_at.clear()
        self._remove_at.clear()
        self._decoy_deployed.clear()
        self._deploy_hosts = []

    # -- Observation parsing for comms_policy (not cheating — deterministic) --

    def _get_blocking_actions(self, obs, phase, mask) -> list[int]:
        """Parse comms_policy from obs and return block actions needed."""
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
                        actions.append(idx)

            base += 27 + 2 * n_hosts
        return actions

    def _get_allow_actions(self, obs, phase, mask) -> list[int]:
        """Parse comms_policy from obs and return allow actions needed."""
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

    def _busy(self, hostname: str) -> bool:
        if self._step <= self._remove_at.get(hostname, -1) + REMOVE_DUR - 1:
            return True
        if self._step <= self._restore_at.get(hostname, -1) + RESTORE_DUR - 1:
            return True
        return False

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
        self._deploy_hosts = []

        controlled_subnets: set[str] = set()
        all_decoy_hosts: list[str] = []

        for idx, raw in enumerate(labels):
            label = raw.strip()
            if label.startswith("[Invalid]"):
                continue

            if label == "Sleep":
                self._sleep_idx = idx

            elif label.startswith("BlockTrafficZone"):
                m = re.match(r"BlockTrafficZone\s+(\S+)\s+\S+\s+<-\s+(\S+)", label)
                if m:
                    to_sn, fr_sn = m.group(1), m.group(2)
                    self._block[(fr_sn, to_sn)] = idx
                    controlled_subnets.add(to_sn)

            elif label.startswith("AllowTrafficZone"):
                m = re.match(r"AllowTrafficZone\s+(\S+)\s+\S+\s+<-\s+(\S+)", label)
                if m:
                    to_sn, fr_sn = m.group(1), m.group(2)
                    self._allow[(fr_sn, to_sn)] = idx

            elif label.startswith("Remove"):
                m = re.match(r"Remove\s+(\S+)", label)
                if m:
                    h = m.group(1)
                    self._remove[h] = idx
                    self._controlled_hosts.add(h)

            elif label.startswith("Restore"):
                m = re.match(r"Restore\s+(\S+)", label)
                if m:
                    h = m.group(1)
                    self._restore[h] = idx
                    self._controlled_hosts.add(h)

            elif label.startswith("DeployDecoy"):
                m = re.match(r"DeployDecoy\s+(\S+)", label)
                if m:
                    h = m.group(1)
                    self._decoy[h] = idx
                    all_decoy_hosts.append(h)
                    self._controlled_hosts.add(h)

        self._deploy_hosts = sorted(all_decoy_hosts, key=_deploy_priority)
        self._subnets_in_obs = sorted(controlled_subnets)
        self._subnet_host_list = {}
        for sn in self._subnets_in_obs:
            self._subnet_host_list[sn] = list(subnet_hosts.get(sn, []))


# -- Module-level helpers (same as heuristic agent) --

def _host_priority(hostname: str, phase: int) -> int:
    if phase == 1:
        if "operational_zone_a" in hostname: return 100
        if "restricted_zone_a" in hostname: return 70
        if "operational_zone_b" in hostname: return 40
    elif phase == 2:
        if "operational_zone_b" in hostname: return 100
        if "restricted_zone_b" in hostname: return 70
        if "operational_zone_a" in hostname: return 40
    elif phase == 0:
        if "operational_zone_b" in hostname: return 40
        if "operational_zone_a" in hostname: return 40
        if "restricted_zone_b" in hostname: return 30
        if "restricted_zone_a" in hostname: return 30
    if any(s in hostname for s in ("admin_network", "office_network", "public_access")):
        return 50
    return 20


def _sorted_by_priority(hosts, phase: int) -> list[str]:
    return sorted(hosts, key=lambda h: _host_priority(h, phase), reverse=True)


def _deploy_priority(hostname: str) -> int:
    if "operational_zone_b" in hostname and "server_host_0" in hostname: return 0
    if "operational_zone_a" in hostname and "server_host_0" in hostname: return 1
    if "operational_zone_b" in hostname and "server" in hostname: return 2
    if "operational_zone_a" in hostname and "server" in hostname: return 3
    if "restricted_zone_b" in hostname and "server" in hostname: return 4
    if "restricted_zone_a" in hostname and "server" in hostname: return 5
    if "server_host_0" in hostname: return 6
    if "server" in hostname: return 7
    if "operational_zone_b" in hostname: return 8
    if "operational_zone_a" in hostname: return 9
    if "restricted_zone_b" in hostname: return 10
    if "restricted_zone_a" in hostname: return 11
    return 20


# -- Factory --

def make_oracle_agents(env) -> dict[str, OracleBlueAgent]:
    """Create one oracle agent per blue team member. Call AFTER env.reset()."""
    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
    agents: dict[str, OracleBlueAgent] = {}
    for agent_name in env.possible_agents:
        ag = OracleBlueAgent(agent_name=agent_name)
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
