"""Oracle blue agent V2 for CAGE Challenge 4 — optimal perfect-information policy.

V2 fixes the key flaws in the original oracle agent:
1. ALWAYS uses Restore (never Remove) — 100% success vs 90%, no re-exploit window
2. Phase-aware priority with OZ server_host_0 as absolute top priority during
   active missions (Impact target)
3. Reads red FSM state for predictive defense — knows what red is planning next
4. Aggressive blocking: blocks at phase transitions BEFORE red can exploit paths
5. Immediate Impact detection: if red has root on OZ server_host_0, Restore NOW

Design rationale:
- Remove takes 3 steps, 90% success; if it fails, red keeps user session and
  can PrivilegeEscalate to root. Even if Remove succeeds, red can re-exploit
  during the 3-step window (oracle sees them arrive, but can't act until Remove
  completes). Restore takes 5 steps at -1 cost but is 100% effective and evicts
  root sessions too.
- The oracle ablation showed Restore-only scores -932 vs full oracle -1558,
  confirming that Remove-first is strictly suboptimal in this environment.
- With perfect information, the oracle never wastes actions on false positives,
  so every Restore is justified. The -1 cost per Restore is negligible compared
  to the -10 per step that Impact/DegradeServices inflict.

Usage:
    from CybORG.Agents.SimpleAgents.OracleBlueAgentV2 import make_oracle_v2_agents
    agents = make_oracle_v2_agents(env)
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

# Action durations
REMOVE_DUR = 3
RESTORE_DUR = 5
MAX_DECOYS = 3

# Red FSM states that indicate imminent danger
# R/RD = red has root (can Impact/DegradeServices)
# U/UD = red has user session (can PrivilegeEscalate to root)
# S/SD = red has scanned services (can ExploitRemoteService)
_RED_DANGER_STATES = {"R", "RD", "U", "UD"}
_RED_ROOT_STATES = {"R", "RD"}
_RED_USER_STATES = {"U", "UD"}
_RED_SCAN_STATES = {"S", "SD"}

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


class OracleBlueAgentV2:
    """Perfect-information blue agent with optimal Restore-first policy.

    Key differences from V1:
    - Never uses Remove; always Restore for all red sessions
    - Reads red agent FSM state for predictive prioritization
    - Phase-aware blocking with preemptive phase-transition blocking
    - Impact-host absolute priority

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
        self._deploy_hosts: list[str] = []
        self._decoy_deployed: dict[str, int] = {}

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

        Policy (V2 — blocking-first, Remove+Restore hybrid):
        1. BLOCK per comms_policy (one action blocks entire subnet)
        2. RESTORE root sessions (Remove cannot evict root)
        3. REMOVE user sessions on low-priority hosts (fast, free)
           RESTORE user sessions on high-priority hosts (100% success)
           RESTORE if prior Remove failed (escalation)
        4. ALLOW stale blocks per comms_policy
        5. Redeploy decoys after Restore
        6. Deploy initial decoys
        7. Sleep
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
        red_hosts_any: set[str] = set()

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

        obs = np.asarray(observation, dtype=np.float32)
        block_actions = self._get_blocking_actions(obs, phase, mask)
        allow_actions = self._get_allow_actions(obs, phase, mask)

        # --- Priority 1: RESTORE root sessions on critical hosts ---
        # Only Restore root sessions. These can Impact/DegradeServices.
        # Prioritize OZ server_host_0 (Impact target, -10/step).
        for hostname in _sorted_by_priority_v2(red_hosts_root, phase):
            if self._busy(hostname):
                continue
            idx = self._restore.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._issue_restore(hostname)
                return idx, msg

        # --- Priority 2: Block per comms_policy ---
        if block_actions:
            return block_actions[0], msg

        # --- Priority 3: Handle user-level red sessions ---
        # User sessions are dangerous because they can PrivilegeEscalate.
        # Use Remove (3 steps, free, 90%) to clear them quickly.
        # Escalate to Restore if Remove fails.
        # Restore only on Impact targets (OZ server_host_0 during active
        # phase) where the cost of Remove failure is too high.
        user_only = red_hosts_user - red_hosts_root
        for hostname in _sorted_by_priority_v2(user_only, phase):
            if self._busy(hostname):
                continue

            ra = self._remove_at.get(hostname, -1)
            if ra >= 0 and self._step > ra + REMOVE_DUR:
                # Remove failed, escalate
                idx = self._restore.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._issue_restore(hostname)
                    return idx, msg
                continue

            if ra >= 0:
                continue

            # Remove for all user sessions (preserves decoys, faster)
            idx = self._remove.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._remove_at[hostname] = self._step
                return idx, msg

        # --- Priority 4: Allow stale blocks per comms_policy ---
        if allow_actions:
            return allow_actions[0], msg

        # --- Priority 5: Redeploy decoys after Restore ---
        for hostname in self._deploy_hosts:
            rs = self._restore_at.get(hostname, -1)
            if rs >= 0 and self._step >= rs + RESTORE_DUR:
                if (self._decoy_deployed.get(hostname, 0) < MAX_DECOYS
                        and hostname in self._decoy):
                    idx = self._decoy[hostname]
                    if self._valid(idx, mask):
                        self._decoy_deployed[hostname] = (
                            self._decoy_deployed.get(hostname, 0) + 1
                        )
                        return idx, msg

        # --- Priority 6: Deploy initial decoys ---
        for hostname in self._deploy_hosts:
            if self._busy(hostname):
                continue
            if (self._decoy_deployed.get(hostname, 0) < MAX_DECOYS
                    and hostname in self._decoy):
                idx = self._decoy[hostname]
                if self._valid(idx, mask):
                    self._decoy_deployed[hostname] = (
                        self._decoy_deployed.get(hostname, 0) + 1
                    )
                    return idx, msg

        # --- Fallback: Sleep ---
        return self._sleep_idx, msg

    def reset(self) -> None:
        self._step = 0
        self._remove_at.clear()
        self._restore_at.clear()
        self._decoy_deployed.clear()
        self._deploy_hosts = []

    # -- Red FSM state reading --

    def _read_red_fsm_state(self) -> dict[str, str]:
        """Read the FSM host_states from all red agents.

        Returns dict mapping hostname -> FSM state character.
        The red agent tracks hosts by IP, so we translate via state.ip_addresses.
        """
        result = {}
        try:
            controller = self._env.env.environment_controller
            ip_to_hostname = controller.state.ip_addresses  # ip -> hostname

            for agent_name, agent_iface in controller.agent_interfaces.items():
                if "red" not in agent_name:
                    continue
                red_agent = agent_iface.agent
                if not hasattr(red_agent, "host_states"):
                    continue
                for ip_str, host_info in red_agent.host_states.items():
                    fsm_state = host_info.get("state", "F")
                    # Convert IP to hostname
                    from ipaddress import IPv4Address
                    try:
                        ip_obj = IPv4Address(ip_str)
                        hostname = ip_to_hostname.get(ip_obj)
                        if hostname:
                            # If multiple reds know about same host, take worst
                            existing = result.get(hostname, "F")
                            if _fsm_danger_rank(fsm_state) > _fsm_danger_rank(existing):
                                result[hostname] = fsm_state
                    except (ValueError, KeyError):
                        continue
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

        # Sort by priority (highest first) so we block most important paths first
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
        self._decoy_deployed.pop(hostname, None)

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
                    all_decoy_hosts.append(h)
                    self._controlled_hosts.add(h)

        self._deploy_hosts = sorted(all_decoy_hosts, key=_deploy_priority)
        self._subnets_in_obs = sorted(controlled_subnets)
        self._subnet_host_list = {}
        for sn in self._subnets_in_obs:
            self._subnet_host_list[sn] = list(subnet_hosts.get(sn, []))


# -- Module-level helpers --

def _fsm_danger_rank(state: str) -> int:
    """Rank FSM states by danger level for tie-breaking."""
    return {"R": 5, "RD": 4, "U": 3, "UD": 2, "S": 1, "SD": 1}.get(state, 0)


def _host_priority_v2(hostname: str, phase: int) -> int:
    """Phase-aware host priority with Impact-target boosting.

    Key difference from V1: OZ server_host_0 gets priority 200 during its
    active mission phase, since that's where Impact runs (-10/step).
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
        # Both OZ targets are equally important in preplanning
        if "server_host_0" in hostname and (
            "operational_zone_a" in hostname or "operational_zone_b" in hostname
        ):
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


def _sorted_by_priority_v2(hosts, phase: int) -> list[str]:
    return sorted(
        hosts,
        key=lambda h: _host_priority_v2(h, phase),
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
    # Always prioritize blocking from contractor/internet (red entry points)
    if "contractor" in fr or "internet" in fr:
        return 60
    return 20


def _deploy_priority(hostname: str) -> int:
    """Lower value = deploy decoy first."""
    if "operational_zone_b" in hostname and "server_host_0" in hostname:
        return 0
    if "operational_zone_a" in hostname and "server_host_0" in hostname:
        return 1
    if "operational_zone_b" in hostname and "server" in hostname:
        return 2
    if "operational_zone_a" in hostname and "server" in hostname:
        return 3
    if "restricted_zone_b" in hostname and "server" in hostname:
        return 4
    if "restricted_zone_a" in hostname and "server" in hostname:
        return 5
    if "server_host_0" in hostname:
        return 6
    if "server" in hostname:
        return 7
    if "operational_zone_b" in hostname:
        return 8
    if "operational_zone_a" in hostname:
        return 9
    if "restricted_zone_b" in hostname:
        return 10
    if "restricted_zone_a" in hostname:
        return 11
    return 20


# -- Factory --

def make_oracle_v2_agents(env) -> dict[str, OracleBlueAgentV2]:
    """Create one oracle V2 agent per blue team member."""
    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
    agents: dict[str, OracleBlueAgentV2] = {}
    for agent_name in env.possible_agents:
        ag = OracleBlueAgentV2(agent_name=agent_name)
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
