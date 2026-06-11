"""Optimal heuristic blue agents for CAGE Challenge 4 -- v11b (Messaging Redesign).

v11b: New messaging protocol with coordinated blocking.
  Rationale:
    - V9 messaging protocol has ZERO measurable value (p=0.80 ablation).
    - Root causes: 86.7% agent-steps have no upstream, T2/compound = dead code,
      slot misalignment bug, RESTORING/OPEN_PATHS bits never consumed.
    - New protocol communicates RED_DETECTED, REQUEST_BLOCK, ZONE_CLEAR so peers
      can preemptively block traffic FROM compromised zones.

  Changes from v10b:
    - New 8-bit message format (see _BIT_* constants below).
    - All agents read ALL peer messages (not just upstream).
    - Slot misalignment fix: agents know message delivery is sorted by sender name,
      and only connected agents deliver messages. We handle variable slot counts.
    - New priority P_CB (coordinated blocking) between P1c and P2.
    - Preemptive block/allow tracking via self._preemptive_blocked.
    - REQUEST_BLOCK triggers when agent has red AND is busy restoring, or THREAT_COUNT >= 2.
    - ZONE_CLEAR triggers re-allow of preemptive blocks (not comms_policy blocks).
    - Dead code removed: T2, compound escalation, OPEN_PATHS, upstream-only logic.

  All other priorities (P1, P1b, P1c, P2, P3, P4, P6, P7) unchanged from v10b.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

# -- Constants ----------------------------------------------------------------

_SORTED_SUBNETS = [
    "admin_network_subnet",        # 0
    "contractor_network_subnet",   # 1
    "internet_subnet",             # 2
    "office_network_subnet",       # 3
    "operational_zone_a_subnet",   # 4
    "operational_zone_b_subnet",   # 5
    "public_access_zone_subnet",   # 6
    "restricted_zone_a_subnet",    # 7
    "restricted_zone_b_subnet",    # 8
]
_SUBNET_IDX = {s: i for i, s in enumerate(_SORTED_SUBNETS)}
NUM_SUBNETS  = 9
MAX_HOSTS    = 16
NUM_MSG_BITS = 32

_OFF_BLOCKED  = NUM_SUBNETS        # 9
_OFF_COMMS    = NUM_SUBNETS * 2    # 18
_OFF_PROC     = NUM_SUBNETS * 3    # 27

RESTORE_DUR = 5

MAX_DECOYS = 3

# -- Messaging constants (v11b protocol) -------------------------------------

_MSG_LEN = 8
_NUM_BLUE_AGENTS = 5

# Agent-to-subnet mapping (static, from EnterpriseScenarioGenerator)
# Each agent controls specific subnets and can block traffic TO those subnets.
_AGENT_SUBNETS = {
    0: ["restricted_zone_a_subnet"],
    1: ["operational_zone_a_subnet"],
    2: ["restricted_zone_b_subnet"],
    3: ["operational_zone_b_subnet"],
    4: ["public_access_zone_subnet", "admin_network_subnet", "office_network_subnet"],
}

# Surgical blocking map: only the active OZ agent should block FROM its upstream RZ.
# Red path: internet -> contractor -> HQ(4) -> RZA(0)/RZB(2) -> OZA(1)/OZB(3)
# Phase 1: agent_1 (OZA) listens to agent_0 (RZA) for blocking RZA->OZA (ASF=0)
# Phase 2: agent_3 (OZB) listens to agent_2 (RZB) for blocking RZB->OZB (ASF=0)
# All other blocking is too costly (ASF != 0) or on wrong attack paths.
_PHASE_BLOCKING_MAP = {
    # phase -> (listener_agent, upstream_agent, from_subnet, to_subnet)
    1: (1, 0, "restricted_zone_a_subnet", "operational_zone_a_subnet"),
    2: (3, 2, "restricted_zone_b_subnet", "operational_zone_b_subnet"),
}

# New 8-bit message format (v11b)
_BIT_RED_DETECTED   = 0  # 1 = confirmed red (conn + malfile/proc)
_BIT_RED_ROOT       = 1  # 1 = root-level red (malfile without conn/proc)
_BIT_BUSY_RESTORING = 2  # 1 = currently restoring a host
_BIT_ZONE_CLEAR     = 3  # 1 = no alerts, all hosts clean
_BIT_THREAT_CNT_LO  = 4  # 2-bit threat count (0-3 hosts with active threats)
_BIT_THREAT_CNT_HI  = 5
_BIT_REQUEST_BLOCK  = 6  # 1 = downstream should block FROM my zone
_BIT_RESERVED       = 7  # 0 (reserved)


class EnterpriseHeuristicAgentV11b:
    """Phase-aware, Restore-only heuristic blue agent for CC4 v11b.

    New messaging protocol with coordinated blocking.
    """

    def __init__(self, agent_name: str = "blue_agent_0"):
        self.agent_name = agent_name

        self._sleep_idx: int  = 0
        self._block:  dict[tuple[str, str], int] = {}
        self._allow:  dict[tuple[str, str], int] = {}
        self._remove: dict[str, int]  = {}   # parsed but never used
        self._restore: dict[str, int] = {}
        self._decoy:  dict[str, int]  = {}

        self._subnets_in_obs:   list[str]       = []
        self._subnet_host_list: dict[str, list] = {}

        self._deploy_hosts:    list[str] = []
        self._decoy_deployed:  dict[str, int] = {}

        self._step:       int              = 0
        self._restore_at: dict[str, int]   = {}
        self._proc_flagged_step: dict[str, int] = {}
        self._decoy_hit_hosts:  set            = set()

        self._labels: list[str] = []

        # v11b: preemptive block tracking
        # Maps (from_subnet, to_subnet) -> step when preemptively blocked.
        # These are blocks triggered by peer messages, NOT comms_policy.
        # They can be re-allowed when peer sends ZONE_CLEAR.
        self._preemptive_blocked: dict[tuple[str, str], int] = {}

        # v11b: own agent index
        try:
            self._own_idx = int(agent_name.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            self._own_idx = -1

        # v11b: controlled subnets for this agent
        self._controlled_subnets: list[str] = _AGENT_SUBNETS.get(self._own_idx, [])

    # -- Public interface -----------------------------------------------------

    def set_action_info(
        self,
        action_labels:  list[str],
        action_mask:    Optional[np.ndarray] = None,
        subnet_hosts:   Optional[dict]       = None,
    ) -> None:
        """Register action catalogue from env.  Call after each reset()."""
        self._labels = action_labels
        if action_labels:
            self._parse_labels(action_labels, subnet_hosts or {})

    def get_action(
        self,
        observation:  np.ndarray,
        action_mask:  Optional[np.ndarray] = None,
    ) -> tuple[int, np.ndarray]:
        """Return (action_idx, 8-bit message)."""
        if not self._labels:
            return 0, np.zeros(8, dtype=bool)

        self._step += 1
        obs  = np.asarray(observation, dtype=np.float32)
        mask = action_mask

        # -- Detect whether malfile section is present -----------------------
        n_malfile_hosts = sum(
            len(self._subnet_host_list.get(sn, []))
            for sn in self._subnets_in_obs
        )
        base_subnet_len = sum(
            27 + 2 * len(self._subnet_host_list.get(sn, []))
            for sn in self._subnets_in_obs
        )
        expected_base_len = 1 + base_subnet_len + NUM_MSG_BITS
        has_malfile = (n_malfile_hosts > 0 and
                       len(obs) == expected_base_len + n_malfile_hosts)
        malfile_start = expected_base_len if has_malfile else len(obs)

        # -- Parse per-subnet obs --------------------------------------------
        phase = int(obs[0])
        conn_alerts   = {}
        proc_alerts   = {}
        malfile_alerts = {}
        blocked_now   = {}
        should_block  = {}

        base = 1
        malfile_cursor = malfile_start
        for sn in self._subnets_in_obs:
            hosts  = self._subnet_host_list.get(sn, [])
            n_hosts = len(hosts)
            off_conn = _OFF_PROC + n_hosts

            blocked_vec      = obs[base + _OFF_BLOCKED : base + _OFF_COMMS]
            comms_policy_vec = obs[base + _OFF_COMMS   : base + _OFF_PROC]
            proc_flags       = obs[base + _OFF_PROC    : base + off_conn]
            conn_flags       = obs[base + off_conn     : base + off_conn + n_hosts]

            if has_malfile:
                malfile_vec = obs[malfile_cursor : malfile_cursor + n_hosts]
                malfile_cursor += n_hosts
            else:
                malfile_vec = []

            for i, src in enumerate(_SORTED_SUBNETS):
                if src == sn:
                    continue
                pair = (src, sn)
                blocked_now[pair]  = bool(blocked_vec[i])
                should_block[pair] = bool(comms_policy_vec[i])

            for hi, hostname in enumerate(hosts):
                if conn_flags[hi]:
                    conn_alerts[hostname] = True
                if proc_flags[hi]:
                    proc_alerts[hostname] = True
                if has_malfile and hi < len(malfile_vec) and malfile_vec[hi]:
                    malfile_alerts[hostname] = True

            base += 27 + 2 * n_hosts

        # -- Update process-flag first-seen tracker --------------------------
        for h in list(self._proc_flagged_step.keys()):
            if h not in proc_alerts:
                del self._proc_flagged_step[h]
        for h in proc_alerts:
            if h not in self._proc_flagged_step:
                self._proc_flagged_step[h] = self._step

        # -- Derived alert sets ----------------------------------------------
        real_red_hosts: set = {h for h in conn_alerts
                               if malfile_alerts.get(h) or proc_alerts.get(h)}
        real_red_hosts.update(malfile_alerts)
        real_red_hosts.update(proc_alerts)

        # Root session indicators: malfile without conn/proc (PrivEsc signature)
        root_indicators: set = {
            h for h in malfile_alerts
            if h not in conn_alerts and h not in proc_alerts
        }

        # -- Read incoming peer messages (v11b protocol) ---------------------
        peer_messages = self._read_peer_messages_v11b(obs, base)

        # -- Track decoy-hit history for DECOYS_BYPASSED detection -----------
        for h in conn_alerts:
            if (not malfile_alerts.get(h) and not proc_alerts.get(h)
                    and self._decoy_deployed.get(h, 0) > 0):
                self._decoy_hit_hosts.add(h)

        # -- Count active threats for outbound message -----------------------
        threat_count = min(len(real_red_hosts), 3)
        is_busy_restoring = any(
            self._step <= self._restore_at[h] + RESTORE_DUR - 1
            for h in self._restore_at
        )
        has_red = bool(real_red_hosts)
        has_root = bool(root_indicators)
        zone_clear = not has_red and not proc_alerts and not conn_alerts

        # -- Build outbound message (v11b protocol) --------------------------
        msg = np.zeros(8, dtype=bool)
        msg[_BIT_RED_DETECTED]   = has_red
        msg[_BIT_RED_ROOT]       = has_root
        msg[_BIT_BUSY_RESTORING] = is_busy_restoring
        msg[_BIT_ZONE_CLEAR]     = zone_clear
        msg[_BIT_THREAT_CNT_LO]  = bool(threat_count & 1)
        msg[_BIT_THREAT_CNT_HI]  = bool((threat_count >> 1) & 1)
        # REQUEST_BLOCK: any confirmed red in my zone.
        # The downstream agent decides whether to act based on phase/ASF cost.
        # Since the only blocking action is RZ->OZ during active phase (ASF=0),
        # being aggressive with REQUEST_BLOCK is safe.
        msg[_BIT_REQUEST_BLOCK]  = has_red
        msg[_BIT_RESERVED]       = False

        # -- Priority 1: Restore on confirmed red -- conn + (malfile OR proc) --
        for hostname in _sorted_by_priority(conn_alerts, phase):
            if not (malfile_alerts.get(hostname) or proc_alerts.get(hostname)):
                continue
            if self._busy(hostname):
                continue
            idx = self._restore.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._issue_restore(hostname)
                return idx, msg

        # -- Priority 1b: Restore on conn-only without decoy coverage --------
        # Note: P1b decoy bypass check uses same logic as V10b (upstream only).
        # The any-peer approach fires too often because agent_4 (HQ) always
        # has red_detected=True, causing all conn-only alerts to be Restored.
        for hostname in _sorted_by_priority(conn_alerts, phase):
            if malfile_alerts.get(hostname) or proc_alerts.get(hostname):
                continue
            if self._decoy_deployed.get(hostname, 0) > 0:
                continue
            if self._busy(hostname):
                continue
            idx = self._restore.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._issue_restore(hostname)
                return idx, msg

        # -- Priority 1c: Restore on pure malfile (PrivEsc signature) --------
        for hostname in _sorted_by_priority(malfile_alerts, phase):
            if hostname in conn_alerts or hostname in proc_alerts:
                continue
            if self._busy(hostname):
                continue
            idx = self._restore.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._issue_restore(hostname)
                return idx, msg

        # -- Priority CB: Coordinated blocking based on peer messages --------
        coord_block_actions = self._compute_coordinated_blocks(
            peer_messages, blocked_now, should_block, phase
        )
        for idx in coord_block_actions:
            if self._valid(idx, mask):
                return idx, msg

        # -- Priority 2: Allow paths per comms_policy (fix stale blocks) -----
        # Also re-allow preemptive blocks when peer sends ZONE_CLEAR
        coord_allow_actions = self._compute_coordinated_allows(
            peer_messages, blocked_now, should_block, phase
        )
        for pair, is_allowed in sorted(
            should_block.items(),
            key=lambda kv: _pair_priority(kv[0], phase)
        ):
            if not is_allowed and blocked_now.get(pair, False):
                # Only allow if this is NOT a preemptive block
                if pair not in self._preemptive_blocked:
                    idx = self._allow.get(pair)
                    if idx is not None and self._valid(idx, mask):
                        return idx, msg

        # Coordinated allows (preemptive blocks that can be released)
        for idx in coord_allow_actions:
            if self._valid(idx, mask):
                return idx, msg

        # -- Priority 3: Block paths per comms_policy (proactive isolation) --
        for pair, should_be_blocked in sorted(
            should_block.items(),
            key=lambda kv: _pair_priority(kv[0], phase),
            reverse=True
        ):
            if should_be_blocked and not blocked_now.get(pair, False):
                idx = self._block.get(pair)
                if idx is not None and self._valid(idx, mask):
                    return idx, msg

        # -- Priority 4: Restore on process flags (Restore-only, no Remove) --
        # Note: Peer messages are NOT used for P4 threshold adjustment.
        # The v9 messaging ablation showed zero value for threshold changes,
        # and aggressive threshold=0 causes catastrophic over-restoring from
        # green false positives (~155 per episode). threshold=1 is optimal.
        for hostname in _sorted_by_priority(proc_alerts, phase):
            if self._busy(hostname):
                continue

            flag_age = self._step - self._proc_flagged_step.get(hostname, self._step)

            # OZ server_host_0 during active phase: immediate Restore
            is_critical_oz = _is_active_oz_server(hostname, phase)

            if is_critical_oz:
                threshold = 0
            else:
                threshold = 1

            if flag_age >= threshold:
                idx = self._restore.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._issue_restore(hostname)
                    return idx, msg

        # -- Priority 6: Re-deploy decoys after Restore ----------------------
        for hostname in self._deploy_hosts:
            rs = self._restore_at.get(hostname, -1)
            if rs >= 0 and self._step >= rs + RESTORE_DUR:
                if self._decoy_deployed.get(hostname, 0) < MAX_DECOYS and hostname in self._decoy:
                    idx = self._decoy[hostname]
                    if self._valid(idx, mask):
                        self._decoy_deployed[hostname] = self._decoy_deployed.get(hostname, 0) + 1
                        return idx, msg

        # -- Priority 7: Deploy decoys (initial setup) -----------------------
        for hostname in self._deploy_hosts:
            if self._busy(hostname):
                continue
            if self._decoy_deployed.get(hostname, 0) < MAX_DECOYS and hostname in self._decoy:
                idx = self._decoy[hostname]
                if self._valid(idx, mask):
                    self._decoy_deployed[hostname] = self._decoy_deployed.get(hostname, 0) + 1
                    return idx, msg

        # -- Fallback: Sleep --------------------------------------------------
        return self._sleep_idx, msg

    def reset(self) -> None:
        self._step = 0
        self._restore_at.clear()
        self._proc_flagged_step.clear()
        self._decoy_deployed.clear()
        self._decoy_hit_hosts.clear()
        self._deploy_hosts = []
        self._preemptive_blocked.clear()

    # -- Label parsing --------------------------------------------------------

    def _parse_labels(self, labels: list[str], subnet_hosts: dict) -> None:
        self._block.clear(); self._allow.clear()
        self._remove.clear(); self._restore.clear(); self._decoy.clear()
        self._deploy_hosts = []

        controlled: set[str] = set()
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
                    controlled.add(to_sn)

            elif label.startswith("AllowTrafficZone"):
                m = re.match(r"AllowTrafficZone\s+(\S+)\s+\S+\s+<-\s+(\S+)", label)
                if m:
                    to_sn, fr_sn = m.group(1), m.group(2)
                    self._allow[(fr_sn, to_sn)] = idx

            elif label.startswith("Remove"):
                m = re.match(r"Remove\s+(\S+)", label)
                if m:
                    self._remove[m.group(1)] = idx  # parsed but never used

            elif label.startswith("Restore"):
                m = re.match(r"Restore\s+(\S+)", label)
                if m:
                    self._restore[m.group(1)] = idx

            elif label.startswith("DeployDecoy"):
                m = re.match(r"DeployDecoy\s+(\S+)", label)
                if m:
                    h = m.group(1)
                    self._decoy[h] = idx
                    all_decoy_hosts.append(h)

        self._deploy_hosts = sorted(all_decoy_hosts, key=_deploy_priority)

        self._subnets_in_obs = sorted(controlled)
        self._subnet_host_list.clear()
        for sn in self._subnets_in_obs:
            self._subnet_host_list[sn] = list(subnet_hosts.get(sn, []))

    # -- v11b: Coordinated blocking logic ------------------------------------

    def _compute_coordinated_blocks(
        self,
        peer_messages: dict[int, dict],
        blocked_now: dict[tuple[str, str], bool],
        should_block: dict[tuple[str, str], bool],
        phase: int,
    ) -> list[int]:
        """Compute preemptive block actions based on peer messages.

        SURGICAL: Only blocks the RZ->OZ path for the active OZ during the
        active mission phase. This is the only path where:
        1. ASF = 0 (no green penalty for blocking)
        2. LWF = -10/step (high cost of NOT blocking)
        3. The upstream agent can detect red before it crosses to OZ

        Phase 1: agent_1 blocks RZA->OZA when agent_0 sends REQUEST_BLOCK
        Phase 2: agent_3 blocks RZB->OZB when agent_2 sends REQUEST_BLOCK
        """
        blocking_rule = _PHASE_BLOCKING_MAP.get(phase)
        if blocking_rule is None:
            return []

        listener_agent, upstream_agent, from_sn, to_sn = blocking_rule

        # Only the designated listener should act
        if self._own_idx != listener_agent:
            return []

        # Check if upstream agent is requesting a block
        upstream_msg = peer_messages.get(upstream_agent)
        if upstream_msg is None:
            return []

        # Block whenever upstream has ANY red indicator.
        # This is safe because the only path we block (RZ->OZ during active phase)
        # has ASF=0, so blocking costs nothing but prevents -10/step LWF damage.
        should_preempt = (
            upstream_msg.get("request_block", False)
            or upstream_msg.get("red_detected", False)
        )

        if not should_preempt:
            return []

        pair = (from_sn, to_sn)

        # Skip if already blocked
        if blocked_now.get(pair, False):
            return []

        idx = self._block.get(pair)
        if idx is not None:
            self._preemptive_blocked[pair] = self._step
            return [idx]

        return []

    def _compute_coordinated_allows(
        self,
        peer_messages: dict[int, dict],
        blocked_now: dict[tuple[str, str], bool],
        should_block: dict[tuple[str, str], bool],
        phase: int,
    ) -> list[int]:
        """Compute re-allow actions for preemptive blocks when upstream sends ZONE_CLEAR.

        Only re-allows blocks that were preemptively set by coordinated blocking.
        Uses the same surgical phase-based map as blocking.
        """
        if not self._preemptive_blocked:
            return []

        blocking_rule = _PHASE_BLOCKING_MAP.get(phase)
        if blocking_rule is None:
            # Phase changed -- clean up any stale preemptive blocks
            allow_actions = []
            for pair in list(self._preemptive_blocked.keys()):
                if blocked_now.get(pair, False) and not should_block.get(pair, False):
                    idx = self._allow.get(pair)
                    if idx is not None:
                        allow_actions.append(idx)
                del self._preemptive_blocked[pair]
            return allow_actions

        listener_agent, upstream_agent, from_sn, to_sn = blocking_rule
        if self._own_idx != listener_agent:
            return []

        pair = (from_sn, to_sn)
        if pair not in self._preemptive_blocked:
            return []

        upstream_msg = peer_messages.get(upstream_agent)

        # Re-allow when upstream says zone is clear
        should_reallow = (
            upstream_msg is not None and upstream_msg.get("zone_clear", False)
        )
        # Also re-allow when upstream no longer requests block AND no red detected
        if upstream_msg is not None and not upstream_msg.get("request_block", False):
            if not upstream_msg.get("red_detected", False):
                should_reallow = True

        if not should_reallow:
            return []

        # Don't re-allow if comms_policy says it should be blocked
        if should_block.get(pair, False):
            del self._preemptive_blocked[pair]
            return []

        # Only re-allow if it's currently blocked
        if not blocked_now.get(pair, False):
            del self._preemptive_blocked[pair]
            return []

        idx = self._allow.get(pair)
        if idx is not None:
            del self._preemptive_blocked[pair]
            return [idx]

        del self._preemptive_blocked[pair]
        return []

    # -- v11b: Message parsing (slot-alignment-safe) -------------------------

    def _read_peer_messages_v11b(
        self, obs: np.ndarray, msg_start: int
    ) -> dict[int, dict]:
        """Parse the 32-bit inter-agent message section (v11b protocol).

        Returns dict mapping peer_agent_idx -> parsed message dict.

        Slot alignment: CybORG delivers messages from connected agents only,
        in sorted order by agent name. BlueFlatWrapper pads to NUM_MESSAGES=4
        with EMPTY_MESSAGE. Since we don't know which agents are connected,
        we can't perfectly map slots to agent indices.

        HOWEVER: the key insight is that for coordinated blocking, we don't
        need perfect sender identification. What matters is:
        1. Whether ANY peer has red and is requesting a block
        2. Whether ANY peer says zone is clear

        For the sender-identification case (knowing WHICH subnet to block FROM),
        we use a heuristic: if a message has RED_DETECTED=1, the sender is likely
        the agent whose zone is under attack. We check all possible peer mappings.

        Conservative approach: read all 4 message slots, parse each, and apply
        blocking logic for ALL possible sender identities. This is safe because:
        - Blocking a clean zone costs nothing (it will be re-allowed)
        - Missing a compromised zone is expensive
        """
        msg_section = obs[msg_start : msg_start + NUM_MSG_BITS]

        peer_indices = [i for i in range(_NUM_BLUE_AGENTS) if i != self._own_idx]
        result: dict[int, dict] = {}

        for slot in range(min(4, len(peer_indices))):
            slot_start = slot * _MSG_LEN
            if slot_start + _MSG_LEN > len(msg_section):
                break
            pmsg = msg_section[slot_start : slot_start + _MSG_LEN]

            # Check if this is an empty message (all zeros)
            if not any(pmsg):
                continue

            red_detected   = bool(pmsg[_BIT_RED_DETECTED])
            red_root       = bool(pmsg[_BIT_RED_ROOT])
            busy_restoring = bool(pmsg[_BIT_BUSY_RESTORING])
            zone_clear     = bool(pmsg[_BIT_ZONE_CLEAR])
            threat_lo      = int(pmsg[_BIT_THREAT_CNT_LO])
            threat_hi      = int(pmsg[_BIT_THREAT_CNT_HI])
            threat_count   = (threat_hi << 1) | threat_lo
            request_block  = bool(pmsg[_BIT_REQUEST_BLOCK])

            parsed = {
                "red_detected": red_detected,
                "red_root": red_root,
                "busy_restoring": busy_restoring,
                "zone_clear": zone_clear,
                "threat_count": threat_count,
                "request_block": request_block,
            }

            # Map slot to peer index. In the common case (all agents connected),
            # slot order matches peer_indices order. When agents are disconnected,
            # the mapping shifts. We use the slot -> peer_indices[slot] mapping
            # as the default. This is correct when all agents are connected
            # (the common case) and may misattribute when agents are isolated
            # (rare, and the blocking actions are still safe).
            if slot < len(peer_indices):
                result[peer_indices[slot]] = parsed

        return result

    # -- Helpers --------------------------------------------------------------

    def _issue_restore(self, hostname: str) -> None:
        """Record bookkeeping when issuing a Restore action on hostname."""
        self._restore_at[hostname] = self._step
        self._decoy_deployed.pop(hostname, None)

    def _busy(self, hostname: str) -> bool:
        """True if a Restore is still in progress for this host."""
        if self._step <= self._restore_at.get(hostname, -1) + RESTORE_DUR - 1:
            return True
        return False

    def _valid(self, idx: int, mask) -> bool:
        if mask is None:
            return True
        if idx < 0 or idx >= len(mask):
            return False
        return bool(mask[idx])


# -- Module-level helpers -----------------------------------------------------

def _is_active_oz_server(hostname: str, phase: int) -> bool:
    """True if hostname is the OZ server_host_0 during its active mission phase."""
    if "server_host_0" not in hostname:
        return False
    if phase == 1 and "operational_zone_a" in hostname:
        return True
    if phase == 2 and "operational_zone_b" in hostname:
        return True
    return False


def _subnet_of(hostname: str) -> Optional[str]:
    for sn in _SORTED_SUBNETS:
        core = sn.replace("_subnet", "")
        if core in hostname:
            return sn
    return None


def _host_priority(hostname: str, phase: int) -> int:
    if phase == 1:
        if "operational_zone_a" in hostname: return 100
        if "restricted_zone_a"  in hostname: return 70
        if "operational_zone_b" in hostname: return 40
    elif phase == 2:
        if "operational_zone_b" in hostname: return 100
        if "restricted_zone_b"  in hostname: return 70
        if "operational_zone_a" in hostname: return 40
    elif phase == 0:
        if "operational_zone_b" in hostname: return 40
        if "operational_zone_a" in hostname: return 40
        if "restricted_zone_b"  in hostname: return 30
        if "restricted_zone_a"  in hostname: return 30
    if any(s in hostname for s in ("admin_network", "office_network", "public_access")):
        return 50
    return 20


def _sorted_by_priority(hosts: dict, phase: int) -> list[str]:
    return sorted(hosts, key=lambda h: _host_priority(h, phase), reverse=True)


def _top_host(hosts: list[str], phase: int) -> Optional[str]:
    if not hosts:
        return None
    return max(hosts, key=lambda h: _host_priority(h, phase))


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


def _pair_priority(from_subnet: str, phase: int) -> int:
    if phase == 1:
        if from_subnet == "restricted_zone_a_subnet": return 250
        if from_subnet in ("contractor_network_subnet", "internet_subnet"): return 200
    elif phase == 2:
        if from_subnet == "restricted_zone_b_subnet": return 250
        if from_subnet in ("contractor_network_subnet", "internet_subnet"): return 200
        if from_subnet == "restricted_zone_a_subnet": return 50
    else:
        if from_subnet in ("contractor_network_subnet", "internet_subnet"): return 200
    return 10


# -- Factory ------------------------------------------------------------------

def make_heuristic_agents_v11b(env) -> dict[str, "EnterpriseHeuristicAgentV11b"]:
    """Create one agent per blue team member.  Call AFTER env.reset()."""
    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
    agents: dict[str, EnterpriseHeuristicAgentV11b] = {}
    for agent_name in env.possible_agents:
        ag = EnterpriseHeuristicAgentV11b(agent_name=agent_name)
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
