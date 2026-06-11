"""Optimal heuristic blue agents for CAGE Challenge 4 -- v9.1.

v9.1 fixes:
  - Clear _remove_at on Restore (host reimaged, Remove history stale)
  - Phase 0 host priorities (OZ=40, RZ=30)
  - Phase 2 _pair_priority fallback for restricted_zone_a (priority 50)
  - Default escalation threshold lowered from 2 to 1 (faster Remove->Restore)

Key insights from full code-level analysis:
------------------------------------------------------------------------------
1. BlockTrafficZone DOES stop ExploitRemoteService via blocking_host() check.
   PhishingEmail bypasses blocks (uses check_routable / connected_components).

2. comms_policy_subvector in obs is INVERTED:
     1 = this source subnet SHOULD BE BLOCKED from this controlled subnet
     0 = allowed (do not block or actively unblock)

   Phase 0 (Preplanning): full graph -- minimal blocks needed
   Phase 1 (MissionA):    rza isolated -- block contractor/internet/rzb/oza<->rza
   Phase 2 (MissionB):    rzb isolated -- block contractor/internet/rza/oza<->rzb

3. Observation layout (BlueFlatWrapperV2):
     obs[0]                  : mission_phase
     Per subnet (variable, n_hosts actual hosts):
       [+0 :+9 ]  subnet one-hot (9 subnets sorted alphabetically)
       [+9 :+18]  blocked_subvector  (1 = subnet i currently blocked TO here)
       [+18:+27]  comms_policy       (1 = subnet i SHOULD BE BLOCKED)
       [+27:+27+n]  process_flags    (n hosts, 1 = has process/malware event)
       [+27+n:+27+2n] connection_flags (n hosts, 1 = has network connection event)
     obs[base:base+32]     : inter-agent messages (4 x 8 bits), where
                             base = 1 + sum(27 + 2*n_hosts_i)
     obs[base+32:]         : malfile flags (M = sum of n_hosts across subnets)
                             1 = host has density>=0.9 unsigned file (red artifact)

   When using plain BlueFlatWrapper (no malfile section), M=0.
   The agent auto-detects which wrapper is in use.

4. Malfile flag properties (BlueFlatWrapperV2 only):
     ExploitRemoteService ALWAYS drops cmd.exe/sh (density=0.9, unsigned),
       including the 5% of exploits that fire no process_creation event.
     PrivilegeEscalate ALWAYS drops escalate.exe/sh (density=0.9, unsigned)
       and fires NO events -- completely invisible without malfile flag.
     Remove NOW clears malware files (density>=0.9 unsigned) as well as killing
       suspicious processes. Root sessions survive Remove even after file removal.
     Green agent actions NEVER drop high-density files -> zero FP rate.

5. Remove clears user sessions ONLY. Root sessions survive Remove.
   Restore clears ALL sessions and reimages host (-1 reward cost, 5 steps).
   Also wipes all deployed decoys (host reimaged to original state).
   DegradeServices is PERMANENT -- only Restore fixes service reliability.

6. OTService only on operational_zone_a/b server_host_0 -- Impact target.

7. Decoy mechanics (v7 key addition):
     DeployDecoy creates a fake vulnerable process (Apache/Tomcat/Haraka/Vsftpd)
       on the host with decoy_type=EXPLOIT.
     When red ExploitRemoteService hits a decoy:
       - Exploit FAILS silently (no session created)
       - A network_connection event IS logged (detectable by blue)
       - Red wastes 4 steps with zero progress
     When red DiscoverDeception on a host:
       - 50% TP rate: decoy correctly identified → red avoids that PID
       - 10% FP rate: legitimate process incorrectly flagged
       - Duration: 2 steps; result: intelligence only (does NOT prevent exploit)
     With 3 decoys + 1 legitimate service: 75% of red's blind exploit attempts fail.

8. Decoy-hit detection (BlueFlatWrapperV2 only):
     Real exploit:  conn_flag=1 AND malfile=1 (cmd.exe always dropped)
     Decoy hit:     conn_flag=1 AND malfile=0 AND proc_flag=0
     Silent exploit: conn_flag=1 AND malfile=1 AND proc_flag=0 (5% case)
     With V2 wrapper + decoys deployed: conn_only + malfile=0 = decoy hit
       -> Skip Restore (decoy defending, no red session to evict).
     Without decoys: conn_only is suspicious -> Restore for safety.

9. Inter-agent messaging (v9 — zero redundancy, 100% bit utilisation):
     Blue agents send 8-bit messages per step; each receives 4 messages from
     peer agents ordered by agent index (sorted, excluding self).  OZA/OZB
     become isolated once Phase 1/2 blocks are in place — their message slots
     are zero-padded in BlueFlatWrapperV2 to maintain the fixed 32-bit length.

     Outgoing message format:
       Bits 0-1  THREAT_LEVEL    2-bit: 0=clean 1=decoy_hit 2=user_session 3=root
       Bits 2-3  OPEN_PATHS      2-bit: count of unblocked required comms paths (0-3+)
       Bits 4-5  RED_HOST_COUNT  2-bit: count of hosts with confirmed red presence (0-3+)
       Bit  6    DECOYS_BYPASSED red has PID knowledge (decoy hit then real exploit seen)
       Bit  7    RESTORING       at least one Restore in progress in my zone

     How receivers use messages (3-tier Priority 4 escalation):
       - T3 (upstream_red_count >= 3): direct Restore on first proc_flag (threshold=0)
       - T2 (any_root OR upstream_threat>=2): lower threshold to flag_age>=1
       - T2 compound (upstream_threat>=1 AND open_paths>0 AND decoys_bypassed):
           Red is advancing through an open path with bypassed decoys → threshold=1
       - Priority 1b: if upstream DECOYS_BYPASSED, don't skip Restore on conn-only
           hits even when own decoys are deployed (red may have PID knowledge).

Optimal strategy (v9):
------------------------------------------------------------------------------
Priority 1:  RESTORE on connection flags WITH malfile OR proc_flag.
Priority 1b: RESTORE on conn-only when NO decoy deployed (decoy hit otherwise).
Priority 1c: RESTORE on pure malfile (no proc/conn) = PrivilegeEscalate signature.
Priority 2:  ALLOW per comms_policy (fix stale blocks).
Priority 3:  BLOCK per comms_policy (proactive isolation).
Priority 4:  REMOVE on process flags; skip to Restore when peer_root or upstream
             advanced (flag_age threshold lowered from 2 to 1).
Priority 5:  RESTORE persistent process flags (no Remove issued yet).
Priority 6:  REDEPLOY DECOYS after Restore.
Priority 7:  DEPLOY DECOYS on all hosts (OTService first).
Priority 8:  SLEEP.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

# -- Constants ----------------------------------------------------------------

# All 9 subnets sorted alphabetically (matches observation one-hot / index order)
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
NUM_SUBNETS  = 9    # one-hot + blocked + comms_policy dimensions
MAX_HOSTS    = 16   # maximum padded process/connection flag dimensions
NUM_MSG_BITS = 32   # 4 messages x 8 bits total

# Offsets within each per-subnet obs block (relative to block base)
_OFF_BLOCKED  = NUM_SUBNETS        # 9
_OFF_COMMS    = NUM_SUBNETS * 2    # 18
_OFF_PROC     = NUM_SUBNETS * 3    # 27

# Action durations
REMOVE_DUR  = 3
RESTORE_DUR = 5

# Deploy up to MAX_DECOYS decoy services per host.
# With 3 decoys + 1 legitimate service, 75% of red's blind exploit attempts fail.
# Red must either spend 2 steps on DiscoverDeception (50% miss) or waste 4 steps.
MAX_DECOYS = 3


# -- Messaging constants ------------------------------------------------------

# Bits per single agent message (= MESSAGE_LENGTH in BlueFixedActionWrapper)
_MSG_LEN = 8

# Total number of blue agents in CC4 (blue_agent_0 .. blue_agent_4)
_NUM_BLUE_AGENTS = 5

# Primary controlled subnet for each blue agent index.
# Used by receivers to interpret what zone a peer's threat level refers to.
_AGENT_PRIMARY_SUBNET = {
    0: "restricted_zone_a_subnet",
    1: "operational_zone_a_subnet",
    2: "restricted_zone_b_subnet",
    3: "operational_zone_b_subnet",
    4: None,   # HQ: controls PAZ + admin + office
}

# Phase-aware upstream agent: in phase P, agent U is "upstream" of agent D
# meaning red must compromise U's subnet before reaching D's subnet.
# Key = (phase, downstream_agent_idx) → upstream_agent_idx
# v10: Added Phase 0 mappings. Red enters via contractor, pivots through RZ
# to OZ. In Phase 0 all subnets are connected, so RZ agents are upstream of
# OZ agents. This enables T3 escalation when a peer zone is saturated,
# addressing the 58% Phase 0 loss gap (FM-12).
_UPSTREAM = {
    (1, 1): 0,   # Phase 1: RZA (agent_0) is upstream of OZA (agent_1)
    (2, 3): 2,   # Phase 2: RZB (agent_2) is upstream of OZB (agent_3)
}

# Outgoing message bit indices — v9 protocol (zero redundancy, all bits consumed)
_BIT_THREAT_LO       = 0   # low bit  of 2-bit threat level
_BIT_THREAT_HI       = 1   # high bit of 2-bit threat level
_BIT_OPEN_PATHS_LO   = 2   # low bit  of 2-bit open comms paths count
_BIT_OPEN_PATHS_HI   = 3   # high bit of 2-bit open comms paths count
_BIT_RED_COUNT_LO    = 4   # low bit  of 2-bit compromised host count
_BIT_RED_COUNT_HI    = 5   # high bit of 2-bit compromised host count
_BIT_DECOYS_BYPASSED = 6   # red has PID knowledge of my decoys (bypassed decoy hit → real exploit)
_BIT_RESTORING       = 7   # at least one Restore in progress in my zone
# Threat level:  0=clean  1=decoy_hit  2=user_session  3=root_session
# Open paths:    0=all blocked  1=one open  2=two open  3=three or more
# Red host count: 0=none  1=one  2=two  3=three or more


class EnterpriseHeuristicAgent:
    """Phase-aware, comms-policy-driven heuristic blue agent for CC4 v6.

    Compatible with both BlueFlatWrapper and BlueFlatWrapperV2.
    When used with BlueFlatWrapperV2 the malfile flags enable detection of:
      - 5% silent ExploitRemoteService exploits
      - PrivilegeEscalate (previously completely invisible)

    Call after each env.reset():
        agent.set_action_info(labels, mask, subnet_hosts)
    """

    def __init__(self, agent_name: str = "blue_agent_0"):
        self.agent_name = agent_name

        # Action catalogues (populated by _parse_labels)
        self._sleep_idx: int  = 0
        self._block:  dict[tuple[str, str], int] = {}  # (from, to) -> idx
        self._allow:  dict[tuple[str, str], int] = {}  # (from, to) -> idx
        self._remove: dict[str, int]  = {}
        self._restore: dict[str, int] = {}
        self._decoy:  dict[str, int]  = {}

        # Observation layout (derived from action labels + subnet_hosts)
        self._subnets_in_obs:   list[str]       = []   # alphabetically sorted controlled subnets
        self._subnet_host_list: dict[str, list] = {}   # subnet -> ordered host list (matches obs)

        # Decoy state
        self._deploy_hosts:    list[str] = []   # ALL hosts eligible for decoys, sorted by priority
        self._decoy_deployed:  dict[str, int] = {}   # hostname -> decoy count (max MAX_DECOYS)

        # Episode state tracking
        self._step:       int              = 0
        self._remove_at:  dict[str, int]   = {}
        self._restore_at: dict[str, int]   = {}
        self._proc_flagged_step: dict[str, int] = {}  # step when process flag first appeared
        self._decoy_hit_hosts:  set            = set()  # hosts where conn-only+decoy hit was seen

        self._labels: list[str] = []

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
        # BlueFlatWrapperV2 appends M extra bits at the end (one per host).
        # BlueFlatWrapper has no such section.
        n_malfile_hosts = sum(
            len(self._subnet_host_list.get(sn, []))
            for sn in self._subnets_in_obs
        )
        # Expected base length (BlueFlatWrapper without malfile):
        #   1 + sum(27 + 2*n_i) + 32
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
        conn_alerts   = {}   # hostname -> True  (connection flag set)
        proc_alerts   = {}   # hostname -> True  (process flag set)
        malfile_alerts = {}  # hostname -> True  (density>=0.9 unsigned file)
        blocked_now   = {}   # (from, to) -> bool
        should_block  = {}   # (from, to) -> bool (comms_policy says block)

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

            # Malfile flags (empty array when using plain BlueFlatWrapper)
            if has_malfile:
                malfile_vec = obs[malfile_cursor : malfile_cursor + n_hosts]
                malfile_cursor += n_hosts
            else:
                malfile_vec = []

            # Record block state and comms_policy for each source subnet
            for i, src in enumerate(_SORTED_SUBNETS):
                if src == sn:
                    continue
                pair = (src, sn)
                blocked_now[pair]  = bool(blocked_vec[i])
                should_block[pair] = bool(comms_policy_vec[i])

            # Collect host-level alerts
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

        # -- Derived alert sets used by both messaging and priorities ---------
        # Hosts with any confirmed real red signal (not just decoy hits)
        real_red_hosts: set = {h for h in conn_alerts
                               if malfile_alerts.get(h) or proc_alerts.get(h)}
        real_red_hosts.update(malfile_alerts)
        real_red_hosts.update(proc_alerts)

        # Root session indicators: malfile without conn/proc (PrivEsc), or proc
        # that survived Remove (root session survived cleanup).
        root_indicators: set = {
            h for h in malfile_alerts
            if h not in conn_alerts and h not in proc_alerts
        }
        for h in proc_alerts:
            ra = self._remove_at.get(h, -1)
            if ra >= 0 and self._step > ra + REMOVE_DUR:
                root_indicators.add(h)

        # -- Read incoming peer messages (msg section sits at obs[base:base+32]) -
        # `base` now points to the start of the message section (after subnet blocks).
        peer_state = self._read_peer_messages(obs, base, phase)

        # -- Track decoy-hit history for DECOYS_BYPASSED detection -----------
        # conn-only (no malfile, no proc) with decoys deployed = decoy hit.
        # Record the host. If that host later shows conn+malfile = red bypassed decoys.
        for h in conn_alerts:
            if (not malfile_alerts.get(h) and not proc_alerts.get(h)
                    and self._decoy_deployed.get(h, 0) > 0):
                self._decoy_hit_hosts.add(h)
        decoys_bypassed = any(
            h in self._decoy_hit_hosts
            for h in real_red_hosts
            if self._decoy_deployed.get(h, 0) > 0
        )

        # -- Compute open comms paths and red host counts ---------------------
        open_paths_count = sum(
            1 for p in should_block
            if should_block[p] and not blocked_now.get(p, False)
        )
        open_paths_enc = min(open_paths_count, 3)
        red_count_enc  = min(len(real_red_hosts), 3)

        # -- Build outbound message (v9 protocol — zero redundancy) ----------
        msg = np.zeros(8, dtype=bool)

        # Bits 0-1 THREAT_LEVEL: worst threat in my zone
        # 0=clean  1=decoy_hit  2=user_session  3=root_session
        if root_indicators:
            out_threat = 3
        elif real_red_hosts:
            out_threat = 2
        elif any(
            conn_alerts.get(h) and not malfile_alerts.get(h)
            and not proc_alerts.get(h) and self._decoy_deployed.get(h, 0) > 0
            for h in conn_alerts
        ):
            out_threat = 1
        else:
            out_threat = 0
        msg[_BIT_THREAT_LO] = bool(out_threat & 1)
        msg[_BIT_THREAT_HI] = bool((out_threat >> 1) & 1)

        # Bits 2-3 OPEN_PATHS: unblocked required comms paths (0-3+)
        msg[_BIT_OPEN_PATHS_LO] = bool(open_paths_enc & 1)
        msg[_BIT_OPEN_PATHS_HI] = bool((open_paths_enc >> 1) & 1)

        # Bits 4-5 RED_HOST_COUNT: hosts with confirmed red presence (0-3+)
        msg[_BIT_RED_COUNT_LO] = bool(red_count_enc & 1)
        msg[_BIT_RED_COUNT_HI] = bool((red_count_enc >> 1) & 1)

        # Bit 6 DECOYS_BYPASSED: red has PID knowledge of my decoys
        msg[_BIT_DECOYS_BYPASSED] = decoys_bypassed

        # Bit 7 RESTORING: at least one Restore in progress in my zone
        msg[_BIT_RESTORING] = any(
            self._step <= self._restore_at[h] + RESTORE_DUR - 1
            for h in self._restore_at
        )

        # -- Priority 1: Restore on confirmed red -- conn + (malfile OR proc) --
        # malfile=1: real exploit (cmd.exe dropped). proc_flag=1: process event.
        # Both together or separately = real red session -> Restore immediately.
        for hostname in _sorted_by_priority(conn_alerts, phase):
            if not (malfile_alerts.get(hostname) or proc_alerts.get(hostname)):
                continue  # conn-only, may be decoy hit -- handled below
            if self._busy(hostname):
                continue
            idx = self._restore.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._issue_restore(hostname)
                return idx, msg

        # -- Priority 1b: Restore on conn-only events WITHOUT decoy coverage --
        # conn_flag=1 AND malfile=0 AND proc_flag=0:
        #   - If a decoy IS deployed AND upstream decoys haven't been bypassed:
        #     almost certainly a decoy hit. Skip Restore (save 5 blue steps).
        #   - If upstream reports DECOYS_BYPASSED: red has PID knowledge; even a
        #     conn-only hit may be a real exploit. Don't skip Restore.
        #   - If NO decoy deployed: unknown source; Restore for safety.
        upstream_decoys_compromised = peer_state.get("upstream_decoys_bypassed", False)
        for hostname in _sorted_by_priority(conn_alerts, phase):
            if malfile_alerts.get(hostname) or proc_alerts.get(hostname):
                continue  # already handled by P1
            if (self._decoy_deployed.get(hostname, 0) > 0
                    and not upstream_decoys_compromised):
                continue  # decoy deployed + no bypass reported -> likely decoy hit
            if self._busy(hostname):
                continue
            idx = self._restore.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._issue_restore(hostname)
                return idx, msg

        # -- Priority 1c: Restore on pure malfile (no proc/conn events) --------
        # malfile=1 AND no proc/conn events = PrivilegeEscalate signature.
        # PrivEscalate fires NO process or connection events — the only indicator
        # is the malware file drop (density=0.9 unsigned). Root session present.
        # Remove cannot evict root sessions; Restore is the only remedy.
        for hostname in _sorted_by_priority(malfile_alerts, phase):
            if hostname in conn_alerts or hostname in proc_alerts:
                continue  # handled by P1 (conn) or P4 (proc + malfile together)
            if self._busy(hostname):
                continue
            idx = self._restore.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._issue_restore(hostname)
                return idx, msg

        # -- Priority 2: Allow paths per comms_policy (fix stale blocks) ----
        for pair, is_allowed in sorted(
            should_block.items(),
            key=lambda kv: _pair_priority(kv[0], phase)
        ):
            if not is_allowed and blocked_now.get(pair, False):
                idx = self._allow.get(pair)
                if idx is not None and self._valid(idx, mask):
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

        # -- Priority 4: Remove on process flags ---------------------------------
        # proc_flag alone has 33-54% FP rate (green). Remove is cheap (3 steps,
        # no reward cost). Remove now also clears malware files, so malfile=1 AND
        # proc_flag=1 (ExploitRemoteService with events) is handled here.
        # If flag persists after Remove, root session is present -> Restore.
        for hostname in _sorted_by_priority(proc_alerts, phase):
            if self._busy(hostname):
                continue
            ra = self._remove_at.get(hostname, -1)
            if ra >= 0:
                # Already tried Remove. Any re-flag after Remove = root session.
                if self._step > ra:
                    idx = self._restore.get(hostname)
                    if idx is not None and self._valid(idx, mask):
                        self._issue_restore(hostname)
                        return idx, msg
                continue

            # High-priority hosts, sticky flags, or peer escalation -> skip Remove
            # 2-tier escalation based on v9 peer message fields:
            #   T3 (threshold 0): upstream zone has 3+ compromised hosts -> Restore immediately
            #   default (threshold 1): Restore if proc_flag persists 1+ step.
            flag_age = self._step - self._proc_flagged_step.get(hostname, self._step)
            peer_escalate_t3 = peer_state.get("upstream_red_count", 0) >= 3
            peer_escalate_t1 = (
                peer_state.get("any_root", False)
                or peer_state.get("upstream_threat", 0) >= 2
            )
            peer_escalate_t2 = (
                peer_state.get("upstream_threat", 0) >= 1
                and peer_state.get("upstream_open_paths", 0) > 0
                and peer_state.get("upstream_decoys_bypassed", False)
            )
            if peer_escalate_t3:
                direct_restore_threshold = 0
            elif peer_escalate_t1 or peer_escalate_t2:
                direct_restore_threshold = 1
            else:
                direct_restore_threshold = 1
            if _host_priority(hostname, phase) >= 100 or flag_age >= direct_restore_threshold:
                idx = self._restore.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._issue_restore(hostname)
                    return idx, msg
            else:
                idx = self._remove.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._remove_at[hostname] = self._step
                    return idx, msg

        # -- Priority 5: Restore for persistent proc flags (no Remove issued yet) -
        for hostname in _sorted_by_priority(proc_alerts, phase):
            if self._busy(hostname):
                continue
            if self._remove_at.get(hostname, -1) < 0 and self._restore_at.get(hostname, -1) < 0:
                idx = self._restore.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._issue_restore(hostname)
                    return idx, msg

        # -- Priority 6: Re-deploy decoys after Restore ----------------------
        # Restore wipes all decoys (host reimaged). Re-arm immediately.
        # Ordered by deploy_priority so critical hosts (OTService) recover first.
        for hostname in self._deploy_hosts:
            rs = self._restore_at.get(hostname, -1)
            if rs >= 0 and self._step >= rs + RESTORE_DUR:
                if self._decoy_deployed.get(hostname, 0) < MAX_DECOYS and hostname in self._decoy:
                    idx = self._decoy[hostname]
                    if self._valid(idx, mask):
                        self._decoy_deployed[hostname] = self._decoy_deployed.get(hostname, 0) + 1
                        return idx, msg

        # -- Priority 7: Deploy decoys (initial setup, all hosts) ------------
        # Deploy MAX_DECOYS per host in priority order: OTService hosts first,
        # then servers, then user hosts. Replaces Sleep when no alert is active.
        # With 3 decoys + 1 legitimate service, red wastes 75% of blind exploit
        # attempts (4 steps each) or must spend 2 steps on DiscoverDeception.
        # _busy() guard prevents deploying on a host mid-Remove or mid-Restore
        # (the decoy would be wiped by Restore anyway, wasting the action).
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
        self._remove_at.clear()
        self._restore_at.clear()
        self._proc_flagged_step.clear()
        self._decoy_deployed.clear()
        self._decoy_hit_hosts.clear()
        self._deploy_hosts = []

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
                    self._remove[m.group(1)] = idx

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

        # Sort ALL decoy hosts by priority: OTService targets first, then servers,
        # then user hosts. This ensures critical hosts are protected before idle
        # steps run out in short episodes.
        self._deploy_hosts = sorted(all_decoy_hosts, key=_deploy_priority)

        # Build observation layout (alphabetically sorted controlled subnets)
        self._subnets_in_obs = sorted(controlled)
        self._subnet_host_list.clear()
        for sn in self._subnets_in_obs:
            self._subnet_host_list[sn] = list(subnet_hosts.get(sn, []))

    # -- Helpers --------------------------------------------------------------

    def _issue_restore(self, hostname: str) -> None:
        """Record bookkeeping when issuing a Restore action on hostname."""
        self._restore_at[hostname] = self._step
        self._remove_at.pop(hostname, None)
        self._decoy_deployed.pop(hostname, None)

    def _busy(self, hostname: str) -> bool:
        """True if a Remove or Restore is still in progress for this host."""
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

    def _read_peer_messages(self, obs: np.ndarray, msg_start: int, phase: int) -> dict:
        """Parse the 32-bit inter-agent message section into actionable state (v9).

        Message slots are ordered by sender index (ascending, excluding self).
        If a peer is network-isolated (e.g. OZA in Phase 1), its slot is zero.

        Returns
        -------
        dict with keys:
          any_real_red          (bool):  any peer has threat_level >= 2 (user/root)
          any_root              (bool):  any peer has threat_level == 3 (root)
          upstream_threat       (int):   threat level 0-3 from upstream peer
          upstream_open_paths   (int):   unblocked comms paths 0-3 from upstream
          upstream_red_count    (int):   compromised host count 0-3 from upstream
          upstream_decoys_bypassed (bool): upstream peer's decoys have been bypassed
          upstream_restoring    (bool):  upstream peer has a Restore in progress
          max_peer_red_count    (int):   highest red_count reported by any peer
        """
        msg_section = obs[msg_start : msg_start + NUM_MSG_BITS]

        try:
            own_idx = int(self.agent_name.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return {
                "any_real_red": False, "any_root": False,
                "upstream_threat": 0, "upstream_open_paths": 0,
                "upstream_red_count": 0, "upstream_decoys_bypassed": False,
                "upstream_restoring": False, "max_peer_red_count": 0,
            }

        # Peer slots: sorted blue_agent indices excluding self
        peer_indices = [i for i in range(_NUM_BLUE_AGENTS) if i != own_idx]

        upstream_idx = _UPSTREAM.get((phase, own_idx))

        any_real_red             = False
        any_root                 = False
        upstream_threat          = 0
        upstream_open_paths      = 0
        upstream_red_count       = 0
        upstream_decoys_bypassed = False
        upstream_restoring       = False
        max_peer_red_count       = 0

        for slot, peer_idx in enumerate(peer_indices):
            slot_start = slot * _MSG_LEN
            if slot_start + _MSG_LEN > len(msg_section):
                break
            pmsg = msg_section[slot_start : slot_start + _MSG_LEN]

            threat_lo    = int(pmsg[_BIT_THREAT_LO])
            threat_hi    = int(pmsg[_BIT_THREAT_HI])
            threat_level = (threat_hi << 1) | threat_lo

            open_lo      = int(pmsg[_BIT_OPEN_PATHS_LO])
            open_hi      = int(pmsg[_BIT_OPEN_PATHS_HI])
            open_paths   = (open_hi << 1) | open_lo

            red_lo       = int(pmsg[_BIT_RED_COUNT_LO])
            red_hi       = int(pmsg[_BIT_RED_COUNT_HI])
            red_count    = (red_hi << 1) | red_lo

            decoys_byp   = bool(pmsg[_BIT_DECOYS_BYPASSED])
            restoring    = bool(pmsg[_BIT_RESTORING])

            if threat_level >= 2:
                any_real_red = True
            if threat_level == 3:
                any_root = True

            if red_count > max_peer_red_count:
                max_peer_red_count = red_count

            if peer_idx == upstream_idx:
                upstream_threat          = threat_level
                upstream_open_paths      = open_paths
                upstream_red_count       = red_count
                upstream_decoys_bypassed = decoys_byp
                upstream_restoring       = restoring

        return {
            "any_real_red":             any_real_red,
            "any_root":                 any_root,
            "upstream_threat":          upstream_threat,
            "upstream_open_paths":      upstream_open_paths,
            "upstream_red_count":       upstream_red_count,
            "upstream_decoys_bypassed": upstream_decoys_bypassed,
            "upstream_restoring":       upstream_restoring,
            "max_peer_red_count":       max_peer_red_count,
        }


# -- Module-level helpers -----------------------------------------------------

def _subnet_of(hostname: str) -> Optional[str]:
    """Return the subnet name that matches this hostname (by substring)."""
    for sn in _SORTED_SUBNETS:
        core = sn.replace("_subnet", "")
        if core in hostname:
            return sn
    return None


def _host_priority(hostname: str, phase: int) -> int:
    """Priority score for alert response -- higher = more urgent."""
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
    """Return hostnames sorted by descending priority."""
    return sorted(hosts, key=lambda h: _host_priority(h, phase), reverse=True)


def _top_host(hosts: list[str], phase: int) -> Optional[str]:
    if not hosts:
        return None
    return max(hosts, key=lambda h: _host_priority(h, phase))


def _deploy_priority(hostname: str) -> int:
    """Deployment order for decoys.

    OTService hosts first (red's Impact targets), then other servers in active
    mission zones, then all remaining hosts. Lower number = higher priority.
    """
    # OTService targets (Impact destination) -- highest priority
    if "operational_zone_b" in hostname and "server_host_0" in hostname: return 0
    if "operational_zone_a" in hostname and "server_host_0" in hostname: return 1
    # Other OZ servers
    if "operational_zone_b" in hostname and "server" in hostname: return 2
    if "operational_zone_a" in hostname and "server" in hostname: return 3
    # Restricted zone servers (red must pass through)
    if "restricted_zone_b" in hostname and "server" in hostname: return 4
    if "restricted_zone_a" in hostname and "server" in hostname: return 5
    # Any other server
    if "server_host_0" in hostname: return 6
    if "server" in hostname: return 7
    # User hosts in high-value zones
    if "operational_zone_b" in hostname: return 8
    if "operational_zone_a" in hostname: return 9
    if "restricted_zone_b" in hostname: return 10
    if "restricted_zone_a" in hostname: return 11
    # All remaining user/HQ hosts
    return 20


def _pair_priority(from_subnet: str, phase: int) -> int:
    """Sort priority for Allow/Block actions.

    v10: Prioritize the direct RZ->OZ attack path (priority 250) over
    contractor/internet entry points (200) during mission phases. The most
    critical block at phase transition is RZ->OZ which cuts the direct
    lateral movement path to mission-critical hosts. Red may already be in
    RZ by the time Phase 1/2 starts.
    """
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

def make_heuristic_agents(env) -> dict[str, "EnterpriseHeuristicAgent"]:
    """Create one agent per blue team member.  Call AFTER env.reset()."""
    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
    agents: dict[str, EnterpriseHeuristicAgent] = {}
    for agent_name in env.possible_agents:
        ag = EnterpriseHeuristicAgent(agent_name=agent_name)
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


# -- Type hint compatibility --------------------------------------------------
from typing import Optional  # noqa: E402 (already imported above; idempotent)
