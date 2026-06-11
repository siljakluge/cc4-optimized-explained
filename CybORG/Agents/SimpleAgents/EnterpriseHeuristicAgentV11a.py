"""Optimal heuristic blue agents for CAGE Challenge 4 -- v11a (Preemptive OZ Blocking).

v11a: Preemptive blocking of paths to OZ subnets before active mission phases.

  Diagnostic findings:
    - Phase 1 comms_policy blocks ALL 8 paths to OZA (including RZA->OZA).
    - Phase 2 comms_policy blocks ALL 8 paths to OZB (including RZB->OZB).
    - In Phase 0, only RZA->OZA is open to OZA. In Phase 1, only RZB->OZB is open to OZB.
    - The V10b agent executes comms_policy blocks via P3, which can be delayed by
      higher-priority Restore actions. This delay leaves paths open at phase transitions.

  Strategy:
    1. PREEMPTIVE BLOCK: Near the end of Phase 0 (before step 167), preemptively
       block RZA->OZA so it's already blocked when Phase 1 starts.
    2. PREEMPTIVE BLOCK: Near the end of Phase 1 (before step 334), preemptively
       block RZB->OZB so it's already blocked when Phase 2 starts.
    3. HIGH-PRIORITY BLOCK: During active phases, elevate blocking of the active
       OZ's remaining open paths above Allow priority (new P_BLOCK_OZ between
       P1c and P2), since ASF=0 for active OZ.
    4. TOTAL LOCKDOWN: During active phases, if any path to the active OZ is
       somehow open and NOT mandated by comms_policy, block it anyway (ASF=0).

  Phase 0 preemptive block cost: ASF=-1/step * ~5 steps = -5 total.
  Phase 1 preemptive block cost: ASF=0 for OZB? No -- OZB is not the active
    target in Phase 1. OZB has ASF=-1 in Phase 1. Cost: ~5 steps * -1 = -5.
  Benefit: prevents red from exploiting through RZ->OZ during the critical
    first few steps of the active phase when the agent may be busy Restoring.

  Changes from v10b:
    - New P_BLOCK_OZ priority between P1c and P2: blocks all open paths to
      active OZ subnet during active mission phase.
    - Preemptive blocking near phase transitions: blocks the soon-to-be-critical
      RZ->OZ path before the phase starts.
    - Tracks preemptive blocks to Allow them if no longer needed.

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

# -- Phase transition steps ---------------------------------------------------
# Phase 0: steps 0-166, Phase 1: steps 167-333, Phase 2: steps 334-499
PHASE_1_START = 167
PHASE_2_START = 334

# How many steps before phase transition to start preemptive blocking
PREEMPTIVE_WINDOW = 10

# -- Messaging constants ------------------------------------------------------

_MSG_LEN = 8
_NUM_BLUE_AGENTS = 5

_AGENT_PRIMARY_SUBNET = {
    0: "restricted_zone_a_subnet",
    1: "operational_zone_a_subnet",
    2: "restricted_zone_b_subnet",
    3: "operational_zone_b_subnet",
    4: None,
}

_UPSTREAM = {
    (1, 1): 0,
    (2, 3): 2,
}

_BIT_THREAT_LO       = 0
_BIT_THREAT_HI       = 1
_BIT_OPEN_PATHS_LO   = 2
_BIT_OPEN_PATHS_HI   = 3
_BIT_RED_COUNT_LO    = 4
_BIT_RED_COUNT_HI    = 5
_BIT_DECOYS_BYPASSED = 6
_BIT_RESTORING       = 7

# -- Preemptive blocking targets per upcoming phase --------------------------

# Before Phase 1 starts, block this path (Phase 0 allows RZA->OZA, Phase 1 blocks it)
_PREEMPTIVE_PHASE1 = [
    ("restricted_zone_a_subnet", "operational_zone_a_subnet"),
]

# Before Phase 2 starts, block this path (Phase 1 allows RZB->OZB, Phase 2 blocks it)
_PREEMPTIVE_PHASE2 = [
    ("restricted_zone_b_subnet", "operational_zone_b_subnet"),
]


class EnterpriseHeuristicAgentV11a:
    """Phase-aware, Restore-only heuristic blue agent for CC4 v11a.

    Adds preemptive OZ blocking to v10b's Restore-only strategy.
    During active mission phases, blocks ALL open paths to the active OZ
    subnet with ASF=0 priority. Near phase transitions, preemptively blocks
    the RZ->OZ path that will become critical in the next phase.
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

        # v11a: Track preemptive blocks we've issued (not mandated by comms_policy)
        self._preemptive_blocks: set[tuple[str, str]] = set()

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
                       len(obs) >= expected_base_len + n_malfile_hosts)
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

        root_indicators: set = {
            h for h in malfile_alerts
            if h not in conn_alerts and h not in proc_alerts
        }

        # -- Read incoming peer messages -------------------------------------
        peer_state = self._read_peer_messages(obs, base, phase)

        # -- Track decoy-hit history for DECOYS_BYPASSED detection -----------
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

        # -- Build outbound message (v9 protocol) ----------------------------
        msg = np.zeros(8, dtype=bool)

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

        msg[_BIT_OPEN_PATHS_LO] = bool(open_paths_enc & 1)
        msg[_BIT_OPEN_PATHS_HI] = bool((open_paths_enc >> 1) & 1)

        msg[_BIT_RED_COUNT_LO] = bool(red_count_enc & 1)
        msg[_BIT_RED_COUNT_HI] = bool((red_count_enc >> 1) & 1)

        msg[_BIT_DECOYS_BYPASSED] = decoys_bypassed

        msg[_BIT_RESTORING] = any(
            self._step <= self._restore_at[h] + RESTORE_DUR - 1
            for h in self._restore_at
        )

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
        upstream_decoys_compromised = peer_state.get("upstream_decoys_bypassed", False)
        for hostname in _sorted_by_priority(conn_alerts, phase):
            if malfile_alerts.get(hostname) or proc_alerts.get(hostname):
                continue
            if (self._decoy_deployed.get(hostname, 0) > 0
                    and not upstream_decoys_compromised):
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

        # -- Priority P_BLOCK_OZ: Preemptive/active OZ blocking (v11a) ------
        #
        # Three blocking scenarios:
        #
        # A) ACTIVE PHASE LOCKDOWN: During Phase 1, block any open paths to
        #    OZA (ASF=0). During Phase 2, block any open paths to OZB (ASF=0).
        #    This catches any paths that comms_policy mandates but haven't been
        #    executed yet, with HIGHER priority than P2 Allow.
        #
        # B) PREEMPTIVE BLOCKING: Near the end of Phase 0, block RZA->OZA
        #    before Phase 1 starts. Near the end of Phase 1, block RZB->OZB
        #    before Phase 2 starts.
        #
        # C) CLEANUP: At phase transitions, allow any preemptive blocks from
        #    previous phase that the new comms_policy says should be allowed.
        #    (Handled in P2 Allow below -- preemptive blocks that comms_policy
        #    says allow will be caught by the existing P2 logic.)

        # Scenario A: Active phase lockdown
        if phase == 1:
            # Block all open paths to OZA
            target_subnet = "operational_zone_a_subnet"
            for src in _SORTED_SUBNETS:
                if src == target_subnet:
                    continue
                pair = (src, target_subnet)
                if not blocked_now.get(pair, False):
                    idx = self._block.get(pair)
                    if idx is not None and self._valid(idx, mask):
                        self._preemptive_blocks.add(pair)
                        return idx, msg

        elif phase == 2:
            # Block all open paths to OZB
            target_subnet = "operational_zone_b_subnet"
            for src in _SORTED_SUBNETS:
                if src == target_subnet:
                    continue
                pair = (src, target_subnet)
                if not blocked_now.get(pair, False):
                    idx = self._block.get(pair)
                    if idx is not None and self._valid(idx, mask):
                        self._preemptive_blocks.add(pair)
                        return idx, msg

        # Scenario B: Preemptive blocking before phase transitions
        elif phase == 0 and self._step >= (PHASE_1_START - PREEMPTIVE_WINDOW):
            # Near end of Phase 0: preemptively block paths that Phase 1 will need
            for pair in _PREEMPTIVE_PHASE1:
                if not blocked_now.get(pair, False):
                    idx = self._block.get(pair)
                    if idx is not None and self._valid(idx, mask):
                        self._preemptive_blocks.add(pair)
                        return idx, msg

        # Note: preemptive Phase 2 blocking during Phase 1 is handled by
        # Scenario A above (Phase 1 already blocks everything to OZA,
        # and RZB->OZB blocking for Phase 2 is done here)
        if phase == 1 and self._step >= (PHASE_2_START - PREEMPTIVE_WINDOW):
            for pair in _PREEMPTIVE_PHASE2:
                if not blocked_now.get(pair, False):
                    idx = self._block.get(pair)
                    if idx is not None and self._valid(idx, mask):
                        self._preemptive_blocks.add(pair)
                        return idx, msg

        # -- Priority 2: Allow paths per comms_policy (fix stale blocks) -----
        # This also handles cleanup of preemptive blocks that are no longer needed
        for pair, is_allowed in sorted(
            should_block.items(),
            key=lambda kv: _pair_priority(kv[0], phase)
        ):
            if not is_allowed and blocked_now.get(pair, False):
                idx = self._allow.get(pair)
                if idx is not None and self._valid(idx, mask):
                    # Remove from preemptive tracking if we're allowing it
                    self._preemptive_blocks.discard(pair)
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
        for hostname in _sorted_by_priority(proc_alerts, phase):
            if self._busy(hostname):
                continue

            flag_age = self._step - self._proc_flagged_step.get(hostname, self._step)

            peer_escalate_t3 = peer_state.get("upstream_red_count", 0) >= 3
            peer_escalate_t2 = (
                peer_state.get("any_root", False)
                or peer_state.get("upstream_threat", 0) >= 2
            )
            peer_escalate_compound = (
                peer_state.get("upstream_threat", 0) >= 1
                and peer_state.get("upstream_open_paths", 0) > 0
                and peer_state.get("upstream_decoys_bypassed", False)
            )

            is_critical_oz = _is_active_oz_server(hostname, phase)

            if peer_escalate_t3 or is_critical_oz:
                threshold = 0
            elif peer_escalate_t2 or peer_escalate_compound:
                threshold = 1
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
        self._preemptive_blocks.clear()

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

        self._deploy_hosts = sorted(all_decoy_hosts, key=_deploy_priority)

        self._subnets_in_obs = sorted(controlled)
        self._subnet_host_list.clear()
        for sn in self._subnets_in_obs:
            self._subnet_host_list[sn] = list(subnet_hosts.get(sn, []))

    # -- Helpers --------------------------------------------------------------

    def _issue_restore(self, hostname: str) -> None:
        self._restore_at[hostname] = self._step
        self._decoy_deployed.pop(hostname, None)

    def _busy(self, hostname: str) -> bool:
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

def _is_active_oz_server(hostname: str, phase: int) -> bool:
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

def make_heuristic_agents_v11a(env) -> dict[str, "EnterpriseHeuristicAgentV11a"]:
    """Create one agent per blue team member.  Call AFTER env.reset()."""
    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
    agents: dict[str, EnterpriseHeuristicAgentV11a] = {}
    for agent_name in env.possible_agents:
        ag = EnterpriseHeuristicAgentV11a(agent_name=agent_name)
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
