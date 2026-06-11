# EnterpriseHeuristicAgent v9.1 -- Gap Analysis and Failure Mode Catalog

**Date:** 2026-04-07
**Agent version:** v9.1
**Evaluation protocol:** 10 episodes x 500 steps, seed 42, BlueFlatWrapperV2
**Baseline performance:** -998.0 +/- 346.1 total reward (5 agents combined)

---

## 1. Executive Summary -- Top Gaps

The v9.1 agent achieves strong performance (~97% improvement over SleepAgent) but
the instrumented evaluation reveals five structural gaps that account for the
majority of remaining negative reward:

1. **Phase 0 dominates losses (58% of total penalty).** Phase 0 averages -3.49
   reward/step compared to -0.85 in Phase 1 and -1.66 in Phase 2. The agent has
   no blocking actions in Phase 0 (comms_policy says nothing needs blocking), and
   red freely scans and exploits across the entire network during the ~33-step
   preplanning window.

2. **HQ agent (blue_agent_4) is overwhelmed.** Agent_4 controls 3 subnets (PAZ,
   admin, office) with ~14 hosts. It issues 218 Removes and 128 escalated Restores
   -- nearly 3x the Remove rate of other agents. Green false positives in HQ drive
   excessive action spending.

3. **Massive decoy redeployment overhead.** 2,131 P6 (redeploy after Restore)
   actions and 1,731 P7 (initial deploy) actions consume significant action budget.
   Each Restore wipes up to 3 decoys, requiring 3 subsequent actions to restore
   them -- a 5+3=8 action cost per Restore cycle.

4. **Phase 2 has high variance (ep5: -1200 in Phase 2 alone).** When red
   establishes a foothold in RZB during Phase 1 or early Phase 2, the agent
   struggles to contain lateral movement. The residual RZA priority fix (v9.1
   Fix 3) helps but does not fully close this gap.

5. **No DegradeServices detection.** DegradeServices is permanent and invisible
   in the observation pipeline. Once red achieves root and degrades a service,
   the reward penalty continues every step until Restore. The agent has no
   signal to prioritize these hosts.

---

## 2. Detailed Per-Priority Analysis

### Priority 1: Restore on conn + (malfile OR proc)

**Trigger:** `conn_flag=1 AND (malfile=1 OR proc_flag=1)` on any host.

**Action:** Immediate Restore (5 steps, -1 reward cost).

**Observed count:** 549 across all agents/episodes (11.0/agent/episode).

**Gaps identified:**

- **Timing window:** Between the step red exploits and the step the conn+malfile
  flags appear in the observation, there is a 1-step delay (action executes, then
  observation is generated). Red's ExploitRemoteService takes 1 step; the flags
  appear next step. If red immediately follows with PrivilegeEscalate (2 steps),
  blue has at most 1 step to issue Restore before red reaches root.

- **No host triage within P1:** All P1 hosts are sorted by `_host_priority`, but
  only one action is taken per step per agent. If 2+ hosts are simultaneously
  flagged (e.g., red hits two hosts in the same subnet on consecutive steps), the
  lower-priority host waits at least 5 steps (Restore duration) before being
  addressed. During that window, red can PrivEsc on the unattended host.

- **Restore cost vs benefit mismatch in Phase 0:** Phase 0 RIA penalties are -1
  for RZA/RZB and -1 for OZA/OZB. The Restore itself costs -1. In Phase 0, a
  Restore on a low-value host is cost-neutral at best. The agent does not
  differentiate Phase-0 low-value Restores from Phase-1/2 high-value ones.

### Priority 1b: Restore on conn-only without decoy coverage

**Trigger:** `conn_flag=1 AND malfile=0 AND proc_flag=0` AND (no decoy deployed OR
upstream reports DECOYS_BYPASSED).

**Action:** Restore (5 steps).

**Gaps identified:**

- **Relies entirely on upstream messaging for bypass detection.** If the upstream
  agent itself is busy (mid-Restore) or has not yet detected the decoy bypass,
  the downstream agent will incorrectly skip the conn-only alert as a decoy hit,
  allowing red through.

- **No phase-based suppression.** In Phase 0, conn-only events are overwhelmingly
  green agent activity. The agent cannot distinguish green GreenAccessService
  events from red decoy hits without malfile flags.

### Priority 1c: Restore on pure malfile (PrivilegeEscalate signature)

**Trigger:** `malfile=1 AND conn=0 AND proc=0`.

**Action:** Restore (root session present, Remove insufficient).

**Gaps identified:**

- **Depends entirely on BlueFlatWrapperV2.** Without the V2 wrapper, PrivEsc is
  completely invisible. The agent docstring mentions compatibility with both
  wrappers, but with plain BlueFlatWrapper, PrivEsc detection is impossible.

- **Timing:** PrivEsc takes 2 steps. The malfile flag appears at the end of step 2.
  If red immediately follows with Impact (2 steps), blue has only the step the
  malfile appears to issue Restore before Impact fires. If the agent is busy with
  another host, Impact succeeds.

### Priority 2: Allow per comms_policy

**Trigger:** A path is blocked that comms_policy says should be open.

**Action:** AllowTrafficZone.

**Observed count:** 43 (mostly blue_agent_0 and agent_1).

**Gaps identified:**

- **Stale block residuals.** When the phase changes (e.g., Phase 1 to Phase 2),
  blocks applied for Phase 1 isolation may no longer be needed. The agent correctly
  detects these via comms_policy changes, but the Allow action costs a step that
  could be spent on threat response.

- **Phase transition race:** The phase transition happens at a fixed step. If red
  has already exploited a host in the old phase's zone, the Allow action may
  re-open a path that red can immediately use for lateral movement.

### Priority 3: Block per comms_policy

**Trigger:** comms_policy says a path should be blocked but it is currently open.

**Action:** BlockTrafficZone.

**Observed count:** 364 across all agents.

**Gaps identified:**

- **Phase 0 has zero blocks to make.** comms_policy in Phase 0 does not require
  any subnet isolation. Red has free lateral movement during the entire ~33-step
  preplanning phase. This is the single largest contributor to Phase 0 losses.

- **Blocking order is not optimized.** `_pair_priority` assigns 200 to
  contractor/internet, 100 to active-zone restricted subnets, but does not
  consider the actual attack path. Red enters via contractor/internet and pivots
  through RZA/RZB. Blocking contractor_network -> restricted_zone_a first would
  cut the entry path before blocking less critical paths.

- **Block actions compete with threat response.** At phase transitions, the agent
  may spend 4-5 steps blocking paths instead of responding to simultaneous alerts.
  There is no mechanism to defer blocking if active threats are present (blocking
  is P3, above P4 Remove).

### Priority 4: Remove on process flags

**Trigger:** `proc_flag=1` on any host, no Remove/Restore already in progress.

**Action:** Remove (3 steps) first, then escalate to Restore based on flag persistence.

**Observed count:** 480 Removes + 293 escalated Restores.

**Gaps identified:**

- **33-54% green false positive rate on proc_flag.** Remove costs 3 steps of the
  agent's action budget per FP. With 480 Removes across 10 episodes (9.6/agent/ep),
  a significant portion are wasted on green activity.

- **Escalation threshold collapse (v9.1 Fix 4).** The default threshold was lowered
  from 2 to 1, and the T2 conditions also resolve to 1. This effectively creates
  only a binary system: T3 (threshold=0, immediate Restore) or everything else
  (threshold=1). The nuance of the original 3-tier system is lost. Whether this
  is optimal depends on the green FP rate at flag_age=1.

- **Branch A timing issue (`ra >= 0` check).** After a Remove is issued at step N,
  any proc_flag reappearing at step N+1 or later triggers Restore. But Remove
  takes 3 steps (completes at step N+3). If the proc_flag reappears at step N+1
  (before Remove finishes), the `_busy()` check prevents action. If it appears at
  step N+3, Branch A triggers Restore. But if the flag disappears at N+3 (green
  FP cleared by Remove timing) and reappears at N+4, it is treated as a new flag
  via `_proc_flagged_step` -- the old `_remove_at` entry causes immediate Restore
  via Branch A. This is correct for red but creates unnecessary Restores for
  intermittent green activity.

### Priority 5: Restore for persistent proc flags (no Remove issued)

**Trigger:** proc_flag present, no Remove or Restore previously issued on this host.

**Observed count:** 0 in all experiments.

**Gaps identified:**

- **Dead code path.** Priority 4 always fires before Priority 5 because it handles
  the same proc_flag set with the same `_busy()` check. The only way P5 fires is
  if P4's `_valid()` check fails (mask blocks Remove and Restore) and then P5
  retries Restore -- but P4 already tried Restore in its escalation path.

- **Potential correctness issue:** If Remove is masked but Restore is not, P4
  would attempt Restore via the threshold path. P5 would then find Restore
  already issued. If both are masked, P5 also cannot act. P5 appears to be
  unreachable in practice.

### Priority 6: Redeploy decoys after Restore

**Trigger:** Host was Restored (step >= restore_at + RESTORE_DUR) and has fewer
than MAX_DECOYS decoys.

**Observed count:** 2,131.

**Gaps identified:**

- **3-action redeployment cost.** Each Restore wipes all decoys. Redeploying 3
  decoys takes 3 separate actions (each DeployDecoy is 2 steps). That is 6 steps
  of pure decoy work per Restored host, during which the agent cannot respond to
  new threats.

- **Deployment order after Restore:** The `_deploy_hosts` list is global priority
  order, not specific to the just-Restored host. If a low-priority host was
  Restored, the agent may redeploy decoys on it before redeploying on a
  high-priority host that was Restored earlier (both need redeployment). However,
  the loop checks `_restore_at`, so it should correctly iterate. The issue is
  subtle: if two hosts were Restored, the one appearing earlier in `_deploy_hosts`
  gets redeployed first, regardless of which was Restored more recently.

### Priority 7: Initial decoy deployment

**Trigger:** Host has < MAX_DECOYS decoys and no prior Restore.

**Observed count:** 1,731.

**Gaps identified:**

- **Deployment takes 3*N_hosts actions.** With ~14 hosts for agent_4 (HQ), initial
  deployment takes 42 actions (84 steps at 2 steps each). Agent_4 spends the
  first ~84 steps of every episode purely deploying decoys. During this time, it
  cannot respond to Phase 0 threats.

- **v9.1 Fix 5 (_busy guard) is correct but insufficient.** If a host gets
  attacked during initial deployment, the agent handles it via P1-P4 (higher
  priority), but then must re-enter P7 and P6 to finish deploying on the
  remaining hosts. The decoy deployment queue does not adapt based on threat
  activity.

### Priority 8: Sleep

**Observed count:** 19,359 (77.4% of all actions).

**Gaps identified:**

- **No monitoring during idle.** When sleeping, the agent takes no action. Monitor
  was evaluated and found functionally identical. However, the agent has no
  proactive behavior during idle -- no preemptive scanning, no information
  gathering. This is by design (reactive-only is better than proactive Restores),
  but it means the agent is blind to DegradeServices and quiet lateral movement.

---

## 3. Experimental Results

### 3.1 Performance Summary

| Metric | Value |
|--------|-------|
| Mean reward (10 ep) | -998.0 +/- 346.1 |
| Per-agent equivalent | -199.6 +/- 69.2 |
| Min / Max episode | -1935.0 / -655.0 |
| Throughput | ~83 steps/sec |

### 3.2 Per-Phase Reward Breakdown

| Phase | Total reward | Steps | Avg reward/step | % of total loss |
|-------|-------------|-------|-----------------|-----------------|
| Phase 0 (Preplanning) | -5,820 | 1,670 | -3.485 | **58.3%** |
| Phase 1 (Mission A) | -1,415 | 1,670 | -0.847 | 14.2% |
| Phase 2 (Mission B) | -2,745 | 1,650 | -1.664 | 27.5% |

**Key finding:** Phase 0 produces 58% of all negative reward despite having the
lowest per-host penalty multipliers. This is because: (a) no blocking is possible,
(b) all subnets are exposed, (c) red is freely scanning and exploiting, and
(d) 5 agents x ~33 steps x 5 subnets means many simultaneous exploitation paths.

### 3.3 Action Distribution

| Priority | Count | % of total | Per agent/ep |
|----------|-------|-----------|--------------|
| P1 Restore | 549 | 2.2% | 11.0 |
| P2 Allow | 43 | 0.2% | 0.9 |
| P3 Block | 364 | 1.5% | 7.3 |
| P4 Remove | 480 | 1.9% | 9.6 |
| P4 Restore (escalated) | 293 | 1.2% | 5.9 |
| P6 Redeploy Decoy | 2,131 | 8.5% | 42.6 |
| P7 Deploy Decoy | 1,731 | 6.9% | 34.6 |
| P8 Sleep | 19,359 | 77.4% | 387.2 |

**Key finding:** Decoy operations (P6 + P7) consume 15.5% of all actions. The
Restore:Remove ratio is 1.75:1, meaning most process flags that trigger Remove
eventually require escalation to Restore. This suggests Remove is often wasted.

### 3.4 Per-Agent Analysis

| Agent | P1 | P4 Remove | P4 Restore | Decoy ops | Sleep |
|-------|-----|----------|------------|-----------|-------|
| agent_0 (RZA) | 101 | 72 | 46 | 594 | 4,086 |
| agent_1 (OZA) | 128 | 66 | 35 | 691 | 3,972 |
| agent_2 (RZB) | 78 | 70 | 43 | 566 | 4,179 |
| agent_3 (OZB) | 107 | 54 | 41 | 666 | 4,034 |
| agent_4 (HQ) | 135 | 218 | 128 | 1,345 | 3,088 |

**Key finding:** Agent_4 handles 3x the zone size of other agents. It spends 45%
of its Remove actions (218) and 44% of its escalated Restores (128) in the HQ
zone. It also deploys 1,345 decoys (vs ~600 for other agents). Agent_4 is the
most action-constrained agent.

### 3.5 Most Compromised Hosts

| Host | Alert count | Zone |
|------|------------|------|
| rza_server_host_0 | 109 | RZA |
| rza_user_host_2 | 90 | RZA |
| rzb_server_host_0 | 86 | RZB |
| ozb_server_host_1 | 72 | OZB |
| rza_server_host_1 | 70 | RZA |
| oza_server_host_0 | 70 | OZA |

**Key finding:** RZA hosts are the most frequently alerted, consistent with
red's Phase 1 attack path (internet -> contractor -> RZA -> OZA). Server hosts
are hit more often than user hosts because red's FSM prioritizes servers (75%
probability in `_choose_host`).

---

## 4. Decoy Strategy Analysis

### 4.1 Current Decoy Configuration

- **Decoy types available:** Apache, Tomcat, Haraka (SMTP), Vsftpd (FTP)
- **MAX_DECOYS:** 3 per host
- **Deployment:** `DeployDecoy` randomly selects from compatible factories
- **Coverage:** One `DeployDecoy` action per host; factory selection is random

### 4.2 Red's Exploit Portfolio vs Decoy Coverage

Red's `ExploitRemoteService` uses these exploits (CC4-active only):
- HTTPRFI (Apache/HTTP)
- HTTPSRFI (HTTPS)
- SSHBruteForce (SSH)
- SQLInjection (SQL)
- HarakaRCE (SMTP/Haraka)
- FTPDirectoryTraversal (FTP)

Decoy factories cover:
- Apache -> catches HTTPRFI
- Tomcat -> catches HTTPRFI/HTTPSRFI (runs on HTTP port)
- Haraka -> catches HarakaRCE
- Vsftpd -> catches FTPDirectoryTraversal

**Not covered by any decoy:**
- SSHBruteForce -- no SSH decoy factory exists in DeployDecoy
- SQLInjection -- no SQL decoy factory exists in DeployDecoy

### 4.3 Gaps

- **SSH and SQL exploits bypass all decoys.** Red's FSM selects exploits randomly
  based on discovered ports. If the random selection lands on SSH or SQL, decoys
  provide zero defense. This means the theoretical 75% decoy hit rate
  (3 decoys / 4 total services) is only correct when the 4th service is HTTP/FTP/SMTP.
  If the legitimate service is SSH or SQL, the effective decoy coverage drops.

- **Random factory selection.** DeployDecoy picks a random compatible factory. The
  agent cannot control which decoy type is deployed. Deploying 3 Apache decoys on
  a host that only exposes SSH provides no coverage.

- **Decoy wipe on Restore.** Every Restore wipes all 3 decoys, requiring 3
  redeployment actions. With the observed 1.75 Restore:Remove ratio, each
  compromise cycle costs 5 (Restore) + 6 (3 x 2 step deploy) = 11 steps.

- **MAX_DECOYS=3 is not clearly optimal.** With 3 decoys and 1 legitimate service,
  blind exploit hit rate is 75%. With 4 decoys, it would be 80%. But the marginal
  gain (5%) costs 1 additional deployment step per host. The current value seems
  reasonable but has not been experimentally validated.

---

## 5. Escalation Logic Review

### 5.1 Priority 4 Escalation -- Current Design

Branch A: `ra >= 0` (Remove was previously issued)
- If `self._step > ra`: Restore immediately (any re-flag after Remove = root)
- Otherwise: continue (Remove still in progress)

Branch B: No prior Remove
- T3: `upstream_red_count >= 3` -> threshold=0 (immediate Restore)
- T2: `any_root OR upstream_threat >= 2` -> threshold=1
- Default: threshold=1 (v9.1 Fix 4)
- If `_host_priority >= 100 OR flag_age >= threshold`: Restore
- Otherwise: Remove

### 5.2 Threshold=0 for T3 vs Threshold=1 for Default

The v9.1 Fix 4 collapsed the default from 2 to 1, creating a 2-tier system.

**Correctness assessment:** The T3 threshold of 0 is correct for saturated upstream
zones (3+ hosts compromised = red is past the decoy layer and advancing rapidly).
Skipping Remove and going directly to Restore saves 3 steps per host.

**Default threshold of 1:** A green false-positive proc_flag typically clears within
1 step. A flag_age >= 1 flag is almost certainly real red. The threshold of 1
correctly filters out most green FPs while responding within 2 steps of detection.

**Missing escalation conditions:**
- No escalation based on `max_peer_red_count` (only `upstream_red_count` is used).
  If a non-upstream peer has 3+ compromised hosts, the agent does not increase its
  urgency. This is by design (only upstream matters for lateral movement), but it
  means Phase 0 threats from any direction do not trigger T3.

- No phase-aware threshold adjustment. In Phase 0, the penalty for a proc_flag host
  is lower (-1 RIA for most zones). The cost of an unnecessary Restore (-1) equals
  the cost of the compromise. A higher threshold in Phase 0 would reduce wasted
  Restores.

### 5.3 `_remove_at` Clearing (v9.1 Fix 1)

**Assessment:** The fix is correct. When a host is Restored, all prior Remove
history is stale. Without clearing `_remove_at`, a subsequent exploit would
immediately trigger Restore via Branch A, skipping the cheaper Remove. The fix
allows Remove to be tried first on a freshly reimaged host.

**Remaining edge case:** If red exploits a host, blue issues Remove, the flag
persists (root session survived), and blue issues Restore -- but Restore fails
(mask blocks it) -- the `_remove_at` entry remains, and the agent will be stuck
in a loop where Branch A keeps trying Restore but it keeps being masked.

---

## 6. Inter-Agent Messaging Effectiveness

### 6.1 Protocol Utilization

The v9 protocol provides actionable intelligence in two places:
1. Priority 1b: upstream DECOYS_BYPASSED suppresses conn-only skip
2. Priority 4: upstream state affects escalation threshold

### 6.2 Gaps

- **Upstream mapping is incomplete.** `_UPSTREAM` only defines:
  - Phase 1: agent_0 (RZA) -> agent_1 (OZA)
  - Phase 2: agent_2 (RZB) -> agent_3 (OZB)

  **Missing mappings:**
  - Phase 0: no upstream defined for any agent. All T3/T2 escalation is disabled
    in Phase 0 because `_UPSTREAM.get((0, any), None)` returns None, so
    `upstream_threat`, `upstream_red_count`, etc. are all 0/False.
  - Phase 1/2: HQ (agent_4) has no upstream. Red can pivot through HQ zones
    (admin, office, PAZ) without triggering escalation in any peer.
  - Phase 1: agent_2 (RZB) has no upstream. If red attacks RZB during Phase 1
    (low priority but possible), agent_3 (OZB) gets no warning.

- **Message latency.** Messages are 1 step delayed (sent at step N, received at
  step N+1). In fast-moving attacks (red exploits + PrivEsc in 3 steps), the
  upstream warning arrives too late for the downstream agent to preempt.

- **`upstream_restoring` is unused.** The field is decoded and available in
  `peer_state` but no decision path uses it. Knowing that an upstream agent is
  busy restoring could trigger more aggressive preemptive blocking or decoy
  redeployment.

- **No cross-phase message continuity.** When the phase changes, the upstream
  mapping changes. Any accumulated intelligence about the old upstream's state is
  lost. If red established a foothold in RZA during Phase 1 and Phase 2 begins,
  agent_1 (OZA) no longer has an upstream and stops receiving T3 escalation even
  though the RZA threat may still be active.

---

## 7. Comms Policy Analysis

### 7.1 Current Blocking Strategy

| Phase | Required blocks |
|-------|----------------|
| Phase 0 | None (full connectivity) |
| Phase 1 | RZA isolated: block contractor/internet/RZB/OZA <-> RZA |
| Phase 2 | RZB isolated: block contractor/internet/RZA/OZA <-> RZB |

### 7.2 Gaps

- **Phase 0 has no blocking.** The comms_policy vector is all-zeros in Phase 0,
  meaning the agent has no blocking to do. Red has unimpeded access to all
  subnets. This is the root cause of the 58% Phase 0 loss share.

- **Proactive blocking in Phase 0 is not supported by the environment.** Even if
  the agent issued BlockTrafficZone in Phase 0, the comms_policy would not
  request it, and the block might be incorrectly unwound by Priority 2 (Allow)
  when the comms_policy says the path should be open.

- **Phase transition blocking delay.** At the Phase 1 transition, the agent must
  block ~8 paths. With Priority 3 handling one block per step, it takes 8 steps
  to fully isolate the active zone. Red can exploit this window.

- **Block priority ordering.** Contractor and internet paths (priority 200) are
  blocked before RZ paths (priority 100). This is correct for cutting entry
  points first. However, the most critical single block is often
  `restricted_zone_a -> operational_zone_a` (the direct attack path), which has
  priority 100 and is blocked after contractor/internet paths.

- **Blocking does not stop PhishingEmail.** PhishingEmail uses `check_routable`
  and `connected_components`, not `blocking_host()`. Blocks only stop
  ExploitRemoteService. Red can still phish into blocked zones.

---

## 8. Failure Mode Catalog

### FM-1: Phase 0 Uncontested Red Activity (CRITICAL)
- **Description:** No blocking possible in Phase 0. Red scans and exploits freely.
- **Impact:** 58% of total reward loss.
- **Mitigation:** Proactive Phase 0 blocking (requires comms_policy changes or
  heuristic override), or faster Phase 0 decoy deployment to slow red.

### FM-2: Simultaneous Multi-Host Exploitation (HIGH)
- **Description:** Red exploits 2+ hosts in the same agent's zone on consecutive
  steps. Agent can only Restore one host per step.
- **Impact:** Second host remains compromised for 5+ steps (Restore duration of
  first host).
- **Mitigation:** None within single-agent design. Would require multi-action
  steps or agent coordination.

### FM-3: PrivEsc Race Condition (HIGH)
- **Description:** Red exploits a host (step N), blue sees alert (step N+1), red
  starts PrivEsc (step N+1, 2 steps). Blue issues Restore (step N+1, 5 steps).
  Restore completes at N+6. PrivEsc completes at N+3. If Restore does not
  interrupt PrivEsc, root session is established before host reimage.
- **Impact:** Root session on a reimaged host would be wiped by Restore (Restore
  clears ALL sessions). So this race is actually won by blue if Restore is issued
  promptly. The risk is when blue is busy with another host and cannot issue
  Restore on step N+1.
- **Mitigation:** Priority 1 already handles this optimally. The gap is when
  multiple hosts are flagged simultaneously.

### FM-4: DegradeServices Invisible Damage (HIGH)
- **Description:** Red issues DegradeServices on a host with root access. This is
  permanent (only Restore fixes it) and produces zero observation signals. The
  agent has no way to detect it.
- **Impact:** Continuous -1 to -10 reward per step depending on phase and zone.
- **Mitigation:** Heuristic: if a host had root indicators (P1c malfile-only) and
  was NOT Restored in time, assume DegradeServices may have occurred. This is
  currently handled indirectly (P1c triggers Restore which fixes Degrade), but if
  the agent is busy, the window is open.

### FM-5: HQ Agent Action Budget Exhaustion (MEDIUM)
- **Description:** Agent_4 manages ~14 hosts across 3 subnets. Initial decoy
  deployment takes ~42 actions. Combined with high green FP rate in HQ zones,
  the agent is frequently overloaded.
- **Impact:** Delayed threat response in HQ, especially during Phase 0.
- **Mitigation:** Reduce MAX_DECOYS for HQ zone, or prioritize decoy deployment
  on server hosts only in HQ.

### FM-6: Decoy Wipe Amplification (MEDIUM)
- **Description:** Each Restore wipes 3 decoys, requiring 3 redeployment actions.
  A host that is repeatedly exploited and Restored cycles through
  Restore(5) -> Deploy(2) -> Deploy(2) -> Deploy(2) = 11 steps per cycle.
- **Impact:** Agent spends 11/500 = 2.2% of episode per compromised host per
  cycle. With 10+ compromise events per agent per episode, this is significant.
- **Mitigation:** Consider deploying only 1-2 decoys on frequently-Restored hosts
  to reduce redeployment cost. Or skip redeployment entirely on hosts that have
  been Restored more than N times.

### FM-7: Green False Positive Removes (MEDIUM)
- **Description:** proc_flag has 33-54% FP rate from green agents. Each Remove
  costs 3 steps of agent budget.
- **Impact:** 480 Removes across 10 episodes. Estimated 160-260 are green FPs.
  Each wastes 3 steps = 480-780 wasted steps.
- **Mitigation:** Raise the proc_flag-only threshold to flag_age >= 2 for
  non-critical hosts in Phase 0. Or use malfile correlation: proc_flag WITHOUT
  malfile is more likely green FP.

### FM-8: SSH/SQL Exploit Bypass of Decoy Layer (MEDIUM)
- **Description:** No SSH or SQL decoy factories exist. Red exploits using
  SSHBruteForce or SQLInjection bypass all deployed decoys.
- **Impact:** Reduces effective decoy coverage from theoretical 75% to lower
  depending on service composition.
- **Mitigation:** Implement SSH and SQL decoy factories. This requires code
  changes to the DecoyActions module, not just the heuristic agent.

### FM-9: Phase Transition Blocking Window (LOW-MEDIUM)
- **Description:** 8 steps needed to fully block all paths at phase transition.
  Red can exploit the partially-blocked network during this window.
- **Impact:** 1-2 exploits per phase transition in worst case.
- **Mitigation:** Pre-compute blocking order to cut the highest-risk path first
  (RZ -> OZ direct path before contractor/internet paths).

### FM-10: Upstream Message Slot Shift (LOW)
- **Description:** When an agent is network-isolated, CybORG may skip its message
  slot, shifting subsequent slot indices. The agent assumes fixed slot ordering.
- **Impact:** Currently benign (isolated agents are the ones with _UPSTREAM
  entries, so their zero-padded slots are correctly handled). But if upstream
  mappings were extended (e.g., Phase 0), slot shifts could cause message
  misinterpretation.
- **Mitigation:** Validate slot assignment against known agent indices rather than
  assuming fixed ordering.

### FM-11: Priority 5 Dead Code (LOW)
- **Description:** Priority 5 (Restore for persistent proc flags without prior
  Remove) never fires in practice because Priority 4 handles the same set.
- **Impact:** None functionally. Increases code complexity without benefit.
- **Mitigation:** Remove or merge into Priority 4.

### FM-12: Incomplete Phase 0 Upstream Mapping (LOW-MEDIUM)
- **Description:** No upstream agent defined in Phase 0. All peer escalation
  is disabled during the highest-loss phase.
- **Impact:** T3 escalation never triggers in Phase 0, even when a peer zone
  has 3+ compromised hosts.
- **Mitigation:** Define Phase 0 upstream mappings (e.g., all restricted zones
  are upstream of their operational zones even in Phase 0). This would enable
  T3 escalation when red is saturating a zone during preplanning.

---

## 9. Ranked Improvement Opportunities

| Rank | Improvement | Est. Impact | Difficulty | Notes |
|------|------------|-------------|------------|-------|
| 1 | Phase 0 proactive blocking | HIGH (20-30% reward improvement) | HARD | Requires overriding comms_policy or environment changes |
| 2 | Phase 0 upstream mappings | MEDIUM (5-10%) | EASY | Add entries to `_UPSTREAM` for Phase 0 |
| 3 | Smarter decoy redeployment | MEDIUM (5-8%) | EASY | Skip redeployment on frequently-Restored hosts; deploy 1 decoy instead of 3 after Restore |
| 4 | Green FP suppression via malfile | MEDIUM (3-7%) | EASY | proc_flag WITHOUT malfile at flag_age=0 -> higher threshold |
| 5 | Block ordering optimization | LOW-MEDIUM (2-5%) | EASY | Block RZ->OZ path before contractor/internet |
| 6 | HQ agent load reduction | LOW-MEDIUM (2-4%) | EASY | Reduce MAX_DECOYS to 1-2 for non-server HQ hosts |
| 7 | Remove Priority 5 dead code | LOW (0%) | TRIVIAL | Code cleanup |
| 8 | `upstream_restoring` usage | LOW (1-3%) | EASY | If upstream is restoring, preemptively increase alert threshold |
| 9 | SSH/SQL decoy factories | MEDIUM (5-10%) | HARD | Requires new DecoyAction subclasses |
| 10 | Adaptive MAX_DECOYS | LOW (1-3%) | MEDIUM | Reduce decoys on hosts with low exploit rates |

---

## 10. Appendix: Red FSM Attack Model

Red's FiniteStateRedAgent uses a finite state machine with 8 states per host:
- K: Known (discovered IP)
- KD: Known + Decoy detected
- S: Scanned (services discovered)
- SD: Scanned + Decoy detected
- U: User session (exploit succeeded)
- UD: User session + Decoy detected
- R: Root session (PrivEsc succeeded)
- RD: Root session + Decoy detected

Key transition probabilities from the FSM:
- State U: 50% PrivilegeEscalate, 50% DiscoverRemoteSystems (re-scan)
- State UD: 100% PrivilegeEscalate
- State R: 25% Impact, 25% DegradeServices, 50% DiscoverRemoteSystems
- State RD: 50% Impact, 50% DegradeServices

**Critical insight:** Once red reaches state UD (user session + decoy knowledge),
it will attempt PrivEsc with 100% probability. The decoy knowledge makes red more
dangerous, not less. This means that after a decoy hit followed by a real exploit,
red becomes deterministic in its escalation path.

Red prioritizes servers with 75% probability (`prioritise_servers`). This explains
why server_host_0 and server_host_1 dominate the compromise count.

---

## 11. Appendix: Observation Pipeline Limitations

BlueFlatWrapperV2 provides:
- Phase indicator (1 bit)
- Per-subnet: one-hot identity, blocked status, comms_policy, proc_flags, conn_flags
- Inter-agent messages (32 bits)
- Malfile flags (1 per host)

**Information NOT available in observation:**
- Which specific exploit red used
- Whether DegradeServices has occurred
- Which specific services are running on a host
- Red's FSM state for any host
- Whether a green agent or red agent caused a proc_flag
- Session types (user vs root) on hosts blue does not control
- Red's DiscoverDeception results (whether red knows decoy PIDs)

These observation limitations are fundamental to the environment design and cannot
be addressed by agent-level changes. The heuristic agent's strategy is bounded by
these information constraints.
