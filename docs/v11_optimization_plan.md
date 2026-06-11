# V11 Optimization Plan: Coordinated Blocking + Messaging Redesign

**Date:** 2026-04-10  
**Baseline:** V10b at -814.0 +/- 247.7 (100 eps, seed 42)  
**Goal:** Break through V10b's local optimum via architectural changes  
**Status:** PLAN (not yet implemented)

---

## The Problem

22 single-parameter experiments found zero improvements to V10b. The agent is at a
local optimum. The remaining performance gap (-814 vs theoretical ~-300) is bounded by:

1. **DegradeServices invisibility** — 83.3% of damage, zero observable flags
2. **Independent agent decisions** — 5 agents act alone, messaging has zero value
3. **Reactive-only posture** — agents wait for flags, then respond

Breaking through requires **architectural changes**, not parameter tuning.

---

## Critical Discovery: ASF=0 in Active Phases

From `BlueRewardMachine.get_phase_rewards()`:

```
Phase 1: operational_zone_a → {"LWF": -10, "ASF": 0, "RIA": -10}
Phase 2: operational_zone_b → {"LWF": -10, "ASF": 0, "RIA": -10}
```

**In active mission phases, blocking traffic to the impact-target OZ subnet has
ZERO Access Service Failure (ASF) cost.** This means we can block paths into OZA
during Phase 1 and OZB during Phase 2 for FREE — no green penalty.

Meanwhile, LWF=-10/step for DegradeServices on those same hosts. This creates an
enormous asymmetry: **blocking costs 0, not-blocking costs -10/step per degraded host.**

### Reward Table Reference

| Phase | Subnet | LWF | ASF | RIA |
|-------|--------|-----|-----|-----|
| 0 | OZA/OZB | -1 | -1 | -1 |
| **1** | **OZA** | **-10** | **0** | **-10** |
| 1 | RZA | -2 | -1 | -3 |
| **2** | **OZB** | **-10** | **0** | **-10** |
| 2 | RZB | -2 | -1 | -3 |
| 0-2 | HQ/admin/office | -1 | -1 | -3 |
| 0-2 | contractor | 0 | 0/-5 | 0/-5 |

---

## Strategy Overview

Three interconnected changes, implemented in phases:

```
Phase A: Preemptive OZ Blocking (zero-cost blocking of active OZ)
Phase B: Messaging Redesign (red-position communication for coordinated blocking)
Phase C: DegradeServices Mitigation (shadow tracking + predictive Restore)
```

Each phase builds on the previous. Phase A is the highest-impact, lowest-risk change.

---

## Phase A: Preemptive OZ Blocking

### Concept

During active mission phases, proactively block ALL traffic into the impact-target
OZ subnet from non-essential sources. Since ASF=0 for the active OZ, this costs
nothing but prevents red from reaching the highest-value targets.

### Blocking Mechanics (from `ControlTraffic.py`)

```python
# BlockTrafficZone adds from_subnet to state.blocks[to_subnet]
state.blocks.setdefault(self.to_subnet, []).append(self.from_subnet)

# ExploitRemoteService checks: is target's subnet blocking source's subnet?
subnet_blocked = other_host_subnet in state.blocks and \
                 src_host_subnet in state.blocks[other_host_subnet]
```

Red's ExploitRemoteService and DiscoverNetworkServices both check for blocks.
If OZA is blocking RZA, red cannot exploit from RZA→OZA.

### What to Block

**Phase 1 (OZA is impact target):**
- Block: contractor→OZA, internet→OZA, admin→OZA, office→OZA, PAZ→OZA
- Keep open: RZA→OZA only if comms_policy says so (it usually does in Phase 1)
- Agent_1 (controls OZA) executes these blocks

**Phase 2 (OZB is impact target):**
- Block: contractor→OZB, internet→OZB, admin→OZB, office→OZB, PAZ→OZB
- Keep open: RZB→OZB only if comms_policy says so
- Agent_3 (controls OZB) executes these blocks

**Phase 0 (no impact target):**
- No preemptive blocking. Follow comms_policy only.
- LWF is only -1/step for OZ, not worth the ASF=-1 cost.

### Implementation

Add a new priority **P1.5** (after confirmed-red Restore, before P1b):

```python
# P1.5: Preemptive OZ blocking in active phases
if phase == 1 and "operational_zone_a" in self._controlled_subnets:
    for src_subnet in non_essential_sources:
        pair = (src_subnet, "operational_zone_a_subnet")
        if not blocked_now.get(pair, False):
            idx = self._block.get(pair)
            if idx is not None and self._valid(idx, mask):
                return idx, msg
```

### Cost/Benefit Analysis

**Cost:** Zero ASF. The only cost is action budget (1 step per block action).
With ~5 non-essential source subnets, this takes 5 steps at the start of Phase 1.

**Benefit:** Prevents red from exploiting OZA hosts via non-RZA paths.
Red attack chain is internet→contractor→HQ→RZ→OZ. Blocking alternative
paths forces red through the single RZ→OZ chokepoint, which comms_policy
and the RZ agent are already monitoring.

**Risk:** If comms_policy already blocks these paths, this is a no-op.
Need to verify which paths comms_policy leaves open in each phase.

### Experiments Needed

1. **Trace comms_policy per phase:** What paths does comms_policy actually
   block/allow in each phase? If it already blocks contractor→OZA in Phase 1,
   preemptive blocking is redundant.

2. **Measure green cross-subnet access:** How often does green call
   GreenAccessService across subnets to OZ? If rarely, blocking is truly free.

3. **Test preemptive OZ blocking:** Implement and measure reward change.

---

## Phase B: Messaging Redesign

### Current Protocol: Dead

The v9 messaging protocol has zero measurable value (p=0.80 ablation test).
Root causes:
- 86.7% of agent-steps have no upstream relationship
- T2/compound escalation = dead code (threshold=1 = default)
- RESTORING and OPEN_PATHS bits never consumed
- Slot misalignment bug when agents are network-isolated

### New Protocol Design: "Red Tracker"

Instead of encoding abstract threat levels, communicate **concrete actionable
information** that directly influences blocking decisions.

**New 8-bit message format:**

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | RED_DETECTED | 1 = I see confirmed red activity (conn+malfile/proc) |
| 1 | RED_ROOT | 1 = malfile-only indicator (PrivEsc signature = root) |
| 2 | RED_PROGRESSING | 1 = red detected AND I'm currently busy Restoring |
| 3 | ZONE_CLEAR | 1 = no alerts, all hosts clean |
| 4-5 | RED_SUBNET_IDX | 2-bit index of which subnet red is in (0-3 = my controlled subnets) |
| 6 | REQUEST_BLOCK | 1 = downstream agent should block traffic FROM my zone |
| 7 | RESTORED_RECENTLY | 1 = I Restored a host in the last 5 steps |

### Consumption Rules

**Every agent reads ALL peer messages (not just upstream):**

```python
# For each peer message:
if peer.RED_DETECTED and peer controls adjacent subnet:
    # Consider preemptive blocking from peer's zone
    if not already_blocked(peer_subnet → my_subnet):
        issue_block(peer_subnet → my_subnet)

if peer.REQUEST_BLOCK:
    # Peer explicitly requests we block traffic FROM their zone
    issue_block(peer_subnet → my_subnet)  # higher priority

if peer.ZONE_CLEAR and previously_blocked(peer_subnet → my_subnet):
    # Peer says zone is clean, can re-allow traffic
    issue_allow(peer_subnet → my_subnet)

if peer.RED_PROGRESSING:
    # Peer is busy Restoring, red may migrate to my zone
    lower_threshold_for_peer_adjacent_hosts()
```

### Key Design Principles

1. **All agents read all peer messages.** No upstream-only restriction.
2. **Messages influence P1-P3 (high-impact priorities)**, not P4 thresholds.
3. **REQUEST_BLOCK is the primary coordination signal.** When an agent detects
   red and is busy Restoring, it requests downstream agents to block.
4. **ZONE_CLEAR enables re-allowing blocked paths** once the threat is resolved.
5. **No complex threshold math.** Binary signals drive binary actions (block/allow).

### Slot Alignment Fix

The slot misalignment bug must be fixed. Options:
1. **Fixed-position slots:** Always reserve slot i for agent i (skip self).
   Pad with zeros if agent is unreachable.
2. **Embed sender ID in message:** Use bits to identify the sender, making
   slot order irrelevant.

Option 1 is simplest. Verify how BlueFlatWrapperV2 handles message layout.

### Experiments Needed

1. **Implement new protocol and measure:** Does coordinated blocking via messaging
   improve over independent blocking?
2. **Compare REQUEST_BLOCK vs autonomous blocking:** Does explicit coordination
   outperform each agent independently deciding to block?
3. **Measure ZONE_CLEAR benefit:** How much does re-allowing traffic help green?

---

## Phase C: DegradeServices Mitigation

### Why This is Hardest

DegradeServices is completely invisible — no conn_flag, no proc_flag, no malfile.
The only evidence is indirect:

1. **Precursor visibility:** Red MUST Exploit (conn_flag) and PrivEsc (malfile)
   before DegradeServices. If we saw those, we know DegradeServices is coming.
2. **Reward signal:** Reward drops without visible alerts = DegradeServices running
   somewhere. But the agent doesn't observe rewards directly.
3. **Service reliability is internal state:** Not exposed in observations.

### Approach 1: Shadow Tracking (Inference)

Track per-host "likely degraded" state based on observed attack chain timing:

```python
# If we saw red indicators and Restore hasn't completed yet:
if host in self._last_red_seen:
    time_since_red = self._step - self._last_red_seen[host]
    if time_since_red >= 4 and host not in self._restore_at:
        # DegradeServices likely running (4 steps = Exploit+PrivEsc+Degrade)
        # This host is silently accumulating damage
        self._likely_degraded.add(host)
```

**Use:** Prioritize Restoring "likely degraded" hosts even without current flags.
After Restore completes, if no new flags appear but the host was previously
compromised, red may have re-exploited silently.

**Risk:** Over-Restoring on false inferences. The flag_age=1 threshold is already
optimal for visible alerts; adding inference-based Restores must be carefully gated.

### Approach 2: Preventive Blocking (The Better Strategy)

Instead of detecting DegradeServices after it starts, **prevent red from reaching
the DegradeServices state** by blocking the network paths it needs:

```
Red needs:  Exploit (network) → PrivEsc (local) → DegradeServices (local)
                ↑
         BLOCK HERE — prevents the entire chain
```

Blocking at the Exploit stage prevents PrivEsc AND DegradeServices.
This is exactly what Phase A (preemptive OZ blocking) achieves.

**Key insight:** DegradeServices mitigation IS coordinated blocking.
If we prevent red from exploiting the host, DegradeServices never happens.
The 83.3% LWF damage that comes from DegradeServices is prevented by
blocking red's network access to high-value hosts.

### Approach 3: Proactive Restore for "Ghost" Infections

For hosts that were previously compromised and Restored:

```python
# Track re-infection risk
if host in self._restore_history:
    last_restore = self._restore_history[host]
    if self._step > last_restore + RESTORE_DUR + 10:
        # 10 steps since Restore — if red re-exploited, DegradeServices
        # could be running invisibly now
        # Only act if no other urgent tasks
        if self._is_idle() and host in high_value_hosts:
            issue_restore(host)  # "refresh" Restore
```

**Risk:** This is essentially the proactive Restore strategy that was tested
and failed (-461.2 worse). The key difference is gating it to:
- Only previously-compromised hosts
- Only when the agent is idle (no other tasks)
- Only high-value hosts (OZ servers during active phase)

This is the most speculative approach and should be tested last.

### Experiments Needed

1. **Shadow tracking value test:** Does tracking "likely degraded" hosts and
   prioritizing them improve over V10b?
2. **Combination test:** Phase A blocking + Phase B messaging + shadow tracking.
3. **Targeted proactive Restore:** Only on previously-compromised OZ server_host_0
   during active phase, only when idle.

---

## Implementation Order

```
Step 1: Trace comms_policy per phase (understand what's already blocked)
Step 2: Measure green cross-subnet OZ access (confirm ASF=0 is truly free)
Step 3: Implement Phase A (preemptive OZ blocking) — highest expected impact
Step 4: Test Phase A in isolation (30 eps, seed 42)
Step 5: Fix slot alignment bug (prerequisite for Phase B)
Step 6: Implement Phase B (messaging redesign with RED_DETECTED/REQUEST_BLOCK)
Step 7: Test Phase A + B combined
Step 8: Implement Phase C shadow tracking (if A+B show improvement)
Step 9: Full validation (100 eps, multi-seed)
```

### Success Criteria

- Phase A alone: >= -750 mean reward (>= 7.8% improvement over -814)
- Phase A+B combined: >= -700 mean reward (>= 14% improvement)
- Phase A+B+C combined: >= -650 mean reward (>= 20% improvement)

### Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| A | comms_policy may already block these paths | Trace first |
| A | Blocking may affect green access (ASF) | ASF=0 in active phase |
| B | Messaging still has no value | New protocol targets P1-P3, not P4 |
| B | Slot misalignment | Fix as prerequisite |
| C | Proactive Restore catastrophic | Gate heavily, test isolated |

---

## Architecture Diagram

```
                    Current V10b                        V11 (Proposed)
                    ============                        ==============

Agent sees          Agent sees red                      Agent sees red
red flags  ──────►  Restore host  ──────►  Done         flags + peer msgs
                                                              │
                                                    ┌────────┴────────┐
                                                    ▼                 ▼
                                              Restore host    Broadcast RED_DETECTED
                                                    │          + REQUEST_BLOCK
                                                    ▼                 │
                                              Redeploy decoys        ▼
                                                              Downstream agent
                                                              blocks traffic
                                                              FROM red zone
                                                                    │
                                                                    ▼
                                                              Red CANNOT exploit
                                                              downstream hosts
                                                                    │
                                                                    ▼
                                                              DegradeServices
                                                              PREVENTED on
                                                              downstream hosts
```

### Red Attack Chain vs Blue Defense

```
Red path:    internet ──► contractor ──► HQ ──► RZA ──► OZA
                                          │              │
Blue:        agent_4 watches HQ    agent_0 watches RZA   agent_1 watches OZA
                    │                      │                     │
             detects red,           detects red,          blocks traffic
             broadcasts             broadcasts            FROM RZA
             RED_DETECTED           REQUEST_BLOCK          (ASF=0 in Phase 1)
                    │                      │                     │
                    └──────────────────────┘                     │
                         messaging                               ▼
                                                          Red BLOCKED
                                                          at RZA→OZA
                                                          chokepoint
```

---

## Appendix: Verification Queries

Before implementing, run these diagnostic queries:

### 1. What does comms_policy actually mandate?

```python
# In evaluation loop, after env.reset():
for step in range(500):
    obs_dict, _, _, _, _ = env.step(actions, messages=messages)
    phase = int(list(obs_dict.values())[0][0])
    # Parse comms_policy from observation for each agent
    # Record which subnet pairs should be blocked per phase
```

### 2. How often does green cross subnets to OZ?

```python
# Trace GreenAccessService actions
# Count how many target OZA hosts from non-OZA sources
# Count how many target OZB hosts from non-OZB sources
```

### 3. Does BlockTrafficZone have a duration?

```python
# Check ControlTraffic for duration attribute
# Block/Allow are instant (duration=1? or 0?)
```

### 4. Can agent_1 block paths it doesn't "own"?

```python
# Each agent controls specific subnets
# Can agent_1 (OZA) issue BlockTrafficZone(from=RZA, to=OZA)?
# Or can it only block paths TO its own subnets?
# Check action labels to verify available block actions
```
