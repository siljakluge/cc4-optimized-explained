# V11b Messaging Redesign: Experimental Results

**Agent:** EnterpriseHeuristicAgentV11b  
**Baseline:** V10b at -735.1 +/- 184.5 (60 eps, seeds 42+123)  
**Date:** 2026-04-08

---

## Executive Summary

The V11b messaging redesign produces results statistically indistinguishable from V10b
(delta = +25.8, p = 0.50, 60 episodes). The coordinated blocking feature never triggers
because **comms_policy already blocks all paths to the active operational zone during
mission phases**. This is a fundamental architectural limitation that no messaging
protocol can overcome.

---

## Key Finding: comms_policy Already Blocks Everything

The central hypothesis of Phase B was that agents could coordinate to block traffic
from compromised zones. Diagnostic inspection revealed:

**Phase 1 (OZA is impact target):**
```
RZA -> OZA: comms_policy_block = True  (already mandated)
All other paths -> OZA: comms_policy_block = True
```

**Phase 2 (OZB is impact target):**
```
RZB -> OZB: comms_policy_block = True  (already mandated)
admin -> OZB: blocked=True, comms_policy_block=True
contractor -> OZB: blocked=True, comms_policy_block=True
internet -> OZB: blocked=True, comms_policy_block=True
office -> OZB: blocked=True, comms_policy_block=True
OZA -> OZB: blocked=True, comms_policy_block=True
PAZ -> OZB: blocked=True, comms_policy_block=True
RZA -> OZB: blocked=True, comms_policy_block=True
RZB -> OZB: blocked=False, comms_policy_block=True  (pending P3 block)
```

Every single path to the active OZ has `comms_policy_block=True`. The existing P3
priority (comms_policy enforcement) already blocks these paths within 1-2 steps of
phase transition. **There are zero unblocked paths for coordinated messaging to add
value.**

---

## Experimental Results

### Setup

- 30 episodes per seed, seeds 42 and 123 (60 total)
- 500 steps per episode, FiniteStateRedAgent, EnterpriseGreenAgent
- BlueFlatWrapperV2 wrapper
- Side-by-side comparison on identical environments

### Results Table

| Metric | V10b | V11b | Delta |
|---|---|---|---|
| **Seed 42** (30 eps) | -717.0 +/- 206.8 | -711.8 +/- 211.3 | +5.2 (p=0.93) |
| **Seed 123** (30 eps) | -753.2 +/- 157.0 | -706.8 +/- 240.8 | +46.3 (p=0.39) |
| **Aggregate** (60 eps) | -735.1 +/- 184.5 | -709.3 +/- 226.5 | +25.8 (p=0.50) |

None of the results are statistically significant. V11b is functionally identical
to V10b because the coordinated blocking feature has zero trigger opportunities.

### Coordinated Blocking Statistics

Across 5 diagnostic episodes (2,500 agent-steps per episode):
- Coordinated blocks issued: **0**
- Coordinated allows issued: **0**
- REQUEST_BLOCK messages sent by agent_0: 12/500 steps (red in RZA)
- REQUEST_BLOCK messages sent by agent_2: 15/500 steps (red in RZB)
- All target paths already blocked by comms_policy when REQUEST_BLOCK arrives

---

## Protocol Design (Implemented but Inert)

### New 8-bit Message Format

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | RED_DETECTED | 1 = confirmed red in my zone |
| 1 | RED_ROOT | 1 = root-level red (malfile only) |
| 2 | BUSY_RESTORING | 1 = currently restoring a host |
| 3 | ZONE_CLEAR | 1 = no alerts, all hosts clean |
| 4-5 | THREAT_COUNT | 0-3 hosts with active threats |
| 6 | REQUEST_BLOCK | 1 = downstream should block FROM my zone |
| 7 | RESERVED | 0 |

### Surgical Blocking Rules

Only two blocking rules exist (phase-aware, ASF=0):
- Phase 1: agent_1 (OZA) blocks RZA->OZA when agent_0 sends REQUEST_BLOCK
- Phase 2: agent_3 (OZB) blocks RZB->OZB when agent_2 sends REQUEST_BLOCK

Both rules are rendered moot by comms_policy.

### Dead Code Removed from V10b

- T2 escalation (threshold=1 = default, dead code)
- Compound escalation (never triggers, dead code)
- RESTORING bit consumption (parsed but never used)
- OPEN_PATHS bit consumption (only used in compound, dead code)
- Upstream-only message routing (replaced by all-peer reading)

---

## Failure Analysis: Three Catastrophic Iterations

### Iteration 1: All-agents-all-peers blocking

Initial implementation allowed ANY agent to block traffic from ANY peer's zone
based on REQUEST_BLOCK. Agent_4 (HQ) sends REQUEST_BLOCK 496/500 steps because
HQ is always under attack. This caused ALL other agents to waste actions blocking
HQ paths, creating massive ASF penalties.

**Result: -5000 to -7000 per episode (7-10x worse than V10b)**

Root cause: HQ agents are always under attack, so broad REQUEST_BLOCK is always on.

### Iteration 2: Any-peer threshold=0 in P4

Using `any_peer_high_threat` (any peer with 3+ threats -> threshold=0) caused
catastrophic over-restoring from green false positives. Agent_4 always has
threat_count=3 (3+ hosts under attack), so ALL agents set threshold=0 for ALL
proc_flag hosts. Green FPs (~155/episode) each trigger a 5-step Restore.

**Result: -5000+ per episode (same catastrophic regression as exp2)**

Root cause: P4 threshold=0 from peer messages is the exact same failure mode as
the "aggressive T2" experiment (-1029, p=0.002 harmful).

### Iteration 3: Surgical blocking (final)

Restricted coordinated blocking to only the RZ->OZ path for the active OZ.
This avoided all ASF penalties but also had zero trigger opportunities because
comms_policy already blocks these paths.

**Result: Identical to V10b (delta = +25.8, p = 0.50)**

---

## Implications for V11 Plan

### Phase A (Preemptive OZ Blocking): Likely Redundant

The same comms_policy that blocks RZ->OZ also blocks all other paths to the active
OZ. Phase A's "block non-essential sources to OZ" is already handled by comms_policy.
However, Phase A may still have value if comms_policy's blocking is delayed (it takes
P3 priority, which is lower than P1/P1b/P1c and may be preempted by Restores).

### Phase B (Messaging Redesign): **Confirmed No Value**

Messaging cannot add blocking value because there are no unblocked paths to protect.
The only potential remaining value is:
1. Threshold adjustment (proven harmful in experiments)
2. Restore coordination (not attempted -- agents rarely overlap)
3. Information that isn't about blocking (unclear what would help)

### Phase C (DegradeServices Mitigation): Independent of Messaging

Shadow tracking and predictive Restore don't depend on inter-agent messages.
They depend on local observation history. Phase C should be pursued independently.

### The Real Performance Bottleneck

The V10b agent already blocks all comms_policy paths, restores all visible red,
and deploys decoys. The remaining -735 reward gap comes from:
1. **DegradeServices invisibility** (83.3% of damage, zero observable flags)
2. **Phase transition delays** (comms_policy changes take 1-2 steps to enforce)
3. **Green false positive Restores** (threshold=1 filters most but not all)
4. **Restore duration** (5 steps, during which host is undefended)

None of these are addressable via inter-agent messaging.

---

## Files

- Agent: `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgentV11b.py`
- Evaluation: `scripts/evaluate_v11b.py`
- Diagnostics: `scripts/diagnose_v11b.py`, `scripts/diagnose_v11b_msg_flow.py`
