# V11 Optimization Synthesis Report

**Date:** 2026-04-08  
**Baseline:** V10b at -814.0 +/- 247.7 (100 eps, seed 42)  
**Best Result:** V11a at -700.0 +/- 160.5 (30 eps, seed 42)  
**Improvement:** +14.0% reward, 35% lower variance  

---

## Executive Summary

V11 targeted three architectural changes to break through V10b's local optimum:
1. **Phase A: Preemptive OZ Blocking** — SUCCESS (+14.0%)
2. **Phase B: Messaging Redesign** — No improvement (NS, p=0.50)
3. **Phase C: DegradeServices Shadow Tracking** — Not implemented (blocked by Phase B failure)

V11a (Phase A only) is the new best agent and has been deployed as the official submission.

---

## Background: Why V10b Was Stuck

22 single-parameter experiments across decoy strategy, priority ordering, messaging protocol, and Restore timing found **zero improvements** to V10b. The agent was at a local optimum.

The remaining performance gap (-814 vs theoretical ~-300) is bounded by:
- **DegradeServices invisibility** — 83.3% of damage, zero observable flags
- **Independent agent decisions** — 5 agents act alone, messaging has zero value
- **Reactive-only posture** — agents wait for flags, then respond

Breaking through required **architectural changes**, not parameter tuning.

---

## Phase A: Preemptive OZ Blocking — SUCCESS

### Critical Discovery

During active mission phases, the Active Service Factor (ASF) for the impact-target OZ subnet is **zero**. This means blocking traffic to that subnet costs nothing — no green penalty. Meanwhile, LWF=-10/step for each degraded host. The asymmetry is enormous: **blocking costs 0, not-blocking costs -10/step per degraded host.**

### Diagnostic Findings

Before implementing, diagnostics revealed that comms_policy **already** mandates blocking all 8 paths to the active OZ during its active phase. The value of V11a is NOT in blocking paths that comms_policy leaves open (there are almost none). The value is in:

1. **Faster compliance with comms_policy blocks at phase transitions** — V10b's agents may be busy Restoring and take 5-10 steps to execute the block
2. **Preemptive blocking of the single RZ->OZ path** ~10 steps before phase transitions
3. **Preventing the vulnerability window** where V10b is busy Restoring and hasn't yet executed the critical block

### Implementation

- **P_BLOCK_OZ priority** (between P1c Restore and P2 Allow): During active phases, blocks any open path to the active OZ above P2 Allow priority
- **Preemptive blocking**: Phase 0 step >= 157: block RZA->OZA. Phase 1 step >= 324: block RZB->OZB
- **Cost**: -7 reward for ~7 steps of preemptive blocking in Phase 0. Prevents potentially -10/step * N_hosts * M_steps of LWF damage

### Results

| Metric | V10b (100ep) | V11a (30ep, seed 42) | V11a (30ep, seed 123) |
|--------|-------------|---------------------|----------------------|
| Mean reward | -814.0 | **-700.0** | **-695.2** |
| Std deviation | 247.7 | 160.5 | 174.7 |
| vs V10b | -- | **+14.0%** | **+14.6%** |

Cross-seed consistency: only 4.8 difference between seeds (0.7%). Highly robust.

---

## Phase B: Messaging Redesign — NO IMPROVEMENT

### Approach

Complete redesign of the 8-bit messaging protocol:
- New signals: RED_DETECTED, RED_ROOT, BUSY_RESTORING, ZONE_CLEAR, THREAT_COUNT, REQUEST_BLOCK
- All agents read all peer messages (not just upstream)
- Coordinated blocking based on peer signals

### Why It Failed

Comms_policy already blocks all paths to the active OZ during active phases. There are **no paths left to block** via messaging coordination. The messaging redesign is solving a problem that doesn't exist — comms_policy already provides the coordination.

### Result

V11b (V11a + messaging redesign): -709.3 +/- 226.5 (seed 42, 30 eps). Delta = -9.3, p = 0.50. Not significant.

**Conclusion:** Messaging remains dead code. The environment's built-in comms_policy provides sufficient coordination without inter-agent communication.

---

## Phase C: DegradeServices Shadow Tracking — NOT IMPLEMENTED

Phase C was contingent on Phase B providing a coordination mechanism. Since Phase B failed, and the primary DegradeServices mitigation strategy (blocking at Exploit stage) is already implemented by Phase A, Phase C was deprioritized.

The remaining DegradeServices damage is from:
1. Red exploiting through the RZ->OZ path during the brief window at phase transitions (mostly eliminated by V11a)
2. Red exploiting hosts in non-OZ subnets (LWF=-1 or -2/step, much lower impact)
3. Re-exploitation of previously Restored hosts

Shadow tracking (Approach 1 from the plan) could still be explored, but the expected benefit is small given that V11a already eliminates the highest-impact vulnerability window.

---

## Performance Ladder (Updated)

```
Agent                         Mean Reward   vs SleepAgent   Notes
----------------------------------------------------------------
SleepAgent                    -30,579       0.0%            Do nothing
Oracle V1 (Remove-first)       -1,558      94.9%           Perfect info, bad strategy
v9.1 heuristic                 -1,039      96.6%           Previous production agent
Oracle V3 (optimal)              -893.5    97.1%           Perfect info, no decoys
V10b (Restore-only)              -814.0    97.3%           Previous best
V11a (Preemptive OZ Blocking)    -700.0    97.7%           CURRENT BEST
Theoretical floor                 ~-300    99.0%           Estimated from reward analysis
```

---

## Key Insights

### 1. Architecture > Parameters

22 parameter experiments found zero improvements. One architectural change (preemptive OZ blocking) yielded +14%. This confirms that V10b's local optimum could only be escaped via structural changes to the decision logic.

### 2. Exploit the Reward Structure

The ASF=0 insight was the key enabler. Understanding the phase-specific reward structure revealed a free action (blocking) that the agent wasn't exploiting. Domain knowledge of the environment's scoring rules was more valuable than any amount of parameter search.

### 3. Timing > Coverage

V11a doesn't block more paths than V10b — it blocks the **same** paths **faster**. The value is entirely in eliminating the transition window, not in expanding blocking coverage. This is a subtle but important distinction.

### 4. Messaging Is Structurally Worthless in CC4

Two independent experiments (v10b M1 ablation, v11b redesign) confirm that inter-agent messaging provides zero value. The environment's comms_policy already coordinates agent blocking decisions. This is a fundamental property of the CC4 environment, not a failure of protocol design.

### 5. Prevention > Detection for Invisible Attacks

DegradeServices is 83.3% of damage and completely invisible. No amount of detection tuning can address it. The only viable strategy is preventing the attack chain at the Exploit stage (network blocking), which is exactly what V11a does for the highest-impact targets.

---

## Remaining Optimization Opportunities

### High Confidence (Small Impact Expected)

1. **Dead code cleanup**: Remove ~80 lines of messaging protocol code that has zero value
2. **Conn-flag persistence tracking**: Persistent conn_flag (2+ steps) = almost certainly real red. Could enable threshold=0 for persistent signals without FP penalty

### Speculative (Unknown Impact)

3. **Shadow tracking for re-exploitation**: Track previously-compromised hosts and issue proactive Restore when idle. Risk of over-Restoring.
4. **Phase 0 optimization**: Phase 0 accounts for 1/3 of the episode. Current strategy is purely reactive. Could preemptively block contractor->HQ during Phase 0 (ASF=-1, low cost).
5. **Decoy-depletion-aware thresholds**: Lower proc_flag threshold on hosts where all decoys have been consumed (red bypassed all decoys = high confidence).

---

## Files

| File | Description |
|------|-------------|
| `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgentV11a.py` | Current best agent |
| `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgentV11b.py` | Messaging redesign (no improvement) |
| `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgentV10b.py` | Previous best agent |
| `CybORG/Evaluation/submission/submission.py` | Official submission (updated to V11a) |
| `docs/v11_optimization_plan.md` | Original 3-phase plan |
| `docs/swarm_analysis/v11a_phase_a_results.md` | Phase A detailed results |
| `docs/swarm_analysis/v11b_messaging_results.md` | Phase B detailed results |
| `docs/swarm_analysis/v10b_comprehensive_critique.md` | V10b 22-experiment review |
| `scripts/evaluate_v11a.py` | V11a evaluation script |
| `scripts/evaluate_v11b.py` | V11b evaluation script |
| `scripts/v11a_diagnostics.py` | Action labels and comms_policy verification |
