# V10b Messaging Protocol Critique and Experimental Analysis

**Agent:** EnterpriseHeuristicAgentV10b  
**Protocol:** v9 (8-bit, 5 agents)  
**Baseline:** -782.2 +/- 195.7 (30 episodes, seeds 42+123)  
**Date:** 2026-04-08

---

## Executive Summary

The v9 inter-agent messaging protocol in V10b is **architecturally sound but operationally
inert**. Five experiments totaling 165 agent-episodes demonstrate that the current
messaging implementation has no statistically significant effect on performance.
Disabling messaging entirely (Experiment 1) produces results indistinguishable from
the baseline (delta = +15.7, p = 0.80). The root cause: 86.7% of agent-steps have
no upstream relationship, the only non-dead escalation path (T3) fires once per
2,500 steps, and two of three escalation tiers are dead code.

---

## Dead Code Inventory

### 1. T2 Escalation (Priority 4, lines 371-372)

```python
elif peer_escalate_t2 or peer_escalate_compound:
    threshold = 1  # DEAD: same as default on line 374
else:
    threshold = 1  # default
```

T2 fires 611 times across 5 episodes (37% of upstream-available steps) but sets
threshold=1, identical to the default. **Zero behavioral impact.** The v9 protocol
documentation describes a default threshold of 2, but V10b changed it to 1 without
updating the T2 tier.

### 2. Compound Escalation (Priority 4, lines 360-364)

```python
peer_escalate_compound = (
    peer_state.get("upstream_threat", 0) >= 1
    and peer_state.get("upstream_open_paths", 0) > 0
    and peer_state.get("upstream_decoys_bypassed", False)
)
```

Triggers 0 times across 5 episodes. Requires all three conditions simultaneously,
which is extremely unlikely given that `decoys_bypassed` detection requires a very
specific sequence of events. Even if it triggered, it sets threshold=1 (dead).

### 3. RESTORING Bit (Bit 7)

Broadcast by every agent but the `upstream_restoring` field is **never referenced**
in any decision logic. The signal is parsed, stored in the peer_state dict, and
discarded. Complete dead signal.

### 4. OPEN_PATHS Signal (Bits 2-3)

Only consumed in the compound escalation condition, which itself is dead code (see
item 2). The open_paths information is encoded, broadcast, and parsed but never
influences any action selection.

### 5. P1b Upstream Decoys Bypassed

The `upstream_decoys_compromised` check in Priority 1b (line 290) is technically
live code, but the diagnostic shows it fires 0 times across 5 episodes. The
upstream decoy bypass signal is too rare in practice to matter.

### Dead Code Summary

| Signal/Path | Status | Trigger Rate | Behavioral Impact |
|---|---|---|---|
| T3 escalation | Barely alive | 0.06% of upstream steps | threshold 1->0 |
| T2 escalation | Dead (threshold=default) | 37% of upstream steps | None |
| Compound escalation | Dead (never triggers) | 0% | None |
| RESTORING bit | Dead (never consumed) | N/A | None |
| OPEN_PATHS bits | Dead (only in compound) | N/A | None |
| P1b decoy bypass | Effectively dead | 0% observed | None |

---

## Structural Flaws

### 1. Upstream Map Too Narrow

```python
_UPSTREAM = {
    (1, 1): 0,   # Phase 1: OZA watches RZA
    (2, 3): 2,   # Phase 2: OZB watches RZB
}
```

Only 2 of 15 possible (phase, agent) combinations have upstream relationships.

| Phase | Agents with upstream | Agents without |
|---|---|---|
| 0 | None | 0, 1, 2, 3, 4 |
| 1 | agent_1 | 0, 2, 3, 4 |
| 2 | agent_3 | 0, 1, 2, 4 |

**86.7% of agent-steps have no upstream.** Agent_4 (HQ/admin/office/PAZ) NEVER
has upstream despite being on the red attack chain (contractor -> HQ -> RZ -> OZ).

### 2. Message Slot Misalignment Bug

When an agent becomes network-isolated (firewall blocks all paths), CybORG
stops delivering its message to peers. The `BlueFlatWrapper` pads with empty
messages to maintain 4 slots, but this creates a slot-shift:

```
Normal (all connected):
  agent_3 receives: [msg_0, msg_1, msg_2, msg_4]  (sorted, self excluded)
  Code maps:        slot 0=agent_0, slot 1=agent_1, slot 2=agent_2, slot 3=agent_4

After agent_1 isolated:
  agent_3 receives: [msg_0, msg_2, msg_4, EMPTY]  (3 msgs + 1 pad)
  Code maps:        slot 0=agent_0, slot 1=agent_1, slot 2=agent_2, slot 3=agent_4
  ACTUAL:           slot 0=agent_0, slot 1=agent_2, slot 2=agent_4, slot 3=EMPTY
```

In Phase 2, agent_3 looks for upstream agent_2 at slot 2 but reads agent_4's
message instead. This is a correctness bug, though its impact is limited because
messages are mostly zeros and the escalation paths are dead code anyway.

### 3. 77% of Messages Are All-Zeros

Agents that see no threats (clean observation) send all-zero messages. Since red
only attacks one zone at a time in a sequential pattern, most agents are idle most
of the time.

```
Threat level distribution (across 2,855 non-zero messages in 5 episodes):
  Level 0 (clean):      243  (8.5%)
  Level 1 (decoy hit):  797  (27.9%)
  Level 2 (real red):   507  (17.8%)
  Level 3 (root):     1,308  (45.8%)
```

The information IS being encoded -- 45.8% of non-zero messages report root-level
threats. But it is almost never consumed.

---

## Experimental Results

### Setup

- 15 episodes per seed, seeds 42 and 123 (30 episodes for baseline, exp1, exp4, exp5)
- 15 episodes seed 42 only for exp2, exp3
- 500 steps per episode
- FiniteStateRedAgent, EnterpriseGreenAgent
- BlueFlatWrapperV2

### Results Table

| Experiment | N | Mean | Std | vs Baseline | p-value | Cohen's d |
|---|---|---|---|---|---|---|
| **baseline** | 30 | -782.2 | 195.7 | --- | --- | --- |
| exp1: no messaging | 30 | -766.5 | 263.5 | +15.7 | 0.798 | +0.07 |
| exp2: aggressive T2 | 15 | -1029.0 | 227.4 | -246.8 | **0.002** | -1.16 |
| exp3: expanded upstream | 15 | -830.3 | 226.1 | -48.2 | 0.501 | -0.23 |
| exp4: RESTORING coord | 30 | -751.3 | 168.4 | +30.8 | 0.523 | +0.17 |
| exp5: OPEN_PATHS priority | 30 | -735.7 | 187.8 | +46.5 | 0.360 | +0.24 |

### Interpretation

**Experiment 1 (ablation):** Disabling all messaging has NO statistically significant
effect (p=0.80). The messaging system's total contribution is indistinguishable from
zero. On seed 42, no-messaging actually scored BETTER (-725.7 vs -808.0); on seed
123, it scored worse (-807.3 vs -756.3). The variance overwhelms the signal.

**Experiment 2 (aggressive T2):** Making T2 escalation trigger threshold=0 is
SIGNIFICANTLY HARMFUL (p=0.002, d=-1.16). Why? Because T2 fires 37% of the time
when upstream is available, and threshold=0 means immediate Restore on ANY proc_flag.
Green agents generate ~155 false-positive proc_flags per episode. Threshold=0 on
37% of steps causes massive over-Restoring (5-step lockout each time) that cripples
the defense.

**Experiment 3 (expanded upstream):** Adding more upstream relationships slightly
worsens performance (-48.2), not significant (p=0.50). More upstream relationships
means more T2 triggers (which are dead code at threshold=1), plus the slot
misalignment bug becomes more likely to cause misattribution.

**Experiment 4 (RESTORING coordination):** Using the RESTORING bit to trigger
immediate Restore shows a non-significant improvement (+30.8, p=0.52) with
notably lower variance (168.4 vs 195.7). The RESTORING signal fires more rarely
than T2, so it doesn't cause the over-Restoring problem that killed exp2.

**Experiment 5 (OPEN_PATHS priority):** Shows a non-significant improvement (+46.5,
p=0.36). On seed 123, it produced IDENTICAL results to baseline (all 15 episodes
matched exactly), confirming the OPEN_PATHS + threat >= 1 condition rarely triggers.
On seed 42, it improved a few episodes where the condition fired.

---

## Root Cause Analysis

The messaging protocol fails to deliver value because of a **design mismatch between
information production and consumption**:

1. **Producers are active:** Agents correctly encode threat state in messages.
   45.8% of non-zero messages report root-level threats.

2. **Consumers are mostly deaf:** Only 2 of 5 agents ever have an upstream
   relationship, and only during 1 of 3 phases each. 86.7% of agent-steps
   cannot consume upstream signals at all.

3. **The one active path (T3) requires extreme conditions:** upstream_red_count >= 3
   means 3+ hosts simultaneously compromised in the upstream zone. Red in CC4
   typically compromises hosts sequentially and moves on, rarely accumulating 3+
   concurrent infections in one zone. T3 fires 0.06% of the time.

4. **The default threshold is already aggressive:** V10b uses threshold=1 by default
   (Restore after 1 step of persistent proc_flag). This is already near-optimal
   for green FP filtering. There is very little room for messaging to improve on it.

5. **The real battles are already won by local observation:** P1 (conn+malfile),
   P1b (conn-only), and P1c (malfile-only) handle the majority of red detections
   without any messaging at all. P4's proc_flag Restore with threshold=1 catches
   the rest. Messaging only matters for P4 threshold adjustment, which is the
   lowest-impact priority.

---

## Recommendations

### Immediate (zero-risk improvements)

1. **Remove dead code.** Delete T2 and compound escalation branches (they set
   threshold=1 = default). Remove RESTORING and OPEN_PATHS parsing since they're
   unused. This reduces code complexity with zero behavioral change.

2. **Fix the slot misalignment bug.** Either:
   - Track which agents are connected and skip missing slots correctly, or
   - Have the wrapper include sender identity in the message section, or
   - Use a fixed-position approach where each agent's slot is always at the same
     index regardless of connectivity (pad per-agent rather than per-list).

3. **Document that messaging has no measurable value** in the current architecture
   so future developers don't waste time optimizing it.

### Medium-term (requires validation)

4. **Expand P1b to use any-peer (not just upstream) decoy bypass.** Change from
   `upstream_decoys_bypassed` to `any_decoys_bypassed` using the `any_real_red`
   aggregation pattern already in the code. This is the most likely path to actual
   messaging value since P1b handles conn-only alerts (common).

5. **If messaging is kept,** repurpose the 8 bits for information that directly
   influences P1-P3 (the high-impact priorities):
   - Bit: "Red is in MY zone right now" (binary, not threat level)
   - Bit: "I just Restored a host" (coordination to avoid double-Restore)
   - Bits: Phase-specific red progress indicator (which subnet red has reached)

6. **Consider removing messaging entirely.** The ablation shows no performance
   loss. Removing messaging simplifies the codebase, eliminates the slot
   misalignment bug, and removes 80+ lines of encoding/decoding logic that
   produces no value. The 8 bits can be set to all-zeros permanently.

### Long-term (research directions)

7. **If cross-agent coordination is desired,** the problem is not the message
   format but the action architecture. The real coordination opportunity is in
   **firewall blocking decisions** (P2/P3): agents should coordinate which paths
   to block based on where red currently is, not just follow local comms_policy.
   This requires redesigning the priority system, not just the message bits.

---

## Appendix: Experimental Code

Variant agent files:
- `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgentV10b_exp1_no_msg.py`
- `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgentV10b_exp2_aggressive.py`
- `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgentV10b_exp3_expanded_upstream.py`
- `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgentV10b_exp4_restoring_coord.py`
- `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgentV10b_exp5_open_paths.py`

Evaluation script:
- `scripts/evaluate_messaging_experiments.py`

### Raw Episode Data

**Seed 42 (15 episodes each):**

| Episode | Baseline | No Msg | Aggressive | Expanded | Restoring | OpenPaths |
|---|---|---|---|---|---|---|
| 1 | -690 | -615 | -705 | -690 | -690 | -690 |
| 2 | -690 | -635 | -910 | -900 | -695 | -690 |
| 3 | -590 | -1020 | -935 | -570 | -590 | -590 |
| 4 | -720 | -665 | -1100 | -850 | -720 | -720 |
| 5 | -825 | -585 | -840 | -870 | -825 | -825 |
| 6 | -1090 | -440 | -1115 | -1305 | -1090 | -1090 |
| 7 | -595 | -675 | -1185 | -810 | -595 | -595 |
| 8 | -565 | -625 | -595 | -875 | -565 | -565 |
| 9 | -495 | -785 | -1500 | -650 | -485 | -490 |
| 10 | -1300 | -870 | -1280 | -730 | -800 | -560 |
| 11 | -1230 | -545 | -985 | -1350 | -915 | -485 |
| 12 | -1050 | -1305 | -1305 | -685 | -770 | -440 |
| 13 | -685 | -460 | -1070 | -575 | -580 | -900 |
| 14 | -910 | -790 | -1030 | -945 | -515 | -775 |
| 15 | -685 | -870 | -880 | -650 | -705 | -1310 |
| **Mean** | **-808.0** | **-725.7** | **-1029.0** | **-830.3** | **-702.7** | **-715.0** |

**Seed 123 (15 episodes, subset):**

| Episode | Baseline | No Msg | Restoring | OpenPaths |
|---|---|---|---|---|
| 1 | -755 | -775 | -755 | -755 |
| 2 | -685 | -1435 | -685 | -685 |
| 3 | -820 | -875 | -820 | -820 |
| 4 | -905 | -1240 | -940 | -905 |
| 5 | -815 | -515 | -940 | -815 |
| 6 | -845 | -415 | -630 | -845 |
| 7 | -720 | -390 | -775 | -720 |
| 8 | -1025 | -635 | -960 | -1025 |
| 9 | -490 | -685 | -895 | -490 |
| 10 | -595 | -590 | -760 | -595 |
| 11 | -630 | -1120 | -775 | -630 |
| 12 | -815 | -575 | -985 | -815 |
| 13 | -825 | -1020 | -520 | -825 |
| 14 | -675 | -865 | -1080 | -675 |
| 15 | -745 | -975 | -480 | -745 |
| **Mean** | **-756.3** | **-807.3** | **-800.0** | **-756.3** |

### Diagnostic Statistics (5 episodes, seed 42)

```
Messages sent:              12,475 (2,495 agent-steps)
Messages all-zero:           9,620 (77.1%)
Upstream available:          1,655 (13.3%)
Upstream unavailable:       10,820 (86.7%)
T3 escalation triggers:         1  (0.06% of upstream steps)
T2 escalation triggers:       611  (36.9% of upstream steps, all dead)
Compound escalation:             0  (never)
Max upstream_red_count:          3  (reached once)
P1b decoy bypass influence:      0  (never)
```
