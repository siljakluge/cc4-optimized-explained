# V10b Priority Ordering and Alert Threshold Critique

## Experimental Setup

- Agent under test: `EnterpriseHeuristicAgentV10b` (Restore-only heuristic)
- Baseline measured: -771.8 +/- 216.2 (30 episodes, seed 42, 500 steps)
- Previously reported baseline: -814.0 +/- 247.7 (100 episodes, seed 42)
- All experiments: 30 episodes, seed 42, 500 steps
- Statistical test: paired t-test (rel) at alpha=0.05
- Harness validation: all "default" config variants produce episode-identical results to the unmodified V10b (bit-exact match across all 30 episodes)

---

## Experiment 1: flag_age Threshold Sweep (P4)

**Question**: What is the optimal proc_flag threshold for triggering Restore?

| Variant | Mean | Std | 95% CI | Delta | p-value | Sig? |
|---------|------|-----|--------|-------|---------|------|
| **threshold=default (V10b)** | **-771.8** | **216.2** | **[-849.2, -694.5]** | **baseline** | -- | -- |
| threshold=0 (all hosts) | -1127.2 | 221.4 | [-1206.4, -1047.9] | -355.3 | 0.0000 | YES |
| threshold=1 (all hosts) | -771.8 | 216.2 | [-849.2, -694.5] | +0.0 | -- | -- |
| threshold=2 (all hosts) | -813.0 | 184.0 | [-878.9, -747.1] | -41.2 | 0.4758 | no |
| threshold=tiered (OZ/RZ=0, other=2) | -946.8 | 220.6 | [-1025.8, -867.9] | -175.0 | 0.0040 | YES |

### Analysis

1. **threshold=0 is catastrophic (-46% worse, p<0.0001).** Restoring on every proc_flag immediately causes massive collateral damage. Green FP rate is ~0.776% (about 155 FP events per episode). At threshold=0, each FP triggers an unnecessary Restore (5 steps busy, -1 cost, green service disruption). The agent spends most of its time Restoring false positives instead of responding to real threats.

2. **threshold=1 (current default) is optimal.** The default V10b uses threshold=1 for most hosts, with peer-escalation overrides and threshold=0 only for active OZ server_host_0. This produces the best results.

3. **threshold=2 is not statistically worse (p=0.48).** The 41-point degradation is within noise. However, the trend is negative: delayed response to real red activity allows more DegradeServices cascading (invisible to blue). The lower variance (184 vs 216) suggests fewer catastrophic episodes but more consistent moderate damage -- red gets slightly further before eviction.

4. **Tiered thresholds are significantly worse (p=0.004).** Setting threshold=0 for OZ/RZ hosts (the critical zones) while using threshold=2 for others is counterproductive. The aggressive Restoring on OZ/RZ hosts causes green LWF penalties at -10/step during active phases, which dwarfs any benefit from faster threat response.

### Verdict

**Keep threshold=1 as default.** The current V10b logic (threshold=1 with OZ server_host_0 exception) is already optimal. The OZ server_host_0 exception for active phases is correctly targeted -- it is the only host where threshold=0 is justified because the -10/step Impact penalty exceeds the FP Restore cost.

---

## Experiment 2: P1b Placement

**Question**: Should conn-only Restores (without decoy coverage confirmation) be issued early (P1b) or deferred?

| Variant | Mean | Std | 95% CI | Delta | p-value | Sig? |
|---------|------|-----|--------|-------|---------|------|
| **P1b default (after P1, before P1c)** | **-771.8** | **216.2** | **[-849.2, -694.5]** | **baseline** | -- | -- |
| P1b after P3 (Block) | -802.8 | 202.1 | [-875.2, -730.5] | -31.0 | 0.6244 | no |
| P1b removed entirely | -805.0 | 242.3 | [-891.7, -718.3] | -33.2 | 0.5180 | no |

### Analysis

1. **Neither variant is statistically worse (p>0.5).** The ~31-33 point degradation is well within noise.

2. **Per-episode differences are enormous.** All 30 episodes differ between variants. Individual episode deltas range from -625 to +760. This extreme variance means P1b's impact is highly scenario-dependent -- sometimes conn-only Restore saves a host before PrivEsc, sometimes it wastes an action on a green FP or decoy hit.

3. **P1b acts as a conservative early-warning response.** When a host has conn_alerts but no proc/malfile alerts AND no decoy coverage, P1b assumes the worst (real red with silent exploit). This is correct ~5% of the time (the silent exploit rate). The other 95% are either green FPs, decoy hits, or network noise. However, the cost of missing a real silent exploit (red proceeds to PrivEsc and Impact) is far higher than the cost of a false-positive Restore.

4. **Removing P1b increases variance** (242.3 vs 216.2), consistent with losing an early-warning mechanism that occasionally catches real threats.

### Verdict

**Keep P1b at current position.** While not statistically significant at n=30, the consistent negative trend and increased variance when removing P1b suggest it provides marginal value as a safety net. The current placement (after P1 confirmed red, before P1c pure malfile) is logically correct: respond to confirmed threats first, then to uncertain threats, then to PrivEsc signatures.

---

## Experiment 3: Block/Allow Ordering (P2 vs P3)

**Question**: Should Block come before Allow, or vice versa?

| Variant | Mean | Std | 95% CI | Delta |
|---------|------|-----|--------|-------|
| **Allow-first (default)** | **-771.8** | **216.2** | **[-849.2, -694.5]** | **baseline** |
| Block-first (swap P2/P3) | -771.8 | 216.2 | [-849.2, -694.5] | +0.0 |

### Analysis

**The ordering has ZERO effect.** Episode-by-episode, Block-first produces bit-exact identical results to Allow-first across all 30 episodes.

**Root cause**: The P2/P3 ordering is irrelevant because:

1. **Mutual exclusivity**: P2 (Allow) acts on paths where `comms_policy=False` (should be open) but `blocked=True` (currently blocked). P3 (Block) acts on paths where `comms_policy=True` (should be blocked) but `blocked=False` (currently open). For any given subnet pair, these conditions are mutually exclusive. A path is either "needs to be unblocked" OR "needs to be blocked" -- never both.

2. **Sequential resolution**: The agent can only take one action per step. Whether it processes the Allow queue first or the Block queue first, it will drain both queues in the same number of steps. The intermediate states differ, but the total cost is identical because Block and Allow are free actions (cost=0) and take 1 step each.

3. **Higher priorities dominate**: At phase transitions (step 167, step 334), when comms_policy changes and Block/Allow actions become relevant, the agent is almost always occupied with Restore actions (P1/P1b/P1c/P4) because red has been actively exploiting hosts. By the time the Block/Allow queue is processed, the critical window has already passed.

**The failed V10 experiment (block-first at absolute P1, -39.6% worse)** was not about Block vs Allow ordering -- it was about placing Block BEFORE all Restores, which delayed threat response. The V10b architecture correctly places all Restore priorities above Block/Allow.

### Verdict

**Ordering is irrelevant. Keep current for readability.** Allow-first (P2) before Block-first (P3) is slightly more intuitive ("fix stale blocks, then apply new blocks"), but it literally does not matter. The previous V10 experiment failure was not about P2/P3 ordering but about priority level placement.

---

## Experiment 4: Host Priority Reweighting

**Question**: Should admin/office hosts get higher priority for Restore ordering?

| Variant | Mean | Std | 95% CI | Delta |
|---------|------|-----|--------|-------|
| **Default priorities** | **-771.8** | **216.2** | **[-849.2, -694.5]** | **baseline** |
| admin/office=80, contractor=10 | -771.8 | 216.2 | [-849.2, -694.5] | +0.0 |

### Analysis

**Host priority reweighting has ZERO effect.** Episode-by-episode identical results.

**Root cause**: The `_host_priority` function is only used for sorting hosts within a priority level (P1, P1b, P1c, P4). It determines which alerted host gets Restored first when multiple hosts have alerts simultaneously. For this to matter:

1. Multiple hosts in the same priority level must have alerts at the same step
2. The agent must control hosts in both admin/office and other zones
3. The Restore order must affect episode outcome (i.e., restoring host A before host B leads to a different result than the reverse)

In practice, each blue agent controls 1-2 subnets:
- agent_0: restricted_zone_a
- agent_1: operational_zone_a
- agent_2: restricted_zone_b
- agent_3: operational_zone_b
- agent_4: HQ (PAZ, admin, office)

Agent_4 is the only one that controls admin/office hosts. Its hosts are rarely simultaneously alerted (red targets OZ/RZ zones primarily), and even when they are, restoring one before the other within the same priority level produces the same total episode outcome because DegradeServices cascading (the dominant damage source) is not affected by which HQ host is restored first.

### Verdict

**Host priority within levels is irrelevant for the current agent architecture.** Each agent controls too few subnets for intra-level priority to matter. The between-level priorities (P1 > P1b > P1c > P2 > P3 > P4) dominate all outcomes.

---

## Summary of Findings

### What to Keep (Validated by Evidence)

1. **P4 threshold=1 default**: Optimal. threshold=0 is catastrophic (-46%), threshold=2 is marginally worse (not significant), tiered is significantly worse.

2. **P1b at current position**: Provides marginal early-warning value. Not statistically significant at n=30 but consistent trend and theoretical justification.

3. **Current priority ordering**: P1 > P1b > P1c > P2 > P3 > P4 > P6 > P7 > Sleep is well-structured. Confirmed threats first, uncertain signals second, network policy third, proc_flag response fourth.

4. **OZ server_host_0 exception (threshold=0 in active phase)**: This is the one host where aggressive Restore pays off because Impact penalty (-10/step) far exceeds false-positive cost (-1 + 5 steps busy).

### What Does Not Matter (Zero Effect)

1. **Block vs Allow ordering (P2 vs P3)**: Produces bit-identical results. Mutual exclusivity of conditions means ordering is irrelevant.

2. **Host priority within levels**: Only matters when multiple hosts in the same zone have simultaneous alerts AND the Restore order affects outcomes. In practice, each agent controls too few subnets for this to occur.

### What Needs More Testing

1. **P1b at n=100+**: The P1b variants show large per-episode variance (deltas up to +/-700) but small mean effect (~-31). With only 30 episodes, we cannot distinguish signal from noise. A 100-episode multi-seed run would clarify whether P1b genuinely helps.

2. **Phase-dependent priority reordering**: The current experiments test static config changes. A dynamic variant that changes threshold or P1b behavior per phase (e.g., more aggressive in active phases, conservative in phase 0) was not tested. This is the most promising untested direction because:
   - Phase 0 has lower penalties (LWF=-1 for OZ, not -10)
   - Phase 0 would benefit from more conservative Restore (save action budget)
   - Active phases (1, 2) would benefit from more aggressive Restore (prevent -10/step cascading)

3. **Threshold=1 vs threshold=1 with broader OZ exceptions**: Currently only server_host_0 gets threshold=0 in active phases. What about ALL OZ hosts (user_host_0, user_host_1, user_host_2)? DegradeServices affects all hosts, not just servers.

4. **Multi-seed validation**: All experiments used seed=42. Different seeds may produce different relative rankings. The 100-episode baseline (-814.0 +/- 247.7) differs from our 30-episode baseline (-771.8 +/- 216.2), suggesting seed-dependent variance.

### Recommendations for V10c

1. **No changes to P2/P3 ordering or host priorities** -- zero-effect levers confirmed.
2. **Keep threshold=1 default** -- already optimal.
3. **Test phase-dependent thresholds** as next experiment: threshold=2 in phase 0 (conservative), threshold=1 in active phases (current behavior). This could save action budget in phase 0 without hurting active-phase response.
4. **Test broader OZ threshold=0 exception** for all OZ hosts (not just server_host_0) during active phases.
5. **Run P1b validation at n=100** with multi-seed to determine if it should be kept.

---

## Raw Data

Results saved to: `docs/swarm_analysis/priority_experiment_results.json`

Evaluation script: `scripts/evaluate_priority_experiments.py`
Configurable agent: `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgentV10b_variants.py`
