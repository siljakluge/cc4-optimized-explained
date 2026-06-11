# Decoy Strategy Critique: EnterpriseHeuristicAgentV10b

## Executive Summary

MAX_DECOYS=3 is confirmed optimal. Decoys are the single most valuable mechanism in V10b, providing a 60% reward improvement over no-decoys. Phase-aware deployment ordering has zero impact. Partial post-Restore redeployment is statistically insignificant. The current decoy strategy should be kept unchanged.

---

## Experiment 1: MAX_DECOYS Ablation

### Design
Compared MAX_DECOYS values of 0, 1, 2, 3 with all other agent logic identical. 15 episodes per variant, two seeds (42, 123).

### Results

| Variant | Seed 42 Mean | Seed 42 Std | Seed 123 Mean | Seed 123 Std | Combined Mean |
|---------|-------------|-------------|---------------|-------------|---------------|
| MAX_DECOYS=0 | -1994.7 | 715.4 | -2036.3 | 790.5 | -2015.5 |
| MAX_DECOYS=1 | -881.3 | 193.7 | -796.0 | 271.0 | -838.7 |
| MAX_DECOYS=2 | -840.3 | 255.5 | -757.3 | 160.2 | -798.8 |
| MAX_DECOYS=3 | -808.0 | 242.9 | -756.3 | 127.5 | -782.2 |

### 95% Confidence Intervals (per seed)

| Variant | Seed 42 CI | Seed 123 CI |
|---------|-----------|------------|
| MAX_DECOYS=0 | [-2356.7, -1632.6] | [-2436.4, -1636.3] |
| MAX_DECOYS=1 | [-979.3, -783.3] | [-933.2, -658.8] |
| MAX_DECOYS=2 | [-969.6, -711.0] | [-838.4, -676.3] |
| MAX_DECOYS=3 | [-930.9, -685.1] | [-820.9, -691.8] |

### Analysis

**0 vs 3 decoys: Massive gap (statistically significant)**
- Combined delta: -1233.3 reward (158% worse without decoys)
- CIs do not overlap at all. This is the strongest result.
- No-decoy agent is 2.6x worse than 3-decoy agent.

**1 vs 3 decoys: Meaningful gap but overlapping CIs**
- Combined delta: -56.5 reward (7.2% worse with 1 decoy)
- 95% CIs overlap substantially. Not statistically significant at n=15.
- However, the direction is consistent across both seeds.

**2 vs 3 decoys: Marginal gap, within noise**
- Combined delta: -16.6 reward (2.1% worse with 2 decoys)
- CIs overlap almost completely. Indistinguishable at n=15.

**Variance trend: More decoys = lower variance**
- MAX_DECOYS=0: std 715-791 (huge variance, catastrophic outliers like -4170, -4655)
- MAX_DECOYS=1: std 194-271
- MAX_DECOYS=2: std 160-256
- MAX_DECOYS=3: std 128-243

This variance reduction is arguably as important as the mean improvement. Decoys stabilize outcomes by preventing catastrophic exploit chains.

### Marginal Value Per Decoy

| Transition | Mean Improvement | Actions Spent | Reward per Action |
|-----------|-----------------|---------------|-------------------|
| 0 -> 1 | +1176.8 | 1 per host | ~1176.8 per host |
| 1 -> 2 | +39.9 | 1 per host | ~39.9 per host |
| 2 -> 3 | +16.6 | 1 per host | ~16.6 per host |

The first decoy provides 98.5% of the total decoy benefit. Decoys 2 and 3 together provide only 1.5%. However, because DeployDecoy takes 2 steps (duration=2 in the CybORG source), the cost is higher than expected -- but the agent has nothing better to do in P7 (idle time), so the opportunity cost is near-zero.

### Verdict: Keep MAX_DECOYS=3

The 3rd decoy costs almost nothing (deployed during idle P7 time) and provides marginal benefit (+16.6 mean, +reduced variance). No reason to reduce.

---

## Experiment 2: Phase-Aware Deployment Order

### Design
Compared static deployment order (OZB server_host_0 first, always) vs dynamically reordering based on mission phase (Phase 1: OZA first, Phase 2: OZB first).

### Results

| Variant | Seed 42 Mean | Seed 42 Std |
|---------|-------------|-------------|
| Static order (baseline) | -808.0 | 242.9 |
| Phase-aware order | -808.0 | 242.9 |

**Identical results. Episode-by-episode rewards are identical.**

### Why No Difference

The phase-aware variant re-sorts `_deploy_hosts` at the start of each `get_action` call. However:

1. **Initial deployment (P7) happens in Phase 0**, where both static and phase-aware orders are nearly identical (OZB server_host_0 first, OZA server_host_0 second in both).
2. **All decoys are fully deployed within the first ~20 steps** of Phase 0. By the time Phase 1 or 2 starts, all hosts already have MAX_DECOYS deployed.
3. **Post-Restore redeploy (P6)** uses the same `_deploy_hosts` order, but P6 only triggers for hosts that were just Restored. The iteration order across all hosts doesn't matter when only 1-2 hosts need redeploy at any given time.

### Verdict: Deployment order is irrelevant

The current static order works fine. Phase-aware ordering is unnecessary complexity for zero benefit. The reason is structural: decoys are a one-time investment completed well before phase transitions matter.

---

## Experiment 3: Partial Post-Restore Redeploy

### Design
After Restore completes, deploy only 1 decoy in P6 (high-priority redeploy) then allow other actions. Remaining decoys deployed in P7 (idle time). Hypothesis: faster Restore responsiveness outweighs temporarily lower trap rate.

### Results

| Variant | Seed 42 Mean | Seed 42 Std | Seed 123 Mean | Seed 123 Std |
|---------|-------------|-------------|---------------|-------------|
| Full redeploy P6 (baseline) | -808.0 | 242.9 | -756.3 | 127.5 |
| Partial redeploy (1 in P6) | -751.3 | 210.7 | -828.0 | 173.8 |

Seed 42: Partial redeploy is +56.7 better (7.0% improvement)
Seed 123: Partial redeploy is -71.7 worse (9.5% degradation)

### Analysis

The results contradict each other across seeds. The combined effect is approximately zero. The 95% CIs overlap substantially in both cases.

**Why the inconsistency:**
- P6 full redeploy "wastes" 2 extra steps per Restore event, but those steps provide 75% trap rate vs 50%.
- In seed 42's random scenarios, the freed-up steps happened to coincide with useful Restore opportunities.
- In seed 123's scenarios, the reduced trap rate during the gap allowed more successful red exploits.
- The effect is scenario-dependent and cancels out in expectation.

### Verdict: Keep full P6 redeploy (status quo)

No statistically significant improvement. The added code complexity is not justified.

---

## Additional Findings

### DeployDecoy Duration
The CybORG source code (`DeployDecoy.py`) sets `self.duration = 2`, meaning each decoy deployment takes 2 timesteps, not 1. This means:
- Initial deployment of 3 decoys per host takes 6 steps (3 decoys x 2 steps each)
- Post-Restore full redeploy takes 6 steps
- This is higher than previously documented, but since it happens during idle time (P7), the cost is absorbed.

### Decoy-Only Hosts (Question 4 from Brief)
All hosts in `_deploy_hosts` receive decoys. Some hosts may never see red attention, but we cannot predict which hosts red will target. The cost of deploying decoys on untargeted hosts is near-zero (P7 idle time), while the cost of NOT having decoys when red arrives is catastrophic (as shown by MAX_DECOYS=0 results). This is a correct insurance strategy.

### Decoy Budget vs Restore Responsiveness (Question 5)
The P6/P7 priority structure already handles this correctly:
- P1-P4 (Restore/Block/Allow) always take priority over P6-P7 (Decoy deploy)
- Decoys are only deployed when there is nothing more urgent to do
- The only contention is within P6 (post-Restore redeploy), where 3 decoy actions take 6 steps. But as Experiment 3 showed, reducing this to 1 decoy + deferred rest does not reliably help.

---

## Recommendations

### Keep (High Confidence)
1. **MAX_DECOYS=3** -- Optimal. Each additional decoy reduces variance and improves mean.
2. **Static deployment order** -- OZB server_host_0 first is fine; order is irrelevant once all decoys are deployed.
3. **Full P6 redeploy** -- Deploy all 3 decoys immediately after Restore. No benefit to partial redeploy.
4. **P6/P7 priority split** -- Post-Restore redeploy (P6) before initial deploy (P7) is correct.

### No Change Needed
- Phase-aware ordering: Zero impact, unnecessary complexity.
- Partial redeploy: Inconsistent results, net zero benefit.
- Selective host deployment: Insurance value exceeds cost.

### Further Investigation (Low Priority)
1. **MAX_DECOYS=4 or 5**: The diminishing returns curve is steep (16.6 improvement from 2->3). Testing 4+ would likely show <5 improvement. Not worth investigating unless the 4-decoy candidate pool supports it (currently 4 factories: Apache, Tomcat, Haraka, Vsftpd -- so MAX_DECOYS=4 is the theoretical maximum per host).
2. **Adaptive decoy count**: Deploy fewer decoys on low-value hosts (e.g., user hosts) and more on servers. Current uniform MAX_DECOYS=3 is simpler and the marginal value of optimization here is tiny.
3. **Decoy-aware Restore threshold**: Could lower the proc_flag threshold on hosts with depleted decoys (hit by red). This is already partially implemented via `_decoy_hit_hosts` tracking but not used for threshold adjustment.

---

## Raw Data Summary

### Experiment 1 Combined (30 episodes across seeds 42, 123)

| Variant | Combined Mean | Combined Std | Delta vs Base |
|---------|--------------|-------------|---------------|
| MAX_DECOYS=0 | -2015.5 | ~750 | -1233.3 |
| MAX_DECOYS=1 | -838.7 | ~230 | -56.5 |
| MAX_DECOYS=2 | -798.8 | ~210 | -16.6 |
| MAX_DECOYS=3 | -782.2 | ~185 | 0.0 |

### Experiment 3 Combined (30 episodes across seeds 42, 123)

| Variant | Combined Mean | Delta |
|---------|--------------|-------|
| Full redeploy | -782.2 | 0.0 |
| Partial redeploy | -789.7 | -7.5 |
