# Oracle Architecture Analysis for CAGE Challenge 4

## 1. Oracle V1 Failure Analysis

Oracle V1 was designed as a perfect-information upper bound: it reads the simulator's ground-truth `State` object to know exactly which hosts have red sessions (user or root level). Despite this advantage, V1 scored **-1678.7** mean reward (30 eps, seed 42), significantly worse than the observation-only heuristic v9.1 at **-1039.3**.

### Root causes (empirical, from `oracle_diagnosis.py` on 30 episodes):

| Metric | Value | Implication |
|--------|-------|-------------|
| Avg removes/episode | 258.1 | Massive Remove usage |
| Avg restores/episode | 255.9 | Escalation cascade from Remove failures |
| Re-exploit during Remove window | 15,424 events | Red re-compromises hosts during 3-step Remove |
| Remove on user-only sessions | 7,743 total | Wrong tool: Remove fails 10%, takes 3 steps |
| Root session response latency | avg 4.6, max 301 steps | Too slow to respond to root |
| User session response latency | avg 20.3, max 464 steps | Extremely slow for user sessions |
| OT server root exposure | 6 events, avg 376.8 steps | Impact target compromised nearly entire episode |

### Why V1 failed despite perfect information:

1. **Remove-first policy for user sessions**: V1 tries Remove (3 steps, 90% success) before Restore. When Remove fails (10% of the time), red still has a user session and can PrivilegeEscalate to root in 2 steps. By the time V1 escalates to Restore, red already has root.

2. **Blocking at priority 3**: V1 handles Restore (priority 1) and Remove (priority 2) before blocking (priority 3). This means red freely spreads to new hosts while V1 is busy removing/restoring existing infections. The heuristic blocks early, preventing red from reaching critical zones.

3. **Re-exploit cascade**: During Remove's 3-step window, red re-exploits the same host. V1 sees the session persist, assumes Remove failed, and escalates to Restore. This creates a Restore cascade (255.9 restores/ep) that consumes all blue agent bandwidth.

4. **No strategic prioritisation**: V1 treats all hosts equally (adjusted by zone). It doesn't recognise that blocking is orders of magnitude more cost-effective than Remove/Restore.

## 2. Oracle V2 Design Rationale and Failure

V2 attempted to fix V1 by:
- Adding phase-aware priority with Impact-target boosting
- Reading red FSM state for predictive defense
- Prioritising blocking before Remove (but AFTER root Restores)
- Using a Remove+Restore hybrid for user sessions

### V2 results: -1722.2 mean reward (30 eps) -- WORSE than V1

V2 failed because:
1. **Root Restore at priority 1, blocking at priority 2**: With 6 red agents constantly creating root sessions, V2 spent every action Restoring instead of blocking. Result: 223.7 restores/ep.
2. **Decoy redeployment**: V2 redeployed decoys after every Restore (564.2 decoys/ep), wasting actions that could prevent damage.
3. **Still used Remove for user sessions**: Same cascade problem as V1.

## 3. Oracle V3 Design -- Optimal Policy

V3 achieves **-893.5** mean reward (100 eps, seed 42), beating the heuristic by **+145.8 points**.

### Key insight: damage comes from GREEN agents, not red sessions directly

The `BlueRewardMachine` calculates damage when:
- **LWF**: `GreenLocalWork` fails on a host where red has an active session
- **ASF**: `GreenAccessService` fails on a host where red has an active session  
- **RIA**: Red `Impact` action succeeds on OT server

This means the optimal strategy minimises the total number of (host x steps) where red has active sessions AND green agents are trying to use those hosts. The cheapest way to do this is prevention (blocking), not remediation (Restore).

### V3 policy (in priority order):

1. **BLOCK per comms_policy** -- Free, instant, prevents red from reaching new hosts. Each block action protects an entire subnet. This is the highest-value action by far.

2. **RESTORE root sessions** -- Only way to evict root (Remove cannot). Prioritised by phase-aware criticality (Impact targets first).

3. **REMOVE user sessions** -- Fast (3 steps), free, 90% success. On normal hosts, Remove-first is correct because it's cheaper than Restore. On Impact targets during active mission, skip directly to Restore (can't afford 10% failure risk).

4. **ALLOW stale blocks** -- Unblock paths that are no longer needed.

5. **No decoys** -- Oracle has perfect information; decoys provide zero detection value. Deploying them wastes actions.

### Why V3 succeeds where V1/V2 failed:

The critical change is **blocking at priority 1** instead of after Remove/Restore. This means:
- Red is contained before it can spread to critical zones
- Fewer hosts get compromised, so fewer Restores are needed
- Blue agents spend most of their time sleeping (2255/ep) rather than firefighting

## 4. Performance Comparison Table

All results on seed 42, 500 steps/episode.

| Agent | Mean Reward | Std | Episodes | Restores/ep | Removes/ep | Blocks/ep | vs Sleep | Capture% |
|-------|------------|-----|----------|-------------|-----------|-----------|----------|----------|
| SleepAgent | -30,578.9 | -- | -- | 0 | 0 | 0 | baseline | 0.0% |
| **Oracle V2** | **-1,722.2** | 960.7 | 30 | 223.7 | 0 | 34.2 | +28,857 | 94.4% |
| **Oracle V1** | **-1,678.7** | 877.0 | 30 | 255.9 | 258.1 | varies | +28,900 | 94.5% |
| **Heuristic v9.1** | **-1,039.3** | -- | 100 | ~85 | ~48 | ~32 | +29,540 | 96.6% |
| **Oracle V3** | **-893.5** | 206.7 | 100 | 99.6 | 101.7 | 33.4 | +29,686 | 97.1% |

### Per-Phase Breakdown (Oracle V3, 100 episodes)

| Phase | Mean Reward | Std | Notes |
|-------|------------|-----|-------|
| Phase 0 (Preplanning) | -502.1 | 165.1 | All subnets connected; hardest phase |
| Phase 1 (Mission A) | -173.3 | 71.9 | OZA isolated; blocking very effective |
| Phase 2 (Mission B) | -217.9 | 83.6 | OZB isolated; blocking very effective |

Phase 0 accounts for 56% of total damage because all subnets are connected and blocking has minimal effect. During Phases 1-2, comms_policy isolates the active mission zone, making blocking extremely effective at containing red.

## 5. Decision Tree for Optimal Oracle Policy

```
Step begins
  |
  v
Read ground truth: red sessions (user/root) on each host
  |
  v
Read comms_policy from observation
  |
  v
Any subnet needs blocking? ----YES----> Block (highest priority pair)
  |                                       |
  NO                                      DONE
  |
  v
Any host has root red session? ---YES---> Restore (highest priority host)
  |                                        |
  NO                                       DONE
  |
  v
Any host has user red session? ---YES---> Is it Impact target in active phase?
  |                                        |           |
  NO                                      YES         NO
  |                                        |           |
  v                                     Restore     Previous Remove failed?
Any stale block to allow? ---YES---->               |           |
  |                        Allow                   YES         NO
  NO                                                |           |
  |                                              Restore      Remove
  v
Sleep
```

## 6. Information Value Analysis: What the Oracle Teaches Us

### The information gap is smaller than expected

The oracle's advantage over the heuristic is only **+145.8** reward points (16.4% of the heuristic's total penalty). This means the heuristic already captures **91.1%** of the oracle's performance from observations alone.

### Where perfect information helps most:

1. **Zero false positives**: The oracle never wastes a Remove/Restore on green noise. The heuristic with malfile flags already achieves near-zero FP rate, so this advantage is small.

2. **Immediate detection**: The oracle sees red sessions the step they appear. The heuristic must wait for observable events (process flags, connection flags, malfile flags), adding 1-3 steps of latency.

3. **Perfect escalation timing**: The oracle knows exactly when to escalate from Remove to Restore (red still present after Remove completes). The heuristic must re-observe to detect escalation.

### Where perfect information does NOT help:

1. **Action bandwidth**: Both oracle and heuristic are constrained to 5 actions/step (one per blue agent). 6 red agents can compromise hosts faster than 5 blue agents can remediate. This is the fundamental bottleneck.

2. **Blocking is observation-derived**: The comms_policy is deterministic from the mission phase and fully visible in observations. The oracle gains no advantage here.

3. **Restore cost**: Both pay -1 per Restore. The oracle can't avoid this cost when root sessions exist.

4. **Phase 0 vulnerability**: During preplanning, all subnets are connected. Blocking provides minimal benefit regardless of information quality. Phase 0 accounts for 56% of Oracle V3's damage.

## 7. Why Even the Oracle Cannot Achieve Zero Damage

### Structural constraints of CC4:

1. **6 red agents vs 5 blue agents**: Red has numerical superiority. Even with perfect information, blue cannot respond to all threats simultaneously.

2. **Action latency**: Remove takes 3 steps, Restore takes 5 steps. During these windows, the host is "busy" but red can exploit OTHER hosts. One blue agent defending one host means 4 agents covering everything else.

3. **Phase 0 is unblockable**: During preplanning (~167 steps), all subnets are connected. Blocking doesn't help. Red agents use this window to establish footholds everywhere.

4. **Green agent damage is continuous**: 54 green agents constantly try to work on all hosts. Any host with a red session generates damage every step a green agent touches it, regardless of blue's awareness.

5. **DegradeServices is permanent**: Once red runs DegradeServices (at 50% probability when in R/RD state), the damage persists until Restore. This creates compounding damage that the oracle cannot prevent faster than it accumulates.

6. **PrivilegeEscalate is invisible and instant**: Takes 2 steps, 100% success, fires no observable events. Even the oracle can only see the RESULT (root session), not prevent the escalation in progress.

### Theoretical minimum damage estimate:

With 6 red agents each taking ~10 steps to reach root on a new host (scan + exploit + privesc), and 5 blue agents each needing ~4 steps average to respond (detect + Remove/Restore), the steady-state infection rate is approximately:
- Red infection rate: ~6 new infections per 10 steps = 0.6/step
- Blue remediation rate: ~5 remediations per 4 steps = 1.25/step
- Net: blue can keep up, but with ~2-3 hosts infected at any given time
- At ~2 damage per infected host per step, minimum steady-state damage is ~4-6/step = ~2000-3000 total

The actual Oracle V3 score of -893.5 is significantly better than this estimate because blocking prevents many infections entirely during Phases 1-2, and because not all infected hosts generate damage every step (depends on green agent activity).

## Appendix: Action Statistics

### Oracle V3 (100 episodes, seed 42)
- Restores/ep: 99.6
- Removes/ep: 101.7
- Blocks/ep: 33.4
- Allows/ep: 5.2
- Sleeps/ep: 2255.0
- Total actions/ep: 2495 (5 agents x 499 steps)

### Heuristic v9.1 (3 episodes, seed 42, for comparison)
- Restores/ep: ~83
- Removes/ep: ~48
- Blocks/ep: ~33
- Allows/ep: ~4
- Decoys/ep: ~384
- Sleeps/ep: ~1948

Key difference: The heuristic deploys ~384 decoys/ep (wasting actions but providing detection value). The oracle skips decoys entirely (no detection needed). The heuristic does fewer Removes/Restores because decoys prevent some infections by tricking red's blind exploit attempts (75% failure rate with 3 decoys).

This suggests a V4 oracle that deploys decoys strategically might perform even better: decoys are a form of "passive defense" that reduce red's infection rate without consuming blue agent bandwidth on every step.
