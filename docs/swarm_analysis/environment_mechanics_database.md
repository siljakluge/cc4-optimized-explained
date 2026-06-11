# Environment Mechanics Database: Empirical Measurements

All data from instrumented experiments. 30 episodes, seed 42, 500 steps each.
SQLite databases in `data/` directory.

---

## 1. Green Agent False Positive Rates

Source: `data/green_trace.db`, 534,733 traced green actions (excluding Sleep).

### 1.1 Action Distribution

Green agents choose uniformly from 3 actions per step. The tracer captures
non-Sleep actions only.

| Action Type       | Count    | % of traced | Est. % of all |
|-------------------|----------|-------------|---------------|
| GreenLocalWork    | 267,129  | 50.0%       | ~33.3%        |
| GreenAccessService| 267,604  | 50.0%       | ~33.3%        |
| Sleep (untraced)  | ~267,000 | -           | ~33.3%        |

61 green agents active per episode, distributed across subnets:
- contractor_network: 10 agents
- public_access_zone: 9 agents
- office_network: 7 agents
- restricted_zone_a: 6 agents
- admin_network: 6 agents
- operational_zone_a: 5 agents
- restricted_zone_b: 3 agents
- operational_zone_b: 3 agents

### 1.2 Process Creation FP Rate (from GreenLocalWork)

Total: 2,074 process_creation FP events out of 267,129 GreenLocalWork actions.

**Overall rate: 0.776%** (configured: 1.000%)

The rate is below 1% because FP only fires on SUCCESSFUL local work (step 2 of
execute). With 23.83% local work failure rate, the effective FP rate is:
`1% * (1 - 23.83%) = 0.762%` -- matching the observed 0.776% within noise.

| Subnet                        | Total LW | FP Count | FP Rate |
|-------------------------------|----------|----------|---------|
| office_network_subnet         | 31,595   | 358      | 1.133%  |
| public_access_zone_subnet     | 35,719   | 341      | 0.955%  |
| admin_network_subnet          | 36,789   | 333      | 0.905%  |
| restricted_zone_b_subnet      | 36,069   | 249      | 0.690%  |
| restricted_zone_a_subnet      | 33,866   | 232      | 0.685%  |
| operational_zone_a_subnet     | 32,623   | 213      | 0.653%  |
| contractor_network_subnet     | 32,158   | 197      | 0.613%  |
| operational_zone_b_subnet     | 28,310   | 151      | 0.533%  |

Higher FP rates on HQ subnets (office, PAZ, admin) correlate with lower
local work failure rates on those hosts (2-5%), leaving more actions
reaching step 2 of execute() where FP fires.

Lower FP rates on OZ/contractor subnets correlate with higher failure rates
(35-44%), as red's DegradeServices/Impact reduce service reliability.

### 1.3 Network Connection FP Rate (from GreenAccessService)

Total: 2,595 network_connection FP events out of 267,604 GreenAccessService actions.

**Overall rate: 0.970%** (configured: 1.000%)

Closer to 1% because FP fires regardless of service availability check
(the code has a bug: `if not self.available_dest_service:` checks the method
object, not calling it -- so the service check never actually executes).

| Subnet (destination)          | Total AS | FP Count | FP Rate |
|-------------------------------|----------|----------|---------|
| office_network_subnet         | 29,664   | 305      | 1.028%  |
| restricted_zone_a_subnet      | 52,310   | 519      | 0.992%  |
| operational_zone_a_subnet     | 24,728   | 244      | 0.987%  |
| restricted_zone_b_subnet      | 52,206   | 503      | 0.963%  |
| contractor_network_subnet     | 35,443   | 340      | 0.959%  |
| public_access_zone_subnet     | 30,229   | 285      | 0.943%  |
| admin_network_subnet          | 22,381   | 209      | 0.934%  |
| operational_zone_b_subnet     | 20,643   | 190      | 0.920%  |

Network FP events land on **server** hosts (since GreenAccessService only
targets servers). By host type:
- Server hosts: 2,595 FP events (100%)
- User hosts: 0 FP events

Process FP events land on **user** hosts (where green agents reside):
- User hosts: 2,074 FP events (100%)
- Server hosts: 0 process FP events

### 1.4 Combined FP Rate per Step

Per step across all green agents:
- Expected proc FP per step: 2,074 / (30 * 500) = **0.138 per step**
- Expected net FP per step: 2,595 / (30 * 500) = **0.173 per step**
- Total FP events per step: **0.311 per step**
- P(at least one FP per step): ~27%

Per episode:
- Process FPs per episode: 2,074 / 30 = **69.1**
- Network FPs per episode: 2,595 / 30 = **86.5**
- Total FPs per episode: **155.6**

---

## 2. Phishing Frequency and Impact

Source: `data/green_trace.db`, phishing_events table.

### 2.1 Overall Phishing Rate

Total phishing events: 846 across 30 episodes.
**Per episode: 28.2 phishing events** creating new red sessions.

Phishing rate as fraction of GreenLocalWork: 846 / 267,129 = **0.317%**

Expected rate: 1% (phishing_error_rate) * 76.17% (success rate of local work)
= 0.762%. Observed is 0.317%, meaning **only 41.6% of phishing attempts
succeed in creating a session**.

The gap is because PhishingEmail._create_new_session() must find a routable
red agent, and fails when:
1. Red is already present on the target host (returns early with success=True
   but no new session)
2. No red agent is routable to the target host

### 2.2 Phishing by Phase

| Phase | Events | Per Episode | % of Total |
|-------|--------|-------------|------------|
| 0     | 569    | 18.97       | 67.3%      |
| 1     | 190    | 6.33        | 22.5%      |
| 2     | 87     | 2.90        | 10.3%      |

Phase 0 has 67% of phishing because:
- More hosts have routable red agents (no blocking)
- Fewer hosts already have red sessions (early in episode)

Phase 2 drops to 10% because:
- Red already occupies many hosts
- Phishing skips hosts where red is already present

### 2.3 Phishing by Subnet

| Subnet                        | Events | Per Episode |
|-------------------------------|--------|-------------|
| admin_network_subnet          | 161    | 5.37        |
| office_network_subnet         | 144    | 4.80        |
| public_access_zone_subnet     | 134    | 4.47        |
| restricted_zone_b_subnet      | 111    | 3.70        |
| restricted_zone_a_subnet      | 98     | 3.27        |
| operational_zone_a_subnet     | 81     | 2.70        |
| contractor_network_subnet     | 60     | 2.00        |
| operational_zone_b_subnet     | 57     | 1.90        |

HQ subnets (admin, office, PAZ) receive the most phishing because they have
more green agents (6-9) and are always routable to red.

### 2.4 Phishing Impact

PhishingEmail creates a `RedAbstractSession` with `username='user'` and
`parent=None`. This is a root-level red session (parent session of type
RedAbstractSession). It **bypasses all BlockTrafficZone** firewall rules
(ADR-CC4-001) because it uses `check_routable()` (link-layer) not
`blocking_host()` (firewall).

**Each phishing event creates a real red foothold that cannot be prevented
by blocking.** The only defense is Restore after detection.

---

## 3. Green Local Work Failure Rates

Source: `data/green_trace.db`.

### 3.1 Overall Failure Rate

Total failures: 63,648 out of 267,129 GreenLocalWork = **23.83%**

### 3.2 Failure Rate by Phase

| Phase | Total LW | Failures | Rate   |
|-------|----------|----------|--------|
| 0     | 88,917   | 4,047    | 4.55%  |
| 1     | 89,530   | 21,461   | 23.97% |
| 2     | 88,682   | 38,140   | 43.01% |

Failure rate increases dramatically across phases because:
- Red's DegradeServices reduces service reliability (20% per hit)
- Red's Impact stops OT services entirely
- Without blue intervention, these effects accumulate

### 3.3 Failure Rate by Subnet

| Subnet                        | Total  | Failures | Rate    |
|-------------------------------|--------|----------|---------|
| operational_zone_b_subnet     | 28,310 | 12,555   | 44.35%  |
| contractor_network_subnet     | 32,158 | 13,424   | 41.74%  |
| operational_zone_a_subnet     | 32,623 | 11,577   | 35.49%  |
| restricted_zone_a_subnet      | 33,866 | 11,135   | 32.88%  |
| restricted_zone_b_subnet      | 36,069 | 11,131   | 30.86%  |
| public_access_zone_subnet     | 35,719 | 1,918    | 5.37%   |
| admin_network_subnet          | 36,789 | 1,108    | 3.01%   |
| office_network_subnet         | 31,595 | 800      | 2.53%   |

Operational and restricted zones have high failure rates because red
focuses DegradeServices/Impact there. HQ subnets (PAZ, admin, office)
have low failure rates because red does not prioritize them for Impact.

---

## 4. Action Execution Order

Source: `data/action_order_trace.db`, 10 episodes, 620,991 action executions.

### 4.1 Priority System

The `sort_action_order()` function in `SimulationController.py` sorts
actions by `action.priority` ascending (lower number = executes first).

| Action Type              | Priority | Duration | Notes                          |
|--------------------------|----------|----------|--------------------------------|
| ControlTraffic (Block/Allow) | 1    | 1        | Always executes first          |
| RemoveOtherSessions      | 5        | 3        | Executes before default actions|
| Sleep                    | 99       | 1        | Default priority               |
| GreenLocalWork           | 99       | 1        | Default priority               |
| GreenAccessService       | 99       | 1        | Default priority               |
| Monitor                  | 99       | 1        | Default (also end_turn_action) |
| ExploitRemoteService     | 99       | varies   | Default priority               |
| PrivilegeEscalate        | 99       | 2        | Default priority               |
| DegradeServices          | 99       | varies   | Default priority               |
| Impact                   | 99       | 2        | Default priority               |
| DiscoverRemoteSystems    | 99       | 1        | Default priority               |
| Restore                  | 99       | 5        | Default priority               |
| Remove                   | 99       | 3        | Default priority (RemoveOtherSessions has 5) |

### 4.2 Execution Order Diagram

```
STEP N begins
|
+-- 1. Mission phase check (update step count, check phase transition)
|       If phase changed: update allowed_subnets for all agents
|
+-- 2. Action collection
|       For each agent: get_action() or use submitted action
|       New actions get remaining_ticks = action.duration
|
+-- 3. Tick decrement
|       remaining_ticks -= 1
|       If remaining_ticks < 1: action ready to execute
|       If remaining_ticks >= 1: Sleep substituted, IN_PROGRESS observation
|
+-- 4. sort_action_order()
|       Sort by priority ascending: Block(1) < RemoveSessions(5) < all others(99)
|       Within same priority: dict iteration order (agent name order)
|       Filter out invalid actions (no session)
|       Check bandwidth (random shuffle, drop if exceeded)
|
+-- 5. Execute actions in sorted order
|       For each (agent, action) in sorted list:
|           obs = action.execute(state)
|           observation[agent].append(obs)
|
+-- 6. Session reassignment (subnet traversal check)
|
+-- 7. End-turn actions
|       Monitor executes for all blue agents
|       Events (process_creation, network_connections) collected and cleared
|
+-- 8. Update agent interfaces and action spaces
|
+-- 9. Increment step_count
|
+-- 10. Calculate reward
|       BlueRewardMachine.calculate_reward() processes:
|         - Green failures (LWF, ASF)
|         - Red Impact successes (RIA)
|       action_cost = sum of all blue action costs
|
+-- 11. Host update (state.hosts[h].update())
```

### 4.3 Key Timing Answers

**Q: Does blue's Block execute before red's Exploit in the same step?**
A: YES. ControlTraffic has priority=1, all other actions have priority=99.
Block/Allow ALWAYS execute before Exploit/Impact/PrivilegeEscalate.

**Q: Does blue's Restore complete before red can re-exploit?**
A: Restore has duration=5 (submitted at step N, executes at step N+4).
During steps N through N+3, Sleep is substituted. At step N+4, RestoreFromBackup
executes, clearing all red sessions. Red can re-exploit starting step N+5.
Restore and Exploit have the same priority (99), so if both fire on the same
step, their relative order depends on agent name alphabetical ordering.
Blue agents are named `blue_agent_0..4`, red agents are `red_agent_0..5`.
Since 'b' < 'r' alphabetically, **blue's Restore executes before red's
Exploit when both fire on the same step**.

**Q: What happens when blue and red act on the same host?**
A: Within the same step, actions execute sequentially in priority order.
Since blue and red share priority=99, agent name ordering determines who
goes first. Blue always goes first due to alphabetical ordering.

### 4.4 Action Counts Per Step

| Team   | Mean actions/step |
|--------|-------------------|
| Blue   | 5.00 (Monitor)    |
| Green  | 34.29             |
| Red    | 7.89              |

Red action distribution across 10 episodes:
- RedSessionCheck: 27,491 (70.0%)
- Impact: 3,469 (8.8%)
- DegradeServices: 3,448 (8.8%)
- ExploitRemoteService: 2,154 (5.5%)
- DiscoverRemoteSystems: 890 (2.3%)
- DiscoverDeception: 730 (1.9%)
- PrivilegeEscalate: 495 (1.3%)
- StealthServiceDiscovery: 413 (1.1%)
- AggressiveServiceDiscovery: 374 (1.0%)

---

## 5. Reward Decomposition

Source: `data/reward_trace.db`, 30 episodes without blue defense.

### 5.1 Unmitigated Damage (No Blue Agent)

Mean episode reward (BRM only, single team): **-6,514.1 +/- 1,322.7**
Equivalent 5-agent total: **~-32,570** (matches SleepAgent benchmark of -30,579)

| Component                | Per Episode | % of Total |
|--------------------------|-------------|------------|
| LWF (Local Work Fails)   | -5,425.0    | 83.3%      |
| RIA (Red Impact/Access)   | -1,089.1    | 16.7%      |
| ASF (Access Service Fails)| 0.0         | 0.0%       |
| Action cost               | 0.0         | 0.0%       |

**LWF dominates at 83.3% of total unmitigated damage.** This is because each
Impact stops a service, causing cascading LWF penalties for all green agents
on that host for all subsequent steps.

### 5.2 Phase Decomposition

| Phase | Total    | Per Step | LWF      | ASF | RIA      |
|-------|----------|----------|----------|-----|----------|
| 0     | -206.5   | -1.24    | -98.5    | 0.0 | -107.9   |
| 1     | -2,213.6 | -13.26   | -1,833.9 | 0.0 | -379.7   |
| 2     | -4,094.0 | -24.66   | -3,492.6 | 0.0 | -601.4   |

Phase 2 is worst (-24.66/step) because:
1. Red has been degrading/impacting services since Phase 0
2. OZ-B LWF penalty is -10 during Phase 2
3. Cumulative service degradation makes most green local work fail

Phase 0 is lightest (-1.24/step) despite no blocking because:
1. Penalty multipliers are low (LWF=-1, RIA=-1 for most zones)
2. Red is still in early exploit phases
3. Services haven't been degraded yet

### 5.3 Penalty by Subnet (All Phases Combined, 30 Episodes)

| Subnet                        | LWF       | RIA       | Total     |
|-------------------------------|-----------|-----------|-----------|
| operational_zone_b_subnet     | -79,803   | -12,397   | -92,200   |
| operational_zone_a_subnet     | -46,326   | -8,766    | -55,092   |
| restricted_zone_b_subnet      | -18,053   | -3,530    | -21,583   |
| restricted_zone_a_subnet      | -14,743   | -4,850    | -19,593   |
| public_access_zone_subnet     | -1,918    | -498      | -2,416    |
| contractor_network_subnet     | 0         | -2,100    | -2,100    |
| admin_network_subnet          | -1,108    | -255      | -1,363    |
| office_network_subnet         | -800      | -276      | -1,076    |

OZ-B (-92,200) is worse than OZ-A (-55,092) because Phase 2 is later in the
episode -- by then, more services are degraded and more hosts are compromised.

### 5.4 Worst Individual Steps

| Episode | Step | Reward | LWF | RIA | Phase |
|---------|------|--------|-----|-----|-------|
| 18      | 487  | -85.0  | 12  | 1   | 2     |
| 16      | 356  | -77.0  | 12  | 2   | 2     |
| 18      | 464  | -75.0  | 15  | 2   | 2     |
| 23      | 482  | -75.0  | 11  | 2   | 2     |
| 16      | 454  | -74.0  | 13  | 3   | 2     |

All worst steps occur in Phase 2. A single step can incur -85 reward from
12 LWF events (many at -10 each) plus 1 RIA event (-10).

### 5.5 RIA Breakdown by Phase and Subnet

| Phase | Subnet                    | Count | Total   | Per Event |
|-------|---------------------------|-------|---------|-----------|
| 0     | contractor_network        | 420   | -2,100  | -5.0      |
| 0     | operational_zone_b        | 278   | -278    | -1.0      |
| 0     | operational_zone_a        | 230   | -230    | -1.0      |
| 1     | operational_zone_a        | 756   | -7,560  | -10.0     |
| 1     | restricted_zone_a         | 648   | -1,944  | -3.0      |
| 1     | contractor_network        | 822   | 0       | 0.0       |
| 2     | operational_zone_b        | 1,125 | -11,250 | -10.0     |
| 2     | restricted_zone_a         | 894   | -2,682  | -3.0      |
| 2     | restricted_zone_b         | 881   | -2,643  | -3.0      |
| 2     | contractor_network        | 990   | 0       | 0.0       |

Average RIA penalty per impact event: **-3.2** (weighted by phase multipliers).

---

## 6. Cost-Benefit Analysis with Real Numbers

### 6.1 Unnecessary Restore vs Missed Impact

On OZ-A during Phase 1 (LWF=-10, RIA=-10):

| Scenario                         | Cost Formula              | Value   |
|----------------------------------|---------------------------|---------|
| Unnecessary Restore              | -1 (action) + -10*5 (LWF)| **-51** |
| Missed Impact for 1 step         | -10 (RIA) + N*-10 (LWF)  | **-60** |
| Missed Impact for 5 steps        | 5*(-10 + N*-10)           | **-300**|
| Missed Impact for 10 steps       | 10*(-10 + N*-10)          | **-600**|

Where N = green agents on the host (~5 for OZ-A).
**Even 1 step of missed Impact (-60) outweighs an unnecessary Restore (-51).**

On HQ hosts (LWF=-1, RIA=-3):

| Scenario                         | Cost Formula              | Value  |
|----------------------------------|---------------------------|--------|
| Unnecessary Restore              | -1 (action) + -1*5 (LWF) | **-6** |
| Missed Impact for 1 step         | -3 (RIA)                  | **-3** |
| Missed Impact for 2 steps        | 2*-3 = -6                 | **-6** |
| Missed Impact for 3 steps        | 3*-3 = -9                 | **-9** |

**Break-even at 2 steps.** Restore pays for itself if Impact persists >2 steps.

### 6.2 True Cost of Green FPs (Empirical)

With the heuristic agent (from optimality_analysis.md):
- 1,474 Remove actions per episode (2% of actions)
- Estimated 50% of Removes are on FP targets (737 wasted)
- Each wasted Remove: 0 direct cost but 3 steps of inaction

Cost of FP-driven unnecessary Restores:
- 88 Restores per episode, estimated 30-50% on FP targets (26-44 wasted)
- Each costs -1 (action) + LWF disruption during 5-step reimage
- On non-mission hosts: -6 each
- On mission hosts: up to -51 each
- Estimated total FP Restore cost: **-150 to -300 per episode**

### 6.3 Theoretical Minimum Reward Bounds

**Perfect oracle with Restore-first strategy:**

Phase 0 (167 steps, no blocking):
- Red starts with footholds, exploits freely
- ~3 red agents actively attacking
- Irreducible: ~18.97 phishing per phase, each creating red session
- Oracle Restores immediately on detection (1-step delay)
- Estimated: -150 to -300 (detection delay + restore costs)

Phase 1 (167 steps, blocking available):
- Block contractor/RZ paths immediately
- Restore all OZ-A hosts pre-emptively at step 167
- Phishing bypasses blocks (~6.33 phishing events)
- Each phishing on OZ-A costs: detect(1-2 steps) + Restore(5 steps)
  = 6-7 steps * -10 LWF = -60 to -70 per phishing on OZ
- Estimated: -100 to -200

Phase 2 (166 steps, similar to Phase 1):
- Same strategy for OZ-B
- Estimated: -80 to -180

**Theoretical oracle floor: approximately -330 to -680 per episode (single team)**
**5-agent total: approximately -1,650 to -3,400**

This matches the best observed episode of -545 (single agent, from
optimality_analysis.md).

---

## 7. GreenAccessService Bug

The `GreenAccessService.execute()` method at line 176 contains:

```python
if not self.available_dest_service:
    return obs
```

This checks the **method object** (always truthy), not calling the method.
The correct code would be `if not self.available_dest_service():`.
As a result, the destination service availability check **never executes**,
and GreenAccessService never fails due to degraded/stopped services.

This explains why `total_access_service_failures = 0` in the green tracer
and why ASF penalties are always 0 in the reward tracer. Green access service
always succeeds (or is blocked by firewall rules, which are not active in
this no-blue-agent run).

Impact: ASF penalties defined in BlueRewardMachine (-1 to -5 per event) are
effectively dead code. They would only fire if this bug were fixed and green
access service actually checked destination service health.

---

## 8. Database Reference

| Database | Tables | Contents |
|----------|--------|----------|
| `data/green_trace.db` | `green_actions`, `phishing_events`, `fp_analysis`, `summary_stats` | 534K traced green actions, 846 phishing events, FP events |
| `data/action_order_trace.db` | `execution_order`, `same_host_conflicts`, `action_priorities`, `timing_analysis`, `summary_stats` | 621K action executions, priority mapping |
| `data/reward_trace.db` | `reward_components`, `step_rewards`, `phase_rewards`, `action_costs`, `restore_cost_analysis`, `impact_cost_analysis`, `summary_stats` | 195K penalty events, per-step reward decomposition |

### Query Examples

```sql
-- FP rate per subnet
SELECT target_subnet,
       ROUND(100.0 * SUM(created_proc_fp) / COUNT(*), 3) as fp_rate
FROM green_actions WHERE action_type='GreenLocalWork'
GROUP BY target_subnet ORDER BY fp_rate DESC;

-- Phishing events per phase
SELECT mission_phase, COUNT(*) FROM phishing_events GROUP BY mission_phase;

-- Worst steps by reward
SELECT episode_id, step, total_reward, n_lwf, n_ria, mission_phase
FROM step_rewards ORDER BY total_reward LIMIT 10;

-- Penalty decomposition by subnet and phase
SELECT subnet, mission_phase, penalty_type, SUM(penalty_value)
FROM reward_components
GROUP BY subnet, mission_phase, penalty_type
ORDER BY mission_phase, SUM(penalty_value);
```

---

## 9. Key Findings Summary

1. **Green FP rate is ~0.78% (process) and ~0.97% (network)**, not 1%.
   Process FP is lower because it only fires on successful local work.

2. **Phishing creates 28.2 new red sessions per episode**, predominantly in
   Phase 0 (67%). Only ~42% of phishing RNG triggers succeed (routing issues).
   Phishing bypasses all firewall blocks.

3. **LWF accounts for 83.3% of unmitigated damage.** RIA is only 16.7%.
   This confirms that service degradation cascade (not direct Impact) is
   the primary penalty driver.

4. **Phase 2 is 20x worse than Phase 0** per step (-24.66 vs -1.24) due to
   cumulative service degradation throughout the episode.

5. **Block/Allow executes before all other actions** (priority=1 vs 99).
   Blue's Restore executes before Red's Exploit when both fire on the same
   step (alphabetical agent ordering).

6. **GreenAccessService has a bug** that makes destination service checks
   inoperative, rendering ASF penalties dead code.

7. **On OZ hosts during mission phases, 1 missed Impact step costs more
   than 1 unnecessary Restore.** Restore-first is correct strategy.

8. **61 green agents** are active per episode across 8 subnets, with
   contractor_network having the most (10) and operational zones the fewest
   (3-5).
