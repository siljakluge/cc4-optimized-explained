# CC4 Reward Structure and Green Agent Behavior Analysis

## 1. Reward Decomposition

### 1.1 Total Reward Formula

The blue agent's per-step reward is the sum of three components:

```
step_reward = BlueRewardMachine_reward + action_cost
```

Where:
- `BlueRewardMachine_reward` = sum of penalties from failed green actions + successful red Impacts
- `action_cost` = sum of `.cost` property across all blue agent actions this step

The final reward returned to the agent is `round(sum(all_reward_components), 1)`.

### 1.2 Blue Action Costs (`.cost` property)

| Action            | Cost | Duration (steps) | Notes |
|-------------------|------|-------------------|-------|
| Sleep             |  0   | 1                 | No-op |
| Monitor           |  0   | 1                 | Runs automatically at end of each step regardless |
| Analyse           |  0   | 2                 | Runs DensityScout + SigCheck on one host |
| Remove            |  0   | 3                 | Kills user-level red sessions + malware files |
| Restore           | -1   | 5                 | Reimages host, clears ALL sessions (red+green) |
| DeployDecoy       |  0   | 2                 | Creates a decoy service |
| BlockTrafficZone  |  0   | 1                 | Blocks subnet-to-subnet traffic |
| AllowTrafficZone  |  0   | 1                 | Unblocks subnet-to-subnet traffic |
| InvalidAction     | -0.1 | 1                 | Submitted invalid action |

Key insight: **Only Restore has a direct action cost (-1).** Remove is free. All monitoring/analysis actions are free.

### 1.3 BlueRewardMachine Penalties (per event, per step)

The BlueRewardMachine scores penalties from two sources:
1. **Green action failures** (LWF = Local Work Fails, ASF = Access Service Fails)
2. **Red Impact successes** (RIA = Red Impact/Access)

A penalty fires when:
- A green agent action (GreenLocalWork or GreenAccessService) returns `success=False`
- A red Impact action returns `success=True`

The penalty value depends on which **subnet** the affected host is in and which **mission phase** is active.

#### Phase 0: Pre-Planning (Steps 0-166)

| Subnet                    | LWF  | ASF  | RIA  |
|---------------------------|------|------|------|
| public_access_zone        |  -1  |  -1  |  -3  |
| admin_network             |  -1  |  -1  |  -3  |
| office_network            |  -1  |  -1  |  -3  |
| contractor_network        |   0  |  -5  |  -5  |
| restricted_zone_a         |  -1  |  -3  |  -1  |
| operational_zone_a        |  -1  |  -1  |  -1  |
| restricted_zone_b         |  -1  |  -3  |  -1  |
| operational_zone_b        |  -1  |  -1  |  -1  |
| internet                  |   0  |   0  |  -1  |

#### Phase 1: Mission A (Steps 167-333)

| Subnet                    | LWF  | ASF  | RIA  |
|---------------------------|------|------|------|
| public_access_zone        |  -1  |  -1  |  -3  |
| admin_network             |  -1  |  -1  |  -3  |
| office_network            |  -1  |  -1  |  -3  |
| contractor_network        |   0  |   0  |   0  |
| **restricted_zone_a**     |  -2  |  -1  | **-3** |
| **operational_zone_a**    |**-10**|  0  |**-10** |
| restricted_zone_b         |  -1  |  -1  |  -1  |
| operational_zone_b        |  -1  |  -1  |  -1  |
| internet                  |   0  |   0  |   0  |

#### Phase 2: Mission B (Steps 334-499)

| Subnet                    | LWF  | ASF  | RIA  |
|---------------------------|------|------|------|
| public_access_zone        |  -1  |  -1  |  -3  |
| admin_network             |  -1  |  -1  |  -3  |
| office_network            |  -1  |  -1  |  -3  |
| contractor_network        |   0  |   0  |   0  |
| restricted_zone_a         |  -1  |  -3  | **-3** |
| operational_zone_a        |  -1  |  -1  |  -1  |
| **restricted_zone_b**     |  -2  |  -1  | **-3** |
| **operational_zone_b**    |**-10**|  0  |**-10** |
| internet                  |   0  |   0  |   0  |

### 1.4 Critical Observations About Reward Sources

1. **There are NO positive rewards.** The best possible step reward is 0. Every penalty pushes the total more negative.

2. **Green service disruption is a major penalty source.** When green's local work fails (LWF) on OZ hosts during active missions, the penalty is -10 per green agent per step. This is as severe as a red Impact.

3. **Red Impact penalties are per-event, not ongoing.** A successful `Impact` action fires the RIA penalty once at the step it succeeds. However, the Impact stops the OT service, which then causes all subsequent green LWF failures on that host (each at -10 in mission zones).

4. **Restore disrupts green services.** Restoring a host reimages it, which temporarily disrupts services and can cause green LWF failures.


## 2. Phase Timing

### 2.1 Phase Boundaries (500-step episode)

```
_generate_mission_phases(500):
  500 / 3 = 166 remainder 2
  remainder == 2: return (167, 167, 166)
```

| Phase | Name          | Steps (inclusive) | Duration |
|-------|---------------|-------------------|----------|
| 0     | Pre-Planning  | 0 - 166           | 167 steps |
| 1     | Mission A     | 167 - 333         | 167 steps |
| 2     | Mission B     | 334 - 499         | 166 steps |

### 2.2 What Changes Between Phases

#### Network Connectivity (Comms Policy / Allowed Subnets)

Each phase defines which subnet pairs are allowed to communicate. Green agents use this to decide where they can send `GreenAccessService` requests.

**Phase 0 (Pre-Planning) allowed connections:**
```
PAZ <-> Contractor, Admin <-> Contractor, Office <-> Contractor
PAZ <-> RZ-A, Admin <-> RZ-A, Office <-> RZ-A
PAZ <-> RZ-B, Admin <-> RZ-B, Office <-> RZ-B
RZ-A <-> Contractor
OZ-A <-> RZ-A
RZ-B <-> Contractor
RZ-B <-> RZ-A
OZ-B <-> RZ-B
```

**Phase 1 (Mission A) -- opens OZ-A path, closes some RZ-B paths:**
```
PAZ <-> Contractor, Admin <-> Contractor, Office <-> Contractor
PAZ <-> RZ-A, Admin <-> RZ-A, Office <-> RZ-A
PAZ <-> RZ-B, Admin <-> RZ-B, Office <-> RZ-B
RZ-B <-> Contractor
OZ-B <-> RZ-B
```
Note: RZ-A <-> Contractor and OZ-A <-> RZ-A are REMOVED. RZ-B <-> RZ-A is REMOVED.
This **isolates** the Mission A zone (RZ-A/OZ-A) from the contractor network.

**Phase 2 (Mission B) -- opens OZ-B path, closes some RZ-A paths:**
```
PAZ <-> Contractor, Admin <-> Contractor, Office <-> Contractor
PAZ <-> RZ-A, Admin <-> RZ-A, Office <-> RZ-A
PAZ <-> RZ-B, Admin <-> RZ-B, Office <-> RZ-B
RZ-A <-> Contractor
OZ-A <-> RZ-A
```
Note: RZ-B <-> Contractor and OZ-B <-> RZ-B are REMOVED.
This **isolates** the Mission B zone (RZ-B/OZ-B) from the contractor network.

#### High-Value Targets Per Phase

| Phase | High-Value Subnet (RIA=-10) | LWF=-10 Subnet | Critical Hosts |
|-------|----------------------------|-----------------|----------------|
| 0     | None above -5              | None above -1   | contractor_network (RIA=-5, ASF=-5) |
| 1     | operational_zone_a (-10)   | operational_zone_a (-10) | OZ-A servers |
| 2     | operational_zone_b (-10)   | operational_zone_b (-10) | OZ-B servers |

The HQ subnets (PAZ, admin, office) maintain a steady -3 RIA penalty across all phases.


## 3. Green Agent Behavior

### 3.1 Action Selection

The `EnterpriseGreenAgent` has 3 possible actions, chosen uniformly at random each step:

| Action             | Probability | Description |
|--------------------|-------------|-------------|
| GreenLocalWork     | ~33.3%      | Access a local service on own host |
| GreenAccessService | ~33.3%      | Access a service on a remote server |
| Sleep              | ~33.3%      | Do nothing |

### 3.2 GreenLocalWork -- Detailed Behavior

**Step 1: Access local service**
- Picks a random active service on the host
- Checks service reliability (default 100%, degraded by red's DegradeServices by 20% per hit)
- If `random(0-99) >= reliability`: action **fails** (triggers LWF penalty)
- If no active services exist: action **fails** (triggers LWF penalty)

**Step 2: False positive alert (fp_detection_rate = 1%)**
- With 1% probability, creates a `process_creation` event on the host
- This event contains `{local_address, local_port}` with an ephemeral port
- Blue's Monitor action collects this as a suspicious process creation event
- **This is the primary false positive signal** that mimics red activity

**Step 3: Phishing error (phishing_error_rate = 1%)**
- With 1% probability, executes a `PhishingEmail` sub-action
- This creates a **real red shell session** on the green agent's host
- The red agent assigned is from the same or a routable subnet
- **This is not a false positive -- it genuinely gives red a foothold**
- PhishingEmail **bypasses BlockTrafficZone** firewall rules (design decision ADR-CC4-001)

### 3.3 GreenAccessService -- Detailed Behavior

**Destination selection:**
- If green's subnet is in `allowed_subnets` for the current mission phase: can reach any server in any allowed subnet
- If green's subnet is NOT in `allowed_subnets`: can only reach servers within its own subnet
- Only targets server hosts (hostname contains "server"), never itself

**Step 1: Check firewall blocks**
- If `BlockTrafficZone` has been used to block the connection path, the action **fails**
- Creates a `network_connection` event on the destination host (local=dest, remote=source, port=8800)
- **This blocked-connection event is visible to blue's Monitor** and looks like suspicious network activity

**Step 2: False positive (fp_detection_rate = 1%)**
- With 1% probability, creates a `network_connection` event on the destination host
- Contains `{local_address=src_ip, remote_address=dest_ip, remote_port=ephemeral_port}`
- Blue's Monitor collects this as a suspicious network connection event

**Step 3: If not blocked and no FP, action succeeds (no event generated)**

### 3.4 False Positive Analysis

#### Process Creation False Positives (from GreenLocalWork)

- **Rate**: 1% per GreenLocalWork action per green agent per step
- **Trigger**: Adds a `process_creation` event to `host.events`
- **What blue sees**: A process creation entry with `{local_address, local_port}` -- identical format to what red exploit/escalation actions create
- **Distinguishing feature**: The process has no PID initially (just address/port), making it hard to distinguish from legitimate red activity events

Expected FP rate per step (assuming ~N green agents, 33% chance of GreenLocalWork):
```
P(at least one proc FP) = 1 - (1 - 0.01 * 0.333)^N
For N=30 hosts: ~9.5% chance of at least one proc FP per step
For N=50 hosts: ~15.3% chance of at least one proc FP per step
```

#### Network Connection False Positives (from GreenAccessService)

- **Rate**: 1% per GreenAccessService action per green agent per step
- **Trigger**: Adds a `network_connection` event to the **destination server** host
- **What blue sees**: A network connection entry on the server -- same format as red lateral movement events
- **Additional noise**: Blocked connections ALSO generate network_connection events (at 100% rate)

Expected FP rate per step from GreenAccessService:
```
P(at least one net FP) = 1 - (1 - 0.01 * 0.333)^N
Same formula as above -- ~9.5% for 30 agents, ~15.3% for 50 agents
```

#### Blocked-Connection Noise

When blue uses `BlockTrafficZone`, every `GreenAccessService` that hits the block creates a `network_connection` event at port 8800. With ~33% of green agents attempting GreenAccessService:
```
Expected blocked-connection events per step = 0.333 * (number of green agents on blocked subnets)
```
This creates significant false positive noise on server hosts if comms are blocked.

### 3.5 Phishing: Green Creating Real Red Footholds

- **Rate**: 1% per GreenLocalWork action per green agent per step
- **Effect**: Creates a real `RedAbstractSession` with user-level access
- **Cannot be blocked by BlockTrafficZone** (bypasses firewall, uses email delivery model)
- **Expected new red footholds per step**: `0.01 * 0.333 * N_green_agents`
  - For 30 agents: ~0.1 new footholds/step, ~50 over 500 steps
  - For 50 agents: ~0.17 new footholds/step, ~83 over 500 steps


## 4. Reward Math for a 500-Step Episode

### 4.1 Theoretical Maximum Reward

**Best case: no red activity, no green failures, all services healthy**
- BlueRewardMachine: 0 every step (no failures, no impacts)
- action_cost: 0 every step (blue does Sleep or Monitor only)
- **Maximum total reward: 0**

### 4.2 Theoretical Minimum Reward (catastrophic scenario)

Assume red Impacts every OZ host every step during mission phases, and all green local work fails:

**Phase 1 (167 steps):**
- Each OZ-A host with green agent: -10 LWF per agent per step
- Each successful Impact on OZ-A: -10 per event
- With ~5-10 OZ-A hosts and ~5-10 green agents: could be -50 to -100+ per step
- Over 167 steps: -8,350 to -16,700+ from OZ-A alone

**Phase 2 (166 steps):**
- Same calculation for OZ-B

**Plus HQ penalties across all phases:**
- HQ hosts (PAZ, admin, office) at -3 RIA and -1 LWF continuously

**Realistic worst case estimate: -30,000 to -50,000+**

### 4.3 Cost-Benefit Decision Matrix

| Scenario | Cost | Notes |
|----------|------|-------|
| Unnecessary Restore on non-mission host | -1 (action cost) + LWF penalties during reimage | Services disrupted during restore |
| Unnecessary Restore on OZ-A during Phase 1 | -1 + (-10 * duration) = **-51** | 5 steps of green LWF at -10 each |
| Missed Impact on OZ-A during Phase 1 | -10 per step it persists | Plus ongoing LWF failures (-10/agent/step) |
| Missed Impact on HQ host | -3 per step | Plus -1 LWF per agent/step |
| False-positive Restore on HQ host | -1 + (-1 * duration) = **-6** | 5 steps of -1 LWF |
| Remove (correct) on any host | 0 (action cost) | No service disruption |
| Remove (unnecessary) on any host | 0 (action cost) | No penalty, just wasted 3 steps |

### 4.4 Critical Cost-Benefit Insight

**On mission-critical OZ hosts during active mission phases:**
- Cost of one unnecessary Restore: ~ -1 (action) + -10 * 5 (green failures) = **-51 worst case**
- Cost of one missed Impact persisting for 10 steps: -10 * 10 = **-100** (plus cascading green failures)
- **Conclusion: On OZ hosts, a missed Impact is far worse than an unnecessary Restore**

**On HQ hosts (PAZ, admin, office):**
- Cost of one unnecessary Restore: -1 (action) + -1 * 5 (green failures) = **-6**
- Cost of one missed Impact persisting for 10 steps: -3 * 10 = **-30**
- **Conclusion: Still better to over-Restore than under-detect, but margin is smaller**

**On non-critical hosts (internet, contractor in Phase 1/2):**
- Cost of unnecessary Restore: -1 (action)
- Cost of missed Impact: 0 (RIA=0 in these zones during mission phases)
- **Conclusion: Do NOT Restore contractor/internet hosts during mission phases**


## 5. Blue Action Details

### 5.1 Remove

- **Duration**: 3 steps
- **Direct cost**: 0
- **What it clears**: 
  - Kills processes with suspicious PIDs tracked by VelociraptorServer
  - Removes malware files (density >= 0.9, unsigned)
- **What survives**: 
  - Root/SYSTEM sessions (privileged red sessions persist)
  - The persistence mechanism assumed for privileged sessions
- **Service impact**: None -- services remain active
- **Use case**: First response to user-level red activity; no reward penalty

### 5.2 Restore

- **Duration**: 5 steps
- **Direct cost**: -1
- **What it clears**: 
  - ALL red sessions (including root/SYSTEM)
  - ALL malware files
  - Restores host to backup state (`host.restore()`)
  - Note: `impact_count` is NOT reset by restore (past impacts already scored)
- **What survives**:
  - Blue sessions are re-injected after restore
  - Host comes back clean
- **Service impact**: Services are restored to healthy state (reliability reset)
- **Use case**: Escalated response when Remove is insufficient (root sessions present)

### 5.3 DeployDecoy

- **Duration**: 2 steps
- **Direct cost**: 0
- **Types**: Apache, Tomcat, HarakaSMTP, Vsftpd
- **Effect**: Creates a fake service that red can discover/exploit, providing blue with detection intel

### 5.4 Monitor

- **Duration**: 1 step
- **Direct cost**: 0
- **Runs automatically**: At the end of each step regardless of blue action
- **If called manually**: No additional effect (events already collected)
- **Collects**: network_connections and process_creation events from all monitored hosts
- **Clears events after collection**: Events are moved to `old_*` lists

### 5.5 Analyse

- **Duration**: 2 steps
- **Direct cost**: 0
- **Effect**: Runs DensityScout (finds files with density > 0.9) and SigCheck (checks signatures)
- **Returns**: Information about suspicious files on the target host
- **Use case**: Confirm presence of malware before deciding Remove vs Restore


## 6. Mission-Specific Impact Analysis

### 6.1 OT Services and Impact

Red's `Impact` action:
1. Requires privileged (root/SYSTEM) session on host
2. Requires active OT service (`ProcessName.OTSERVICE`) on host
3. Stops the OT service via `StopService`
4. Increments `impact_count` on the host (never reset)
5. Returns success=True, triggering RIA reward penalty

After Impact:
- OT service is stopped (`active=False`)
- Green's GreenLocalWork on that host will fail (no active services) -> triggers LWF penalty every step
- Green's GreenAccessService to that host will fail (service check fails) -> triggers ASF penalty

### 6.2 DegradeServices vs Impact

Red's `DegradeServices` action:
- Requires privileged session
- Reduces reliability of ALL active services by 20% per execution
- Does NOT stop services outright
- After 5 DegradeServices: reliability drops to 0%, causing all green local work to fail probabilistically
- **Does NOT trigger RIA penalty** (it is not an Impact action)
- But it indirectly causes green LWF/ASF failures, which DO incur penalties

### 6.3 Per-Step Penalty Comparison by Zone

| Zone | Phase 0 | Phase 1 | Phase 2 | When critical |
|------|---------|---------|---------|---------------|
| OZ-A | -1 (RIA), -1 (LWF) | **-10 (RIA), -10 (LWF)** | -1 (RIA), -1 (LWF) | Phase 1 only |
| OZ-B | -1 (RIA), -1 (LWF) | -1 (RIA), -1 (LWF) | **-10 (RIA), -10 (LWF)** | Phase 2 only |
| RZ-A | -1 (RIA), -1 (LWF) | **-3 (RIA), -2 (LWF)** | **-3 (RIA)**, -1 (LWF) | Phase 1-2 |
| RZ-B | -1 (RIA), -1 (LWF) | -1 (RIA), -1 (LWF) | **-3 (RIA), -2 (LWF)** | Phase 2 |
| HQ (PAZ/Admin/Office) | -3 (RIA), -1 (LWF) | -3 (RIA), -1 (LWF) | -3 (RIA), -1 (LWF) | Always |
| Contractor | -5 (RIA), 0 (LWF) | 0 (all) | 0 (all) | Phase 0 only |
| Internet | -1 (RIA), 0 (LWF) | 0 (all) | 0 (all) | Phase 0 only |


## 7. Comms Policy Effects on Rewards

### 7.1 BlockTrafficZone Reward Effects

Blocking traffic has **no direct reward cost** (action cost = 0). However:

**Indirect penalty from blocking:**
- Green agents on blocked subnets attempting `GreenAccessService` to blocked destinations will fail
- Each failure generates an ASF penalty based on the green agent's subnet
- Additionally, blocked connections create `network_connection` events that pollute blue's observation space

**ASF penalties when blocking in Phase 0:**
- Blocking RZ-A <-> anything: -3 ASF per failed green access from RZ-A
- Blocking contractor <-> anything: -5 ASF per failed green access from contractor

**ASF penalties when blocking in Phase 1:**
- Most ASF values are -1 or 0 for mission-zone subnets
- OZ-A: ASF = 0 (no penalty for blocking access services to/from OZ-A)
- Contractor: ASF = 0
- This means **blocking OZ-A comms during Phase 1 incurs no ASF penalty**

**ASF penalties when blocking in Phase 2:**
- OZ-B: ASF = 0 (no penalty for blocking access services to/from OZ-B)
- Same logic: **blocking OZ-B comms during Phase 2 incurs no ASF penalty**

### 7.2 Key Blocking Insight

The reward structure is designed so that blocking the mission-critical operational zones during their active phases has **ASF=0**, meaning green access service failures from those zones carry no penalty. This intentionally makes blocking a viable defensive strategy without reward cost.

However, blocking can still cause **LWF penalties** if it disrupts local services (though BlockTrafficZone only affects cross-subnet traffic, not local service access).

### 7.3 AllowTrafficZone

- **Cost**: 0
- **Purpose**: Reverses a previous BlockTrafficZone
- **When needed**: To restore green access service connectivity after a threat is cleared
- **Reward consideration**: Unblocking too early re-opens attack paths; unblocking too late continues to block legitimate green traffic (though ASF=0 in most mission scenarios)


## 8. Decision Trees for Blue Strategy

### 8.1 Per-Step Decision: Respond to Alert

```
Alert detected on host H in subnet S at step T
|
+-- Determine current phase P from T
|   P=0 if T < 167, P=1 if 167 <= T < 334, P=2 if T >= 334
|
+-- Look up reward weights for (S, P)
|   |
|   +-- Is RIA >= -3 for this (S,P)?  [High-value target]
|   |   |
|   |   +-- YES: Prioritize immediate response
|   |   |   +-- Is there confirmed root access? -> Restore (-1 cost, clears everything)
|   |   |   +-- Is there only user-level access? -> Remove (0 cost, clears user sessions)
|   |   |
|   |   +-- NO (RIA = 0 or -1): Low priority
|   |       +-- Queue for later or Skip
|   |
|   +-- Is this a false positive?
|       +-- Check: Did Analyse find malware (density >= 0.9, unsigned)?
|       +-- Check: Is there a red session on this host?
|       +-- If neither: likely green FP -> do nothing (save 3-5 steps)
```

### 8.2 Phase Transition Checklist

```
At step 167 (Phase 0 -> Phase 1):
  - OZ-A becomes critical (RIA/LWF = -10)
  - RZ-A becomes important (RIA = -3)
  - Contractor/Internet become zero-penalty
  - Consider pre-emptive Restore on OZ-A/RZ-A hosts
  - Block contractor <-> RZ-A if not already blocked

At step 334 (Phase 1 -> Phase 2):
  - OZ-B becomes critical (RIA/LWF = -10)
  - RZ-B becomes important (RIA = -3)
  - OZ-A drops to -1 (less critical)
  - Consider pre-emptive Restore on OZ-B/RZ-B hosts
  - Block contractor <-> RZ-B if not already blocked
```


## 9. Summary Tables

### 9.1 Reward Sensitivity Rankings by Phase

**Phase 0 -- Defend everywhere lightly:**
1. Contractor network (RIA=-5, ASF=-5)
2. HQ subnets (RIA=-3)
3. RZ-A/RZ-B (ASF=-3)
4. Everything else (RIA=-1)

**Phase 1 -- All-in on Mission A zone:**
1. Operational Zone A (RIA=-10, LWF=-10)
2. HQ subnets (RIA=-3)
3. Restricted Zone A (RIA=-3, LWF=-2)
4. Everything else (-1 or 0)

**Phase 2 -- All-in on Mission B zone:**
1. Operational Zone B (RIA=-10, LWF=-10)
2. HQ subnets (RIA=-3)
3. Restricted Zone B (RIA=-3, LWF=-2)
4. Restricted Zone A (RIA=-3)
5. Everything else (-1 or 0)

### 9.2 Green Noise Summary

| Source | Rate | Event Type | Where Appears | Distinguishable? |
|--------|------|------------|---------------|-------------------|
| GreenLocalWork FP | 1% per action | process_creation | Same host as green agent | No -- same format as red |
| GreenAccessService FP | 1% per action | network_connection | Destination server host | No -- same format as red |
| Blocked GreenAccessService | 100% when blocked | network_connection (port 8800) | Destination server | Partially -- fixed port 8800 |
| Phishing (real threat) | 1% per GreenLocalWork | Creates red session | Green agent's host | N/A -- actual red access |

### 9.3 Break-Even Analysis

**Question: How many steps of Impact must be prevented to justify one unnecessary Restore?**

On OZ-A during Phase 1:
- Unnecessary Restore cost: -1 (action) + up to -10 * 5 = **-51** worst case
- Impact penalty per step: -10 (RIA) + potential -10 * N_green (LWF)
- Break-even: **1 step of prevented Impact on OZ-A justifies up to 5 unnecessary Restores**

On HQ hosts:
- Unnecessary Restore cost: -1 (action) + up to -1 * 5 = **-6**
- Impact penalty per step: -3
- Break-even: **2 steps of prevented Impact justifies one unnecessary Restore**

On contractor during Phase 1/2:
- Unnecessary Restore cost: -1 (action)
- Impact penalty: 0
- **Never Restore contractor hosts during mission phases**
