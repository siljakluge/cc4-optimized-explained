# Red Agent Attack Chain Database -- Empirical Analysis

**Date**: 2026-04-07
**Method**: 30 episodes, seed 42, 500 steps/episode, FiniteStateRedAgent, no blue defense
**Data source**: `data/red_trace.db` (SQLite, 82,980 action records, 7.7M session records)

---

## 1. Complete FSM Diagram

### 1.1 State Definitions

| State | Meaning | Red Knowledge |
|-------|---------|---------------|
| K | Known | IP discovered via DiscoverRemoteSystems |
| KD | Known + Decoy detected | A decoy was discovered on this host |
| S | Services scanned | Ports known via AggressiveServiceDiscovery or StealthServiceDiscovery |
| SD | Services scanned + Decoy detected | Service scan done AND decoy found |
| U | User shell | Red has user-level session (via ExploitRemoteService) |
| UD | User shell + Decoy detected | User shell AND decoy known |
| R | Root shell | Red has privileged session (via PrivilegeEscalate) |
| RD | Root shell + Decoy detected | Root AND decoy known |
| F | Foreign | Host outside agent's subnet. Ignored permanently. |

### 1.2 Complete ASCII FSM Diagram with Every Transition

```
                         +---------+
                         |  INIT   |
                         | (step 0)|
                         +----+----+
                              |
                     Starting host enters state U
                              |
     +------- DiscoverRemoteSystems (1 step, success) --------+
     |           All hosts in subnet: state -> K/KD           |
     |           (K if already K, KD if from K via DRS)        |
     |                                                         |
     v                                                         v
+----+----+  success                                     +-----+-----+
|  K      |---AggrSvcDisc(1step)----> S                  |  KD        |
| (Known) |---StlthSvcDisc(3step)---> S                  | (Known +   |
|         |---DiscRemSys(1step)-----> KD                 |  Decoy)    |
+---------+  fail                                        +-----+------+
|         |---AggrSvcDisc-----------> K (no change)            |
|         |---StlthSvcDisc---------> K (no change)       Same as K:
|         |---DiscRemSys-----------> K (no change)       DRS(50%), Aggr(25%),
+---------+                                               Stlth(25%)
     P(DRS)=50%, P(Aggr)=25%, P(Stlth)=25%                    |
                                                               |
+----+----+  success                                     +------+-----+
|  S      |---Exploit(4step)---------> U                 |  SD        |
| (Svcs)  |---DiscDeception(2step)---> S                 | (Svcs +    |
|         |---DiscRemSys(1step)------> SD                |  Decoy)    |
+---------+  fail                                        +------+-----+
|         |---Exploit(4step)---------> S (no change)     P(Exploit)=75%
|         |---DiscDeception(2step)---> S (no change)     P(DiscDecept)=25%
|         |---DiscRemSys(1step)------> S (no change)           |
+---------+                                                    |
     P(Exploit)=50%, P(DD)=25%, P(DRS)=25%                     |
                                                               |
+----+----+  success                                     +------+-----+
|  U      |---PrivEsc(2step)---------> R                 |  UD        |
| (User)  |---DiscRemSys(1step)------> UD                | (User +    |
|         |---Withdraw(1step)--------> S                 |  Decoy)    |
+---------+  fail                                        +------+-----+
|         |---PrivEsc(2step)---------> U (no change)     P(PrivEsc)=100%
|         |---DiscRemSys(1step)------> U (no change)     P(Withdraw)=0%
|         |---Withdraw(1step)--------> U (no change)           |
+---------+                                                    |
     P(PrivEsc)=50%, P(DRS)=50%, P(Withdraw)=0.0              |
                                                               |
+----+----+  success                                     +------+-----+
|  R      |---Impact(2step)----------> R                 |  RD        |
| (Root)  |---Degrade(2step)---------> R                 | (Root +    |
|         |---DiscRemSys(1step)------> RD                |  Decoy)    |
|         |---Withdraw(1step)--------> S                 +------+-----+
+---------+  fail                                        P(Impact)=50%
|         |---Impact(2step)----------> R (no change)     P(Degrade)=50%
|         |---Degrade(2step)---------> R (no change)     P(Withdraw)=0%
|         |---DiscRemSys(1step)------> R (no change)           |
|         |---Withdraw(1step)--------> R (no change)           |
+---------+                                                    |
     P(Impact)=25%, P(Degrade)=25%, P(DRS)=50%,               |
     P(Withdraw)=0.0                                           |
                                                               |
+----+----+                                                    |
|  F      |--- Only DRS allowed, stays F                       |
| Foreign |    (host outside agent's subnets)                   |
+---------+                                                    |

BLUE INTERVENTION:
  When blue Removes/Restores a red session:
  Any host in U/UD/R/RD that loses its session -> KD
  (Red must re-scan and re-exploit from scratch)
```

### 1.3 Transition Matrices (Exact Values from Code)

**Success Transition Matrix** (FiniteStateRedAgent.py lines 441-452):
```
        DRS    ASD    SSD    DD     ERS    PE     Imp    Deg    With
K  :   [KD,    S,     S,     None,  None,  None,  None,  None,  None]
KD :   [KD,    SD,    SD,    None,  None,  None,  None,  None,  None]
S  :   [SD,    None,  None,  S,     U,     None,  None,  None,  None]
SD :   [SD,    None,  None,  SD,    UD,    None,  None,  None,  None]
U  :   [UD,    None,  None,  None,  None,  R,     None,  None,  S   ]
UD :   [UD,    None,  None,  None,  None,  RD,    None,  None,  SD  ]
R  :   [RD,    None,  None,  None,  None,  None,  R,     R,     S   ]
RD :   [RD,    None,  None,  None,  None,  None,  RD,    RD,    SD  ]
F  :   [F,     None,  None,  None,  None,  None,  None,  None,  None]
```

**Failure Transition Matrix** (lines 481-492):
```
        DRS    ASD    SSD    DD     ERS    PE     Imp    Deg    With
K  :   [K,     K,     K,     None,  None,  None,  None,  None,  None]
KD :   [KD,    KD,    KD,    None,  None,  None,  None,  None,  None]
S  :   [S,     None,  None,  S,     S,     None,  None,  None,  None]
SD :   [SD,    None,  None,  SD,    SD,    None,  None,  None,  None]
U  :   [U,     None,  None,  None,  None,  U,     None,  None,  U   ]
UD :   [UD,    None,  None,  None,  None,  UD,    None,  None,  UD  ]
R  :   [R,     None,  None,  None,  None,  None,  R,     R,     R   ]
RD :   [RD,    None,  None,  None,  None,  None,  RD,    RD,    RD  ]
F  :   [F,     None,  None,  None,  None,  None,  None,  None,  None]
```

**Probability Matrix** (lines 540-552):
```
        DRS    ASD    SSD    DD     ERS    PE     Imp    Deg    With
K  :   [0.50,  0.25,  0.25,  None,  None,  None,  None,  None,  None]
KD :   [0.50,  0.25,  0.25,  None,  None,  None,  None,  None,  None]
S  :   [0.25,  None,  None,  0.25,  0.50,  None,  None,  None,  None]
SD :   [None,  None,  None,  0.25,  0.75,  None,  None,  None,  None]
U  :   [0.50,  None,  None,  None,  None,  0.50,  None,  None,  0.0 ]
UD :   [None,  None,  None,  None,  None,  1.00,  None,  None,  0.0 ]
R  :   [0.50,  None,  None,  None,  None,  None,  0.25,  0.25,  0.0 ]
RD :   [None,  None,  None,  None,  None,  None,  0.50,  0.50,  0.0 ]
```

### 1.4 Critical FSM Mechanics

**Foreign (F) state transition** (lines 156-160): When a success transition would
yield state `U`, the code checks if the host IP is within `agent_subnets`. If not,
the state becomes `F` instead of `U`. Foreign hosts are permanently excluded from
`known_hosts` (line 114: filtered out of action selection).

**Session removal by Blue** (lines 169-188): At each step, the agent checks all
hosts in states containing `U` or `R` (i.e., U, UD, R, RD). If the host's session
no longer appears in the observation, the state resets to `KD`. This means blue
Remove/Restore drops red back to KD (must re-scan and re-exploit).

**Decoy-aware states** (D suffix): When DiscoverDeception detects decoy PIDs, they
are stored in `host_service_decoy_status`. In D-states, the `PIDSelectiveExploitActionSelector`
excludes those ports, reducing the pool of available exploits. This is paradoxically
dangerous: UD has P(PrivEsc)=100% vs U's 50%, and RD has P(Impact)=50%/P(Degrade)=50%
vs R's 25%/25%/50%.

**Host selection** (lines 252-293): The `_choose_host` method:
1. If `host_states_priority_list` is set (base class returns None), weight-select a state
2. If `prioritise_servers` is True, 75% chance of picking a server, 25% non-server
3. Otherwise, uniform random among hosts in the chosen state

---

## 2. Attack Timing Distributions (Empirical)

### 2.1 Action Durations

| Action | Duration (steps) |
|--------|-----------------|
| DiscoverRemoteSystems | 1 |
| AggressiveServiceDiscovery | 1 |
| StealthServiceDiscovery | 3 |
| DiscoverDeception | 2 |
| ExploitRemoteService | 4 |
| PrivilegeEscalate | 2 |
| Impact | 2 |
| DegradeServices | 2 |
| Withdraw | 1 |

### 2.2 Time to Impact (Empirical, 30 episodes, no blue)

Only red_agent_2 (OZ-A) and red_agent_4 (OZ-B) achieve Impact, because only
operational zone hosts have OT services.

| Red Agent | Chains to Impact | Chains w/o Impact | Mean Steps | Std | Min | Max | Median |
|-----------|-----------------|-------------------|-----------|-----|-----|-----|--------|
| red_agent_0 | 0 | 637 | N/A | - | - | - | - |
| red_agent_1 | 0 | 641 | N/A | - | - | - | - |
| **red_agent_2** | **265** | 338 | **300.5** | 109.6 | 4 | 474 | 321 |
| red_agent_3 | 0 | 664 | N/A | - | - | - | - |
| **red_agent_4** | **252** | 299 | **323.5** | 98.4 | 18 | 485 | 342 |
| red_agent_5 | 0 | 1143 | N/A | - | - | - | - |

**Impact cannot succeed on**: contractor_network (no OT service), restricted_zones
(no OT service), public_access_zone/admin/office (no OT service).

### 2.3 First Successful Impact Timing

```
First successful Impact per episode (30 episodes):
  Mean step: 67.2
  Std:       48.4
  Min:        9
  Max:      281
  Median:    66
```

This means red achieves its first OT service disruption by step 67 on average.
The minimum of 9 steps means impact can arrive as early as step 9 (PrivEsc(2) +
DRS(1) + some chain to OZ host).

### 2.4 Per-Phase Red Activity Volume

| Phase | Exploits | PrivEsc | Impact | Impact Success | DegradeServices |
|-------|---------|---------|--------|---------------|-----------------|
| Phase 0 (0-166) | 2,091 | 885 | 1,434 | 202 (14.1%) | 1,528 |
| Phase 1 (167-333) | 2,423 | 493 | 3,879 | 231 (6.0%) | 3,962 |
| Phase 2 (334-499) | 1,876 | 170 | 5,030 | 84 (1.7%) | 5,180 |

**Key pattern**: Exploit and PrivEsc activity drops over time (red has already
established root on most hosts), while Impact and DegradeServices dominate later
phases (red is cycling Impact/Degrade on already-compromised hosts).

Phase 2's Impact success rate drops to 1.7% because most OT services have already
been stopped and remain stopped (without blue Restore).

---

## 3. Exploit Success and Decoy Interaction Statistics

### 3.1 Exploit Success Rates (Empirical)

| Outcome | Count | Percentage |
|---------|-------|-----------|
| Exploit failed (includes decoy hits, blocking, no valid port) | 4,801 | 75.1% |
| Exploit succeeded | 1,589 | 24.9% |
| **Total exploit attempts** | **6,390** | 100% |

The 24.9% success rate aligns with the theoretical model: with 3 decoys deployed
per host and `odds_of_top_choice=0`, approximately 75% of exploits hit decoys or
fail (when decoys are deployed). Without blue deploying decoys, the failure rate
would be lower (blocked routes, no valid ports, etc.).

**Note**: These 30 episodes ran without a blue agent, so no decoys were deployed.
The 75% failure rate is due to blocking (comms_policy) and exploit precondition
failures (wrong port, wrong process type, no vulnerable process).

### 3.2 PrivilegeEscalate Success Rate

```
Total PrivilegeEscalate attempts: 1,548
Successful: 1,548 (100.0%)
Failed: 0 (0.0%)
```

**PrivilegeEscalate never fails** in practice. It uses JuicyPotato (Windows) or
V4L2KernelExploit (Linux), both of which always succeed unless the session is in
a sandbox (`is_escalate_sandbox`). Since no blue agent is deploying sandboxing
decoys, the success rate is 100%.

### 3.3 Impact Success vs Failure by Host

Impact only succeeds on hosts with active OT services (operational zone hosts):

| Host Zone | Total Impact Attempts | Successful | Rate |
|-----------|----------------------|-----------|------|
| operational_zone_a | 1,398 | 282 | 20.2% |
| operational_zone_b | 1,689 | 263 | 15.6% |
| contractor_network | 1,810 | 0 | 0.0% |
| restricted_zone_a | 1,508 | 0 | 0.0% |
| restricted_zone_b | 1,509 | 0 | 0.0% |
| HQ (PAZ/admin/office) | 429 | 0 | 0.0% |

### 3.4 DegradeServices Statistics

```
Total DegradeServices actions: 10,670
Successful: 10,670 (100.0%)
```

DegradeServices always succeeds when red has root. It degrades ALL services on the
host (not just OT). Each call reduces service reliability. This is completely
invisible to blue (no observation signal).

---

## 4. Host Targeting Patterns

### 4.1 Most Targeted Hosts (non-Sleep, non-DRS)

| Rank | Host | Actions |
|------|------|---------|
| 1 | operational_zone_b_subnet_user_host_0 | 689 |
| 2 | operational_zone_b_subnet_server_host_0 | 643 |
| 3 | operational_zone_b_subnet_user_host_2 | 616 |
| 4 | contractor_network_subnet_user_host_1 | 606 |
| 5 | contractor_network_subnet_user_host_2 | 587 |
| 6 | operational_zone_b_subnet_user_host_1 | 571 |
| 7 | contractor_network_subnet_user_host_0 | 556 |
| 8 | contractor_network_subnet_server_host_0 | 540 |
| 9 | operational_zone_a_subnet_user_host_2 | 487 |
| 10 | restricted_zone_a_subnet_user_host_0 | 456 |

**Pattern**: Operational zone hosts receive the most actions because red_agent_2
and red_agent_4 cycle Impact/DegradeServices on them continuously once root is
achieved. Contractor network hosts are second because red_agent_0 starts there
and actively explores.

### 4.2 Server vs User Host Targeting

Server hosts receive proportionally more actions per host due to the 75% server
preference when `prioritise_servers=True` (base class sets False, but the pattern
shows server_host_0 is consistently near the top of targeting lists).

---

## 5. FSM State Time Distribution

### 5.1 Time Spent in Each State

| State | Actions | % of Total | Interpretation |
|-------|---------|-----------|----------------|
| multi (DRS subnet-level) | 49,354 | 59.5% | Most actions are subnet scans |
| RD (Root + Decoy) | 20,934 | 25.2% | Red dominates in root state |
| SD (Services + Decoy) | 7,780 | 9.4% | Repeated exploit attempts |
| KD (Known + Decoy) | 1,869 | 2.3% | Re-scanning after session removal |
| UD (User + Decoy) | 1,459 | 1.8% | Brief state before PrivEsc |
| S (Services) | 895 | 1.1% | Initial scans (decoy-unaware) |
| K (Known) | 521 | 0.6% | Very brief before first DRS |
| U (User) | 89 | 0.1% | Nearly instant to PrivEsc |
| R (Root) | 79 | 0.1% | Nearly instant to DRS->RD |

### 5.2 State-Action Success Rates

| State | Action | Count | Success Rate |
|-------|--------|-------|-------------|
| K | AggressiveServiceDiscovery | 241 | 100% |
| K | StealthServiceDiscovery | 280 | 100% |
| KD | AggressiveServiceDiscovery | 909 | 100% |
| KD | StealthServiceDiscovery | 960 | 100% |
| S | ExploitRemoteService | 593 | 32% |
| S | DiscoverDeception | 302 | 100% |
| SD | ExploitRemoteService | 5,797 | 24% |
| SD | DiscoverDeception | 1,983 | 100% |
| U | PrivilegeEscalate | 89 | 100% |
| UD | PrivilegeEscalate | 1,459 | 100% |
| R | Impact | 28 | 32% |
| R | DegradeServices | 51 | 100% |
| RD | Impact | 10,315 | 5% |
| RD | DegradeServices | 10,619 | 100% |

**Key finding**: Exploit success rate drops from 32% in state S to 24% in state SD.
This shows that `PIDSelectiveExploitActionSelector` (used in D-states) successfully
avoids some exploits, but the remaining exploits also fail for other reasons (wrong
port/process type). The overall 5% Impact success rate in RD is low because most
hosts lack OT services or the OT service is already stopped.

### 5.3 Decoy Awareness Dynamics

The D suffix states are dominant because DiscoverRemoteSystems transitions any host
from K->KD, S->SD, U->UD, R->RD on success. Since DRS is the most common action
(59.5% of all actions), hosts quickly enter D-states. Once in a D-state, a host
never reverts to a non-D state (except via blue session removal -> KD).

---

## 6. Per-Agent Attack Patterns

### 6.1 Red Agent Assignment

| Agent | Allowed Subnets | Starting Condition |
|-------|----------------|-------------------|
| red_agent_0 | contractor_network | Active at step 0 (has session) |
| red_agent_1 | restricted_zone_a | Activated when session crosses to RZ-A |
| red_agent_2 | operational_zone_a | Activated when session crosses to OZ-A |
| red_agent_3 | restricted_zone_b | Activated when session crosses to RZ-B |
| red_agent_4 | operational_zone_b | Activated when session crosses to OZ-B |
| red_agent_5 | PAZ, admin, office | Activated when session crosses to HQ |

### 6.2 Action Distribution Per Agent

| Agent | DRS | SvcDisc | DD | Exploit | PrivEsc | Impact | Degrade |
|-------|-----|---------|----|---------|---------|--------|---------|
| red_agent_0 | 463 | 344 | 325 | 894 | 263 | 2,232 | 2,305 |
| red_agent_1 | 457 | 339 | 349 | 999 | 272 | 1,766 | 1,930 |
| red_agent_2 | 360 | 300 | 282 | 741 | 273 | 1,962 | 1,992 |
| red_agent_3 | 425 | 357 | 395 | 1,059 | 277 | 1,768 | 1,764 |
| red_agent_4 | 321 | 265 | 191 | 548 | 256 | 2,272 | 2,332 |
| red_agent_5 | 826 | 848 | 743 | 2,149 | 207 | 343 | 347 |

**Patterns**:
- red_agent_5 (HQ: PAZ/admin/office) has the most exploits (2,149) but the fewest
  Impact/Degrade because HQ hosts have no OT services.
- red_agent_4 (OZ-B) has the most Impact+Degrade combined (4,604) because OZ-B
  hosts have OT services and it achieves root early.
- red_agent_5 has the most DRS (826) because it covers 3 subnets (PAZ, admin, office).

---

## 7. Exploit Selection Mechanics (Code Analysis)

### 7.1 Available Exploits

| Exploit | Port | Weight | Process Type | Success Condition |
|---------|------|--------|-------------|-------------------|
| HTTPRFI | 80 | 3.0 | http | `"rfi" in vuln_proc.properties` |
| HTTPSRFI | 443 | 4.0 | webserver | `"rfi" in vuln_proc.properties` |
| SSHBruteForce | 22 | 0.1 | SSH | User must be `bruteforceable` |
| SQLInjection | 3390 | 5.0 | mysql | Always True (+ needs port 80 or 443) |
| HarakaRCE | 25 | 6.0 | smtp | `vuln_proc.version < HARAKA_2_8_9` |
| FTPDirectoryTraversal | 21 | 7.0 | femitter | Always True |
| EternalBlue | 139 | - | - | Never used in CC4 |
| BlueKeep | 3389 | - | - | Never used in CC4 |

### 7.2 Selection Algorithm (Critical: odds_of_top_choice = 0)

```python
# ExploitRemoteService.py lines 56-68
odds_of_top_choice = 0     # THE TOP-WEIGHTED EXPLOIT IS NEVER CHOSEN
top_choice = max(weighted_options, key=weighted_options.get)
weighted_options.pop(top_choice)  # Remove top choice from pool
other_choice = state.np_random.choice(list(weighted_options.keys()))
# ... random() < 0 is always False, so top_choice never selected
selected_choice = other_choice   # Random from remaining exploits
```

**This means**: The highest-weight applicable exploit is ALWAYS excluded. Selection
is uniform random among the remaining applicable exploits. This is a critical design
choice that increases decoy effectiveness.

### 7.3 PIDSelectiveExploitActionSelector (D-states)

When a host has detected decoys (stored as PIDs in `host_service_decoy_status`):
1. Get available ports from session
2. Exclude ports associated with detected decoy PIDs
3. Run same exploit selection on reduced port set
4. If all ports excluded, exploit fails (returns None)

**Detection limitations**: DiscoverDeception has only 50% true positive rate per
decoy process. On average, only half of deployed decoys are detected. The 10%
false positive rate occasionally flags legitimate services as decoys, which can
cause the selector to skip valid exploit targets.

---

## 8. PhishingEmail Mechanics

### 8.1 How Green Phishing Creates Red Footholds

PhishingEmail (executed by EnterpriseGreenAgent at 1% probability per step):
1. Green agent "opens a phishing email" on its host
2. The action creates a new `RedAbstractSession` on the green host
3. The red agent assigned to that subnet gets the session
4. The session bypasses ALL BlockTrafficZone rules (uses physical routing, not IP)

**Critical implications**:
- PhishingEmail can create red footholds inside blocked zones
- Even during Phase 1 when contractor->RZ-A is blocked, green agents in RZ-A can
  still "open phishing emails" that give red_agent_1 a session
- This is the primary mechanism for red to enter isolated zones during missions
- Red cannot prevent this; blue cannot prevent this
- Expected phishing events: ~5 per episode (1% * 500 steps * ~1 green per host)

### 8.2 Session Handoff Between Agents

When red_agent_0 exploits a host in restricted_zone_a_subnet, the new session
belongs to red_agent_0 but is on a host in red_agent_1's territory. The
environment's `different_subnet_agent_reassignment()` (SimulationController line 343)
handles this by transferring the session to the appropriate subnet agent.

---

## 9. DiscoverDeception Detailed Analysis

### 9.1 Parameters

| Parameter | Value |
|-----------|-------|
| Duration | 2 steps |
| True Positive Rate | 50% (detects actual EXPLOIT decoys) |
| False Positive Rate | 10% (flags legitimate processes as decoys) |

### 9.2 How It Changes Red Behavior

1. DiscoverDeception iterates over ALL processes on target host
2. For each EXPLOIT-type decoy: 50% chance of correct detection -> adds PID to `host_service_decoy_status`
3. For each non-decoy process: 10% chance of false flag -> incorrectly adds PID
4. Detected PIDs cause state to enter D-variant (S->S stays S, but the decoy info is stored)
5. When ExploitRemoteService runs on a host with detected decoys, PIDSelectiveExploitActionSelector excludes those ports

### 9.3 Empirical Statistics

```
Total DiscoverDeception actions: 2,285
All succeeded (DiscoverDeception always returns success=True,
even if no decoys are found)
```

Per-agent DD usage correlates with time spent in S/SD states:
- red_agent_5: 743 (most, because HQ has many hosts to scan)
- red_agent_3: 395
- red_agent_1: 349
- red_agent_0: 325
- red_agent_2: 282
- red_agent_4: 191 (least, because OZ-B has fewer hosts)

---

## 10. Recovery After Blue Intervention

### 10.1 Session Removal Mechanics

When blue executes Remove or Restore on a host:
1. The red session on that host is destroyed
2. At the next step, `_session_removal_state_change()` detects the missing session
3. The host state transitions from U/UD/R/RD -> **KD**
4. Red must re-scan (AggrSvcDisc or StlthSvcDisc) and re-exploit from scratch

### 10.2 Theoretical Recovery Time

From KD state, the minimum recovery path is:
```
KD -> AggrSvcDisc(1) -> SD -> Exploit(4) -> UD -> PrivEsc(2) -> RD
Total: 7 steps minimum (1 + 4 + 2)
```

Expected recovery time (accounting for probabilistic action selection):
```
KD: P(DRS)=50%, P(AggrSvc)=25%, P(StlthSvc)=25%
  Expected steps to scan: 2 actions * avg(1,1,3) = ~3 steps

SD: P(Exploit)=75%, P(DD)=25%
  Expected steps to exploit: 1.33 actions * 4 steps = ~5.3 steps

UD: P(PrivEsc)=100%
  Expected steps to PrivEsc: 1 action * 2 steps = 2 steps

Total expected recovery: ~10-12 steps
```

### 10.3 Remove vs Restore Effect

**Remove**: Kills user sessions only. Root sessions survive.
- If red has only a user session: state drops to KD. Recovery ~10-12 steps.
- If red has a root session: session survives Remove. State stays R/RD.

**Restore**: Kills ALL sessions (user AND root), wipes all files, restores services.
- State always drops to KD regardless of session type.
- Also wipes deployed decoys (blue must redeploy, costing 3 more actions).
- Recovery ~10-12 steps.

---

## 11. Network Topology and Attack Paths

### 11.1 Inter-Subnet Links

```
contractor_server_0 <---> restricted_zone_a_server_0 <---> ops_zone_a_server_0
contractor_server_0 <---> restricted_zone_b_server_0 <---> ops_zone_b_server_0
contractor_server_0 <---> public_access_zone_server_0 <---> admin_server_0
                                                       <---> office_server_0
```

**All inter-subnet links go through server_host_0**. These are the ONLY chokepoints.

### 11.2 Comms Policy by Phase

| Phase | Steps (500-step ep) | Blocked Routes |
|-------|-------|----------------|
| Phase 0 | 0-166 | Many routes blocked but contractor->RZ-A, contractor->RZ-B open |
| Phase 1 | 167-333 | RZ-A open, RZ-B blocked from contractor |
| Phase 2 | 334-499 | RZ-B open, RZ-A blocked from contractor |

### 11.3 Attack Path to Impact (Required Minimum)

The fastest path to Impact on an OZ host from contractor_network:
```
Step 0:  red_agent_0 starts at contractor host (state U)
Step 1:  PrivEsc on starting host (2 steps)
Step 2:  (in progress)
Step 3:  DRS on contractor subnet (1 step) -> discovers contractor_server_0
Step 4:  AggrSvcDisc on contractor_server_0 (1 step)
Step 5:  ExploitRemoteService on contractor_server_0 (4 steps)
Step 6-8: (in progress)
Step 9:  Session on contractor_server_0 -> DRS discovers RZ hosts
Step 10: AggrSvcDisc on RZ server_host_0 (1 step)
Step 11: Exploit on RZ server_host_0 (4 steps) -> session transfers to red_agent_1
Step 12-14: (in progress)
Step 15: red_agent_1 does DRS on RZ-A -> discovers OZ-A hosts
Step 16: AggrSvcDisc on OZ-A server_host_0 (1 step)
Step 17: Exploit on OZ-A server_host_0 (4 steps) -> session to red_agent_2
Step 18-20: (in progress)
Step 21: red_agent_2 PrivEsc on OZ-A host (2 steps)
Step 22: (in progress)
Step 23: Impact on OZ-A host (2 steps)
Step 24: (in progress) -> Impact succeeds

Minimum theoretical: ~24 steps from start to first OZ Impact
```

**But empirically, first Impact arrives at step 9 (min)**. This is possible because
the red_agent_0 starting host may already be in a subnet with OZ access, or
phishing creates shortcuts. The median of 66 steps aligns with the expected
probabilistic path accounting for action selection randomness.

---

## 12. Data Reference

All raw data is stored in `data/red_trace.db` (SQLite).

### Tables

| Table | Rows (30 ep) | Description |
|-------|-------------|-------------|
| episodes | 30 | Per-episode reward summary |
| red_actions | ~83,000 | Every red action with FSM state, target, success |
| red_sessions | ~7.7M | Session snapshots per step per agent |
| attack_chains | ~4,600 | Aggregated host-level attack sequences |

### Useful Queries

```sql
-- Impact success by phase and zone
SELECT
  CASE WHEN step < 167 THEN 'Phase0'
       WHEN step < 334 THEN 'Phase1'
       ELSE 'Phase2' END as phase,
  target_host,
  COUNT(*) as total,
  SUM(CASE WHEN success='TRUE' THEN 1 ELSE 0 END) as successes
FROM red_actions
WHERE action_type = 'Impact'
GROUP BY phase, target_host
HAVING successes > 0
ORDER BY phase, successes DESC;

-- Time from exploit to PrivEsc per host
SELECT a1.episode_id, a1.target_host,
       a2.step - a1.step as exploit_to_privesc_steps
FROM red_actions a1
JOIN red_actions a2
  ON a1.episode_id = a2.episode_id
  AND a1.target_host = a2.target_host
  AND a1.red_agent = a2.red_agent
WHERE a1.action_type = 'ExploitRemoteService' AND a1.success = 'TRUE'
  AND a2.action_type = 'PrivilegeEscalate' AND a2.success = 'TRUE'
  AND a2.step > a1.step
  AND a2.step - a1.step < 50;

-- FSM state distribution over time
SELECT step / 50 as time_bucket, fsm_state, COUNT(*) as cnt
FROM red_actions
WHERE fsm_state IS NOT NULL AND fsm_state != 'multi'
GROUP BY time_bucket, fsm_state
ORDER BY time_bucket, cnt DESC;
```

---

## 13. Implications for Optimal Blue Strategy

### 13.1 Key Findings for Blue Defense

1. **PrivilegeEscalate always succeeds**: Once red has a user session, it WILL
   get root within 2-4 steps (2 steps if PrivEsc chosen immediately, 4 if DRS
   chosen first). Blue must respond to user-level exploits within 2 steps.

2. **DegradeServices is invisible and always succeeds**: This is the most
   damaging action from a defensive perspective. Once red has root, DegradeServices
   fires with 50% probability (in RD state) and always succeeds. There is NO
   observation signal for this action.

3. **Impact success depends on OT service presence**: Impact only succeeds on OZ
   hosts with active OT services. Defending OZ server_host_0 (the inter-subnet link)
   is critical.

4. **Restore is mandatory for root sessions**: Remove only kills user sessions.
   If red has root (which it always will within 2-4 steps of exploitation), only
   Restore can evict it. The oracle's strategy of Remove-first wastes 3 steps.

5. **Recovery time is ~10-12 steps**: After Restore, red needs ~10-12 steps to
   re-exploit. This gives blue a window to redeploy decoys and respond to other
   threats.

6. **Phase 0 is the worst for blue**: 14.1% Impact success rate in Phase 0 vs
   6.0% in Phase 1 and 1.7% in Phase 2. This is because all zones are accessible
   and blue cannot block traffic.

### 13.2 Why Oracle Scores Worse Than Heuristic

The oracle (with perfect state info) scores -1558 vs heuristic's -1039 because:

1. **Oracle uses Remove-first for user sessions**: Remove takes 3 steps and only
   kills user sessions. By the time Remove completes, red has already escalated
   to root (PrivEsc takes only 2 steps). The root session survives Remove.

2. **Heuristic goes straight to Restore**: When malfile or confirmed signals are
   detected, the heuristic immediately Restores (5 steps), which kills ALL sessions
   including root. This is -1 reward cost but prevents the DegradeServices cascade.

3. **The 90% Remove success rate is misleading**: Remove "succeeds" at killing the
   user session, but the root session (created during the 3-step Remove delay)
   survives. The oracle sees the user session gone and thinks it succeeded.

### 13.3 Optimal Response Timing

Given the empirical data:
- Exploit detection: step N (blue sees proc_flag/conn_flag/malfile)
- PrivEsc timing: step N+1 to N+3 (red escalates within 2-4 steps)
- Blue must Restore by step N+1 to prevent root establishment
- If blue uses Remove (3 steps), root is established by step N+3 when Remove completes

**Therefore**: Immediate Restore on confirmed red signals (conn+malfile, malfile-only)
is optimal. Remove should only be used for ambiguous signals (proc_flag-only with
flag_age=0, which has 33-54% green FP rate).
