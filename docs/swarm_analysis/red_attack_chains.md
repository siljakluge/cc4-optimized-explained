# Red Agent Attack Chain Analysis -- FiniteStateRedAgent (CC4)

This document provides a complete analysis of every attack chain the
FiniteStateRedAgent can execute, with exact timing, probabilities, and
blue-observable signals.  All line references point to the optimized
codebase at the time of writing.

---

## 1. FSM State Diagram

### 1.1 State Definitions

| State | Meaning | Red Knowledge |
|-------|---------|---------------|
| K | Known -- IP discovered, no service info | Host found via DiscoverRemoteSystems |
| KD | Known + Decoy detected | A decoy was discovered on this host |
| S | Services scanned -- port info available | After AggressiveServiceDiscovery or StealthServiceDiscovery |
| SD | Services scanned + Decoy detected | Service scan done AND decoy found |
| U | User shell -- red has user-level session | After successful ExploitRemoteService |
| UD | User shell + Decoy detected | User shell AND decoy known |
| R | Root shell -- red has privileged session | After successful PrivilegeEscalate |
| RD | Root shell + Decoy detected | Root AND decoy known |
| F | Foreign -- host outside agent's subnet | Cannot act on this host |

### 1.2 FSM Transition Diagram (ASCII)

```
                         +---------+
                         |  INIT   |
                         | (step 0)|
                         +----+----+
                              |
                     Starting host = U
                              |
     +------- DiscoverRemoteSystems (pingsweep) ----------+
     |           discovers new IPs -> state K             |
     v                                                    v
+----+----+                                         +-----+-----+
|  K      |---AggrSvcDisc(OK)--->  S                |  KD        |
| (Known) |---StlthSvcDisc(OK)->   S                | (Known +   |
|         |---DiscRemSys(OK)--->  KD (all in subnet)|  Decoy)    |
|         |---AggrSvcDisc(FAIL)->  K                |            |
|         |---StlthSvcDisc(FAIL)-> K                +-----+------+
+---------+                                               |
     Probs: DiscRemSys 50%, AggrSvc 25%, StlthSvc 25%     |
                                                    Probs: DiscRemSys 50%
                                                           AggrSvc 25%
                                                           StlthSvc 25%
                                                           |
+----+----+                                         +------+-----+
|  S      |---Exploit(OK)-------> U                 |  SD        |
| (Svcs)  |---DiscDeception(OK)-> S                 | (Svcs +    |
|         |---DiscRemSys(OK)--->  SD                |  Decoy)    |
|         |---Exploit(FAIL)----->  S                |            |
|         |---DiscDeception(FAIL)> S                +------+-----+
+---------+                                                |
     Probs: Exploit 50%, DiscDeception 25%,          Probs: Exploit 75%
             DiscRemSys 25%                                 DiscDeception 25%
                                                           |
+----+----+                                         +------+-----+
|  U      |---PrivEsc(OK)-------> R                 |  UD        |
| (User)  |---DiscRemSys(OK)--->  UD                | (User +    |
|         |---PrivEsc(FAIL)----->  U                |  Decoy)    |
|         |---Withdraw(OK)------>  S                |            |
+---------+                                         +------+-----+
     Probs: PrivEsc 50%, DiscRemSys 50%              Probs: PrivEsc 100%
             Withdraw 0% (always available                   Withdraw 0%
             but probability is 0.0)
                                                           |
+----+----+                                         +------+-----+
|  R      |---Impact(OK)-------> R                  |  RD        |
| (Root)  |---Degrade(OK)------> R                  | (Root +    |
|         |---DiscRemSys(OK)--->  RD                |  Decoy)    |
|         |---Withdraw(OK)------>  S                |            |
+---------+                                         +------+-----+
     Probs: Impact 25%, Degrade 25%,                 Probs: Impact 50%
             DiscRemSys 50%, Withdraw 0%                    Degrade 50%
                                                            Withdraw 0%
```

### 1.3 Complete Transition Matrices

Source: `FiniteStateRedAgent.py` lines 441-492, 540-552.

**Action Index Mapping** (line 401-412):
```
0 = DiscoverRemoteSystems      4 = ExploitRemoteService     8 = Withdraw
1 = AggressiveServiceDiscovery 5 = PrivilegeEscalate
2 = StealthServiceDiscovery    6 = Impact
3 = DiscoverDeception          7 = DegradeServices
```

**Success Transition Matrix** (lines 441-452):
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

**Key observation about KD state (line 543-544):** The KD probability row was
previously missing, causing an `AttributeError` crash when all hosts entered KD.
The fix gives KD identical probabilities to K: `[0.50, 0.25, 0.25, ...]`.

### 1.4 Special Transitions

**Foreign (F) state** (lines 156-160): When the success transition yields `U`,
the agent checks if the host IP is in its assigned `agent_subnets`. If not, the
state becomes `F` instead.  An `F` host is filtered from `known_hosts` (line
114) and never acted upon.

**Session removal by Blue** (lines 169-188): At each step,
`_session_removal_state_change()` checks all hosts in states `U`, `UD`, `R`,
`RD`. If the host's session no longer appears in the observation, the host state
is **reset to `KD`**. This means blue Remove/Restore drops the red agent back
to the "known but need to re-scan" state.

---

## 2. Attack Chain Timing

### 2.1 Action Durations

| Action | Duration (steps) | Source |
|--------|-----------------|--------|
| DiscoverRemoteSystems | 1 (default) | `Action.py` line 19: `DEFAULT_DURATION = 1` |
| AggressiveServiceDiscovery | 1 (default) | `DiscoverNetworkServices.py` line 170: no override |
| StealthServiceDiscovery | 3 | `DiscoverNetworkServices.py` line 136: `self.duration = 3` |
| DiscoverDeception | 2 | `DiscoverDeception.py` line 39: `self.duration = 2` |
| ExploitRemoteService | 4 | `ExploitRemoteService.py` line 147: `self.duration = 4` |
| PrivilegeEscalate | 2 | `PrivilegeEscalate.py` line 82: `self.duration = 2` |
| Impact | 2 | `Impact.py` line 36: `self.duration = 2` |
| DegradeServices | 2 | `DegradeServices.py` line 36: `self.duration = 2` |
| Withdraw | 1 (default) | No duration override |

**Multi-step action behavior** (lines 110-112 of `FiniteStateRedAgent.py`):
When `success == IN_PROGRESS`, the agent returns `Sleep()` and no new action is
chosen. The duration counter is managed by the environment's action execution
logic. During `IN_PROGRESS` steps, the agent increments `self.step` but does
nothing else -- no state transitions occur, and no host selection happens.

### 2.2 Minimum Attack Chains (Steps to Impact)

#### Fastest Path: Aggressive Scan -> Exploit -> PrivEsc -> Impact

| Step | Action | Duration | Cumulative Steps | FSM State After |
|------|--------|----------|-----------------|-----------------|
| 0 | Initial observation | 0 | 0 | U (starting host) |
| 1 | PrivilegeEscalate | 2 | 1 | (in progress) |
| 2 | (Sleep -- in progress) | - | 2 | R |
| 3 | Impact | 2 | 3 | (in progress) |
| 4 | (Sleep -- in progress) | - | 4 | R (Impact succeeded) |

**Minimum steps to first Impact on starting host: 4 steps** (action at step 1,
completes at step 2, then Impact at step 3, completes at step 4).

The starting host begins at state `U`. Red can immediately PrivilegeEscalate (2
steps), then Impact (2 steps). Total: 4 steps.

#### Fastest Path to Impact on a DIFFERENT Host

| Step | Action | Duration | Cumulative | FSM State |
|------|--------|----------|------------|-----------|
| 0 | Initial obs | - | 0 | Starting host = U |
| 1 | DiscoverRemoteSystems | 1 | 1 | New hosts = K |
| 2 | AggressiveServiceDiscovery | 1 | 2 | Target = S |
| 3 | ExploitRemoteService | 4 | 3 | (in progress) |
| 4-6 | (Sleep x3) | - | 4-6 | Target = U |
| 7 | PrivilegeEscalate | 2 | 7 | (in progress) |
| 8 | (Sleep) | - | 8 | Target = R |
| 9 | Impact | 2 | 9 | (in progress) |
| 10 | (Sleep) | - | 10 | Target = R |

**Minimum steps to Impact on a new host: 10 steps** (assuming all actions
succeed on first try and aggressive scan is chosen).

#### With Stealth Scan Instead

Replace AggressiveServiceDiscovery (1 step) with StealthServiceDiscovery (3
steps): adds 2 extra steps. **Minimum: 12 steps.**

### 2.3 Expected Steps (Probabilistic)

From state K, the expected actions to reach Impact:

1. **K -> S**: 50% chance DiscoverRemoteSystems (wastes a step, stays K or goes
   KD), 25% AggressiveSvcDisc, 25% StealthSvcDisc.
   - Expected attempts to reach S: ~2 steps (50% chance of non-scan action per
     step, geometric distribution)
   - With stealth scan factored in: expected ~2.75 steps

2. **S -> U**: 50% chance of ExploitRemoteService, 25% DiscoverDeception, 25%
   DiscoverRemoteSystems.
   - Expected attempts to pick Exploit: ~2 (geometric with p=0.5)
   - Each exploit takes 4 steps. Expected ~8 steps at this stage.
   - Exploit can fail due to decoys, blocking, no valid port. On failure stays
     at S.

3. **U -> R**: 50% PrivEsc, 50% DiscoverRemoteSystems.
   - Expected attempts to pick PrivEsc: ~2
   - Each PrivEsc takes 2 steps. Expected ~4 steps.

4. **R -> Impact**: 25% Impact, 25% DegradeServices, 50% DiscoverRemoteSystems.
   - Expected attempts to pick Impact: ~4
   - Each Impact takes 2 steps. Expected ~8 steps at this stage.

**Rough expected steps from discovery to Impact on a single host: ~23 steps**
(without failures or blue interference).

---

## 3. Exploit Success Conditions

### 3.1 ExploitRemoteService (ExploitRemoteService.py, ExploitAction.py)

The exploit chain:
1. **Session check**: Agent must have an active `RedAbstractSession` (line 164-172)
2. **Port knowledge**: Target IP must be in `session.ports` (line 175) -- set by prior service discovery
3. **Blocking check**: `blocking_host()` checks if there is a comms policy block between source and target subnet (line 180)
4. **Exploit selection**: `ExploitActionSelector.get_exploit_action()` picks an exploit based on known ports

### 3.2 Available Exploits and Their Conditions

| Exploit | Port | Weight | Process Type | Success Check | Source |
|---------|------|--------|-------------|---------------|--------|
| HTTPRFI | 80 | 3.0 | http | `"rfi" in vuln_proc.properties` | HTTPRFI.py:17-19 |
| HTTPSRFI | 443 | 4.0 | webserver | `"rfi" in vuln_proc.properties` | HTTPSRFI.py:17-19 |
| SSHBruteForce | 22 | 0.1 | SSH | User must be `bruteforceable` | SSHBruteForce.py:86-91 |
| SQLInjection | 3390 | 5.0 | mysql | Always True (+ needs port 80 or 443) | SQLInjection.py:16-24 |
| HarakaRCE | 25 | 6.0 | smtp | `vuln_proc.version < HARAKA_2_8_9` | HarakaRCE.py:18-20 |
| FTPDirectoryTraversal | 21 | 7.0 | femitter | Always True | FTPDirectoryTraversal.py:14-17 |
| EternalBlue | 139 | - | - | Never used in CC4 | ExploitRemoteService.py:47 |
| BlueKeep | 3389 | - | - | Never used in CC4 | ExploitRemoteService.py:47 |

**Selection algorithm** (ExploitRemoteService.py lines 46-69):
- Filter exploits by port availability (`get_weight` returns 0 if port not in known ports)
- `odds_of_top_choice = 0` (line 58) -- the top-weighted exploit is **never** chosen preferentially
- Instead, a **random** non-top exploit is selected from the remaining weighted options
- This means the exploit chosen is effectively random among available options (excluding highest-weight)

**Detection rate**: All exploits have `self.detection_rate = 0.95` (ExploitAction.py line 33). This means 95% chance of creating a `process_creation` event on the target host (line 102-103 of ExploitAction.py).

**Decoy interaction** (ExploitAction.py lines 90-99): If the vulnerable process has `DecoyType.EXPLOIT`:
- A `network_connections` event IS created on the target host (observable by blue)
- The exploit returns `success=False` (no session created)
- The FSM stays in state S/SD

**Sandbox interaction** (ExploitAction.py lines 249-258): If the process has `DecoyType.SANDBOXING_EXPLOIT`:
- The session IS created, but `is_escalate_sandbox = True`
- When red later tries PrivilegeEscalate, the sandbox kills the session (PrivilegeEscalate.py lines 110-112)

### 3.3 DiscoverDeception (DiscoverDeception.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| Duration | 2 steps | line 39 |
| True Positive Rate | 50% | line 40: `self.detection_rate = 0.5` |
| False Positive Rate | 10% | line 41: `self.fp_rate = 0.1` |

**How it works** (lines 84-106):
- Iterates over ALL processes on the target host
- For each EXPLOIT-type decoy process: 50% chance of correctly detecting it
- For each NON-decoy process: 10% chance of falsely flagging it as a decoy
- Detected decoys report `pid` and `properties=['decoy']`
- The FSM processes these (lines 243-250 of FiniteStateRedAgent.py) and stores PIDs in `host_service_decoy_status`

### 3.4 PIDSelectiveExploitActionSelector (ExploitRemoteService.py lines 72-115)

When ExploitRemoteService is selected for a host that has detected decoys:
- The selector excludes ports associated with known-decoy PIDs
- This filters the port list before exploit selection
- If all ports are excluded, returns None and the exploit fails
- **Critical gap**: Only PIDs detected by DiscoverDeception are excluded. 50% TP rate means half of decoys are missed.

---

## 4. Network Topology and Attack Paths

### 4.1 Subnet Layout

```
                              +------------+
                              |  INTERNET  |
                              | (inactive) |
                              +------+-----+
                                     |
            +------------+-----------+-----------+------------+
            |            |                       |            |
       +----+----+  +----+----+            +-----+-----+ +---+---+
       | RESTRIC | |CONTRACTOR|            | PUBLIC    | |RESTRIC|
       | ZONE A  | | NETWORK  |            | ACCESS    | |ZONE B |
       |         | |(Red Start)|           | ZONE      | |       |
       +----+----+ +----+-----+            +-----+-----+ +---+---+
            |            |                       |            |
       +----+----+       |                  +----+----+  +----+----+
       |  OPS    |       |                  | ADMIN   |  |  OPS    |
       | ZONE A  |       |                  | NETWORK |  | ZONE B  |
       |(OT Svc) |       |                  +---------+  |(OT Svc) |
       +---------+       |                               +---------+
                         |                  +---------+
                         |                  | OFFICE  |
                         |                  | NETWORK |
                         |                  +---------+
```

### 4.2 Between-Subnet Routing (Server-to-Server Links)

Source: `EnterpriseScenarioGenerator.py` lines 150-180 (`_BETWEEN_SUBNET_LINKS`).

The only hosts that link subnets are `server_host_0` in each subnet:

```
contractor_server_0 <---> restricted_zone_a_server_0 <---> ops_zone_a_server_0
contractor_server_0 <---> restricted_zone_b_server_0 <---> ops_zone_b_server_0
contractor_server_0 <---> public_access_zone_server_0 <---> admin_server_0
                                                       <---> office_server_0
```

**These server_host_0 nodes are the chokepoints.** Red MUST compromise
`server_host_0` in each subnet to pivot to the next.

### 4.3 Red Agent Assignment and Allowed Subnets

Source: `EnterpriseScenarioGenerator.py` lines 790-796.

| Agent | Allowed Subnets | Starting Condition |
|-------|----------------|-------------------|
| red_agent_0 | contractor_network | **Active** (has RedAbstractSession) |
| red_agent_1 | restricted_zone_a | Inactive (SleepAgent until activated) |
| red_agent_2 | operational_zone_a | Inactive |
| red_agent_3 | restricted_zone_b | Inactive |
| red_agent_4 | operational_zone_b | Inactive |
| red_agent_5 | public_access_zone, admin_network, office_network | Inactive |

Only `red_agent_0` starts active with a session (lines 811-822). The other
agents start as `SleepAgent` instances unless the `red_agent_class` parameter
overrides them.

When `red_agent_class` is `FiniteStateRedAgent`, ALL red agents get FSM behavior
(lines 833-835), receiving their `agent_subnets` as CIDRs. But only agent_0
starts active with an actual session.

**Implication**: In practice with the default FSM, only red_agent_0 operates
on contractor_network. The other agents need to be activated by the environment
(e.g., via `RedSessionCheck`). Hosts discovered outside an agent's subnets
are marked `F` (foreign) and ignored.

### 4.4 Mission Phases and Comms Policy

Source: `EnterpriseScenarioGenerator.py` lines 183-211, 878-884.

For a 100-step episode: phases are (34, 33, 33) steps.

| Phase | Steps | Key Policy (Blocked Comms) |
|-------|-------|---------------------------|
| 0: Pre-planning | 0-33 | Many routes blocked. RestrictedA<->Contractor open. RestrictedB<->Contractor open. OpsA<->RestrictedA open. OpsB<->RestrictedB open. |
| 1: Mission A | 34-66 | RestrictedA<->Contractor OPEN (unblocked). OpsA<->RestrictedA OPEN. RestrictedB<->Contractor BLOCKED. OpsB<->RestrictedB BLOCKED. |
| 2: Mission B | 67-99 | RestrictedB<->Contractor OPEN. OpsB<->RestrictedB OPEN. RestrictedA<->Contractor BLOCKED. OpsA<->RestrictedA BLOCKED. |

The comms policy tuples define which subnet pairs CAN communicate. Blue's
`BlockTrafficZone` / `AllowTrafficZone` actions can override these.

**Critical for attack timing**: Red in contractor_network can only reach
restricted_zone_a during phases 0 and 1, and restricted_zone_b during phases 0
and 2. This constrains the lateral movement windows.

---

## 5. Observable Signals for Blue

### 5.1 Monitor Action (Monitor.py)

Monitor runs automatically at the end of each step. It collects:
- `host.events.network_connections` -- any network activity on blue-watched hosts
- `host.events.process_creation` -- any new process creation events

Events are moved to `old_network_connections` / `old_process_creation` and then
cleared. Blue's observation wrapper reads the `old_*` lists.

### 5.2 Per-Action Observability

| Red Action | Creates Network Connection Events? | Creates Process Creation Events? | Detection Rate | Notes |
|------------|-----------------------------------|--------------------------------|----------------|-------|
| DiscoverRemoteSystems (Pingsweep) | No events on target | No | 0% | Pingsweep only checks route existence; no events generated on targets (Pingsweep.py line 57 only checks `respond_to_ping`) |
| AggressiveServiceDiscovery (Portscan) | YES -- on each open port | No | 75% (fixed random) | Portscan.py line 58: one random draw, if `<= detection_rate`, ALL ports get connection events. Decoy ports (EXPLOIT type) ALWAYS create events regardless. |
| StealthServiceDiscovery (Portscan) | YES -- on each open port | No | 25% | Same as Aggressive but `detection_rate = 0.25` (line 136). Lower chance of connection events. |
| DiscoverDeception | No | No | 0% | Only reads process list, creates no events on target (DiscoverDeception.py line 84-106) |
| ExploitRemoteService | YES -- on exploit target | YES -- process creation (95%) | 95% | ExploitAction.py line 102: `process_creation` event with PID and connection info. Network connections also created on route hosts. SSHBruteForce creates 10 connection events (wordlist_length, line 95-101). |
| PrivilegeEscalate | No | No | 0% | No events generated (EscalateAction.py). Creates `escalate.sh`/`escalate.exe` file (density=0.9, unsigned) but no host events. |
| Impact | No | No | 0% | StopService only modifies service state; no network/process events. **Impact is invisible to Monitor.** |
| DegradeServices | No | No | 0% | Same -- modifies service reliability only. **Invisible to Monitor.** |
| Withdraw | No | No | 0% | StopProcess removes sessions; no events. |

### 5.3 What Blue Can See vs What Is Invisible

**Visible (via Monitor events):**
- Port scanning (connection events on target ports)
- Exploit attempts (process creation + connections, 95% rate)
- SSH brute force (10 connection events per attempt)
- Decoy interactions (EXPLOIT decoys always generate events)

**Invisible to Monitor:**
- Pingsweep / DiscoverRemoteSystems
- DiscoverDeception probing
- Privilege escalation
- Impact (OT service shutdown)
- DegradeServices

**Visible via Analyse (file inspection):**
- `cmd.sh` / `cmd.exe` files (density=0.9, unsigned) dropped by ExploitRemoteService
- `escalate.sh` / `escalate.exe` files (density=0.9, unsigned) dropped by PrivilegeEscalate

### 5.4 Decoy Detection Signals

When red exploits a decoy:
- EXPLOIT-type decoy: connection event IS created (ExploitAction.py line 92-98). Blue can see this.
- SANDBOXING decoy: session is created (looks like success to red) but when PrivEsc is attempted, session is killed. Blue sees the initial exploit event.

When red runs DiscoverDeception:
- No events are created on the target host. Blue cannot detect that red is probing for decoys.
- Red gets PID info with `properties=['decoy']` flag at 50% TP rate.

---

## 6. Red Action Durations (Summary Table)

| Action | Steps | During In-Progress | Can Blue Act? |
|--------|-------|-------------------|---------------|
| DiscoverRemoteSystems | 1 | N/A (instant) | No in-progress window |
| AggressiveServiceDiscovery | 1 | N/A (instant) | No in-progress window |
| StealthServiceDiscovery | 3 | Steps 2-3: Sleep | Yes -- blue has 2 steps to respond |
| DiscoverDeception | 2 | Step 2: Sleep | Yes -- blue has 1 step |
| ExploitRemoteService | 4 | Steps 2-4: Sleep | Yes -- blue has 3 steps to respond |
| PrivilegeEscalate | 2 | Step 2: Sleep | Yes -- blue has 1 step |
| Impact | 2 | Step 2: Sleep | Yes -- but Impact creates no alerts |
| DegradeServices | 2 | Step 2: Sleep | Yes -- but Degrade creates no alerts |
| Withdraw | 1 | N/A (instant) | No in-progress window |

**Key insight**: ExploitRemoteService has the longest multi-step window (4
steps). Blue can potentially see the exploit event on step 1 and respond with
Remove/Restore before the exploit resolves. However, the action has already been
submitted to the environment -- the multi-step mechanism is about the agent
sleeping, not about the action being interruptible mid-execution. The action
executes atomically when first submitted; the duration represents how long
before the agent can act again.

---

## 7. Lateral Movement

### 7.1 Subnet-to-Subnet Movement Chain

Red must follow the server_host_0 chain to move between subnets:

```
contractor_network --> restricted_zone_a --> operational_zone_a
                   \-> restricted_zone_b --> operational_zone_b
                   \-> public_access_zone --> admin_network
                                          \-> office_network
```

Each hop requires:
1. DiscoverRemoteSystems on the target subnet (1 step)
2. Service discovery on server_host_0 (1-3 steps)
3. ExploitRemoteService on server_host_0 (4 steps)
4. PrivilegeEscalate on server_host_0 (2 steps) -- needed for `ExploreHost` which reveals `info` (between-subnet links)

### 7.2 How Between-Subnet Discovery Works

When PrivilegeEscalate succeeds, `ExploreHost` is called (PrivilegeEscalate.py
line 163-165). This returns `target_host.info`, which for server_host_0 nodes
contains the `_between_subnet_links` data (lines 473-493 of
EnterpriseScenarioGenerator.py). This reveals IP addresses of server_host_0
nodes in adjacent subnets.

### 7.3 Route Computation

The `get_route()` method in RemoteAction traces the path through
`interface.data_links` from source to target host. All hosts within a subnet
link to `{subnet_name}_router`. Routers connect to adjacent subnet routers
as defined by `_generate_data_links()` (lines 440-471).

Traffic is checked against the comms policy (mission-phase-dependent blocks).
If any link in the route is blocked, the action fails.

---

## 8. Impact Action

### 8.1 What Impact Does (Impact.py)

1. Finds a session on the target host (line 60-61)
2. Checks that the session has privileged access (root/SYSTEM) (line 65)
3. Checks for an active OTService process (line 70)
4. Calls `StopService(ProcessName.OTSERVICE)` (lines 80-82)
5. Calls `host.increment_impact_count()` (line 90)

**OTService only exists on operational zone hosts** (EnterpriseScenarioGenerator.py
line 573: `if "operational" in hostname`).

### 8.2 Reward Penalty

Source: `BlueRewardMachine.py` lines 35-65.

The `RIA` (Red Impact/Access) penalty is phase-dependent:

| Subnet | Phase 0 | Phase 1 (Mission A) | Phase 2 (Mission B) |
|--------|---------|---------------------|---------------------|
| operational_zone_a | -1 | **-10** | -1 |
| operational_zone_b | -1 | -1 | **-10** |
| contractor_network | -5 | 0 | 0 |
| restricted_zone_a | -1 | -3 | -3 |
| restricted_zone_b | -1 | -1 | -3 |
| public_access_zone | -3 | -3 | -3 |
| admin_network | -3 | -3 | -3 |
| office_network | -3 | -3 | -3 |

**Impact is scored per successful Impact action per step** (line 118-119).
Impact can be repeated each step on the same host as long as OTService is active.

### 8.3 Impact vs Restore Interaction

`host.restore()` resets services to original state, re-enabling OTService.
However, `impact_count` is **never reset** by Restore (Host.py line 351-353).
Restore prevents future Impacts (by evicting red sessions) but does not undo
scored penalties.

Impact is **invisible to Monitor** -- no events are generated. Blue can only
infer Impact from green agent failures (GreenAccessService/GreenLocalWork
failing on hosts with stopped services).

### 8.4 DegradeServices vs Impact

DegradeServices (DegradeServices.py) calls `service.degrade_service_reliability()`
on ALL active services. This reduces service reliability, making green actions
more likely to fail. Unlike Impact, it does not stop the service entirely. From
the R/RD state, Impact and Degrade are chosen with equal probability (25%/25%
from R; 50%/50% from RD).

---

## 9. Red Recovery After Blue Response

### 9.1 Blue Remove (Remove.py)

- **Duration**: 3 steps
- **Effect**: Kills suspicious processes (tracked PIDs), removes malware files (density >= 0.9, unsigned)
- **Does NOT remove**: Root/SYSTEM sessions (line 13: "Will not remove privileged sessions"). If red has already PrivEsc'd, the root session **survives** Remove.
- **Red FSM impact**: If the user session is gone, `_session_removal_state_change` detects it and sets state to `KD`. But if root session exists, session is still visible and state stays U/R.

### 9.2 Blue Restore (Restore.py)

- **Duration**: 5 steps
- **Effect**: Full host reimage via `RestoreFromBackup`. Removes ALL red sessions (user AND root), restores original processes, files, services.
- **Cost**: -1 penalty per Restore action (line 70)
- **Red FSM impact**: Session disappears, `_session_removal_state_change` sets state to `KD`.

### 9.3 Recovery Speed After Blue Intervention

After blue Remove (if it actually evicted the session):
```
KD -> AggrSvcDisc(1 step) -> SD -> Exploit(4 steps) -> UD -> PrivEsc(2 steps) -> RD
Minimum recovery: 7 steps
```

After blue Restore:
```
KD -> AggrSvcDisc(1 step) -> SD -> Exploit(4 steps) -> UD -> PrivEsc(2 steps) -> RD
Minimum recovery: 7 steps (same as Remove since both drop to KD)
```

But since red is in KD, it chooses DiscoverRemoteSystems 50% of the time,
wasting steps. Expected recovery from KD:

- KD -> S/SD: ~2 attempts (50% DRS, 25% Aggr, 25% Stealth). Expected ~2.5 steps.
- S/SD -> U/UD: Exploit at 50%/75%. Expected ~1.3-2 attempts, each 4 steps. ~6-8 steps.
- U/UD -> R/RD: PrivEsc at 50%/100%. Expected ~1-2 attempts, each 2 steps. ~2-4 steps.

**Expected full recovery: ~11-15 steps from KD to R/RD.**

### 9.4 Critical: Remove Does Not Stop Root Sessions

This is the most important tactical fact. If red has reached state R (root) and
blue uses Remove instead of Restore, the root session survives. Red can
immediately continue with Impact. **Blue MUST use Restore to evict privileged
red sessions.**

---

## 10. Host Selection and Priority

### 10.1 Default Priority

In the base `FiniteStateRedAgent`, `set_host_state_priority_list()` returns
`None` (line 513). This means no state prioritization -- when choosing a host,
the agent picks uniformly at random from all known non-F hosts.

### 10.2 Server Prioritization

`self.prioritise_servers = False` by default (line 50). If enabled, servers
get 75% selection probability when available (lines 275-291).

### 10.3 Host Selection Flow

1. Get all hosts not in state F (line 114)
2. If priority list exists, choose a state weighted by priority, then pick a random host in that state
3. If `prioritise_servers`, 75% chance of picking a server host
4. Otherwise, uniform random among hosts in chosen state

---

## 11. Summary: Key Tactical Insights for Blue Defense

### 11.1 Detection Windows

| Red Phase | Blue Detection Opportunity | Response Window |
|-----------|---------------------------|-----------------|
| Discovery (DRS) | NONE -- pingsweep is invisible | Cannot detect |
| Service Scan (Aggressive) | Network connections on ALL ports (75% chance) | Immediate -- scan is 1 step |
| Service Scan (Stealth) | Network connections (25% chance) | 2 extra Sleep steps |
| Exploit | Process creation (95%) + connections | 3 Sleep steps after initiation |
| PrivEsc | NONE -- no events generated | Cannot detect directly |
| Impact | NONE -- no events generated | Cannot detect directly |

### 11.2 Optimal Blue Response Strategy

1. **Deploy decoys on server_host_0 nodes** -- these are the chokepoints for lateral movement
2. **Use Restore, not Remove, against root-level compromise** -- Remove cannot evict root sessions
3. **Monitor for exploit events (process_creation)** -- this is the primary detection signal
4. **Act during exploit in-progress windows** -- 3 steps of Sleep give blue time to respond
5. **Block traffic during mission-critical phases** -- restrict lateral movement paths
6. **DiscoverDeception is invisible to blue** -- blue cannot know red is probing for decoys

### 11.3 Red Vulnerabilities

- 50% of actions from K/KD/U/R are DiscoverRemoteSystems (wasted if subnet is fully mapped)
- ExploitRemoteService takes 4 steps -- longest vulnerability window
- Decoys with EXPLOIT type both fail the exploit AND generate events for blue
- PrivEsc on sandboxed sessions kills the session (both exploit decoy event visible + time wasted)
- Withdraw probability is 0.0 from all states -- red never voluntarily retreats
- Recovery from KD takes ~11-15 steps expected -- blue Restore buys significant time
