# Scientific Analysis of Red and Green Agents in CAGE Challenge 4

## Table of Contents

1. [Red Agent: FiniteStateRedAgent](#1-red-agent-finitestateredagent)
   - 1.1 [FSM Architecture](#11-fsm-architecture)
   - 1.2 [State Definitions](#12-state-definitions)
   - 1.3 [Complete FSM Diagram](#13-complete-fsm-diagram)
   - 1.4 [Probability Matrices](#14-probability-matrices)
   - 1.5 [State Transition Matrices](#15-state-transition-matrices)
   - 1.6 [Action Durations](#16-action-durations)
   - 1.7 [Attack Chains and Timing Analysis](#17-attack-chains-and-timing-analysis)
   - 1.8 [Empirical Performance Data](#18-empirical-performance-data)
   - 1.9 [Blue Observable Signals](#19-blue-observable-signals)
   - 1.10 [Session Removal Mechanics](#110-session-removal-mechanics)
   - 1.11 [Host Selection Algorithm](#111-host-selection-algorithm)
   - 1.12 [Six Independent Red Agents](#112-six-independent-red-agents)
2. [Green Agent: EnterpriseGreenAgent](#2-green-agent-enterprisegreenagent)
   - 2.1 [Action Selection](#21-action-selection)
   - 2.2 [GreenLocalWork Mechanics](#22-greenlocalwork-mechanics)
   - 2.3 [GreenAccessService Mechanics](#23-greenaccessservice-mechanics)
   - 2.4 [False Positive Analysis](#24-false-positive-analysis)
   - 2.5 [Phishing Mechanics](#25-phishing-mechanics)
   - 2.6 [GreenLocalWork Failure Rates](#26-greenlocalwork-failure-rates)
3. [Blue Agent Decision Tree](#3-blue-agent-decision-tree)

---

## 1. Red Agent: FiniteStateRedAgent

### 1.1 FSM Architecture

The `FiniteStateRedAgent` (source: `CybORG/Agents/SimpleAgents/FiniteStateRedAgent.py`) is
a finite-state automaton that maintains a per-host state dictionary tracking its level of
knowledge and access. On each simulation step the agent:

1. Processes the previous action's success/failure and transitions the acted-upon host's
   state via `_host_state_transition()`.
2. Ingests new observations for newly discovered hosts and decoy detections via
   `_process_new_observations()`.
3. Checks for session removals by blue and transitions affected hosts to `KD` via
   `_session_removal_state_change()`.
4. If the previous action is still `IN_PROGRESS` (multi-step actions), returns `Sleep()`.
5. Otherwise, selects a host and action via `_choose_host_and_action()` using the
   probability matrix for the chosen host's current state.

### 1.2 State Definitions

The FSM has 9 states. Each host tracked by a red agent instance occupies exactly one state
at any given time.

| State | Name             | Meaning                                                        |
|-------|------------------|----------------------------------------------------------------|
| `K`   | Known            | IP address discovered; no services scanned, no decoys found    |
| `KD`  | Known+Decoy      | IP known and at least one decoy detected on host               |
| `S`   | Services         | Services have been scanned (ports enumerated)                  |
| `SD`  | Services+Decoy   | Services scanned and decoys detected                           |
| `U`   | User Shell       | Red has a user-level session on the host                       |
| `UD`  | User+Decoy       | User-level session with decoy awareness                        |
| `R`   | Root             | Red has root/admin privileges on the host                      |
| `RD`  | Root+Decoy       | Root privileges with decoy awareness                           |
| `F`   | Foreign          | Host is outside the agent's assigned subnets (unreachable)     |

The starting host begins in state `U` (red starts with a pre-existing user-level session).
All newly discovered hosts enter state `K`.

### 1.3 Complete FSM Diagram

```
                           COMPLETE FSM STATE DIAGRAM
                    FiniteStateRedAgent (CAGE Challenge 4)

  Action Key:
    DRS  = DiscoverRemoteSystems        ERS  = ExploitRemoteService
    ASD  = AggressiveServiceDiscovery   PE   = PrivilegeEscalate
    SSD  = StealthServiceDiscovery      Imp  = Impact
    DD   = DiscoverDeception            Deg  = DegradeServices
    With = Withdraw

  Notation: action/ok = success transition, action/fail = failure transition
            [p=X.XX] = probability of choosing this action from this state
            (Nd) = action duration in steps

  ============================================================================

                           +---+
                           | F |  (Foreign -- unreachable)
                           +---+
                             ^
                             | ERS/ok when target IP outside agent_subnets
                             |
  +---------------------------------------------------------------------------+
  |                    KNOWLEDGE LAYER (no session)                            |
  |                                                                           |
  |   +-----+    ASD/ok (1d)     +-----+                                      |
  |   |     |------------------>|     |                                       |
  |   |  K  |    SSD/ok (3d)     |  S  |                                      |
  |   |     |------------------>|     |                                       |
  |   +-----+                   +-----+                                       |
  |     | ^                       | ^                                         |
  |     | |                       | |                                         |
  |     | | ASD/fail              | | ERS/fail                                |
  |     | | SSD/fail              | | DD/ok or DD/fail                        |
  |     | +---+                   | +---+                                     |
  |     |                         |                                           |
  |     | DRS/ok on               | ERS/ok (4d)                              |
  |     | any host in subnet      | (gains user session)                     |
  |     | adds 'D' suffix         v                                           |
  |     |                                                                     |
  |   +-----+   ASD/ok (1d)     +-----+                                      |
  |   |     |------------------>|     |                                       |
  |   | KD  |   SSD/ok (3d)     | SD  |                                      |
  |   |     |------------------>|     |                                       |
  |   +-----+                   +-----+                                       |
  |     ^ ^                       |                                           |
  |     | |                       | ERS/ok (4d)                               |
  |     | | ASD/fail              | (gains user session + decoy)              |
  |     | | SSD/fail              |                                           |
  |     | +---+                   |                                           |
  |     |                         |                                           |
  |     | Session removed by      |                                           |
  |     | Blue (Remove/Restore)   |                                           |
  |     | from U/UD/R/RD          |                                           |
  +---------------------------------------------------------------------------+
            |                     |
            |                     v
  +---------------------------------------------------------------------------+
  |                     ACCESS LAYER (has session)                             |
  |                                                                           |
  |                  +-----+   PE/ok (2d)    +-----+                          |
  |        ERS/ok    |     |--------------->|     |    Imp/ok (2d)            |
  |       --------->|  U  |                 |  R  |-----> (stays R)           |
  |                  |     |                 |     |    Deg/ok (2d)            |
  |                  +-----+                 +-----+-----> (stays R)          |
  |                    | |                     | |                             |
  |          PE/fail   | | With/ok             | | With/ok                    |
  |          (stays U) | | (goes to S)         | | (goes to S)               |
  |                    | |                     | |                             |
  |                    | |    DRS/ok           | |    DRS/ok                  |
  |                    | |  adds 'D' suffix    | |  adds 'D' suffix          |
  |                    | |  to ALL hosts       | |  to ALL hosts             |
  |                    | |  in scanned subnet  | |  in scanned subnet        |
  |                    v v                     v v                             |
  |                                                                           |
  |                  +-----+   PE/ok (2d)    +-----+                          |
  |        ERS/ok    |     |--------------->|     |    Imp/ok (2d)            |
  |       --------->| UD  |                 | RD  |-----> (stays RD)          |
  |                  |     |                 |     |    Deg/ok (2d)            |
  |                  +-----+                 +-----+-----> (stays RD)         |
  |                    |                       |                              |
  |          PE/fail   |  With/ok              | With/ok                      |
  |          (stays UD)|  (goes to SD)         | (goes to SD)                |
  |                    |                       |                              |
  +---------------------------------------------------------------------------+

  DRS BROADCAST EFFECT:
  =====================
  When DRS is executed from ANY state, ALL hosts in the target subnet
  that are in the agent's host_states get the 'D' suffix appended:
    K  --> KD       S  --> SD       U  --> UD       R  --> RD
    KD --> KD       SD --> SD       UD --> UD       RD --> RD
```

### 1.4 Probability Matrices

The probability matrix determines which action the agent selects given a host's current
state. Columns correspond to the action list indices:

```
Index:  0=DRS   1=ASD   2=SSD   3=DD    4=ERS   5=PE    6=Imp   7=Deg   8=With
```

**Action Selection Probability Matrix** (extracted from `state_transitions_probability()`):

```
         DRS    ASD    SSD     DD    ERS     PE    Imp    Deg   With
  K  : [ 0.50,  0.25,  0.25,    _,    _,     _,    _,     _,    _  ]
  KD : [ 0.50,  0.25,  0.25,    _,    _,     _,    _,     _,    _  ]
  S  : [ 0.25,    _,     _,   0.25,  0.50,   _,    _,     _,    _  ]
  SD : [   _,     _,     _,   0.25,  0.75,   _,    _,     _,    _  ]
  U  : [ 0.50,    _,     _,     _,    _,   0.50,   _,     _,   0.0 ]
  UD : [   _,     _,     _,     _,    _,   1.00,   _,     _,   0.0 ]
  R  : [ 0.50,    _,     _,     _,    _,     _,  0.25,  0.25,  0.0 ]
  RD : [   _,     _,     _,     _,    _,     _,  0.50,  0.50,  0.0 ]
```

`_` = `None` (action not available from this state). `0.0` = listed but never selected.

**Key insight -- decoy-aware states make red MORE focused and MORE dangerous:**

| Transition | Without Decoy Awareness   | With Decoy Awareness        | Effect                  |
|------------|---------------------------|-----------------------------|-------------------------|
| U -> ?     | 50% PE, 50% DRS          | UD: 100% PE                 | Eliminates exploration  |
| R -> ?     | 25% Imp, 25% Deg, 50% DRS| RD: 50% Imp, 50% Deg       | Doubles damage rate     |
| S -> ?     | 50% ERS, 25% DD, 25% DRS | SD: 75% ERS, 25% DD        | 50% more exploit attempts|

This is a critical design feature: **deploying decoys that red detects (via DiscoverDeception)
actually accelerates the attack** by eliminating the exploratory DRS action from the
probability distribution. A decoy-aware red agent wastes no time on reconnaissance and
focuses entirely on exploitation and privilege escalation.

### 1.5 State Transition Matrices

**Success Transitions** (host moves to this state when the action succeeds):

```
              DRS    ASD    SSD     DD    ERS     PE    Imp    Deg   With
  K  :      [ KD,    S,     S,      _,    _,      _,    _,     _,    _  ]
  KD :      [ KD,    SD,    SD,     _,    _,      _,    _,     _,    _  ]
  S  :      [ SD,    _,     _,      S,    U*,     _,    _,     _,    _  ]
  SD :      [ SD,    _,     _,      SD,   UD*,    _,    _,     _,    _  ]
  U  :      [ UD,    _,     _,      _,    _,      R,    _,     _,    S  ]
  UD :      [ UD,    _,     _,      _,    _,      RD,   _,     _,    SD ]
  R  :      [ RD,    _,     _,      _,    _,      _,    R,     R,    S  ]
  RD :      [ RD,    _,     _,      _,    _,      _,    RD,    RD,   SD ]
  F  :      [ F,     _,     _,      _,    _,      _,    _,     _,    _  ]
```

`*` = ERS success transitions to `U`, but if the target IP is outside `agent_subnets`, it
becomes `F` instead (foreign host, unreachable for further actions).

**Failure Transitions** (host stays in its current state):

```
              DRS    ASD    SSD     DD    ERS     PE    Imp    Deg   With
  K  :      [ K,     K,     K,      _,    _,      _,    _,     _,    _  ]
  KD :      [ KD,    KD,    KD,     _,    _,      _,    _,     _,    _  ]
  S  :      [ S,     _,     _,      S,    S,      _,    _,     _,    _  ]
  SD :      [ SD,    _,     _,      SD,   SD,     _,    _,     _,    _  ]
  U  :      [ U,     _,     _,      _,    _,      U,    _,     _,    U  ]
  UD :      [ UD,    _,     _,      _,    _,      UD,   _,     _,    UD ]
  R  :      [ R,     _,     _,      _,    _,      _,    R,     R,    R  ]
  RD :      [ RD,    _,     _,      _,    _,      _,    RD,    RD,   RD ]
  F  :      [ F,     _,     _,      _,    _,      _,    _,     _,    _  ]
```

All failure transitions preserve the current state. The agent does not lose progress on
failure -- it merely fails to advance.

### 1.6 Action Durations

Action duration is the number of simulation steps an action takes to complete. During
multi-step actions, the agent returns `Sleep()` on each intermediate step (`IN_PROGRESS`).

| Action                     | Duration (steps) | Source Location                              |
|----------------------------|:----------------:|----------------------------------------------|
| DiscoverRemoteSystems      | 1                | Default (no `self.duration` set)             |
| AggressiveServiceDiscovery | 1                | Default (inherits base, no override)         |
| StealthServiceDiscovery    | 3                | `DiscoverNetworkServices.py:136`             |
| DiscoverDeception          | 2                | `DiscoverDeception.py:39`                    |
| ExploitRemoteService       | 4                | `ExploitRemoteService.py:147`                |
| PrivilegeEscalate          | 2                | `PrivilegeEscalate.py:82`                    |
| Impact                     | 2                | `Impact.py:36`                               |
| DegradeServices            | 2                | `DegradeServices.py:36`                      |
| Withdraw                   | 1                | Default (no `self.duration` set)             |

### 1.7 Attack Chains and Timing Analysis

#### Fastest Path to Impact on Starting Host

The starting host begins in state `U`. The minimum path to Impact:

```
Step  0: Agent receives initial observation, host in state U
Step  1: PrivilegeEscalate selected (p=0.50)  -- begins (duration=2)
Step  2: Sleep (IN_PROGRESS)
Step  3: PE completes, host transitions U -> R
Step  3: Impact selected (p=0.25) -- begins (duration=2)
Step  4: Sleep (IN_PROGRESS)
Step  5: Impact completes, host stays R

Total: 4 simulation steps (2 for PE + 2 for Impact)
```

The probability of this exact fastest path: `0.50 * 0.25 = 12.5%` per attempt.

#### Fastest Path to Impact on Remote Host

The minimum attack chain for a previously unknown remote host requires at least 5 actions
spanning 10 simulation steps under optimal conditions:

```
Step   Action                  Duration  State Transition   Probability
----   ------                  --------  ----------------   -----------
1      DiscoverRemoteSystems      1      (new) -> K         from U: p=0.50
2-2    AggressiveServiceDiscovery 1      K -> S             from K: p=0.25
3-6    ExploitRemoteService       4      S -> U             from S: p=0.50
7-8    PrivilegeEscalate          2      U -> R             from U: p=0.50
9-10   Impact                     2      R -> R (Impact!)   from R: p=0.25
```

Minimum steps: `1 + 1 + 4 + 2 + 2 = 10 steps` (but requires service scan
to succeed, exploit to succeed at ~24.9% rate, and each action to be selected).

In practice the path is longer because:
- DRS reveals hosts in K state, but the agent must select THAT host next
- ERS has approximately 24.9% empirical success rate (includes decoy hits, blocked routes,
  exploit failures, and wrong-port selection)
- The agent may choose DRS instead of PE from U state (50/50)
- The agent may choose DRS instead of Impact from R state (50% DRS vs 25% Imp + 25% Deg)

#### ExploitRemoteService Success Rate: 24.9% Overall

The low success rate is a compound of multiple failure modes:
1. **No valid exploit available** -- target ports do not match any exploit's required ports
2. **Decoy port selected** -- exploit hits a decoy service, returns failure
3. **BlockTrafficZone active** -- firewall blocks the connection (`blocking_host()` check)
4. **Exploit-specific failure** -- the concrete exploit's own success probability
5. **Session lost during duration** -- blue removes the session during the 4-step action

#### Time Budget Analysis

An episode has 500 steps. With 6 red agents operating independently, each agent has 500
steps of wall-clock time. Given action durations, the theoretical maximum number of
completed actions per agent per episode:

| Action Sequence                    | Steps per Cycle | Max Cycles/Episode |
|------------------------------------|:---------------:|:------------------:|
| DRS only (1 step each)             | 1               | 500                |
| ASD then ERS (1+4 steps)           | 5               | 100                |
| Full chain DRS+ASD+ERS+PE (1+1+4+2)| 8              | 62                 |
| Full chain to Impact (8 + 2)       | 10              | 50                 |

### 1.8 Empirical Performance Data

Based on 30 episodes with seed 42 and no blue defense:

#### Impact Achievement by Red Agent

| Red Agent    | Subnet      | Impact Chains | Mean Steps to Impact | Min  | Max  |
|-------------|-------------|:-------------:|:--------------------:|:----:|:----:|
| red_agent_2 | OZA         | 265           | 300.5                | 4    | 474  |
| red_agent_4 | OZB         | 252           | 323.5                | 18   | 485  |
| red_agent_0 | Other       | 0             | --                   | --   | --   |
| red_agent_1 | Other       | 0             | --                   | --   | --   |
| red_agent_3 | Other       | 0             | --                   | --   | --   |
| red_agent_5 | Other       | 0             | --                   | --   | --   |

Only `red_agent_2` (OZA subnet) and `red_agent_4` (OZB subnet) ever achieve Impact because
only Operational Zone (OZ) hosts contain OT services. Other subnets lack the OT services
required for Impact/DegradeServices to cause reward penalties.

This means that from a blue-team optimization perspective, **the defense problem reduces to
protecting OZA and OZB subnets**, with the other 4 red agents being nuisances that consume
blue attention but cause no direct reward loss.

### 1.9 Blue Observable Signals per Red Action

This table is critical for designing blue-agent detection logic. It shows which observable
events each red action generates:

| Red Action                | conn_flag | proc_flag  | malfile | Notes                                    |
|--------------------------|:---------:|:----------:|:-------:|------------------------------------------|
| DiscoverRemoteSystems    | 0         | 0          | 0       | **Completely invisible** to blue          |
| AggressiveServiceDiscovery| 0        | 0          | 0       | **Completely invisible** to blue          |
| StealthServiceDiscovery  | 0         | 0          | 0       | **Completely invisible** to blue          |
| ExploitRemoteService     | 1         | 1 (95%)    | 1       | Always drops cmd.exe/sh                   |
| ExploitRemoteService (silent)| 1     | 0 (5%)     | 1       | No proc event (5% of exploits)           |
| ExploitRemoteService (decoy)| 1      | 0          | 0       | Exploit fails on decoy, no file dropped  |
| PrivilegeEscalate        | 0         | 0          | 1       | Drops escalate.exe/sh, NO events fired   |
| Impact                   | 0         | 0          | 0       | **Completely invisible** to blue          |
| DegradeServices          | 0         | 0          | 0       | **Completely invisible**, 83.3% of damage|

**Critical implications for blue-agent design:**

1. **The majority of damage-causing actions are invisible.** DegradeServices accounts for
   approximately 83.3% of reward penalties and generates zero observable signals.

2. **The only reliable detection window is ExploitRemoteService**, which produces a
   `network_connection` event on the target host. This is the primary (and often only)
   signal that red has gained access.

3. **PrivilegeEscalate drops a malicious file** but does not generate process_creation or
   network_connection events. Blue can only detect this through file-based monitoring
   (which is not directly available in the CC4 observation space).

4. **Blue must act on exploit detection, not damage detection.** By the time Impact or
   DegradeServices occur, blue has already missed the optimal intervention point.

### 1.10 Session Removal Mechanics

When blue executes `Remove` or `Restore` on a host where red has a session (state `U`,
`UD`, `R`, or `RD`), the session is destroyed. The `_session_removal_state_change()`
method detects this by checking which hosts should have sessions but are no longer present
in the observation's session list.

**All removed hosts transition to state `KD`**, regardless of their previous state.

```
U  --[blue Remove/Restore]--> KD
UD --[blue Remove/Restore]--> KD
R  --[blue Remove/Restore]--> KD
RD --[blue Remove/Restore]--> KD
```

From `KD`, red must repeat the full attack chain:
1. ASD or SSD (1 or 3 steps) to reach `SD`
2. ERS (4 steps, ~24.9% success) to reach `UD`
3. PE (2 steps) to reach `RD`
4. Impact/DegradeServices (2 steps)

Minimum re-attack time after session removal: `1 + 4 + 2 + 2 = 9 steps` (if ASD chosen
and ERS succeeds on first attempt). Expected re-attack time with the 24.9% ERS success
rate: approximately 30-40 steps.

**Blue's Restore also resets service reliability**, undoing all DegradeServices damage on
that host. This makes Restore strictly more powerful than Remove, at the cost of higher
action duration (5 steps vs 3 steps).

### 1.11 Host Selection Algorithm

The `_choose_host()` method selects which host to act upon:

1. **State priority selection**: If `host_states_priority_list` is defined (in child
   classes), hosts are grouped by state and a state is chosen probabilistically based
   on priority weights that sum to 100. The base `FiniteStateRedAgent` returns `None`
   for the priority list, meaning all states are equally likely.

2. **Server preference**: If `prioritise_servers` is `True` (default is `False`):
   - 75% chance of selecting a server host (hostname contains "server")
   - 25% chance of selecting a non-server host
   - If only servers or only non-servers are available, selection is from what exists

3. **Random selection within group**: Within the selected state and server/non-server
   group, a host is chosen uniformly at random.

### 1.12 Six Independent Red Agents

The simulation spawns 6 independent `FiniteStateRedAgent` instances, each assigned to
different subnets:

| Agent        | Starting Subnet | Reach                                      |
|-------------|-----------------|---------------------------------------------|
| red_agent_0 | HQ              | HQ subnets                                  |
| red_agent_1 | Contractor      | Contractor subnets                          |
| red_agent_2 | OZA             | Operational Zone A subnets (CRITICAL)        |
| red_agent_3 | Restricted      | Restricted subnets                          |
| red_agent_4 | OZB             | Operational Zone B subnets (CRITICAL)        |
| red_agent_5 | Public          | Public-facing subnets                       |

Each agent maintains its own independent `host_states` dictionary. **Red agents do not
coordinate** -- there is no information sharing, joint planning, or synchronized attacks.
Each agent independently executes the FSM logic within its own subnet scope.

The `agent_subnets` parameter constrains each agent's reach: if an ERS exploit succeeds
on a host outside the agent's subnets, the host transitions to state `F` (Foreign) rather
than `U`, and the agent cannot act on it further.

---

## 2. Green Agent: EnterpriseGreenAgent

### 2.1 Action Selection

Source: `CybORG/Agents/SimpleAgents/EnterpriseGreenAgent.py`

Each step, the `EnterpriseGreenAgent` selects uniformly at random from 3 actions:

| Action             | Probability | Effect                                        |
|-------------------|:-----------:|-----------------------------------------------|
| GreenLocalWork     | 33.3%       | Attempts to use a local service on own host   |
| GreenAccessService | 33.3%       | Attempts to connect to a remote server        |
| Sleep              | 33.3%       | Does nothing                                  |

There are 61 green agents per episode, distributed across all subnets. Each green agent
operates independently on its assigned host.

### 2.2 GreenLocalWork Mechanics

Source: `CybORG/Simulator/Actions/GreenActions/GreenLocalWork.py`

Execution proceeds in 3 sequential phases:

**Phase 1: Service Access Attempt**
1. Enumerate all active services on the host.
2. Select one service uniformly at random.
3. Roll `np_random.integers(100)` and compare to `service.get_service_reliability()`.
4. If `roll >= reliability` --> the local work **fails** (LWF penalty applied to blue).
5. If no active services exist --> the local work **fails**.

The reliability check means that as red degrades services (reducing reliability from 100%
toward 0%), green agents fail more often, generating continuous reward penalties for blue.

**Phase 2: False Positive Alert (1% probability)**

```python
if state.np_random.random() < self.fp_detection_rate:  # default 0.01
    host_port = host.get_ephemeral_port()
    pc = {'local_address': self.ip_address, 'local_port': host_port}
    host.events.process_creation.append(pc)
```

This creates a `process_creation` event on the host -- the same event type that
ExploitRemoteService generates. Blue agents cannot distinguish this false positive from
a genuine red exploit.

**Phase 3: Phishing Email (1% probability)**

```python
if state.np_random.random() < self.phishing_error_rate:  # default 0.01
    sub_action = PhishingEmail(agent=self.agent, session=self.session, ip_address=self.ip_address)
    sub_obs = sub_action.execute(state)
```

This creates a REAL red session on the green host. The `PhishingEmail` action:
1. Checks if red already has a session on the host (if so, no-op).
2. Finds a red agent in the same subnet (or any routable red agent).
3. Creates a `RedAbstractSession` with user-level access.
4. The targeted red agent now has a foothold on this host in state `U`.

### 2.3 GreenAccessService Mechanics

Source: `CybORG/Simulator/Actions/GreenActions/GreenAccessService.py`

Execution:

1. **Destination Selection** (`random_reachable_ip()`):
   - If the green agent's subnet is in `allowed_subnets` for the current phase, it can
     reach any server in any allowed subnet.
   - If the green agent's subnet is NOT in `allowed_subnets`, it can only reach servers
     within its own subnet.
   - Only server hosts are targeted (hostname contains "server").
   - The green agent never targets itself.

2. **BlockTrafficZone Check**:
   - If a firewall block exists between source and destination subnets:
     - A `network_connection` event is added to the destination server host.
     - The action fails (returns unsuccessful observation).
   - This blocked-connection event looks identical to a red exploit's network event to
     blue agents.

3. **False Positive Detection (1% probability)**:
   - On success (no block), with probability `fp_detection_rate`:
     - A `network_connection` event is added to the destination server.
     - This mimics a red agent's ExploitRemoteService connection event.

### 2.4 False Positive Analysis

**Empirical false positive rates (per episode, 500 steps):**

| FP Source               | Rate per Action | Actions/Episode | FP Events/Episode |
|------------------------|:---------------:|:---------------:|:-----------------:|
| GreenLocalWork (proc)  | 0.776%          | ~8,900          | ~69.1             |
| GreenAccessService (net)| 0.970%         | ~8,900          | ~86.5             |
| **Total**              | --              | --              | **~155.6**        |

- P(at least one FP per step) is approximately 27%.
- Process FPs occur on **USER hosts only** (where green agents reside).
- Network FPs occur on **SERVER hosts only** (the connection destinations).

**Signal-to-Noise Ratio:**

With approximately 155.6 false positives per episode and red generating approximately
50-100 true positive exploit events per episode (across all 6 agents), the false positive
rate creates a signal-to-noise ratio of roughly 1:2 to 1:3. A naive blue agent that
reacts to every alert will waste approximately 60-75% of its actions on false positives.

### 2.5 Phishing Mechanics

Source: `CybORG/Simulator/Actions/ConcreteActions/PhishingEmail.py`

**Empirical phishing statistics (per episode):**

| Metric                          | Value    |
|---------------------------------|----------|
| Phishing sessions per episode   | ~28.2    |
| Phase 0 share                   | ~67%     |
| Phase 1 share                   | ~22%     |
| Phase 2 share                   | ~11%     |

**Critical properties of phishing:**

1. **Bypasses BlockTrafficZone**: PhishingEmail uses `check_routable()` (physical
   connectivity) instead of `blocking_host()` (firewall rules). The design rationale
   (documented in ADR-CC4-001) is that phishing emails travel via external SMTP
   infrastructure, not through the modeled IP routing fabric.

2. **Cannot be prevented by blocking**: No blue action can prevent phishing. The only
   defense is to detect the resulting red session (via subsequent exploit or priv-esc
   events) and Remove/Restore the host.

3. **Phase 0 dominance**: 67% of phishing occurs in Phase 0 because:
   - Fewer existing red sessions (red already on a host blocks new phishing there)
   - More green agents have routable paths to red agents
   - No BlockTrafficZone restrictions yet

4. **Red agent selection**: The PhishingEmail action preferentially assigns the session
   to a red agent in the same subnet as the green agent. If none exists, it finds any
   routable red agent.

### 2.6 GreenLocalWork Failure Rates

As red agents execute DegradeServices, service reliability decreases. This directly
increases green LocalWork failure rates, which impose reward penalties on blue:

| Phase   | Steps     | Mean LWF Rate | Cause                                    |
|---------|-----------|:-------------:|------------------------------------------|
| Phase 0 | 0-99      | 4.55%         | Healthy services, minimal degradation     |
| Phase 1 | 100-299   | 23.97%        | DegradeServices accumulating              |
| Phase 2 | 300-499   | 43.01%        | Heavy degradation, many unreliable services|

The escalation from 4.55% to 43.01% demonstrates that **DegradeServices is the primary
reward-destroying action**, not Impact. DegradeServices:
- Is invisible to blue (no observable signals)
- Has cumulative effects (each execution further reduces reliability)
- Affects ALL green agents on the host (multiplied by number of green users)
- Accounts for approximately 83.3% of total reward penalty

This makes DegradeServices strictly more damaging per execution than Impact, despite
both being available from state R/RD with equal probability.

---

## 3. Blue Agent Decision Tree

Based on the red and green agent analysis, the following decision tree captures the
optimal blue response to each observable signal type.

```
                    BLUE AGENT DECISION TREE
                    ========================

  Step begins
      |
      v
  [Check observations for all hosts]
      |
      +---> network_connection event on SERVER host?
      |         |
      |         +---> YES
      |         |       |
      |         |       v
      |         |   [Is this host in OZA or OZB subnet?]
      |         |       |
      |         |       +---> YES (critical OT host)
      |         |       |       |
      |         |       |       v
      |         |       |   *** IMMEDIATE RESTORE ***
      |         |       |   (Removes red session + resets service reliability)
      |         |       |   (Duration: 5 steps, but prevents Impact/Degrade)
      |         |       |
      |         |       +---> NO (non-OZ subnet)
      |         |               |
      |         |               v
      |         |           [Has this host had multiple events recently?]
      |         |               |
      |         |               +---> YES: REMOVE (3 steps)
      |         |               |     (Cheaper than Restore, OT not at risk)
      |         |               |
      |         |               +---> NO: MONITOR
      |         |                     (Likely green FP, ~60-75% chance)
      |         |                     (React only if repeated)
      |         |
      |         +---> NO (no network event)
      |                 |
      |                 v
      |             [Continue to next check]
      |
      +---> process_creation event on USER host?
      |         |
      |         +---> YES
      |         |       |
      |         |       v
      |         |   [Is the source IP in OZA or OZB subnet?]
      |         |       |
      |         |       +---> YES: REMOVE (red may be pivoting toward OT)
      |         |       |
      |         |       +---> NO: LOW PRIORITY
      |         |               (1% FP rate from GreenLocalWork)
      |         |               (True red exploits also generate net events)
      |         |               (Proc-only event = likely FP or phishing)
      |         |
      |         +---> NO
      |                 |
      |                 v
      |             [Continue to next check]
      |
      +---> No events detected?
              |
              v
          [PROACTIVE DEFENSE]
              |
              +---> [BlockTrafficZone on OZA/OZB subnets]
              |     (Prevents red lateral movement INTO OZ)
              |     (Does NOT prevent phishing)
              |     (Generates green FP net events on blocked connections)
              |
              +---> [Deploy decoys strategically]
              |     WARNING: Decoys that red detects via DiscoverDeception
              |     make red MORE focused (see Section 1.4)
              |     Only deploy if decoys waste red's ERS attempts (4 steps)
              |
              +---> [Preemptive Restore on OZ hosts]
                    (Periodic Restore on OZ servers to reset reliability)
                    (Counters invisible DegradeServices damage)
                    (Cost: 5 steps of inaction)
```

### Blue Response Priority Matrix

| Signal Type              | Host Location | Likely Cause          | Recommended Action | Priority |
|-------------------------|---------------|----------------------|-------------------|:--------:|
| network_connection      | OZA/OZB server| Red ExploitRemoteService | **RESTORE**    | CRITICAL |
| network_connection      | Other server  | 60% green FP, 40% red| Monitor/Remove   | MEDIUM   |
| process_creation        | OZ user host  | Red exploit or green FP| Remove          | HIGH     |
| process_creation        | Other user    | 99% green FP         | Ignore           | LOW      |
| No signal, OZ host      | OZA/OZB       | DegradeServices (invisible)| Periodic Restore| HIGH   |
| No signal, other host   | Non-OZ        | No damage possible   | Sleep            | NONE     |
| Blocked green connection| Any server    | GreenAccessService blocked| Ignore (FP)  | NONE     |

### Key Strategic Insights for Blue

1. **Reactive-only on non-OZ subnets.** Since only OZA/OZB can suffer Impact/Degrade
   damage, resources spent defending other subnets are wasted.

2. **BlockTrafficZone on OZ subnets early.** This prevents red's ExploitRemoteService
   from reaching OZ servers. The cost is green network FP events, but these are harmless.

3. **Periodic Restore beats reactive Remove.** Because DegradeServices is invisible and
   accounts for 83.3% of damage, the only counter is periodic Restore to reset service
   reliability. Remove only evicts the session; Restore also undoes degradation.

4. **Decoys are a double-edged sword.** Detected decoys make red skip DRS and focus
   100% on exploitation/damage. Undetected decoys waste red's ERS attempts (4 steps
   each). The optimal decoy strategy maximizes ERS waste while minimizing detection.

5. **Phishing is unblockable background noise.** Approximately 28 phishing sessions per
   episode bypass all defenses. Blue must detect and remediate the resulting sessions
   after the fact. This is a fixed cost of the environment.

6. **False positive management is critical.** With a 1:2 to 1:3 signal-to-noise ratio,
   blue must avoid over-reacting to every alert. Focusing defensive actions on OZ
   subnets inherently reduces FP impact since non-OZ alerts can be safely ignored.
