# CybORG CC4 Environment Mechanics -- Comprehensive Analysis

Source files analysed (all paths relative to repo root):

- `CybORG/Simulator/Actions/ConcreteActions/DecoyActions/*.py`
- `CybORG/Simulator/Actions/AbstractActions/Misinform.py`
- `CybORG/Simulator/Actions/AbstractActions/ExploitRemoteService.py`
- `CybORG/Simulator/Actions/ConcreteActions/ExploitActions/*.py`
- `CybORG/Simulator/Actions/AbstractActions/Monitor.py`
- `CybORG/Simulator/Actions/AbstractActions/Remove.py`
- `CybORG/Simulator/Actions/AbstractActions/Restore.py`
- `CybORG/Simulator/Actions/ConcreteActions/RestoreFromBackup.py`
- `CybORG/Simulator/Actions/ConcreteActions/RemoveOtherSessions.py`
- `CybORG/Simulator/Actions/ConcreteActions/ControlTraffic.py`
- `CybORG/Simulator/Actions/AbstractActions/DiscoverDeception.py`
- `CybORG/Simulator/Actions/ConcreteActions/DecoyActions/DeployDecoy.py`
- `CybORG/Simulator/Actions/Action.py` (RemoteAction, blocking_host)
- `CybORG/Simulator/SimulationController.py` (step, sort_action_order)
- `CybORG/Simulator/Host.py`
- `CybORG/Simulator/Process.py`
- `CybORG/Simulator/File.py`
- `CybORG/Simulator/State.py`
- `CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py`
- `CybORG/Agents/Wrappers/BlueFlatWrapper.py`
- `CybORG/Agents/Wrappers/BlueFlatWrapperV2.py`
- `CybORG/Agents/Wrappers/BlueFixedActionWrapper.py`
- `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgent.py`

---

## 1. Decoy System Deep Dive

### 1.1 Available Decoy Types in CC4

The `DeployDecoy` action (the one available to blue in CC4's `_BLUE_ACTIONS`) uses
exactly four candidate factories:

| Decoy Class         | Factory              | Process Name | Port  | Service Name | Process Type | Properties   | OS Constraint | Extra Constraint       |
|---------------------|----------------------|-------------|-------|-------------|-------------|-------------|---------------|------------------------|
| `DecoyApache`       | `ApacheDecoyFactory` | `apache2`    | 80    | `apache2`    | `webserver`  | `["rfi"]`    | None          | Port 80 not in use     |
| `DecoyTomcat`       | `TomcatDecoyFactory` | `Tomcat.exe` | 443   | `tomcat`     | `webserver`  | `["rfi"]`    | None          | Port 443 not in use    |
| `DecoyHarakaSMPT`   | `HarakaDecoyFactory` | `haraka`     | 25    | `haraka`     | `smtp`       | None         | Linux only    | Port 25 not in use     |
| `DecoyVsftpd`       | `VsftpdDecoyFactory` | `vsftpd`     | 80    | `vsftpd`     | `webserver`  | `["rfi"]`    | Linux only    | Port 21 not in use (!) |

**Additional decoy types exist in code but are NOT in CC4's DeployDecoy:**

| Decoy Class    | Factory                | Process Name | Port  | Service Name | Process Type | OS Constraint |
|----------------|------------------------|-------------|-------|-------------|-------------|---------------|
| `DecoySSHD`    | `SSHDDecoyFactory`     | `Sshd.exe`   | 22    | `sshd`       | `sshd`       | None          |
| `DecoySmss`    | `SMSSDecoyFactory`     | `Smss.exe`   | 139   | `smss`       | `smb`        | Windows only  |
| `DecoySvchost` | `SvchostDecoyFactory`  | `Svchost.exe`| 3389  | `svchost`    | `rdp`        | Windows only  |
| `DecoyFemitter`| `FemitterDecoyFactory` | `femitter`   | 21    | `femitter`   | `femitter`   | Windows only  |

Since CC4 hosts are ALL Linux, the Windows-only decoys (Smss, Svchost, Femitter)
would never be compatible anyway. SSHD is not included in DeployDecoy's candidate
list despite being Linux-compatible.

### 1.2 Decoy Compatibility Check

`DecoyFactory.is_host_compatible(host)` checks `not host.is_using_port(PORT)`.
Subclasses may add OS checks. The `DecoyAction.execute()` method:

1. Collects all factories where `factory.is_host_compatible(host)` is True.
2. If none are compatible, the action **fails** (returns `obs_fail`).
3. Otherwise, picks one **randomly** from the compatible set.

**Key implication**: Deploying multiple decoys on the same host is possible as long
as they use different ports. With Apache on 80, Tomcat on 443, and Haraka on 25,
you can deploy up to 3 decoys per host (assuming those ports are not already in
use by legitimate services).

**Vsftpd quirk**: Despite being labelled as a webserver on port 80 in the factory,
the compatibility check is `not host.is_using_port(21)`, meaning it checks port 21
availability but actually binds to port 80. If Apache is already deployed (port 80
occupied), Vsftpd will fail because port 80 is in use. And vice versa. So Apache
and Vsftpd are mutually exclusive.

### 1.3 Decoy Limits Per Host

There is **no explicit limit** on the number of decoys per host. The constraint is
purely port-based: each factory requires its port to be free. Practical maximum
per Linux host:

- Apache (port 80) OR Vsftpd (port 80) -- mutually exclusive
- Tomcat (port 443)
- Haraka (port 25)

So **maximum 3 decoys per host** (e.g., Apache + Tomcat + Haraka, or Vsftpd +
Tomcat + Haraka).

If a host already has a legitimate service occupying one of those ports (e.g.,
Apache on port 80, SMTP/Haraka on port 25), those decoy slots are unavailable.

### 1.4 How Decoys Catch Exploits

The exploit interception mechanism lives in `ExploitAction._get_vulnerable_process()`:

```
For each process on the target host:
    If process.process_type matches the exploit's expected type:
        For each connection in process.connections:
            If connection.local_port matches the exploit's PORT:
                vuln_proc = this process
                If vuln_proc.decoy_type != NONE:
                    Log a NetworkConnection event on the host
                Break (first match wins)
```

Then in `ExploitAction.sim_exploit()`:

```python
is_exploit_decoy = bool(vuln_proc.decoy_type & DecoyType.EXPLOIT)
if is_exploit_decoy:
    # Log network_connection event (blue can detect this)
    # Return failure observation (no session created)
    return obs  # success=False
```

**Critical finding**: The exploit searches processes **in list order** (iteration
order of `target_host.processes`). The first process matching `process_type` AND
`port` wins. Decoy processes are **appended** to the host's process list by
`DecoyAction.__create_process()`, so they go at the end. Legitimate processes are
created at scenario start and appear earlier in the list.

**This means**: If a legitimate service occupies the same port AND process type,
the legitimate one is found first and the decoy is never checked. But since the
compatibility check prevents deploying a decoy on an already-occupied port, this
scenario should not arise in normal operation.

### 1.5 DiscoverDeception

Red can use `DiscoverDeception` to probe a host for decoys:

- **Duration**: 2 steps
- **True Positive rate**: 50% -- each EXPLOIT-type decoy process has a 50% chance
  of being correctly identified
- **False Positive rate**: 10% -- each non-decoy process has a 10% chance of being
  incorrectly flagged as a decoy
- **Result**: Returns process info with `'decoy'` added to properties
- **Effect on future exploits**: Intelligence only. DiscoverDeception does NOT
  prevent the exploit from hitting the decoy. However, the `PIDSelectiveExploitActionSelector`
  exists in code and could exclude discovered PIDs; the `FiniteStateRedAgent`
  uses this after discovering deception.

---

## 2. Decoy-Exploit Interaction Matrix

### 2.1 Exploit Types and Matching Process Types

| Exploit              | PORT  | Process Type Searched | Weight | OS Required | Extra Check                        |
|----------------------|-------|----------------------|--------|-------------|-------------------------------------|
| `HTTPRFI`            | 80    | `http` (webserver)   | 3.0    | Any         | `"rfi" in vuln_proc.properties`     |
| `HTTPSRFI`           | 443   | `webserver`          | 4.0    | Any         | `"rfi" in vuln_proc.properties`     |
| `SSHBruteForce`      | 22    | `SSH` (sshd)         | 0.1    | Any         | User must be bruteforceable         |
| `EternalBlue`        | 139   | `smb`                | 2.0    | Windows     | MS17_010 patch not present          |
| `BlueKeep`           | 3389  | `rdp`                | 1.0    | Windows     | Specific KB patches not present     |
| `FTPDirTraversal`    | 21    | `femitter`           | 7.0    | Any         | Always succeeds if process found    |
| `HarakaRCE`          | 25    | `smtp`               | 6.0    | Any         | Version < HARAKA_2_8_9              |
| `SQLInjection`       | 3390  | `mysql`              | 5.0    | Any         | Requires port 80 or 443 also open  |

**CC4-relevant exploits only** (EternalBlue and BlueKeep are "never used in CC4"
per code comment -- they require Windows hosts, and CC4 is all-Linux):

| Exploit              | PORT  | CC4 Weight | Decoy That Catches It                     |
|----------------------|-------|-----------|-------------------------------------------|
| `HTTPRFI`            | 80    | 3.0       | DecoyApache (port 80, webserver, has rfi)  |
| `HTTPSRFI`           | 443   | 4.0       | DecoyTomcat (port 443, webserver, has rfi) |
| `SSHBruteForce`      | 22    | 0.1       | **No CC4 decoy** (SSHD decoy not in list) |
| `FTPDirTraversal`    | 21    | 7.0       | **No CC4 decoy** (Femitter is Windows)    |
| `HarakaRCE`          | 25    | 6.0       | DecoyHarakaSMPT (port 25, smtp)           |
| `SQLInjection`       | 3390  | 5.0       | **No CC4 decoy** (no mysql decoy exists)  |

### 2.2 Decoy-Exploit Match Table

```
Decoy           Port  Type        Props    Catches Exploit?
-----------     ----  ----------  -------  -----------------------------------------
DecoyApache     80    webserver   [rfi]    HTTPRFI (port=80, type=http/webserver, needs rfi) -> YES
DecoyTomcat     443   webserver   [rfi]    HTTPSRFI (port=443, type=webserver, needs rfi)   -> YES
DecoyHaraka     25    smtp        None     HarakaRCE (port=25, type=smtp)                   -> DEPENDS*
DecoyVsftpd     80    webserver   [rfi]    HTTPRFI (port=80, type=http/webserver, needs rfi) -> YES
```

*HarakaRCE has an additional check: `vuln_proc.version.value < HARAKA_2_8_9.value`.
The `HarakaDecoyFactory` sets `VERSION = "haraka 2.7.0"`, which IS vulnerable
(2.7.0 < 2.8.9). So the exploit passes `test_exploit_works()` but then fails at
the `is_exploit_decoy` check. **The decoy catches it successfully.**

**Important nuance on HTTPRFI vs Apache decoy**: `HTTPRFI.execute()` calls
`sim_exploit(state, 'http')`. The ProcessType parse converts 'http' to what
matches. The Apache decoy's process_type is 'webserver'. In `ProcessType.parse_string()`,
'http' maps to `ProcessType.WEBSERVER` -- so the types match.

### 2.3 Exploit Selection Algorithm

`DefaultExploitActionSelector.get_exploit_action()`:

1. For each exploit class, compute `get_weight(ports)` -- returns weight if the
   exploit's PORT is in the discovered ports, else 0.
2. Filter to non-zero weights.
3. Find the `top_choice` (highest weight), remove it.
4. Pick `other_choice` randomly from remaining.
5. `odds_of_top_choice = 0` -- **the top choice is never selected!**
6. The randomly chosen `other_choice` is always used (unless priority override).

**Key insight**: The exploit selection is effectively random among all applicable
exploits EXCEPT the highest-weighted one. This is counterintuitive but confirmed
by the code: `odds_of_top_choice = 0`.

### 2.4 Probability of Hitting a Decoy

Given N applicable exploit types on a host, the selection is random among N-1
(excluding the top-weighted one). If K of those are caught by decoys:

```
P(decoy hit) = K / (N-1)    when N > 1
P(decoy hit) = 1.0           when N = 1 and that exploit targets a decoy
```

**Example**: A host with SSH (port 22), Apache (port 80), MySQL (port 3390), plus
3 decoys (Apache/80, Tomcat/443, Haraka/25):

Available ports: 22, 80, 443, 25, 3390
Applicable exploits:
- SSHBruteForce (22, weight 0.1)
- HTTPRFI (80, weight 3.0)
- HTTPSRFI (443, weight 4.0) -- hits decoy
- HarakaRCE (25, weight 6.0) -- hits decoy
- SQLInjection (3390, weight 5.0, also needs 80/443 -> yes)
- FTPDirTraversal -- no port 21, excluded

Top choice: HarakaRCE (weight 6.0) -- removed from pool.
Pool: SSHBruteForce, HTTPRFI, HTTPSRFI, SQLInjection (4 options).
Decoy-caught in pool: HTTPSRFI only (HTTPRFI hits the REAL apache first on port 80).

P(decoy hit) = 1/4 = 25%.

But if the host does NOT have a real apache (port 80 from decoy only):
Applicable exploits with decoy ports:
- HTTPRFI (80, weight 3.0) -> hits decoy
- HTTPSRFI (443, weight 4.0) -> hits decoy
- HarakaRCE (25, weight 6.0) -> hits decoy
- SSHBruteForce (22, weight 0.1) -> no decoy

Top: HarakaRCE (removed). Pool: HTTPRFI, HTTPSRFI, SSH.
Decoy-caught: HTTPRFI, HTTPSRFI = 2/3 = 67%.

With ONLY decoy services (no real services except SSH):
**2 out of 3 random picks = 67% chance of wasting red's exploit attempt.**

---

## 3. Observation Pipeline

### 3.1 Monitor Action

`Monitor` is executed automatically at the END of each step as a default action
for blue agents. It collects events from all hosts the agent has sessions on.

**What Monitor returns:**

For each host with a blue session (parent + children):

1. **Network Connection Events** (`host.events.network_connections`):
   - `local_address`, `local_port`, `remote_address`, `remote_port`
   - `pid` (if available)
   - `application_protocol`, `transport_protocol`
   - PIDs are added to `session.sus_pids[hostname]` for later Remove actions.
   - After collection, events are moved to `old_network_connections` and cleared.

2. **Process Creation Events** (`host.events.process_creation`):
   - Dict with `pid`, `Connections` (list of connection dicts)
   - PIDs are added to `session.sus_pids[hostname]`.
   - After collection, events are moved to `old_process_creation` and cleared.

**When events fire** (source: ExploitAction code):
- Successful exploit with detection (95% rate): creates process_creation event
- Decoy hit: creates network_connection event (always)
- SSHBruteForce: creates 10 network_connection events (wordlist_length=10)
- Blocked exploit: creates network_connection event at the blocking host

### 3.2 BlueFlatWrapper Observation Layout

The observation vector has the following structure:

```
obs[0]                      : mission_phase (0=Preplanning, 1=MissionA, 2=MissionB)

For each controlled subnet (in alphabetical order):
  [subnet_one_hot]           : 9 bits, one-hot identifying which subnet
  [blocked_subvector]        : 9 bits, 1 = subnet i is currently blocked TO this subnet
  [comms_policy_subvector]   : 9 bits, 1 = subnet i SHOULD be blocked (inverted adjacency)
  [process_flags]            : n_hosts bits, 1 = host has process_creation events
  [connection_flags]         : n_hosts bits, 1 = host has network_connection events

[messages]                   : 32 bits (4 messages x 8 bits each)
```

**Process/connection flag logic** (`_get_procesess` / `_get_connections`):
- Returns True if `old_process_creation OR process_creation` is non-empty
- Returns True if `old_network_connections OR network_connections` is non-empty
- These include BOTH the "old" (already consumed by Monitor) and "new" (not yet
  consumed) events. This means flags persist for one step after Monitor clears them.

**Subnet ordering**: Alphabetical sort of `state.subnet_name_to_cidr.items()`:
```
0: admin_network_subnet
1: contractor_network_subnet
2: internet_subnet
3: office_network_subnet
4: operational_zone_a_subnet
5: operational_zone_b_subnet
6: public_access_zone_subnet
7: restricted_zone_a_subnet
8: restricted_zone_b_subnet
```

**Host ordering within each subnet**: Hosts matching `subnet_name in hostname AND
"router" not in hostname`, in whatever order `state.hosts` provides. These are
user_host_0..N and server_host_0..M.

**Agent-to-subnet mapping (which subnets each agent sees):**

| Agent          | Controlled Subnets                                              | # Subnet Blocks |
|----------------|----------------------------------------------------------------|-----------------|
| `blue_agent_0` | restricted_zone_a_subnet                                       | 1               |
| `blue_agent_1` | operational_zone_a_subnet                                      | 1               |
| `blue_agent_2` | restricted_zone_b_subnet                                       | 1               |
| `blue_agent_3` | operational_zone_b_subnet                                      | 1               |
| `blue_agent_4` | public_access_zone, admin_network, office_network              | 3               |

### 3.3 BlueFlatWrapperV2 Extensions

Appends `sum(n_hosts_i)` malfile bits after the base observation:

```
obs[:base_len]     : identical to BlueFlatWrapper
obs[base_len:]     : one malfile bit per host (same subnet/host order as proc/conn flags)
```

**Malfile detection criteria** (`_get_malfile`):
```python
any(f.density >= 0.9 and not f.signed for f in state.hosts[hostname].files)
```

**What triggers malfile=1:**
- `ExploitRemoteService` ALWAYS drops `cmd.exe`/`cmd.sh` with density=0.9, signed=False
- `PrivilegeEscalate` ALWAYS drops `escalate.exe`/`escalate.sh` with density=0.9, signed=False
- Green agent actions NEVER create high-density unsigned files
- **Zero false positives** from green activity

**Malfile detection patterns:**
```
conn_flag=1, proc_flag=1, malfile=1 : Standard exploit (95% case)
conn_flag=1, proc_flag=0, malfile=1 : Silent exploit (5% undetected proc)
conn_flag=1, proc_flag=0, malfile=0 : Decoy hit (exploit failed, no file dropped)
conn_flag=0, proc_flag=0, malfile=1 : PrivilegeEscalate (no events fired at all)
conn_flag=0, proc_flag=0, malfile=0 : Clean host
```

### 3.4 Information Gaps and Delays

1. **PrivilegeEscalate is invisible** to standard BlueFlatWrapper -- it drops a
   file but fires NO process_creation or network_connection events. Only
   BlueFlatWrapperV2's malfile flag detects it.

2. **One-step persistence**: Process and connection flags include `old_*` events,
   so flags persist for one extra step after Monitor clears the queue.

3. **5% silent exploit rate**: `ExploitAction` fires a process_creation event with
   95% probability (`detection_rate = 0.95`). The remaining 5% create a red session
   with no process_creation event, only a network_connection event.

4. **No cross-zone visibility**: Each blue agent only sees events on hosts it has
   sessions on (its controlled subnets). Inter-agent messaging is the only way to
   share threat intelligence.

---

## 4. Action Execution Mechanics

### 4.1 Remove Action

**What Remove does** (from `Remove.execute()` and `StopProcess`):

1. Finds the VelociraptorServer session for the blue agent on the target host.
2. For each PID in `parent_session.sus_pids[hostname]` (populated by Monitor):
   - Executes `StopProcess` which kills the process and removes the associated
     red session from `state.sessions`.
3. Removes malware files: `host.files = [f for f in host.files if not (f.density >= 0.9 and not f.signed)]`

**What survives Remove:**
- **Root/SYSTEM sessions**: `RemoveOtherSessions` only removes sessions where
  `session.username not in ['root', 'SYSTEM', 'hardware']` at the 'user' level.
  Privileged red sessions from PrivilegeEscalate persist.
- **Processes not in sus_pids**: If Monitor missed a process creation (5% case),
  its PID is not in sus_pids and it will not be killed.

**Duration**: 3 steps.
**Success rate**: 90% per session (`RemoveOtherSessions` success_rate=0.9).

### 4.2 Restore Action

**What Restore does** (from `Restore.execute()` -> `RestoreFromBackup` -> `Host.restore()`):

1. Collects ALL sessions referencing this host from `state.sessions` (comprehensive).
2. Calls `target_host.restore()`:
   - Clears `events.network_connections` and `events.process_creation`
   - Preserves `old_network_connections` and `old_process_creation` (for current obs)
   - Resets `files` to `original_files` (scenario start state)
   - Resets `sessions` to `original_sessions` (blue sessions only)
   - Resets `processes` to `original_processes` (scenario start processes)
   - Resets `services` to `original_services`
   - Resets `ephemeral_ports`
   - Increments `restore_count`
   - Does NOT reset `impact_count`
3. Re-injects only sessions that survived the restore (present in restored
   `host.sessions`), discarding red sessions.

**What Restore clears that Remove cannot:**
- Root/SYSTEM-level red sessions
- Any red processes not in sus_pids
- Malware files (also cleared by Remove v9+)
- DegradeServices effects (service reliability)
- Deployed decoys (host reimaged to original state!)

**Duration**: 5 steps.
**Cost**: -1 flat reward penalty.

### 4.3 Action Queuing and Duration System

Each agent submits one action per step. The simulation maintains an
`actions_in_progress` dict:

1. On each step, the agent's action is placed in `actions_in_progress` with
   `remaining_ticks = action.duration`.
2. Each step, `remaining_ticks` is decremented.
3. When `remaining_ticks < 1`, the action moves to `actions_to_execute` and fires.
4. While in progress, the agent receives `IN_PROGRESS` observation and a `Sleep`
   action is substituted.

**Action durations:**
```
Sleep:           1 step  (instant)
Monitor:         1 step  (instant, also runs automatically at end of each turn)
BlockTrafficZone: 1 step (priority=1, executes before most other actions)
AllowTrafficZone: 1 step (priority=1)
Remove:          3 steps
Restore:         5 steps
DeployDecoy:     2 steps
DiscoverDeception: 2 steps (red)
ExploitRemoteService: 4 steps (red)
```

**An agent can only have one action in progress at a time.** While waiting for a
multi-step action to complete, the agent is locked out of taking any other action.

### 4.4 Execution Order Within a Step

From `SimulationController.sort_action_order()`:

1. All agent actions are flattened into a list of (agent, action) tuples.
2. Sorted by `action.priority` (ascending = higher priority first).
3. Action indices are **shuffled randomly** for bandwidth calculation.
4. Remote actions have routes computed, blocking checked, and bandwidth enforced.
5. Actions execute in the sorted-by-priority order.

**Priority values (from code):**
```
ControlTraffic (Block/Allow):  priority = 1  (executes first)
RemoveOtherSessions:           priority = 5
Sleep:                         priority = 0  (default from Action base)
ExploitRemoteService:          not set (default 0)
```

**Blue vs Red same-host same-step**: Both actions execute in the same step.
Blocking actions (priority 1) execute before exploit actions (priority 0),
so a BlockTrafficZone in the same step will block a concurrent exploit. However,
Remove (priority 5) executes AFTER the exploit, meaning a Remove and an exploit
on the same host in the same step results in the exploit succeeding first, then
Remove cleaning up (if it finds the process via sus_pids from Monitor, which runs
at end of step).

**Post-action processing**: After all actions execute, Monitor runs for all blue
agents (end-of-turn default action), collecting events generated during the step.

---

## 5. Subnet Architecture

### 5.1 Subnet Topology

```
                            INTERNET
                       (root_internet_host_0)
                        /       |        \        \
                       /        |         \        \
           RZA_router   RZB_router  CONTRACTOR_router  PAZ_router
              |             |            |              /    |    \
              |             |            |           ADMIN  OFFICE PAZ
           OZA_router    OZB_router      |          router  router hosts
              |             |            |             |      |
           OZA hosts     OZB hosts    CONTR hosts   ADMIN  OFFICE
                                                    hosts  hosts
           RZA hosts     RZB hosts
```

### 5.2 Subnet Definitions

Each subnet (except internet) contains:
- 1 router host (Linux, no services, does not respond to ping)
- 3-10 user hosts (randomized per episode, Linux)
- 1-6 server hosts (randomized per episode, Linux)

Internet subnet has only `root_internet_host_0` (Linux, no services).

**Between-subnet links** (how red traverses between zones via server_host_0):

```
contractor_network_server_host_0 <-> restricted_zone_a_server_host_0
contractor_network_server_host_0 <-> restricted_zone_b_server_host_0
contractor_network_server_host_0 <-> public_access_zone_server_host_0
restricted_zone_a_server_host_0  <-> operational_zone_a_server_host_0
restricted_zone_b_server_host_0  <-> operational_zone_b_server_host_0
public_access_zone_server_host_0 <-> admin_network_server_host_0
public_access_zone_server_host_0 <-> office_network_server_host_0
```

These links provide red with IP address intelligence when gaining root control
of the source host, enabling cross-subnet pivoting.

### 5.3 Agent-Subnet Mapping

| Blue Agent     | Subnets Controlled                                             |
|----------------|---------------------------------------------------------------|
| `blue_agent_0` | `restricted_zone_a_subnet`                                     |
| `blue_agent_1` | `operational_zone_a_subnet`                                    |
| `blue_agent_2` | `restricted_zone_b_subnet`                                     |
| `blue_agent_3` | `operational_zone_b_subnet`                                    |
| `blue_agent_4` | `public_access_zone_subnet`, `admin_network_subnet`, `office_network_subnet` |

| Red Agent      | Starting Subnet                                                |
|----------------|---------------------------------------------------------------|
| `red_agent_0`  | `contractor_network_subnet` (ONLY initially active agent)      |
| `red_agent_1`  | `restricted_zone_a_subnet` (starts inactive)                   |
| `red_agent_2`  | `operational_zone_a_subnet` (starts inactive)                  |
| `red_agent_3`  | `restricted_zone_b_subnet` (starts inactive)                   |
| `red_agent_4`  | `operational_zone_b_subnet` (starts inactive)                  |
| `red_agent_5`  | `public_access_zone`, `admin_network`, `office_network` (inactive) |

### 5.4 Host Services at Scenario Start

All non-router, non-internet hosts are Linux and have:

**Mandatory services:**
- `sshd` (port 22, type SSH) -- always present

**Conditional services:**
- `OTService` (port 1, type UNKNOWN) -- only on `operational_zone_*` hosts

**Optional services (0 to 3 chosen randomly per host):**
- `apache2` (port 80, type WEBSERVER, with `rfi` property at low probability)
- `mysqld` (port 3390, type MYSQL)
- `smtp` / haraka (port 25, type SMTP, version HARAKA_2_8_9 = not vulnerable;
  vulnerable 2.7.0 only appears with low probability)

**Service vulnerability probability**: `prob_vuln_proc_occurs = 1.0`. The
conditional to give a process the `rfi` property or vulnerable version is
`if prob_vuln_proc_occurs < state.np_random.random()`, which with prob=1.0
means `1.0 < random()` is ALWAYS FALSE. So by default:
- Apache processes do NOT have `rfi` property (HTTPRFI exploit will fail
  `test_exploit_works`)
- SMTP processes have version HARAKA_2_8_9 (HarakaRCE will fail
  `test_exploit_works`)

**This means**: On default-generated hosts, HTTPRFI and HarakaRCE exploits
**always fail** against real services. Decoys with `rfi` property and
vulnerable Haraka version are actually MORE exploitable than real services
-- but they catch the exploit via the decoy_type flag.

---

## 6. ControlTraffic Mechanics

### 6.1 BlockTrafficZone

`BlockTrafficZone(session, agent, from_subnet, to_subnet)`:

- Adds `from_subnet` to `state.blocks[to_subnet]` list.
- Blocks are directional: blocks traffic FROM `from_subnet` TO `to_subnet`.
- **Persists across steps** -- blocks remain until explicitly removed by
  `AllowTrafficZone`.
- Duration: 1 step (instant).
- Priority: 1 (executes before most other actions).

### 6.2 How Blocking Affects Red

`RemoteAction.blocking_host(state, src_hostname, other_hostname)`:

```python
src_host_subnet = state.hostname_subnet_map[src_hostname]
other_host_subnet = state.hostname_subnet_map[other_hostname]
ip_blocked = other_hostname in state.blocks and src_hostname in state.blocks[other_hostname]
subnet_blocked = other_host_subnet in state.blocks and src_host_subnet in state.blocks[other_host_subnet]
return subnet_blocked or ip_blocked
```

This is checked in `ExploitRemoteService.execute()` before the exploit fires:
```python
if self.blocking_host(state=state, src_hostname=src_hostname,
                      other_hostname=state.ip_addresses[self.ip_address]):
    return Observation(success=False)
```

And in `sort_action_order()`, blocking is applied to the route. If any host
along the route is blocked, the action's `blocked` attribute is set and a
network_connection event is logged at the blocking host.

### 6.3 AllowTrafficZone

`AllowTrafficZone(session, agent, from_subnet, to_subnet)`:

- Removes `from_subnet` from `state.blocks[to_subnet]`.
- Fails (returns False) if the block does not exist.
- Duration: 1 step, Priority: 1.
- No cost beyond the action slot for that step.

### 6.4 Effect on Green Services

**Blocking does NOT affect green agents directly.** Green agent actions
(`GreenAccessService`, `GreenLocalWork`) are local actions that do not use
`RemoteAction.blocking_host()`. However, PhishingEmail (green -> red vector)
uses `check_routable()` / `connected_components`, which BYPASSES the block
mechanism entirely. So blocking does not prevent phishing.

### 6.5 Comms Policy (Required Block Pattern)

The observation includes a `comms_policy_subvector` that indicates which subnets
SHOULD be blocked. This is inverted: 1 = should be blocked, 0 = allowed.

The policies change with mission phase:

**Phase 0 (Preplanning)**: Most subnets can communicate. Required blocks:
- `operational_zone_a <-> restricted_zone_a` (one-way: OZA->RZA allowed, RZA->OZA not)
- `operational_zone_b <-> restricted_zone_b` (same pattern)

**Phase 1 (MissionA)**: RZA becomes isolated:
- Remove `restricted_zone_a <-> contractor_network`
- Remove `restricted_zone_a <-> restricted_zone_b`
- Remove `restricted_zone_a <-> internet`
- Remove `operational_zone_a <-> restricted_zone_a`

**Phase 2 (MissionB)**: RZB becomes isolated (similar pattern for zone B).

### 6.6 NACL (Network Access Control Lists)

The scenario defines static NACLs that constrain inter-subnet routing:

| From Subnet     | To Subnet      | Inbound  | Outbound |
|-----------------|----------------|----------|----------|
| RZA             | OZA            | None     | all      |
| OZA             | RZA            | all      | None     |
| RZB             | OZB            | None     | all      |
| OZB             | RZB            | all      | None     |
| RZA             | Contractor     | all      | all      |
| RZA             | PAZ            | all      | all      |
| RZB             | Contractor     | all      | all      |
| RZB             | PAZ            | all      | all      |
| Contractor      | RZA, RZB, PAZ  | all      | all      |
| PAZ             | RZA, RZB, Contr, Admin, Office | all | all |
| Admin           | PAZ, Office    | all      | all      |
| Office          | PAZ, Admin     | all      | all      |
| Internet        | ALL subnets    | all      | all      |

---

## 7. Wrapper Message System

### 7.1 Message Constants

```python
NUM_MESSAGES = 4        # Each agent receives 4 messages per step
MESSAGE_LENGTH = 8      # Each message is 8 bits
EMPTY_MESSAGE = np.zeros(8, dtype=bool)
```

### 7.2 Message Layout in Observation

Messages are appended at the end of the observation vector, before the malfile
section (V2 only):

```
obs[base_len - 32 : base_len]    : 32 message bits (4 x 8)
obs[base_len:]                   : malfile flags (V2 only)
```

Where `base_len = 1 + sum(27 + 2*n_hosts_i for each controlled subnet) + 32`.

Messages from the 4 OTHER blue agents (excluding self) are concatenated in sorted
agent index order. If fewer than 4 messages arrive (e.g., isolated subnet agents),
the remaining slots are zero-padded with `EMPTY_MESSAGE`.

### 7.3 V9 Message Protocol

The heuristic agent (v9) encodes 8 bits per outgoing message:

| Bits  | Field              | Values                                         |
|-------|--------------------|-------------------------------------------------|
| 0-1   | THREAT_LEVEL       | 0=clean, 1=decoy_hit, 2=user_session, 3=root   |
| 2-3   | OPEN_PATHS         | 0-3+: count of unblocked required comms paths   |
| 4-5   | RED_HOST_COUNT     | 0-3+: count of hosts with confirmed red presence|
| 6     | DECOYS_BYPASSED    | Red has PID knowledge (decoy then real exploit)  |
| 7     | RESTORING          | At least one Restore in progress in this zone    |

### 7.4 How BlueFlatWrapperV2 Handles Messages

The wrapper does not perform any encoding/decoding of message content. It simply:
1. Reads `observation.get("message", [])` from the raw observation dict.
2. Pads to `NUM_MESSAGES` entries with `EMPTY_MESSAGE`.
3. Concatenates all messages into a flat 32-element array.
4. Appends this to the observation vector.

The semantic interpretation of message bits is entirely up to the agent
implementation. The wrapper treats messages as opaque bit arrays.

---

## 8. Step Execution Timeline

A single simulation step proceeds as follows:

```
1. Mission phase check: update step counter, check if phase transition needed
2. Collect actions from all agents (or use defaults for missing agents)
3. Place actions in actions_in_progress with remaining_ticks = duration
4. Decrement remaining_ticks for all in-progress actions
5. Actions with remaining_ticks < 1 move to actions_to_execute
6. Sort actions_to_execute by priority (ascending)
7. Shuffle for bandwidth randomization
8. Compute routes, check blocks, enforce bandwidth limits
9. Execute actions in priority order:
   a. ControlTraffic (priority 1) -- blocks/allows first
   b. Sleep/default actions (priority 0)
   c. Remote actions (priority 0) -- exploits, service discovery
   d. RemoveOtherSessions (priority 5) -- last
10. Session reassignment for subnet traversal
11. End-of-turn actions: Monitor for all active blue agents
    - Collects network_connection and process_creation events
    - Moves events to old_* lists, clears active event lists
12. Observations returned to agents
```

---

## 9. Strategy-Relevant Takeaways

### 9.1 Decoy Placement Priorities

1. **server_host_0 in each zone**: These are the pivoting nodes between subnets.
   Red MUST exploit these to advance. 3 decoys here maximizes waste.

2. **Operational zone hosts**: OTService targets (Impact/DegradeServices).
   Protecting these prevents reward loss.

3. **User hosts**: Many targets but lower priority. Red often phishes these first
   and the phishing vector bypasses blocks and decoys.

### 9.2 Decoy Effectiveness

With 3 decoys on a host with only SSH as a real service:
- 3 exploit types hit decoys (HTTPRFI, HTTPSRFI, HarakaRCE)
- 1 real exploit (SSHBruteForce, weight 0.1)
- FTP/SQL may apply depending on ports
- Red wastes 4 steps per failed exploit + 2 steps for DiscoverDeception (50% miss)

### 9.3 Remove vs Restore Decision

| Scenario                        | Remove Sufficient? | Restore Needed? |
|---------------------------------|-------------------|-----------------|
| User-level red session          | Yes (90% chance)  | Backup option   |
| Root-level red session          | No                | Yes             |
| DegradeServices applied         | No                | Yes             |
| PrivilegeEscalate (malfile only)| No (clears file, not session) | Yes |
| Decoy hit (conn_flag, no malfile)| Neither needed   | Skip            |
| Post-Restore decoy redeployment | N/A               | Decoys wiped    |

### 9.4 Blocking Strategy

- Block BEFORE red reaches a subnet (priority 1 = executes first in step)
- Follow comms_policy for phase-appropriate blocks
- PhishingEmail BYPASSES blocks (uses connected_components, not blocks dict)
- Blocking is free (no reward penalty, instant duration)
- AllowTrafficZone is also free and instant

### 9.5 Observation Diagnostic Patterns

```
Event Pattern (V2)            | Meaning                    | Best Response
------------------------------|----------------------------|------------------
conn=1, proc=1, malfile=1     | Standard exploit (95%)     | Restore
conn=1, proc=0, malfile=1     | Silent exploit (5%)        | Restore
conn=1, proc=0, malfile=0     | Decoy hit                  | No action needed
conn=0, proc=0, malfile=1     | PrivilegeEscalate          | Restore (root)
conn=0, proc=0, malfile=0     | Clean                      | Deploy/Block
conn=1, proc=1, malfile=0     | Should not occur normally  | Investigate
```
