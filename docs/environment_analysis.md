# CAGE Challenge 4: Definitive Environment Mechanics Reference

This document synthesizes all empirical measurements and source-code-level findings about
the CybORG CC4 simulation environment. It is intended as a single authoritative reference
for researchers developing blue-team agents.

---

## 1. Network Topology

CC4 models an enterprise network with 9 subnets connected through routers in a
hierarchical topology. The internet serves as the root, with four top-level routers
branching into restricted zones, operational zones, and headquarters subnets.

```
                                INTERNET
                           (root_internet_host_0)
                          /       |         \          \
                         /        |          \          \
              RZA_router    RZB_router   CONTRACTOR_router   PAZ_router
                  |              |              |                |
           +------+------+ +----+----+   +-----+-----+   +-----+--------+
           |             | |         |   |           |    |     |        |
         RZA           OZA_router  RZB  OZB_router  CONTR  PAZ  ADMIN  OFFICE
      (3 hosts)          |      (3 hosts)  |       (3 hosts)(3h) (3h)  (3h)
                        OZA              OZB
                     (3 hosts)        (3 hosts)
```

### Subnet Summary

| Subnet          | Abbreviation | Hosts | Role                              |
|-----------------|--------------|-------|-----------------------------------|
| restricted_zone_a | RZA        | 3     | High-value servers, restricted    |
| operational_zone_a | OZA       | 3     | Mission-critical ops (Phase 1)    |
| restricted_zone_b | RZB        | 3     | High-value servers, restricted    |
| operational_zone_b | OZB       | 3     | Mission-critical ops (Phase 2)    |
| contractor_network | CONTR     | 3     | External contractor access        |
| public_access_zone | PAZ       | 3     | Public-facing services            |
| admin_network   | ADMIN        | 3     | Administrative hosts              |
| office_network  | OFFICE       | 3     | General office workstations       |
| internet        | INET         | 1     | External (red entry point)        |

Total: 25 internal hosts + 1 internet host = 26 hosts.

### Blue Agent Assignments

| Agent        | Controlled Subnets            | Notes                       |
|--------------|-------------------------------|-----------------------------|
| blue_agent_0 | RZA                           | Single-subnet agent         |
| blue_agent_1 | OZA                           | Single-subnet agent         |
| blue_agent_2 | RZB                           | Single-subnet agent         |
| blue_agent_3 | OZB                           | Single-subnet agent         |
| blue_agent_4 | PAZ, admin, office            | HQ agent (3 subnets)        |

### Red Agent Assignments

6 `FiniteStateRedAgent` instances (`red_agent_0` through `red_agent_5`), each assigned
to specific target subnets. Red agents operate a finite state machine that progresses
through: `Discover -> Exploit -> PrivilegeEscalate -> DegradeServices/Impact`, with
lateral movement between subnets.

---

## 2. Phase System

Each episode is 500 steps, divided into 3 phases:

```
 Step:  0                  167                 334                 500
        |--- Phase 0 -------|--- Phase 1 -------|--- Phase 2 -------|
        |   167 steps       |   167 steps       |   166 steps       |
        |   Baseline        |   Mission A       |   Mission B       |
        |   (all subnets    |   (OZA critical)  |   (OZB critical)  |
        |    matter)        |                   |                   |
```

### Reward Tables (per event, per step)

Rewards are always zero or negative. The best achievable step reward is 0.
Three penalty types:
- **LWF** = Loss of Work Function (service failure)
- **ASF** = Allowed Service Failure (blocked traffic)
- **RIA** = Red Impact Action (active compromise)

#### Phase 0 (Steps 0-166): Baseline

```
 Subnet          | LWF  | ASF  | RIA
 ----------------+------+------+------
 OZA / OZB       |  -1  |  -1  |  -1
 RZA / RZB       |  -1  |  -3  |  -1
 contractor      |   0  |  -5  |  -5
 PAZ/admin/office|  -1  |  -1  |  -3
```

#### Phase 1 (Steps 167-333): Mission A -- OZA Critical

```
 Subnet          | LWF  | ASF  | RIA
 ----------------+------+------+------
 OZA             | -10  |   0  | -10       <-- ASF = 0 (blocking is FREE)
 RZA             |  -2  |  -1  |  -3
 OZB             |  -1  |  -1  |  -1
 contractor      |   0  |   0  |   0       <-- fully deprioritized
```

#### Phase 2 (Steps 334-499): Mission B -- OZB Critical

```
 Subnet          | LWF  | ASF  | RIA
 ----------------+------+------+------
 OZB             | -10  |   0  | -10       <-- ASF = 0 (blocking is FREE)
 RZB             |  -2  |  -1  |  -3
 OZA             |  -1  |  -1  |  -1
 contractor      |   0  |   0  |   0       <-- fully deprioritized
```

**Critical insight**: ASF = 0 for the active operational zone means that
BlockTrafficZone on the critical OZ incurs zero penalty. This makes preemptive
blocking of the active OZ the single most impactful defensive action.

### Reward Heatmap (ASCII)

Darker = more severe penalty. Dot = zero penalty.

```
                Phase 0          Phase 1          Phase 2
            LWF  ASF  RIA   LWF  ASF  RIA   LWF  ASF  RIA
 OZA        [#] [#]  [#]   [X]  [.]  [X]   [#]  [#]  [#]
 OZB        [#] [#]  [#]   [#]  [#]  [#]   [X]  [.]  [X]
 RZA        [#] [##] [#]   [##] [#]  [##]  ---- ---- ----
 RZB        [#] [##] [#]   ---- ---- ----  [##] [#]  [##]
 contractor [.] [XX] [XX]  [.]  [.]  [.]   [.]  [.]  [.]
 PAZ/adm/of [#] [#]  [##]  ---- ---- ----  ---- ---- ----

 Legend:  [.]  = 0    [#]  = -1 to -2    [##] = -3 to -5    [X]/[XX] = -10
          ---- = not scored in this phase
```

### Communications Policy per Phase

The comms_policy dictates which subnet-to-subnet links are administratively open
or closed. This changes at each phase transition.

**Phase 0**: Most paths open.
- RZA <-> OZA: open
- RZB <-> OZB: open
- Contractor connected to most subnets

**Phase 1**: OZA fully isolated.
- All 8 paths to/from OZA blocked by policy
- RZA <-> Contractor: removed
- RZB <-> OZB: still open

**Phase 2**: OZB fully isolated.
- All 8 paths to/from OZB blocked by policy
- RZB <-> Contractor: removed
- OZA <-> RZA: re-opened

---

## 3. Blue Agent Actions

Each blue agent may execute exactly one action per step. Multi-step actions lock
the agent for their full duration.

```
 Action            | Reward | Duration | Priority | Effect
                   | Cost   | (steps)  |          |
 ------------------+--------+----------+----------+------------------------------
 Sleep             |   0    |    1     |    0     | No-op
 Monitor           |   0    |    1     |   n/a    | Auto-runs end of every step.
                   |        |          |          | Collects alert events.
 Remove            |   0    |    3     |    5     | Kills user sessions + malware.
                   |        |          |          | Root sessions SURVIVE.
                   |        |          |          | 90% success rate.
 Restore           |  -1    |    5     |   n/a    | Full host reimage.
                   |        |          |          | Kills ALL sessions (incl root).
                   |        |          |          | Resets services to 100%.
                   |        |          |          | Wipes deployed decoys.
 DeployDecoy       |   0    |    2     |   n/a    | Creates fake service on host.
 BlockTrafficZone  |   0    |    1     |    1     | Blocks subnet-to-subnet link.
                   |        |          |          | PERSISTENT until reversed.
 AllowTrafficZone  |   0    |    1     |    1     | Reverses a block.
```

### Action Execution Timeline

```
 Step N begins
   |
   v
 [Priority 1] ControlTraffic actions execute (Block/Allow)
   |
   v
 [Priority 0] Sleep, Exploit (red), other default-priority actions
   |
   v
 [Priority 5] RemoveOtherSessions executes
   |
   v
 [End of step] Monitor auto-runs for all blue agents
   |
   v
 Observations generated, rewards calculated
   |
   v
 Step N+1 begins
```

**Key ordering consequences**:
- Blue Block in same step as Red Exploit: block wins (priority 1 < priority 0,
  lower number = executes first)
- Remove in same step as Exploit: exploit lands first (priority 0), then Remove
  cleans it up (priority 5) -- but the exploit damage is already done for that step

---

## 4. Decoy System

### Available Decoy Types

All applicable to Linux hosts only:

| Decoy Type       | Port | Service Type | Catches Exploit   |
|------------------|------|--------------|-------------------|
| DecoyApache      | 80   | webserver    | HTTPRFI           |
| DecoyTomcat      | 443  | webserver    | HTTPSRFI          |
| DecoyHarakaSMPT  | 25   | smtp         | HarakaRCE         |
| DecoyVsftpd      | 80   | webserver    | (port conflict w/ Apache) |

**Constraint**: Maximum 3 decoys per host. DecoyVsftpd and DecoyApache are mutually
exclusive (both use port 80). The optimal triple is Apache + Tomcat + Haraka.

### Exploit Selection and Decoy Effectiveness

Red uses `DefaultExploitActionSelector`, which:
1. Identifies all exploits applicable to the target host's services/ports
2. Removes the single highest-weight exploit
3. Randomly selects from the remaining candidates

With 3 decoys covering 3 of approximately 5 applicable ports, roughly 67% of
exploit selections will hit a decoy, resulting in an automatic failure. Empirically
measured effectiveness is approximately 75% when accounting for real service
interactions and port overlap.

### DiscoverDeception

Red can probe for decoys using `DiscoverDeception`:
- Duration: 2 steps
- True positive rate: 50%
- False positive rate: 10%

**Paradox**: Decoy-aware red states (UD = User+Decoy, RD = Root+Decoy) make red
MORE aggressive. In state UD, red attempts PrivilegeEscalate with 100% probability,
versus only 50% in state U (User without decoy knowledge). This means that
revealing decoys can accelerate the attack chain rather than deter it.

---

## 5. Observation Pipeline

### BlueFlatWrapperV2 Observation Vector Layout

```
 Index   | Width        | Content
 --------+--------------+------------------------------------------
 0       | 1 bit        | mission_phase (0, 1, or 2)

 --- Repeated per controlled subnet (alphabetical order) ---

 +0      | 9 bits       | Subnet identity one-hot
 +9      | 9 bits       | Blocked vector (1 = subnet i is blocked
         |              |   TO this subnet)
 +18     | 9 bits       | comms_policy (1 = SHOULD be blocked)
         |              |   NOTE: semantics are INVERTED
 +27     | n_hosts bits | proc_flags (1 = process creation event
         |              |   detected on host)
 +27+n   | n_hosts bits | conn_flags (1 = network connection event
         |              |   detected on host)

 --- After all subnets ---

 +0      | 32 bits      | Messages (4 peers x 8 bits each)
 +32     | n_total bits | malfile flags (V2 extension)
         |              | 1 = high-density unsigned file on host
```

### Full Vector Diagram (blue_agent_1 controlling OZA, 3 hosts)

```
 [phase] [subnet_onehot(9)] [blocked(9)] [comms(9)] [proc(3)] [conn(3)] [msgs(32)] [malfile(25)]
  ^         ^                  ^            ^          ^          ^          ^           ^
  |         |                  |            |          |          |          |           |
  1 bit     OZA identity      who is       policy     per-host   per-host   peer        V2:
            in 9-subnet       blocked      mask       process    network    messages    all hosts
            space             to OZA       (inverted) alerts     alerts                 in network
```

### Alert Pattern Signatures

These patterns in `(conn_flag, proc_flag, malfile)` tuples are diagnostic:

```
 conn  proc  malfile  |  Interpretation                    | Frequency
 ------+------+-------+------------------------------------+-----------
   1   |  1   |   1   | Standard exploit (red Exploit)     | ~95%
   1   |  0   |   1   | Silent exploit (process undetected)| ~5%
   1   |  0   |   0   | Decoy hit (exploit caught by fake) | varies
   0   |  0   |   1   | PrivilegeEscalate (no network sig) | post-exploit
   0   |  1   |   0   | Green false positive (benign)      | clears in 1 step
   0   |  0   |   0   | Clean / no activity                | baseline
```

**Key distinction**: Without the V2 malfile extension, PrivilegeEscalate (0,0,1)
is completely invisible -- it produces no conn_flag and no proc_flag. The malfile
bit is the only observable signal.

---

## 6. Green Agent Behavior (Empirical Measurements)

### Population and Activity

- 61 green agents active per episode
- Each green agent per step: 33% GreenLocalWork, 33% GreenAccessService, 33% Sleep

### False Positive Rates

| Event Type     | Source Action     | Rate per action | Total per episode |
|----------------|-------------------|-----------------|-------------------|
| Process FP     | GreenLocalWork    | 0.776%          | ~155.6            |
| Network FP     | GreenAccessService| 0.970%          | varies            |

- Process FPs appear on USER hosts only
- Network FPs appear on SERVER hosts only
- P(at least one FP in any given step) ~ 27%
- Green FPs clear after 1 step (no persistence)

### Phishing

- 1% of GreenLocalWork actions create a REAL red session on the host
- ~28.2 phishing events per episode (expected value)
- Phishing **bypasses BlockTrafficZone entirely** (local action, no network traversal)
- 67% of phishing occurs in Phase 0 (most green activity before service degradation)

### GreenLocalWork Failure Rates by Phase

```
 Phase 0:   4.55%  failure rate   (services mostly healthy)
 Phase 1:  23.97%  failure rate   (DegradeServices accumulating)
 Phase 2:  43.01%  failure rate   (services severely degraded)
                    ^
                    |
                    Each failure triggers an LWF penalty
```

The steep increase across phases reflects cumulative DegradeServices calls by red
agents reducing service reliability toward 0%.

### GreenAccessService and Traffic Blocking

GreenAccessService checks BOTH:
1. `comms_policy` (administrative policy for the phase)
2. `BlockTrafficZone` (blue agent blocks)

If either blocks the path, the service access fails. Failed access still generates
a `network_connection` event (contributing to observation noise / false positives).

---

## 7. Action Execution Order

All actions within a step are sorted by `action.priority` (ascending = executes first):

```
 Priority | Action                  | Notes
 ---------+-------------------------+----------------------------------
    1     | ControlTraffic          | Block/Allow execute FIRST
          | (BlockTrafficZone,      |
          |  AllowTrafficZone)      |
    0     | Sleep, Exploit,         | Default priority actions
          | DiscoverNetworkServices,|
          | other red/green actions |
    5     | RemoveOtherSessions     | Executes AFTER exploits
```

**Note on priority semantics**: Lower number = higher priority = executes first.
This is critical for understanding race conditions:

```
 Scenario 1: Blue Block vs Red Exploit (same step)
 ================================================================
 Step N: Blue issues BlockTrafficZone, Red issues Exploit
   -> Priority 1 (Block) executes first
   -> Priority 0 (Exploit) executes second, but traffic is now blocked
   -> Result: BLOCK WINS. Red exploit fails.

 Scenario 2: Blue Remove vs Red Exploit (same step)
 ================================================================
 Step N: Blue issues Remove, Red issues Exploit
   -> Priority 0 (Exploit) executes first, red gains session
   -> Priority 5 (Remove) executes second, cleans up session
   -> Result: EXPLOIT LANDS FIRST. Damage taken for this step,
      then Remove cleans up. Net: 1 step of penalty.
```

### Monitor Behavior

Monitor is NOT a queued action. It runs automatically at the end of every step
for all blue agents, regardless of what action they selected. Explicitly calling
Monitor is equivalent to Sleep (wastes the action slot).

---

## 8. Service Reliability and DegradeServices

### Mechanism

Each host's services have a `_percent_reliable` attribute starting at 100.

```
 DegradeServices (Red action):
   - Duration: 2 steps
   - Success rate: 100%
   - Requires: root access on host
   - Effect: _percent_reliable -= 20 (per call)
   - After 5 calls: reliability = 0%
```

### Reliability Check

When a green agent performs GreenLocalWork:
```python
if random.randint(0, 99) >= service._percent_reliable:
    # Service fails -> triggers LWF penalty
```

At 100% reliability: 0% failure rate
At 80% reliability: 20% failure rate
At 0% reliability: 100% failure rate

### Invisibility Problem

DegradeServices is **completely invisible** to blue agents:
- No `conn_flag` generated
- No `proc_flag` generated
- No `malfile` indicator
- No direct observation of service reliability values

The only evidence is indirect:
- Increasing service failure rates (observed via reward signal)
- Reward drops that cannot be attributed to known compromises
- GreenLocalWork failure rate climbing across phases

### Damage Attribution

Empirically, **83.3% of total cumulative reward loss** comes from LWF penalties
caused by degraded services, not from direct red impact actions. This makes
DegradeServices the primary damage mechanism in CC4, and the invisibility of this
action is the central challenge for blue-team agents.

### Restore as Counter

`Restore` is the only action that resets service reliability:
- Reimages the host completely
- Sets `_percent_reliable` back to 100
- Also kills ALL sessions (including root -- the only way to remove root access)
- Also wipes any deployed decoys on that host
- Duration: 5 steps
- Cost: -1 reward penalty per Restore

---

## 9. Strategic Implications

### Derived Principles

1. **Block early, block free**: During Phases 1 and 2, blocking traffic to/from
   the active OZ costs zero (ASF = 0). This should be done immediately at phase
   transitions.

2. **Reactive beats proactive**: Issuing Restore preemptively wastes 5 steps of
   agent time and incurs a -1 penalty. Since 90% of red sessions are user-level
   (removable via Remove), Restore should be reserved for confirmed root access
   or degraded services.

3. **Phishing is unblockable**: No network-level defense prevents phishing.
   The only counter is monitoring for the resulting sessions and removing them.

4. **Decoys are a double-edged sword**: While they block ~67-75% of exploits,
   red agents that discover them become more aggressive. Deploy-and-forget is
   acceptable; do not rely on decoys as a primary defense.

5. **Contractor is a trap**: In Phases 1 and 2, contractor penalties are zero.
   Blue resources spent defending contractor are wasted.

6. **DegradeServices dominates damage**: Since 83% of damage comes from degraded
   services, the blue agent's primary objective is preventing or reversing root
   access on high-value hosts where DegradeServices will be called.

---

## 10. Appendix: Numerical Constants

| Parameter                          | Value       |
|------------------------------------|-------------|
| Episode length                     | 500 steps   |
| Phase 0 duration                   | 167 steps   |
| Phase 1 duration                   | 167 steps   |
| Phase 2 duration                   | 166 steps   |
| Green agents                       | 61          |
| Red agents                         | 6           |
| Blue agents                        | 5           |
| Total subnets                      | 9           |
| Hosts per subnet (typical)         | 3           |
| Remove success rate                | 90%         |
| Remove duration                    | 3 steps     |
| Restore duration                   | 5 steps     |
| Restore cost                       | -1 reward   |
| DeployDecoy duration               | 2 steps     |
| Max decoys per host                | 3           |
| DegradeServices reliability drop   | 20 per call |
| DegradeServices duration           | 2 steps     |
| DiscoverDeception duration         | 2 steps     |
| DiscoverDeception true positive    | 50%         |
| DiscoverDeception false positive   | 10%         |
| Phishing rate                      | 1% of GreenLocalWork |
| Process FP rate                    | 0.776% of GreenLocalWork |
| Network FP rate                    | 0.970% of GreenAccessService |
| Green action distribution          | 33/33/33 (work/access/sleep) |
| Phase 0 service failure rate       | 4.55%       |
| Phase 1 service failure rate       | 23.97%      |
| Phase 2 service failure rate       | 43.01%      |
| LWF share of total damage          | 83.3%       |
| Phishing events per episode        | ~28.2       |
| Process FPs per episode            | ~155.6      |
| P(any FP in a step)                | ~27%        |
