No# EnterpriseHeuristicAgentV11a -- Architecture Reference

## Complete Technical Documentation for CAGE Challenge 4 Blue-Team Agent

**Version**: V11a (Preemptive OZ Blocking)
**File**: `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgentV11a.py`
**Performance**: -717.0 +/- 202.9 (100 episodes, seed 42)
**Improvement**: +11.9% over V10b (-814.0 +/- 247.7)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Performance Context](#2-performance-context)
3. [Network Topology and Agent Assignments](#3-network-topology-and-agent-assignments)
4. [Observation Vector Layout](#4-observation-vector-layout)
5. [Priority System Architecture](#5-priority-system-architecture)
6. [Alert Classification Engine](#6-alert-classification-engine)
7. [Flag Age Tracking and Green FP Filtering](#7-flag-age-tracking-and-green-fp-filtering)
8. [Threat Response: Restore-Only Strategy](#8-threat-response-restore-only-strategy)
9. [Preemptive OZ Blocking (V11a Innovation)](#9-preemptive-oz-blocking-v11a-innovation)
10. [Comms Policy Management](#10-comms-policy-management)
11. [Decoy System](#11-decoy-system)
12. [Messaging Protocol](#12-messaging-protocol)
13. [Host Priority Ordering](#13-host-priority-ordering)
14. [Phase-Aware Behavior](#14-phase-aware-behavior)
15. [Decision Flowchart](#15-decision-flowchart)
16. [Attack-Defense Timing Analysis](#16-attack-defense-timing-analysis)
17. [Known Limitations and Future Work](#17-known-limitations-and-future-work)
18. [Appendix: Constants and Configuration](#18-appendix-constants-and-configuration)

---

## 1. Executive Summary

EnterpriseHeuristicAgentV11a is a rule-based, priority-driven heuristic agent
designed to defend a 9-subnet enterprise network in CAGE Challenge 4 (CC4).
It is deployed as 5 independent agent instances, one per blue team member,
each controlling a subset of the network.

The agent operates on a strict priority cascade: at each step, it evaluates
conditions top-to-bottom and executes the first matching action. There is no
learning, no neural network, and no stochastic component. Despite this
simplicity, V11a outperforms both its predecessor V10b (+11.9%) and an
oracle agent with perfect information about red sessions (Oracle V3: -893.5).

### Core Design Principles

1. **Restore-only threat response.** The Remove action is never issued.
   PrivilegeEscalate completes in 2 steps; Remove takes 3 steps. Red reaches
   root before Remove finishes. Restore (5 steps, -1 reward) is the only
   reliable eviction mechanism.

2. **Preemptive blocking.** V11a blocks paths to operational zone subnets
   BEFORE mission phases begin, eliminating the vulnerability window that
   exists while agents are busy with higher-priority Restores.

3. **Decoy-first defense.** Deploying 3 decoys per host causes 75% of
   blind exploit attempts to fail. The first decoy provides 98.5% of all
   decoy value. Decoys prevent compromise rather than merely detecting it.

4. **Flag-age filtering.** Green agent false positives (proc_flag) clear
   within 1 step. By requiring flag_age >= 1 before acting on proc_flag,
   the agent eliminates ALL green FPs with zero false negatives.

---

## 2. Performance Context

```
Agent                     Mean Reward   Std Dev   Notes
----------------------    -----------   -------   -------------------------
SleepAgent (do nothing)   -30,579       --        Baseline: no defense
Oracle V3 (perfect info)     -893.5     --        Knows all red sessions
V10b (previous best)         -814.0     247.7     Restore-only + decoys
V11a (current best)          -717.0     202.9     + preemptive OZ blocking
Theoretical floor            ~-300      --        Unavoidable red damage
```

Key observations:
- V11a is ~97.7% better than doing nothing.
- V11a BEATS the perfect-information oracle, proving that heuristic
  structure (priority ordering, preemptive blocking) matters more than
  information completeness.
- The gap to theoretical floor (~-300) represents unavoidable red damage
  from PhishingEmail (bypasses blocks), green FP restoration costs, and
  structural timing constraints.

---

## 3. Network Topology and Agent Assignments

### 3.1 Subnet Topology

```
                         +------------------+
                         |    INTERNET       |
                         |  (internet_       |
                         |   subnet)         |
                         +--------+---------+
                                  |
                         +--------+---------+
                         |   CONTRACTOR     |
                         |  (contractor_    |
                         |   network_subnet)|
                         +--------+---------+
                                  |
              +-------------------+-------------------+
              |                                       |
    +---------+---------+                   +---------+---------+
    | PUBLIC ACCESS ZONE|                   |  OFFICE NETWORK   |
    | (public_access_   |                   | (office_network_  |
    |  zone_subnet)     |                   |  subnet)          |
    +---------+---------+                   +---------+---------+
              |                                       |
              |                             +---------+---------+
              |                             |  ADMIN NETWORK    |
              |                             | (admin_network_   |
              |                             |  subnet)          |
              |                             +---------+---------+
              |                                       |
    +---------+-------------------+-------------------+---------+
    |                             |                             |
    |         ZONE A              |          ZONE B             |
    |                             |                             |
    v                             |                             v
+---+-------------------+        |        +--------------------+---+
| RESTRICTED ZONE A     |        |        | RESTRICTED ZONE B      |
| (restricted_zone_a_   |        |        | (restricted_zone_b_    |
|  subnet)              |        |        |  subnet)               |
| ~6 hosts              |        |        | ~6 hosts               |
+-----------+-----------+        |        +-----------+------------+
            |                    |                    |
+-----------+-----------+        |        +-----------+------------+
| OPERATIONAL ZONE A    |        |        | OPERATIONAL ZONE B     |
| (operational_zone_a_  |        |        | (operational_zone_b_   |
|  subnet)              |        |        |  subnet)               |
| ~3 hosts              |        |        | ~3 hosts               |
| ** server_host_0:     |        |        | ** server_host_0:      |
|    OTService (Impact) |        |        |    OTService (Impact)  |
+-----------------------+        |        +------------------------+
```

### 3.2 Agent Assignments

```
+------------------+----------------------------+------------------+
|  Agent           |  Controlled Subnets        |  Primary Role    |
+------------------+----------------------------+------------------+
|  blue_agent_0    |  restricted_zone_a_subnet  |  Guard RZA       |
|  blue_agent_1    |  operational_zone_a_subnet |  Guard OZA       |
|  blue_agent_2    |  restricted_zone_b_subnet  |  Guard RZB       |
|  blue_agent_3    |  operational_zone_b_subnet |  Guard OZB       |
|  blue_agent_4    |  public_access_zone_subnet |  Guard HQ        |
|                  |  admin_network_subnet      |                  |
|                  |  office_network_subnet     |                  |
+------------------+----------------------------+------------------+
```

Each agent operates independently with its own observation vector, action
space, and internal state. Coordination occurs only through the 8-bit
inter-agent messaging system (Section 12).

### 3.3 Alphabetically Sorted Subnet Indices

The observation vector uses alphabetical ordering for all 9 subnets. This
ordering determines the positions of blocked/comms_policy bits:

```
Index   Subnet Name                    Abbreviation
-----   ---------------------------    ------------
  0     admin_network_subnet           ADMIN
  1     contractor_network_subnet      CONT
  2     internet_subnet                INET
  3     office_network_subnet          OFF
  4     operational_zone_a_subnet      OZA
  5     operational_zone_b_subnet      OZB
  6     public_access_zone_subnet      PAZ
  7     restricted_zone_a_subnet       RZA
  8     restricted_zone_b_subnet       RZB
```

---

## 4. Observation Vector Layout

The agent receives a flat numpy array from BlueFlatWrapperV2. The layout
is variable-length, depending on how many hosts exist in each controlled
subnet.

### 4.1 High-Level Structure

```
+-------+---------------------+---------------------+-----+----------+----------+
| Phase | Subnet Block 0      | Subnet Block 1      | ... | Messages | Malfiles |
| (1)   | (27 + 2*n_hosts_0)  | (27 + 2*n_hosts_1)  |     | (32)     | (M)      |
+-------+---------------------+---------------------+-----+----------+----------+
  obs[0]                                                     base      base+32

  base = 1 + sum(27 + 2*n_hosts_i)  for each controlled subnet i
  M    = sum(n_hosts_i)             total hosts across all controlled subnets
```

### 4.2 Per-Subnet Block Detail

Each controlled subnet contributes one block to the observation. Subnets
appear in alphabetical order. Within each block:

```
Offset        Size    Content
-----------   -----   ----------------------------------------
+0  to +8       9     Subnet one-hot encoding (9 subnets)
+9  to +17      9     blocked_subvector:
                        bit i = 1 if traffic FROM subnet i
                        TO this subnet is currently blocked
+18 to +26      9     comms_policy_subvector:
                        bit i = 1 if subnet i SHOULD BE
                        BLOCKED from reaching this subnet
                        (INVERTED: 1=block, 0=allow)
+27 to +27+n    n     process_flags:
                        bit j = 1 if host j has a suspicious
                        process event (green OR red)
+27+n to +27+2n n     connection_flags:
                        bit j = 1 if host j has a network
                        connection event (exploit attempt)

Total block size: 27 + 2*n_hosts
```

### 4.3 Full Observation Diagram

For an agent controlling 2 subnets with 6 and 3 hosts respectively:

```
Index:  0     1              36    37             65     66        97   98   106
        |     |               |     |              |      |         |    |     |
        v     v               v     v              v      v         v    v     v
       [P] [--Subnet0(39)--] [---Subnet1(30)---] [---Messages(32)---] [Malfile(9)]
        |   |                 |                   |                   |
        |   +-- one_hot(9)    +-- one_hot(9)      +-- 4x8-bit msgs   +-- 6+3 bits
        |   +-- blocked(9)   +-- blocked(9)      |    from peers     |   per host
        |   +-- comms(9)     +-- comms(9)        |                   |
        |   +-- proc(6)      +-- proc(3)         |                   |
        |   +-- conn(6)      +-- conn(3)         |                   |
        |                                        |                   |
        phase (0,1,2)                            base              malfile_start
```

### 4.4 Malfile Detection

BlueFlatWrapperV2 appends one bit per host at the end of the observation.
The malfile bit is 1.0 if the host holds any file with `density >= 0.9`
and `signed == False`. This fingerprint is left by:

- **ExploitRemoteService**: Always drops cmd.exe/cmd.sh (density=0.9, unsigned)
- **PrivilegeEscalate**: Always drops escalate.exe/escalate.sh (same signature)

Green agent actions NEVER drop high-density files, giving the malfile flag
a **zero false positive rate**.

The agent auto-detects whether BlueFlatWrapperV2 is in use by checking if
`len(obs) == expected_base_len + n_malfile_hosts`. If the standard
BlueFlatWrapper is used, the malfile section is absent and the agent falls
back to proc/conn-only detection.

---

## 5. Priority System Architecture

The agent evaluates a strict priority cascade on every step. The first
matching condition triggers the corresponding action. If no condition
matches, the agent sleeps.

### 5.1 Priority Hierarchy Diagram

```
+============================================================+
|                    get_action() called                       |
+============================================================+
        |
        v
+-------+--------------------------------------------------+
| P1:   RESTORE on confirmed red                           |
|       conn_flag AND (malfile OR proc_flag) = real exploit |
|       Iterate hosts by descending priority               |
+-------+--------------------------------------------------+
        |  no match
        v
+-------+--------------------------------------------------+
| P1b:  RESTORE on conn-only without decoy coverage       |
|       conn_flag AND NOT malfile AND NOT proc_flag         |
|       Skip if decoy deployed (likely decoy hit)          |
|       Unless upstream reports DECOYS_BYPASSED            |
+-------+--------------------------------------------------+
        |  no match
        v
+-------+--------------------------------------------------+
| P1c:  RESTORE on pure malfile (PrivEsc signature)        |
|       malfile AND NOT conn AND NOT proc                   |
|       = PrivilegeEscalate completed (root session)       |
+-------+--------------------------------------------------+
        |  no match
        v
+-------+--------------------------------------------------+
| P_BLOCK_OZ: [V11a] Preemptive/active OZ blocking        |
|  A) Active phase: block ALL open paths to active OZ      |
|     Phase 1 -> block everything to OZA (ASF=0)           |
|     Phase 2 -> block everything to OZB (ASF=0)           |
|  B) Preemptive: 10 steps before transition, block        |
|     RZA->OZA (before Phase 1) or RZB->OZB (before P2)   |
+-------+--------------------------------------------------+
        |  no match
        v
+-------+--------------------------------------------------+
| P2:   ALLOW paths per comms_policy                       |
|       Fix stale blocks: comms_policy says allow but      |
|       path is currently blocked -> AllowTrafficZone      |
|       Also cleans up preemptive blocks no longer needed  |
+-------+--------------------------------------------------+
        |  no match
        v
+-------+--------------------------------------------------+
| P3:   BLOCK paths per comms_policy                       |
|       Enforce missing blocks: comms_policy says block    |
|       but path is currently open -> BlockTrafficZone     |
|       Sorted by pair_priority (descending)               |
+-------+--------------------------------------------------+
        |  no match
        v
+-------+--------------------------------------------------+
| P4:   RESTORE on proc_flag with flag_age threshold       |
|       Default threshold = 1 (filters green FPs)          |
|       T3 escalation (upstream 3+ red): threshold = 0     |
|       Active OZ server_host_0: threshold = 0             |
+-------+--------------------------------------------------+
        |  no match
        v
+-------+--------------------------------------------------+
| P6:   RE-DEPLOY decoys after Restore                     |
|       Restore wipes decoys; re-arm in priority order     |
|       Only after Restore duration has elapsed            |
+-------+--------------------------------------------------+
        |  no match
        v
+-------+--------------------------------------------------+
| P7:   DEPLOY decoys (initial setup)                      |
|       Up to MAX_DECOYS=3 per host                        |
|       OZ servers first, then RZ, then others             |
+-------+--------------------------------------------------+
        |  no match
        v
+-------+--------------------------------------------------+
| SLEEP: No action needed. Return Sleep action.            |
+-------+--------------------------------------------------+
```

### 5.2 Priority Level Rationale

**Why P1 > P_BLOCK_OZ > P2?**

- Confirmed red sessions (P1/P1b/P1c) cause immediate ongoing damage and
  must be addressed before anything else.
- Blocking open paths to the active OZ (P_BLOCK_OZ) prevents NEW red
  intrusions at zero ASF cost, making it more valuable than fixing stale
  Allow states (P2).
- P2 Allow must run below P_BLOCK_OZ because during active phases, the
  OZ lockdown takes priority over comms_policy compliance.

**Why no P5 in V11a?**

V11a uses a Restore-only strategy. The Remove action is never issued. P5
(Restore persistent proc_flags after Remove) existed in earlier versions
that used Remove. In V11a, P4 handles all proc_flag responses directly
with Restore. The P5 numbering gap is retained for historical traceability.

---

## 6. Alert Classification Engine

The agent classifies host-level alerts into mutually exclusive categories
that drive the priority cascade.

### 6.1 Alert Categories

```
+-------+----------+----------+----------------------------------+
| conn  | proc     | malfile  | Classification                   |
+-------+----------+----------+----------------------------------+
|   1   |   1      |   1      | Confirmed exploit + events       |
|       |          |          | -> P1 RESTORE                    |
+-------+----------+----------+----------------------------------+
|   1   |   0      |   1      | Silent exploit (5% case)         |
|       |          |          | cmd.exe dropped, no proc event   |
|       |          |          | -> P1 RESTORE                    |
+-------+----------+----------+----------------------------------+
|   1   |   1      |   0      | Exploit with proc but no file    |
|       |          |          | (should not occur with V2)       |
|       |          |          | -> P1 RESTORE                    |
+-------+----------+----------+----------------------------------+
|   1   |   0      |   0      | Connection only, no artifacts    |
|       |          |          | If decoy deployed: likely decoy  |
|       |          |          |   hit -> SKIP (save 5 steps)     |
|       |          |          | If no decoy: unknown -> P1b      |
|       |          |          |   RESTORE for safety             |
+-------+----------+----------+----------------------------------+
|   0   |   0      |   1      | Pure malfile = PrivilegeEscalate |
|       |          |          | Root session, invisible to other |
|       |          |          | signals -> P1c RESTORE           |
+-------+----------+----------+----------------------------------+
|   0   |   1      |   1      | Proc + malfile, no connection    |
|       |          |          | -> Enters real_red_hosts set     |
|       |          |          | -> P4 RESTORE (flag_age check)   |
+-------+----------+----------+----------------------------------+
|   0   |   1      |   0      | Process flag only                |
|       |          |          | 33-54% are green FPs             |
|       |          |          | -> P4 with flag_age >= threshold |
+-------+----------+----------+----------------------------------+
|   0   |   0      |   0      | Clean host. No action needed.    |
+-------+----------+----------+----------------------------------+
```

### 6.2 Derived Alert Sets

The agent computes two derived sets used throughout the priority logic:

**`real_red_hosts`**: Union of:
- Hosts with `conn_flag AND (malfile OR proc_flag)`
- All hosts with `malfile_flag` (any malfile = real red)
- All hosts with `proc_flag` (conservative; filtered by flag_age later)

**`root_indicators`**: Hosts where root sessions are confirmed:
- `malfile AND NOT conn AND NOT proc` = PrivilegeEscalate completed
- `proc_flag` persisting after Remove completion = root survived Remove

---

## 7. Flag Age Tracking and Green FP Filtering

### 7.1 The Green FP Problem

Green agents (legitimate users) occasionally trigger `proc_flag` events
through normal activity. These false positives occur at a rate of 33-54%
depending on the subnet. Without filtering, the agent would waste 5 steps
Restoring clean hosts.

### 7.2 Flag Age Mechanism

The agent tracks when each proc_flag first appeared:

```python
_proc_flagged_step[hostname] = step_when_flag_first_seen
flag_age = current_step - _proc_flagged_step[hostname]
```

**Key insight**: Green FPs clear within 1 step. A proc_flag that persists
for >= 1 step is always a real red indicator.

```
Step N  :  proc_flag appears on host X
Step N+1:  Green FP:  proc_flag GONE  -> flag_age = 1, but flag deleted
           Real red:  proc_flag STILL -> flag_age = 1, RESTORE triggered

Timeline for Green FP:
  Step N  : flag appears -> _proc_flagged_step[X] = N
  Step N+1: flag gone    -> delete _proc_flagged_step[X]
  Result: never reaches threshold

Timeline for Real Red:
  Step N  : flag appears -> _proc_flagged_step[X] = N
  Step N+1: flag persists, flag_age = 1 >= threshold -> RESTORE
```

### 7.3 Threshold Variations

```
Context                                 Threshold   Rationale
--------------------------------------  ---------   ---------------------------
Default                                     1       Filters all green FPs
T3 escalation (upstream 3+ red hosts)       0       Emergency: skip FP filter
Active OZ server_host_0                     0       RIA=-10 >> FP Restore cost
T2 escalation (any_root or threat>=2)       1       Standard but confirmed
```

The `_is_active_oz_server()` function identifies the critical OTService
host (`server_host_0` in the active operational zone) where the cost of
missing a real threat (RIA = -10 per step) far exceeds the cost of a false
Restore (-1 reward + 5 steps downtime).

---

## 8. Threat Response: Restore-Only Strategy

### 8.1 Why Remove is Never Used in V11a

```
Red Attack: ExploitRemoteService -> PrivilegeEscalate
            (user session)          (root session)

            +-- 4 steps --+-- 2 steps --+
            |  Exploit     | PrivEsc     |
            +--------------+-------------+
Red:        |====EXPLOIT===|==PRIVESC==|
            t              t+4          t+6

Blue Remove attempt:
            |===REMOVE (3 steps)==|
            t                    t+3
            Result: clears user session at t+3
            But PrivEsc started at t+4, completes at t+6
            Root session survives Remove -> WASTED 3 steps

Blue Restore:
            |=====RESTORE (5 steps)======|
            t                            t+5
            Result: full reimage, clears EVERYTHING
            Cost: -1 reward, 5 steps busy, decoys wiped
```

Even if Remove is issued the instant ExploitRemoteService completes (best
case), PrivilegeEscalate has already begun or will begin before Remove
takes effect. Root sessions survive Remove entirely. Only Restore, which
re-images the entire host, reliably evicts red.

### 8.2 Restore Mechanics

| Property           | Value                                        |
|--------------------|----------------------------------------------|
| Duration           | 5 steps                                      |
| Reward cost        | -1 per Restore issued                        |
| Clears             | ALL sessions (red + green), all malware       |
| Side effects       | Wipes deployed decoys, resets service state   |
| Busy guard         | `_busy(hostname)` prevents duplicate actions  |
| Bookkeeping        | `_restore_at[hostname] = current_step`        |
|                    | `_decoy_deployed[hostname]` cleared           |

### 8.3 The _busy() Guard

```python
def _busy(self, hostname: str) -> bool:
    if self._step <= self._restore_at.get(hostname, -1) + RESTORE_DUR - 1:
        return True
    return False
```

V11a only tracks Restore busy state (no Remove). This prevents issuing
redundant Restore actions on a host that is already mid-Restore.

---

## 9. Preemptive OZ Blocking (V11a Innovation)

This is the key architectural change in V11a that yields +11.9% improvement
over V10b. It addresses a critical vulnerability window at phase transitions.

### 9.1 The Problem

At phase transitions, the comms_policy changes to require new blocks. In
V10b, these blocks were handled by P3 (Block per comms_policy), which runs
at low priority. If the agent was busy Restoring a host (P1-P1c), the
critical RZ->OZ block could be delayed by 5-10 steps. During this window,
red could freely move from RZ to OZ.

```
V10b Vulnerability Window:

Step 166: Phase 0. RZA->OZA is OPEN (allowed by Phase 0 comms_policy)
Step 167: Phase 1 begins. comms_policy now says BLOCK RZA->OZA
          But agent is mid-Restore on another host (busy until step 172)
Steps 167-172: RZA->OZA is OPEN while agent cannot act
          Red exploits this window to reach OZA (LWF = -10/step)
          Cost: 6 steps * -10 = -60 reward LOST

V11a Solution:

Step 157: Phase 0. Agent preemptively blocks RZA->OZA (10 steps early)
          Cost: 10 steps * ASF=-1 = -10 (blocking during Phase 0)
Step 167: Phase 1 begins. RZA->OZA is ALREADY BLOCKED
          Zero-delay transition. No vulnerability window.
          Net benefit: -10 (preemptive cost) vs -60 (window cost) = +50
```

### 9.2 Three Blocking Scenarios

```
+=====================================================================+
|                        P_BLOCK_OZ Logic                             |
+=====================================================================+
|                                                                     |
|  Scenario A: ACTIVE PHASE LOCKDOWN                                  |
|  ---------------------------------                                  |
|  During Phase 1: block ALL 8 open paths to OZA                      |
|  During Phase 2: block ALL 8 open paths to OZB                      |
|  Rationale: ASF=0 for active OZ, so blocking is FREE                |
|  Iterates all source subnets, blocks any unblocked path             |
|                                                                     |
|  Scenario B: PREEMPTIVE BLOCKING                                    |
|  --------------------------------                                   |
|  Phase 0, step >= 157: block RZA->OZA (Phase 1 will need it)       |
|  Phase 1, step >= 324: block RZB->OZB (Phase 2 will need it)       |
|  Cost: ~10 steps * ASF=-1 = -10 total (negligible)                  |
|  Benefit: eliminates 5-10 step vulnerability window                 |
|                                                                     |
|  Scenario C: CLEANUP (handled by P2 Allow)                          |
|  ------------------------------------------                        |
|  When phase changes, old preemptive blocks may need removal         |
|  P2 Allow catches these: comms_policy says allow + path blocked     |
|  -> AllowTrafficZone issued, preemptive_blocks.discard(pair)        |
|                                                                     |
+=====================================================================+
```

### 9.3 Preemptive Block Tracking

The agent maintains `_preemptive_blocks: set[tuple[str, str]]` to track
which blocks it issued preemptively (not mandated by comms_policy at the
time of issuance). When P2 Allow detects that comms_policy now says a
preemptively-blocked path should be open, it issues AllowTrafficZone and
removes the pair from the tracking set.

### 9.4 Preemptive Block Targets

```python
# Before Phase 1: block the one path Phase 0 allows that Phase 1 blocks
_PREEMPTIVE_PHASE1 = [
    ("restricted_zone_a_subnet", "operational_zone_a_subnet"),
]

# Before Phase 2: block the one path Phase 1 allows that Phase 2 blocks
_PREEMPTIVE_PHASE2 = [
    ("restricted_zone_b_subnet", "operational_zone_b_subnet"),
]

PREEMPTIVE_WINDOW = 10  # steps before transition to start blocking
PHASE_1_START = 167     # step where Phase 0 -> Phase 1
PHASE_2_START = 334     # step where Phase 1 -> Phase 2
```

### 9.5 Why P_BLOCK_OZ Sits Above P2

```
Priority ordering around the innovation:

  P1c  (Restore on PrivEsc)
    |
    v
  P_BLOCK_OZ  <-- V11a insertion point
    |
    v
  P2   (Allow per comms_policy)
    |
    v
  P3   (Block per comms_policy)
```

If P_BLOCK_OZ were below P2, the agent would first try to Allow paths
(fixing stale blocks from the previous phase) before blocking the active
OZ. This would leave the OZ exposed during the Allow operations. By
placing P_BLOCK_OZ above P2, the active OZ is locked down first, and
stale Allow corrections happen only after security is ensured.

---

## 10. Comms Policy Management

### 10.1 Comms Policy Encoding

The comms_policy_subvector has a **counterintuitive encoding**:

```
comms_policy[i] = 1  -->  Subnet i SHOULD BE BLOCKED from this subnet
comms_policy[i] = 0  -->  Subnet i is ALLOWED (do NOT block)
```

This is INVERTED from the natural reading where 1 might mean "allowed."
The code checks `should_block[pair] = bool(comms_policy_vec[i])`.

### 10.2 Phase-Dependent Comms Policy

```
Phase 0 (Preplanning):
  - Mostly open network. Minimal blocking required.
  - contractor/internet may need blocking to some subnets.

Phase 1 (Mission A):
  - RZA isolated: block contractor, internet, RZB, and OZA from RZA.
  - OZA fully locked down: ALL 8 source subnets blocked.
  - RZB and OZB remain relatively open.

Phase 2 (Mission B):
  - RZB isolated: block contractor, internet, RZA, and OZB from RZB.
  - OZB fully locked down: ALL 8 source subnets blocked.
  - RZA and OZA revert to less restricted state.
```

### 10.3 P2 Allow: Fix Stale Blocks

After a phase transition, blocks from the previous phase may violate the
new comms_policy. P2 iterates all pairs sorted by ascending pair_priority
(lowest priority paths fixed first) and issues AllowTrafficZone where:

```
should_block[pair] == False  AND  blocked_now[pair] == True
```

### 10.4 P3 Block: Enforce Required Blocks

P3 iterates all pairs sorted by descending pair_priority (highest priority
paths blocked first) and issues BlockTrafficZone where:

```
should_block[pair] == True  AND  blocked_now[pair] == False
```

### 10.5 Pair Priority

The `_pair_priority()` function determines which source subnets are most
important to block first:

```
Phase 1:
  restricted_zone_a_subnet  -> 250  (direct RZ->OZ attack path)
  contractor/internet       -> 200  (red entry points)
  all others                ->  10

Phase 2:
  restricted_zone_b_subnet  -> 250  (direct RZ->OZ attack path)
  contractor/internet       -> 200  (red entry points)
  restricted_zone_a_subnet  ->  50  (secondary path)
  all others                ->  10

Phase 0:
  contractor/internet       -> 200  (red entry points)
  all others                ->  10
```

---

## 11. Decoy System

### 11.1 Decoy Mechanics

DeployDecoy creates a fake vulnerable process (Apache port 80, Tomcat
port 443, or Haraka port 25) on a host. When red's ExploitRemoteService
randomly targets a decoy instead of a real service:

```
Red selects target PID randomly from all services on the host.
With 3 decoys + 1 legitimate service = 4 total services:
  P(hit decoy) = 3/4 = 75%
  P(hit real)  = 1/4 = 25%

When exploit hits decoy:
  - Exploit FAILS silently (no session created)
  - network_connection event IS logged (blue sees conn_flag)
  - Red wasted 4 steps (exploit duration) with zero progress

When red uses DiscoverDeception first:
  - Duration: 2 steps
  - 50% true positive rate: correctly identifies decoy PID
  - 10% false positive rate: flags legitimate PID as decoy
  - Result: intelligence only; does NOT remove the decoy
```

### 11.2 Decoy Value Analysis

```
Decoy Count   Exploit Failure Rate   Marginal Value
-----------   --------------------   --------------------------
    0                0%              --
    1               50%              +98.5% of total decoy value
    2               67%              Small marginal gain
    3               75%              Diminishing returns

First decoy: +1176.8 cumulative reward improvement (from ablation)
```

The massive value of the first decoy comes from the PREVENTION mechanism:
75% fewer successful exploits means 75% fewer Restores needed, each of
which saves 5 steps of downtime plus avoids the cascade of re-deploying
wiped decoys.

### 11.3 Deployment Priority

Decoys are deployed in priority order determined by `_deploy_priority()`:

```
Priority   Host Pattern                        Rationale
--------   ---------------------------------   ---------------------------
   0       OZB server_host_0                   Impact target (Phase 2)
   1       OZA server_host_0                   Impact target (Phase 1)
   2       OZB other servers                   OZ servers critical
   3       OZA other servers                   OZ servers critical
   4       RZB servers                         Red transit path
   5       RZA servers                         Red transit path
   6       Other server_host_0                 Servers > user hosts
   7       Other servers                       Generic server coverage
  8-11     OZ/RZ user hosts                    Lower priority
  20       All remaining hosts                 Lowest priority
```

### 11.4 Decoy Lifecycle

```
Initial deployment (P7):
  Step 0+: Deploy up to 3 decoys per host in priority order
  Each DeployDecoy takes 2 steps
  3 decoys = 6 steps per host

After Restore (P6):
  Restore wipes ALL decoys (host re-imaged to original state)
  P6 detects: restore_at[host] exists AND step >= restore_at + RESTORE_DUR
  Re-deploys decoys in same priority order as initial deployment

Busy guard:
  _busy(hostname) prevents deploying on a host mid-Restore
  (the decoy would be immediately wiped, wasting the action)
```

### 11.5 Decoy-Hit Detection

With BlueFlatWrapperV2, the agent can distinguish decoy hits from real
exploits:

```
Observation signals:   conn=1, malfile=0, proc=0, decoy_deployed > 0
Classification:        Decoy hit (exploit failed, no session created)
Action:                SKIP Restore (save 5 steps, decoy is protecting)

Exception:             If upstream reports DECOYS_BYPASSED, even a
                       conn-only hit may be a real exploit (red used
                       DiscoverDeception to identify real PIDs).
                       -> P1b RESTORE for safety.
```

---

## 12. Messaging Protocol

### 12.1 Protocol Overview (v9)

Each agent sends an 8-bit message per step. Each agent receives 4
messages (from the 4 other agents), totaling 32 bits in the observation.

**Empirically proven zero-value**: Ablation testing (p=0.80) shows the
messaging system has no statistically significant effect on performance.
The only marginally live code path (T3 escalation when `upstream_red_count
>= 3`) fires approximately 0.06% of the time. The messaging system is
retained for compatibility but is effectively dead code.

### 12.2 Message Format

```
Bit    Field               Encoding
----   -----------------   ------------------------------------
0-1    THREAT_LEVEL        2-bit: 0=clean, 1=decoy_hit,
                                  2=user_session, 3=root_session
2-3    OPEN_PATHS          2-bit: count of unblocked required
                                  comms paths (0-3, 3=three+)
4-5    RED_HOST_COUNT      2-bit: count of hosts with confirmed
                                  red presence (0-3, 3=three+)
6      DECOYS_BYPASSED     1-bit: red has PID knowledge (saw
                                  decoy hit then real exploit)
7      RESTORING           1-bit: at least one Restore in
                                  progress in sender's zone
```

### 12.3 Message Slot Layout

Messages are ordered by sender agent index (ascending, excluding self):

```
For blue_agent_1 (own_idx=1), peer slots are:
  Slot 0: blue_agent_0's message (bits 0-7)
  Slot 1: blue_agent_2's message (bits 8-15)
  Slot 2: blue_agent_3's message (bits 16-23)
  Slot 3: blue_agent_4's message (bits 24-31)

Known bug: When agents are network-isolated (e.g., OZA in Phase 1),
their message slots are zero-padded. The slot-to-agent mapping assumes
all agents are present, which can cause misalignment if the wrapper
re-orders or drops slots for isolated agents.
```

### 12.4 Message Reception and Escalation Tiers

```
Tier    Condition                            Effect
----    -----------------------------------  ---------------------------
T3      upstream_red_count >= 3              threshold = 0 in P4
                                             (immediate Restore on any
                                             proc_flag, skip FP filter)

T2      any_root OR upstream_threat >= 2     threshold = 1 in P4
                                             (standard with confirmation)

T2c     upstream_threat >= 1                 threshold = 1 in P4
        AND upstream_open_paths > 0          (compound condition)
        AND upstream_decoys_bypassed

P1b     upstream_decoys_bypassed             Don't skip Restore on
                                             conn-only even with decoys
```

### 12.5 Upstream Agent Mapping

```python
_UPSTREAM = {
    (1, 1): 0,   # Phase 1: RZA agent (0) is upstream of OZA agent (1)
    (2, 3): 2,   # Phase 2: RZB agent (2) is upstream of OZB agent (3)
}
```

"Upstream" means red must compromise that zone before reaching the
downstream zone. In Phase 1, red enters via contractor, pivots through
RZA, then reaches OZA. So agent_0 (RZA) is upstream of agent_1 (OZA).

---

## 13. Host Priority Ordering

### 13.1 Priority Scores by Phase

The `_host_priority()` function returns an integer score used to sort
hosts for processing within each priority level. Higher score = processed
first.

```
                      Phase 0    Phase 1    Phase 2
                      -------    -------    -------
OZA hosts                40       100         40
OZB hosts                40        40        100
RZA hosts                30        70         20*
RZB hosts                30        20*        70
HQ hosts (admin/         50        50         50
  office/PAZ)
Other hosts              20        20         20

* Falls through to default (20) unless matched by HQ pattern
```

### 13.2 Priority Rationale

During Phase 1, OZA hosts are mission-critical (LWF = -10 per step per
degraded host), so they receive the highest priority (100). RZA hosts are
elevated (70) because they are the gateway to OZA and RIA = -3. Phase 2
mirrors this for the B zones.

HQ hosts maintain a constant priority of 50 across all phases because they
are always accessible to red and have moderate impact scores.

### 13.3 Sorting Functions

```python
_sorted_by_priority(hosts, phase)  # returns list sorted by descending priority
_top_host(hosts, phase)            # returns single highest-priority host
```

These are used throughout the priority cascade to ensure the most critical
hosts are processed first when multiple hosts have alerts.

---

## 14. Phase-Aware Behavior

### 14.1 Phase Timeline

```
Step:   0         100        157  167        250        324  334        450  499
        |          |          |    |          |          |    |          |    |
        v          v          v    v          v          v    v          v    v
Phase:  |<------- Phase 0 ------->|<------- Phase 1 ------->|<-- Phase 2 -->|
        |     Preplanning         |     Mission A            |  Mission B   |
        |                         |                          |              |
Events: |  Deploy decoys          |  OZA critical            | OZB critical |
        |  Enforce comms_policy   |  LWF=-10 for OZA         | LWF=-10 OZB |
        |  Respond to alerts      |  ASF=0 for OZA           | ASF=0 OZB   |
        |                         |  RZA elevated (RIA=-3)   | RZB (RIA=-3)|
        |                    [Preemptive]                [Preemptive]       |
        |                    block RZA->OZA              block RZB->OZB    |
        |                    starts at 157               starts at 324     |
        |                         |                          |              |
        +-------------------------+--------------------------+--------------+

Key Scoring Parameters:
  LWF  = Loss of Workforce        (degraded service on OZ host)
  ASF  = Availability of Services (cost of blocking traffic)
  RIA  = Red Impact Assessment    (red session on host)
```

### 14.2 Phase 0: Preplanning (Steps 0-166)

- Deploy decoys on all hosts in priority order (P7)
- Enforce comms_policy blocks (P3) and allows (P2)
- Respond to red alerts (P1/P1b/P1c/P4)
- No preemptive OZ blocking (except in the last 10 steps)
- Host priorities: OZ=40, RZ=30, HQ=50

### 14.3 Phase 1: Mission A (Steps 167-333)

- OZA is mission-critical: LWF=-10/step, ASF=0 (blocking is free)
- ALL 8 paths to OZA should be blocked (P_BLOCK_OZ Scenario A)
- RZA elevated to priority 70 (gateway to OZA, RIA=-3)
- OZA server_host_0 gets threshold=0 for proc_flag (P4)
- Preemptive blocking of RZB->OZB starts at step 324 (Scenario B)

### 14.4 Phase 2: Mission B (Steps 334-499)

- OZB is mission-critical: LWF=-10/step, ASF=0 (blocking is free)
- ALL 8 paths to OZB should be blocked (P_BLOCK_OZ Scenario A)
- RZB elevated to priority 70 (gateway to OZB, RIA=-3)
- OZB server_host_0 gets threshold=0 for proc_flag (P4)

---

## 15. Decision Flowchart

### 15.1 Main get_action() Loop

```
                    +---------------------------+
                    |  get_action(obs, mask)     |
                    +---------------------------+
                                |
                    +-----------v-----------+
                    |  Increment _step      |
                    |  Parse observation    |
                    |  Detect malfile sect  |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |  For each subnet:     |
                    |    Extract blocked,   |
                    |    comms_policy,      |
                    |    proc, conn, mal    |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |  Update flag_age      |
                    |  tracker              |
                    |  Compute derived sets |
                    |  Read peer messages   |
                    |  Track decoy hits     |
                    |  Build outbound msg   |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |  P1: conn + (mal|proc)|-----> RESTORE
                    +-----------+-----------+
                                |  no
                    +-----------v-----------+
                    |  P1b: conn-only,      |
                    |  no decoy coverage    |-----> RESTORE
                    +-----------+-----------+
                                |  no
                    +-----------v-----------+
                    |  P1c: pure malfile    |-----> RESTORE
                    +-----------+-----------+
                                |  no
                    +-----------v-----------+
                    |  P_BLOCK_OZ:          |
                    |  Active OZ lockdown?  |-----> BLOCK
                    |  Preemptive window?   |-----> BLOCK
                    +-----------+-----------+
                                |  no
                    +-----------v-----------+
                    |  P2: stale block      |
                    |  violates comms_pol?  |-----> ALLOW
                    +-----------+-----------+
                                |  no
                    +-----------v-----------+
                    |  P3: missing block    |
                    |  per comms_policy?    |-----> BLOCK
                    +-----------+-----------+
                                |  no
                    +-----------v-----------+
                    |  P4: proc_flag with   |
                    |  flag_age >= threshold|-----> RESTORE
                    +-----------+-----------+
                                |  no
                    +-----------v-----------+
                    |  P6: decoys wiped by  |
                    |  completed Restore?   |-----> DEPLOY DECOY
                    +-----------+-----------+
                                |  no
                    +-----------v-----------+
                    |  P7: host needs more  |
                    |  decoys (< MAX)?      |-----> DEPLOY DECOY
                    +-----------+-----------+
                                |  no
                    +-----------v-----------+
                    |  SLEEP                |
                    +-----------+-----------+
```

### 15.2 Action Validity Check

Every candidate action passes through `_valid(idx, mask)` before execution.
The action mask (provided by the environment) encodes which actions are
currently legal. Invalid actions are skipped, and the agent falls through
to the next priority level.

---

## 16. Attack-Defense Timing Analysis

### 16.1 Red Attack Chain

```
Red's attack sequence for a single host:

Step    Action                 Duration   Result
------  ---------------------  --------   --------------------------------
1-4     ExploitRemoteService   4 steps    User-level session (if not decoy)
                                          Drops cmd.exe (malfile=1)
                                          conn_flag=1, proc_flag=1 (95%)
                                          conn_flag=1, proc_flag=0 (5%)
5-6     PrivilegeEscalate      2 steps    Root-level session
                                          Drops escalate.exe (malfile=1)
                                          NO conn or proc events
7+      Impact / lateral       ongoing    DegradeServices, spread to OZ
```

### 16.2 Blue Detection Windows

```
                      Red Timeline
                      ============
              t     t+1    t+2    t+3    t+4    t+5    t+6    t+7
              |------|------|------|------|------|------|------|
              |<-- ExploitRemoteService -->|<- PrivEsc ->|
              |                           |              |
Blue sees:    |                           |              |
  conn_flag:  ......1.......................              |
  proc_flag:  ......1 (95%) or 0 (5%)......              |
  malfile:    ......1.......................1..............|
              |                           |              |
Detection     |<--- WINDOW 1 ----------->|              |
windows:      |  conn+malfile+proc (P1)  |              |
              |  or conn+malfile (P1)    |              |
              |                          |<-- WINDOW 2 ->|
              |                          | pure malfile  |
              |                          | (P1c)         |
              |                          |               |
Best case     |  Restore issued at t+1   |               |
response:     |  Completes at t+6        |               |
              |  Host clean by t+6       |               |
```

### 16.3 Defense Timing Budget

```
Episode length: 500 steps
Phase 0: 167 steps (decoy deployment + initial defense)
Phase 1: 167 steps (defend OZA)
Phase 2: 166 steps (defend OZB)

Decoy deployment: 3 decoys * 2 steps = 6 steps per host
  For ~6 hosts: 36 steps of Phase 0 spent deploying

Restore: 5 steps per host
  During Restore, host is unavailable (green services also down)
  After Restore, 6 more steps to re-deploy 3 decoys

Total recovery cycle per host: 5 (Restore) + 6 (re-deploy) = 11 steps
  During 167-step phase: max ~15 Restore cycles per phase
  In practice: 2-5 Restores per subnet per phase typical
```

### 16.4 Red vs Blue Action Duration Comparison

```
Action                    Duration    Side Effects
------------------------  --------    ---------------------------------
ExploitRemoteService       4 steps    Creates user session + malware
PrivilegeEscalate          2 steps    Creates root session + malware
DiscoverDeception          2 steps    Intelligence only (50% TP rate)
Impact (DegradeServices)   1 step     Permanent service degradation
PhishingEmail             variable    Bypasses all BlockTrafficZone

Restore                    5 steps    Full re-image, clears everything
Remove                     3 steps    Clears user sessions only (UNUSED)
BlockTrafficZone           1 step     Blocks src->dst subnet traffic
AllowTrafficZone           1 step     Unblocks src->dst subnet traffic
DeployDecoy                2 steps    Creates fake service on host
Sleep                      0 steps    No operation
```

---

## 17. Known Limitations and Future Work

### 17.1 Current Limitations

1. **PhishingEmail bypass.** BlockTrafficZone does not prevent
   PhishingEmail, which uses `check_routable` / connected components
   instead of `blocking_host()`. Red can always phish into any subnet
   regardless of blocks.

2. **Messaging ineffectiveness.** The inter-agent messaging system has
   zero measurable impact (p=0.80 ablation). The 32 bits of observation
   space are wasted.

3. **Single-action-per-step constraint.** Each agent can only execute one
   action per step. When multiple hosts need Restore simultaneously, the
   agent must triage, leaving lower-priority hosts compromised longer.

4. **DegradeServices permanence.** Once red executes Impact
   (DegradeServices) on the OTService host, the service degradation is
   permanent. Only Restore can fix it, but by then the damage to the
   reward signal has already occurred.

5. **Remove action parsed but unused.** V11a still parses Remove action
   labels but never issues them. This is dead code retained for potential
   future use.

### 17.2 Potential Improvements

1. **Adaptive flag_age threshold.** Currently hardcoded at 1. Could be
   tuned per-subnet based on observed green FP rates.

2. **Multi-host Restore coordination.** When multiple hosts need Restore,
   the agent could batch them more intelligently rather than strict
   priority ordering.

3. **Predictive phase-transition blocking.** Currently uses a fixed
   10-step window. Could be dynamic based on agent busyness.

4. **Decoy rotation.** After DiscoverDeception, red knows which PIDs are
   decoys. Rotating decoy types (Apache->Tomcat->Haraka) could reset
   red's knowledge, though CC4 does not support this.

5. **Messaging protocol redesign.** Since the current protocol is proven
   ineffective, the 32 bits could be repurposed for a fundamentally
   different coordination mechanism (e.g., explicit action coordination
   rather than state broadcasting).

---

## 18. Appendix: Constants and Configuration

### 18.1 Core Constants

```python
NUM_SUBNETS   = 9     # Total subnets in the network
MAX_HOSTS     = 16    # Maximum hosts per subnet (padded)
NUM_MSG_BITS  = 32    # Inter-agent message section size (4 * 8)
RESTORE_DUR   = 5     # Steps for Restore to complete
MAX_DECOYS    = 3     # Maximum decoys per host

# Observation offsets within per-subnet block
_OFF_BLOCKED  = 9     # Blocked subvector starts at offset 9
_OFF_COMMS    = 18    # Comms policy starts at offset 18
_OFF_PROC     = 27    # Process flags start at offset 27
# Connection flags start at offset 27 + n_hosts

# Phase transition timing
PHASE_1_START      = 167   # Step where Phase 0 -> Phase 1
PHASE_2_START      = 334   # Step where Phase 1 -> Phase 2
PREEMPTIVE_WINDOW  = 10    # Steps before transition to start blocking
```

### 18.2 Messaging Constants

```python
_MSG_LEN           = 8    # Bits per agent message
_NUM_BLUE_AGENTS   = 5    # Total blue agents

# Bit indices within 8-bit message
_BIT_THREAT_LO       = 0
_BIT_THREAT_HI       = 1
_BIT_OPEN_PATHS_LO   = 2
_BIT_OPEN_PATHS_HI   = 3
_BIT_RED_COUNT_LO    = 4
_BIT_RED_COUNT_HI    = 5
_BIT_DECOYS_BYPASSED = 6
_BIT_RESTORING       = 7
```

### 18.3 Agent-Subnet Mapping

```python
_AGENT_PRIMARY_SUBNET = {
    0: "restricted_zone_a_subnet",
    1: "operational_zone_a_subnet",
    2: "restricted_zone_b_subnet",
    3: "operational_zone_b_subnet",
    4: None,   # HQ agent: controls PAZ + admin + office
}

_UPSTREAM = {
    (1, 1): 0,   # Phase 1: agent 0 (RZA) upstream of agent 1 (OZA)
    (2, 3): 2,   # Phase 2: agent 2 (RZB) upstream of agent 3 (OZB)
}
```

### 18.4 Preemptive Block Targets

```python
_PREEMPTIVE_PHASE1 = [
    ("restricted_zone_a_subnet", "operational_zone_a_subnet"),
]

_PREEMPTIVE_PHASE2 = [
    ("restricted_zone_b_subnet", "operational_zone_b_subnet"),
]
```

### 18.5 Internal State Variables

```
Variable                   Type                          Purpose
-------------------------  ----------------------------  ---------------------------
_step                      int                           Current episode step (1-indexed)
_restore_at                dict[str, int]                Step when Restore was issued per host
_proc_flagged_step         dict[str, int]                Step when proc_flag first appeared
_decoy_deployed            dict[str, int]                Decoy count per host (0-3)
_decoy_hit_hosts           set[str]                      Hosts where decoy hit was observed
_preemptive_blocks         set[tuple[str, str]]          Pairs blocked preemptively by V11a

_sleep_idx                 int                           Action index for Sleep
_block                     dict[tuple[str,str], int]     (from,to) -> action index
_allow                     dict[tuple[str,str], int]     (from,to) -> action index
_remove                    dict[str, int]                hostname -> action index (unused)
_restore                   dict[str, int]                hostname -> action index
_decoy                     dict[str, int]                hostname -> action index

_subnets_in_obs            list[str]                     Controlled subnets (alphabetical)
_subnet_host_list          dict[str, list]               Subnet -> ordered host list
_deploy_hosts              list[str]                     All decoy hosts, sorted by priority
_labels                    list[str]                     Raw action labels from env
```

---

## Revision History

| Version | Key Change                           | Performance Impact |
|---------|--------------------------------------|--------------------|
| V1-V6   | Initial heuristic development        | --                 |
| V7      | Decoy system added                   | Major improvement  |
| V9      | Messaging protocol, flag_age         | Minor improvement  |
| V9.1    | Bug fixes, threshold tuning          | Minor improvement  |
| V10b    | Remove eliminated, Restore-only      | -814.0 baseline    |
| V11a    | Preemptive OZ blocking               | -717.0 (+11.9%)    |
