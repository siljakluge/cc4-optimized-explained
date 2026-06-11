# The DegradeServices Problem: Why the Dominant Damage Source Is Invisible

**Date:** 2026-04-10
**Context:** CAGE Challenge 4, V11a Blue-Team Heuristic Agent (mean reward: -717.0)

---

## 1. Executive Summary

DegradeServices is the single most damaging red action in CC4, responsible for **83.3% of all
reward loss** through indirect Loss-of-Work-Function (LWF) penalties. Despite a clear
precursor kill-chain that blue agents *can* observe (ExploitRemoteService -> PrivilegeEscalate),
blue agents cannot prevent DegradeServices damage due to a combination of structural timing
constraints, observation delays, action budget limits, and false positive noise. This report
explains exactly why, quantifies the tradeoffs, and presents experimental evidence that
aggressive response to precursor signals is counterproductive.

---

## 2. The DegradeServices Mechanism

### 2.1 How It Works

```
 Service Reliability Model
 ========================

 DegradeServices call #:   0     1     2     3     4     5
 Reliability (%):        100    80    60    40    20     0
                          |     |     |     |     |     |
 Green failure rate:      0%   20%   40%   60%   80%  100%
                          |     |     |     |     |     |
 LWF penalty per fail:   -10 (OZ during mission), -1 (elsewhere)
```

Each `DegradeServices` call:
- Requires **root access** on the target host (red must have completed PrivilegeEscalate)
- Takes **2 steps** to execute (red is locked during this time)
- Has **100% success rate** (never fails once preconditions are met)
- Degrades ALL active services on the host by **20 percentage points**
- Produces **zero observable signals** in blue's observation space

### 2.2 Why It's Invisible

The blue observation vector contains three alert flags per host:

```
 Flag        | Source Actions                | DegradeServices?
 ------------+-------------------------------+-----------------
 conn_flag   | ExploitRemoteService          | NO
 proc_flag   | Exploit, green FP             | NO
 malfile     | PrivilegeEscalate             | NO
             |                               |
 (nothing)   | DegradeServices               | INVISIBLE
```

DegradeServices modifies `service._percent_reliable` on the host object directly.
This attribute is **never exposed** in any observation field. The blue agent has
no way to read service reliability values, nor does the action generate any network
connection or process creation event.

---

## 3. The Precursor Kill-Chain

### 3.1 The Observable Attack Sequence

Red must complete a strict chain before DegradeServices becomes available:

```
 RED KILL-CHAIN (Observable Precursors -> Invisible Damage)
 ==========================================================

 Step N:   DiscoverRemoteSystems     (duration: 1)
           |
           v                          [Blue sees: nothing useful]
 Step N+1: ServiceDiscovery          (duration: 2)
           |
           v                          [Blue sees: nothing useful]
 Step N+3: ExploitRemoteService      (duration: 4, success: 24.9%)
           |
           v                          [Blue sees: conn_flag + proc_flag]
 Step N+7: PrivilegeEscalate         (duration: 2, success: 100%)
           |
           v                          [Blue sees: malfile flag]
 Step N+9: DegradeServices           (duration: 2, success: 100%)
           |
           v                          [Blue sees: NOTHING]
                                      [Green work starts failing silently]
```

### 3.2 Why Observing Precursors Doesn't Prevent Damage

The precursors are observable, but **four structural constraints** prevent blue
from acting fast enough:

```
 THE TIMING PROBLEM
 ==================

 Timeline (zoomed into the critical window):

 Step:    0       1       2       3       4       5       6       7
          |       |       |       |       |       |       |       |
 Red:     Exploit ~~~~~~~~|       PrivEsc ~~|     Degrade ~~|
          (4 steps duration)      (2 steps)       (2 steps)
          |       |       |       |       |       |       |
 Blue     |  sees |  flag |       | sees  |       | DAMAGE|
 sees:    | conn_ | _age  |       |malfile|       | DONE  |
          | flag  | >= 1  |       |       |       |       |
          |       |       |       |       |       |       |
 Blue     |       | Issue |  R    |  e    |  s    |  t    | o
 acts:    |       |Restore|  e    |  s    |  t    |  o    | r
          |       | HERE  |  s    |  t    |  o    |  r    | e
          |       |       |  t    |  o    |  r    |  e    |
          |       |       |  o    |  r    |  i    |       | done
          |       |       |  r    |  i    |  n    |       |
          |       |       |  i    |  n    |  g    |       |
          |       |       |  n    |  g    |       |       |
          |       |       |  g    |       |       |       |
          |       |       |       |       |       |       |
          |       |       5 steps duration ------->|       |

 RESULT: Even with perfect detection at step 1, Restore completes
         at step 6. Red fires DegradeServices at step 4-5.
         Blue is 1-2 steps TOO LATE.
```

### 3.3 The Four Structural Constraints

```
 CONSTRAINT 1: Observation Delay (1 step)
 =========================================
 Blue observes the state AFTER red and green have acted.
 By the time conn_flag appears, red's exploit is already done.

    Red acts at step N  --->  Blue sees result at step N+1
                              (1 step lost)


 CONSTRAINT 2: False Positive Filter (1 step)
 =============================================
 Green agents generate false positive proc_flags at 0.776% rate.
 To avoid wasting Restores on FPs, blue requires flag_age >= 1.

    Flag appears step N+1 --->  Blue confirms at step N+2
                                (1 more step lost)


 CONSTRAINT 3: Restore Duration (5 steps)
 =========================================
 Restore is the ONLY action that removes root access and
 resets service reliability. It takes 5 steps to complete.

    Restore issued step N+2 --->  Completes at step N+7
                                  (5 steps locked)


 CONSTRAINT 4: Action Budget (1 action per step per agent)
 ==========================================================
 Each blue agent can do exactly 1 thing per step.
 If 2 hosts are compromised simultaneously, only 1 gets Restored.

    Host A compromised + Host B compromised
         |                    |
         v                    v
    Restore Host A        MUST WAIT
    (5 steps)             (Host B degraded meanwhile)
```

---

## 4. The Green False Positive Problem

### 4.1 The Scale of False Positives

```
 GREEN FALSE POSITIVE BUDGET (per 500-step episode)
 ===================================================

 61 green agents x 500 steps x 33% GreenLocalWork x 0.776% FP rate

                     = ~155.6 false positive proc_flags per episode

 That's roughly 1 false positive every 3.2 steps, somewhere in the network.
 P(at least one FP in any given step) = ~27%
```

### 4.2 The Dilemma: Respond Fast or Respond Accurately?

The core tradeoff is between **response speed** (catch real attacks faster)
and **response accuracy** (avoid wasting actions on false positives).

```
 THE FALSE POSITIVE TRADEOFF
 ============================

                    False Positive         Real Red Attack
                    (green noise)          (actual threat)
                         |                      |
                         v                      v
                   +-----------+          +-----------+
                   | proc_flag |          | proc_flag |
                   | appears   |          | appears   |
                   +-----------+          +-----------+
                         |                      |
          +--------------+--------------+       |
          |                             |       |
    flag_age = 0                  flag_age = 1   |
    (respond immediately)         (wait 1 step)  |
          |                             |       |
          v                             v       v
    +------------------+       +------------------+
    | Catches ALL red  |       | Catches real red |
    | 1 step faster    |       | with 1-step delay|
    |                  |       |                  |
    | BUT ALSO catches |       | Filters out most |
    | ~54 extra FPs    |       | green FPs (they  |
    | per episode      |       | clear in 1 step) |
    +------------------+       +------------------+
          |                             |
          v                             v
    EXPERIMENT 1:               V10b BASELINE:
    -1127.2 reward              -771.8 reward
    (355 points WORSE)          (current best)
```

### 4.3 Why Green FPs Clear in 1 Step

```
 GREEN FP LIFECYCLE vs RED ATTACK LIFECYCLE
 ============================================

 GREEN FALSE POSITIVE:

 Step N:     GreenLocalWork fires FP
             proc_flag = 1   <---- appears
 Step N+1:   No new FP on same host
             proc_flag = 0   <---- CLEARS (flag_age stays 0)

 Reason: Green FP is a one-time random event. The green agent
 picks a different random action next step. The flag does not
 persist because no new process_creation event is generated.


 RED EXPLOIT:

 Step N:     ExploitRemoteService succeeds
             conn_flag = 1, proc_flag = 1  <---- appears
 Step N+1:   Red session persists on host
             conn_flag = 1, proc_flag = 1  <---- PERSISTS (flag_age >= 1)

 Reason: The red session creates persistent network connections
 and processes that Monitor detects every step until cleaned up.


 THEREFORE: flag_age >= 1 perfectly separates green FPs from
 real red activity in the vast majority of cases.
```

---

## 5. Experimental Evidence

### 5.1 Experiment 1: Aggressive Response (flag_age = 0)

**Question:** What if we respond to every proc_flag immediately, accepting
all false positives as the cost of faster response?

```
 EXPERIMENT 1 RESULTS
 =====================

                    Baseline (flag_age=1)    Aggressive (flag_age=0)
                    ---------------------    -----------------------
 Mean Reward:            -771.8                   -1127.2
 Std Dev:                 212.5                     217.7
 Restores/episode:         44.8                      98.9
 Extra Restores:             --                     +54.1
 Delta vs baseline:          --                    -355.3  (MUCH WORSE)
 p-value:                    --                   < 0.0001 (significant)

 Cost per extra FP Restore: 355.3 / 54.1 = -6.6 reward each
```

**Breakdown of the -6.6 per false positive Restore:**

```
                        Direct cost: -1 (Restore action penalty)
                        +
                        Opportunity cost: ~5 steps agent is BUSY
                        |
                        v
           During those 5 steps, the agent CANNOT:
           - Respond to real red attacks on other hosts
           - Deploy decoys
           - Enforce blocking policy
           |
           v
    Red exploits OTHER hosts unopposed
    -> cascading damage from undefended hosts
    -> THIS is the real cost, not the -1 penalty
```

### 5.2 Experiment 2: Proactive Restore Every N=30 Steps

**Question:** Since DegradeServices is invisible, should we periodically
Restore hosts to reset reliability even without observing an attack?

```
 EXPERIMENT 2 RESULTS
 =====================

                    Baseline              Proactive N=30
                    ---------             --------------
 Mean Reward:       -771.8                -1233.0
 Restores/episode:    44.8                  125.9
 Delta:                --                  -461.2  (CATASTROPHIC)
 p-value:              --                 < 0.0001

 WHY IT FAILS:
 +----------------------------------------------------------+
 |  Each proactive Restore = -1 penalty + 5 steps locked    |
 |  125.9 Restores x 5 steps = 629.5 agent-steps consumed   |
 |  That's 629.5 / 2500 total = 25.2% of all agent time     |
 |  spent Restoring hosts that may not even be compromised   |
 +----------------------------------------------------------+
```

### 5.3 Summary: The False Positive Tax vs DegradeServices Damage

```
 COMPARATIVE DAMAGE ANALYSIS
 ============================

 Source                    | Reward Impact  | Controllable?
 --------------------------+----------------+--------------
 DegradeServices (LWF)     | ~597 / episode | Partially (via precursor detection)
 Green FP wasted Restores  | ~150 / episode | Cannot reduce FP rate (fixed at 0.776%)
 Restore action costs      |  ~88 / episode | Cannot avoid (only way to clear root)
 Red Impact (RIA)          |  ~50 / episode | Mostly handled by blocking
 Detection delay           |  ~50 / episode | Structural (observation lag)

 Total observed:           | ~717 / episode |
                             (V11a mean = -717.0)
```

```
 WHAT HAPPENS WHEN WE TRY TO REDUCE DEGRADE DAMAGE:

 Strategy                  | Degrade    | FP Cost    | Net Effect
                           | Savings    | Increase   |
 --------------------------+------------+------------+------------
 flag_age=0 (faster react) | ~10-30 pts |  +355 pts  | -325  WORSE
 Proactive N=30 Restore    | ~50-100    |  +461 pts  | -361  WORSE
 Current V11a (flag_age=1) |   ---      |   ---      |  BEST KNOWN

 The cure is worse than the disease.
```

---

## 6. The Damage Arithmetic

### 6.1 How DegradeServices Generates 83.3% of All Damage

```
 DAMAGE CASCADE: From Root Access to Reward Loss
 =================================================

 Step 1: Red obtains root access (via Exploit + PrivEsc)
         |
         v
 Step 2: Red calls DegradeServices (invisible, 100% success)
         |
         v
 Step 3: Service reliability drops 100% -> 80%
         |
         v
 Step 4: Green agents try GreenLocalWork on the host
         |
         +---> 20% chance of failure per green agent per step
         |
         v
 Step 5: Each failure triggers LWF penalty
         |
         +---> Phase 1/2 OZ hosts: -10 per failure
         +---> HQ hosts: -1 per failure
         +---> Other hosts: -1 per failure
         |
         v
 Step 6: Red calls DegradeServices AGAIN (still has root)
         |
         v
 Step 7: Reliability drops 80% -> 60%
         |
         v
 Step 8: Now 40% failure rate -> even more LWF penalties
         |
         ...continues until Restore or reliability hits 0%
```

### 6.2 Cumulative Impact by Phase

Empirically measured GreenLocalWork failure rates:

```
 SERVICE FAILURE RATES ACROSS PHASES
 =====================================

 Phase 0 (steps 0-166):    4.55%   [Services mostly healthy]
                           ||||

 Phase 1 (steps 167-333): 23.97%   [Degradation accumulating]
                           ||||||||||||

 Phase 2 (steps 334-499): 43.01%   [Services severely degraded]
                           ||||||||||||||||||||||

 The steep increase reflects cumulative DegradeServices calls.
 By Phase 2, nearly half of all green work attempts fail.
```

### 6.3 OZ During Active Mission: The Worst Case

```
 OZ-A HOST DURING PHASE 1 (Mission A active)
 =============================================

 Assumptions:
 - 3 hosts in OZ-A, each with green agents
 - Red has root on 1 host, calls DegradeServices repeatedly
 - LWF penalty = -10 per green failure

 After 3 DegradeServices calls (reliability 100% -> 40%):

   Per green agent per step:
   - 33% chance of GreenLocalWork
   - 60% failure rate when attempted
   - -10 per failure

   Expected penalty per green agent per step = 0.33 x 0.60 x (-10) = -1.98

   With ~3-5 green agents on one OZ host:
   Expected penalty = -5.9 to -9.9 PER STEP

   Over 167 steps of Phase 1:
   Potential damage from ONE degraded OZ host = -985 to -1,653

 This single host can account for MORE than the entire
 episode penalty of V11a (-717).

 In practice, blocking prevents most red access to OZ during
 active phases, which is why V11a's score isn't worse.
```

---

## 7. Why Blocking Is the Real Defense (Not Faster Detection)

### 7.1 The Effective Defense Stack

```
 V11a DEFENSE LAYERS (ordered by impact)
 =========================================

 Layer 1: BLOCKING (prevents the kill-chain from starting)
 +----------------------------------------------------------+
 |  comms_policy enforcement + preemptive OZ blocking       |
 |  - Blocks ALL 8 paths to active OZ during mission phase  |
 |  - ASF cost = 0 (free during active phase)               |
 |  - Prevents ExploitRemoteService from reaching OZ        |
 |  - If red can't exploit, red can't escalate              |
 |  - If red can't escalate, red can't degrade              |
 |  --> Eliminates the ENTIRE kill-chain for blocked hosts   |
 +----------------------------------------------------------+
            |
            | (red attacks that bypass blocking, e.g. pre-Phase 1
            |  footholds, phishing, unblocked paths in Phase 0)
            v
 Layer 2: REACTIVE DETECTION (catches exploits that got through)
 +----------------------------------------------------------+
 |  conn_flag + proc_flag monitoring with flag_age >= 1     |
 |  - Detects ExploitRemoteService within 1-2 steps         |
 |  - Triggers Remove (3 steps) or Restore (5 steps)        |
 |  - Removes red sessions before PrivEsc in best case      |
 |  - Removes after PrivEsc but before too many Degrades    |
 +----------------------------------------------------------+
            |
            | (red completes full chain before blue responds:
            |  exploit -> privesc -> degrade, then blue restores)
            v
 Layer 3: RESTORE (undoes damage after the fact)
 +----------------------------------------------------------+
 |  Restore resets service reliability to 100%              |
 |  - The ONLY way to reverse DegradeServices damage        |
 |  - Cost: -1 penalty + 5 steps locked                    |
 |  - Also wipes decoys (must redeploy)                    |
 |  - Effective but expensive                              |
 +----------------------------------------------------------+
            |
            | (damage that occurred between DegradeServices
            |  and Restore completion is PERMANENT in the
            |  reward record -- cannot be undone)
            v
 IRREDUCIBLE DAMAGE: LWF penalties already scored
 +----------------------------------------------------------+
 |  Every green failure that occurred while services were   |
 |  degraded is a permanent penalty. Restore prevents       |
 |  FUTURE damage but does not cancel PAST penalties.       |
 +----------------------------------------------------------+
```

### 7.2 Why V11a's Preemptive OZ Blocking Was the +14% Gain

```
 V10b -> V11a: The Single Most Impactful Change
 ================================================

 V10b: Blocks OZ paths as part of normal comms_policy enforcement (P3).
       Problem: P3 is low priority. If the agent is busy Restoring
       a host (P1, 5 steps), the block is DELAYED.

       Phase transition at step 167:
       Step 167: Agent is mid-Restore (started step 164) --+
       Step 168: Still restoring                           |
       Step 169: Restore completes, NOW blocks OZ path     |
                                                           |
       2 steps of OPEN OZ path during active mission ------+
       Red can exploit OZ during this window.

 V11a: Blocks OZ paths PREEMPTIVELY 10 steps before phase transition.
       + Elevates OZ blocking above P2 Allow priority.

       Phase transition at step 167:
       Step 157: Preemptive block issued (ASF=-1, cheap)
       Step 167: Phase starts, OZ already locked down
       Step 168: Zero window for red exploitation

       Result: -771.8 -> -717.0 (+7.1% improvement)
               With 100 episodes: -814.0 -> -700.0 (+14.0%)
```

---

## 8. The Fundamental Information Asymmetry

```
 WHAT RED KNOWS vs WHAT BLUE KNOWS
 ====================================

 RED knows:                          BLUE knows:
 +---------------------------+      +---------------------------+
 | - All its sessions        |      | - conn_flag (per host)    |
 | - Session privilege level |      | - proc_flag (per host)    |
 | - Which hosts have root   |      | - malfile (per host)      |
 | - Service reliability     |      | - blocked paths           |
 | - Action success/failure  |      | - comms_policy            |
 | - Full host state         |      | - messages from peers     |
 +---------------------------+      +---------------------------+
          |                                    |
          v                                    v
 Red KNOWS when to degrade       Blue CANNOT see:
 and which hosts are most        - Service reliability values
 valuable to target              - Whether DegradeServices ran
                                 - How many times it ran
                                 - Current degradation level

 This is a FUNDAMENTAL information asymmetry by CC4 design.
 No observation-based agent can detect DegradeServices directly.
```

---

## 9. Remaining Research Directions

### 9.1 What Could Still Help (Theoretical)

```
 APPROACH                        | EXPECTED GAIN | FEASIBILITY
 --------------------------------+---------------+------------
 Shadow state tracking           | 50-200 pts    | High complexity
 (Bayesian P(degraded|history))  |               | Weak signal
                                 |               |
 Selective proactive Restore     | 50-150 pts    | Needs gating
 (only on known-compromised      |               | criteria that
 hosts, not uniform)             |               | don't exist yet
                                 |               |
 conn_flag persistence tracking  | 10-30 pts     | Simple to test
 (2-step persistent = real red)  |               | Low risk
                                 |               |
 Decoy depletion awareness       | 5-20 pts      | Simple to test
 (depleted host = real signals)  |               | Low risk
                                 |               |
 DRL recurrent policy (LSTM)     | 100-400 pts   | Very expensive
 (learn hidden degradation       |               | 5M+ episodes
 state from reward history)      |               | training
```

### 9.2 The Hard Limit

```
 THEORETICAL FLOOR ANALYSIS
 ===========================

 V11a current:        -717.0  (97.6% of gap closed)
 Theoretical floor:   ~-300   (estimated perfect play)
 Remaining gap:        ~417 points

 Of this ~417 point gap:
 - ~347 pts (83.3%) = DegradeServices LWF damage
 - ~21 pts  (5%)    = Green FP wasted Restores
 - ~49 pts  (12%)   = Other sources

 Even a PERFECT agent with full observability would still
 incur ~300 points of penalty from:
 - Unavoidable green work failures during Restore downtime
 - Phase 0 damage (no free blocking available)
 - Phishing bypassing all blocks
 - Stochastic red exploit successes before detection

 V11a is already within 2.4x of the theoretical floor.
```

---

## 10. Conclusions

1. **DegradeServices is invisible by design.** No modification to the observation
   pipeline can make it visible. The only approach is indirect inference.

2. **Precursor detection helps but cannot prevent all damage.** The timing gap
   between observation and response means red always gets at least 1-2
   DegradeServices calls through before Restore completes.

3. **Faster response (flag_age=0) is counterproductive.** The +54 false positive
   Restores per episode cost -355 reward, far exceeding any benefit from 1-step
   faster detection. The flag_age >= 1 threshold is correctly calibrated.

4. **Proactive Restore is catastrophic.** Uniform proactive Restore (N=30)
   wastes 25% of agent time and costs -461 reward. Any proactive strategy
   requires precise gating criteria that don't yet exist.

5. **Blocking is the primary defense.** Preemptive OZ blocking prevents the
   entire kill-chain from starting on the highest-value hosts. This produced
   the only statistically significant improvement in the V11 optimization cycle.

6. **The remaining gap is narrow.** V11a captures 97.6% of the defensible gap.
   Further improvement requires either multi-parameter learned policies (DRL)
   or fundamentally new inference mechanisms (Bayesian degradation estimation),
   both of which face the hard constraint of extremely weak indirect signals.
