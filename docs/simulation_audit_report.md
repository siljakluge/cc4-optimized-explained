# CC4 Simulation Audit Report

**Auditors:** Network Architecture Expert, Simulation Consistency Checker, Red Agent FSM Analyst  
**Scope:** Full codebase audit — blocking mechanisms, PhishingEmail consistency, action semantics,
observation timing, FSM realism, and logical errors.

---

## Executive Summary

The audit identified **7 bugs** (3 critical), **4 inconsistencies**, **4 design gaps**, and **6
design choices**. The most severe findings cluster around two areas:

1. **PhishingEmail bypasses the entire blocking system** — not by design, but because it calls
   `check_routable()` instead of `blocking_host()`. Every other remote action uses `blocking_host()`.
   This is a coding error with large consequences for blue agent strategy.

2. **The FSM red agent has a latent crash path** — when all known hosts enter state KD,
   `_choose_host_and_action` calls a non-existent method `self._choose_action()`, raising
   `AttributeError` at runtime.

---

## Critical Bugs

### C1: PhishingEmail uses the wrong routability check [BUG]
**File:** `PhishingEmail.py:77,90`

Every `RemoteAction` subclass (`ExploitRemoteService`, `DiscoverNetworkServices`, etc.) enforces
firewall rules by calling `self.blocking_host(state, src, target)`, which reads `state.blocks`.
`PhishingEmail` instead calls `self.check_routable(state, green_hostname, hostname)`, which reads
`state.connected_components` — a physical topology graph that is **never updated when
`BlockTrafficZone` is applied**.

**Consequence:** All `BlockTrafficZone` rules are silently invisible to PhishingEmail. Blue's entire
firewall system provides zero protection against the phishing vector. This is not documented as
intentional; it contradicts every other remote action in the codebase.

**Realistic?** Accidentally realistic: phishing emails do travel via SMTP out-of-band, not IP
routing. But the mechanism is wrong — `check_routable()` checks physical link adjacency, not email
transport semantics. The result is correct by coincidence, not design.

**Fix options:**
- Apply `blocking_host()` for consistency (makes phishing blockable by firewall)
- Keep `check_routable()` but add an explicit ADR comment documenting the out-of-band SMTP intent

---

### C2: PhishingEmail fallback loop has no list-shrinking — potential infinite loop [BUG]
**File:** `PhishingEmail.py:83-91`

The `while red_agent_src == "":` fallback loop iterates `red_agents` (all red agents in any subnet)
and calls `state.np_random.choice(red_agents, replace=False)`. The chosen agent is never removed
from `red_agents` if `check_routable()` fails. If no red agent is ever routable, this loop runs
indefinitely. In practice `replace=False` on a shrinking internal selection is used but the outer
list is never modified, making termination depend on implementation details of the RNG.

Additionally: a red agent in a completely separate zone can phish into any green host's subnet via
physical connected_components adjacency. The docstring says "the red agent should be in the same
subnet" but the fallback explicitly contradicts this, allowing cross-zone phishing.

---

### C3: FSM crashes when all known hosts are in KD state [BUG]
**File:** `FiniteStateRedAgent.py:315`

When `_choose_host_and_action` exhausts all host options, the fallback at line 315 calls
`self._choose_action(action_space, new_options)`. The method `_choose_action` **does not exist** on
`FiniteStateRedAgent` — only `_choose_host_and_action` exists. This raises `AttributeError`.

The KD state is also absent from `state_transitions_probability`, meaning there is no probability
vector for the agent to use even if the name typo were fixed. KD is a true dead-end state that
crashes the agent rather than gracefully idling or sleeping.

**Trigger condition:** After blue issues aggressive Remove/Restore on all hosts red has visited, red
may find every known host in KD simultaneously. In long episodes with an effective blue strategy
this is achievable.

---

## Bugs

### B1: Remove claims to delete files but does not [BUG]
**File:** `Remove.py:70-71`

The comment at line 70 reads `# remove suspicious files` but the next line is `return
Observation(True)` with no file-removal logic. Only `StopProcess` is called on suspicious PIDs
(lines 68-69). No code path removes any file from `host.files`.

**Impact on v6 strategy:** This confirms that `cmd.exe/sh` and `escalate.exe/sh` (density=0.9,
unsigned) persist on the host after Remove. Our v6 Priority 1b (Restore on malfile_flag) is
correct — once malfile=1, Remove is useless and only Restore clears the file.

---

### B2: Restore silently destroys pending detection events [BUG]
**File:** `Host.py:340-343`, `RestoreFromBackup.py:15`

`Monitor.execute()` moves events from `process_creation` to `old_process_creation` each step.
`host.restore()` calls `.clear()` on all four event lists — including `old_process_creation`.
Any events already moved to `old_process_creation` in the current step are destroyed before
BlueFlatWrapper reads them, causing silent event loss during Restore steps.

**Impact:** Blue may miss detection events on the step a Restore fires. The malfile_flag
(v6) is unaffected since it reads `host.files`, not events.

---

### B3: Multi-hop red sessions orphaned after Restore — potential KeyError [BUG]
**File:** `RestoreFromBackup.py:10-18`

`RestoreFromBackup` saves sessions from `target_host.sessions` and re-injects them after
`host.restore()`. If red has a session that was **pivoted through** the restored host (session
exists in `state.sessions['red_agent_X']` but not in the host's local `sessions` dict), it is
popped from `state.sessions` at line 14 but never re-inserted. The red agent then holds a session
key pointing to nothing — any subsequent action using that session raises `KeyError`.

---

## Inconsistencies

### I1: `check_routable()` models physical topology, not email routing [QUESTIONABLE]
**File:** `PhishingEmail.py`, `State.py:328`

`state.connected_components` is derived from the raw `link_diagram` graph at episode start and is
never updated. It represents physical link-layer connectivity, not email delivery paths. Using it
for PhishingEmail provides accidentally realistic behavior (phishing bypasses firewall) via the
wrong abstraction. Neither the code nor documentation states the intent.

---

### I2: ExploitRemoteService network-path detection is commented out [INCONSISTENCY]
**File:** `ExploitAction.py:68-70`

The code that would add a `network_connections` event along the route (giving blue visibility into
lateral movement paths, not just successful exploit endpoints) is commented out. Only post-exploit
session creation generates a detection event. This silently reduces blue's observability compared
to what the architecture implies.

---

### I3: DiscoverDeception TP/FP branches are not mutually exclusive [INCONSISTENCY]
**File:** `DiscoverDeception.py:89-100`

Line 89 gates on `process.decoy_type.name == 'EXPLOIT'` for the true-positive branch (50%
detection rate). Line 95 uses a **separate independent RNG draw** for the false-positive branch.
For non-EXPLOIT decoy types, a process can be missed by the TP branch (name check fails) and then
falsely flagged by the FP branch in the same call. The branches should be mutually exclusive by
decoy type but are not.

---

### I4: `impact_count` not decremented by Restore [INCONSISTENCY / DESIGN CHOICE]
**File:** `Impact.py:87`, `Host.py:127,363`

A successful Impact increments `host.impact_count`. `host.restore()` does not reset
`impact_count` (`Host.py:363` — field is not listed in restore). A blue Restore on a host after
Impact completes still scores the reward penalty. Fast Restore does not fully undo Impact damage.
This is likely intentional (attack already succeeded) but is undocumented.

---

## Design Gaps

### D1: No persistence mechanism for red [DESIGN GAP]
After Remove or Restore, red must fully re-exploit to regain access. There is no simulated
persistence (cron jobs, registry run keys, secondary implants). Real APT actors maintain 30–90 day
dwell times with multiple fallback mechanisms. The simulation underrepresents this resilience.

---

### D2: No inter-red-agent coordination [DESIGN GAP]
The messaging API (`SimulationController.py:891+`) exists and blue uses it. `FiniteStateRedAgent`
silently skips `message` keys in observations (lines 178, 197) and never sends messages. Multiple
red agents operate in complete isolation. Real coordinated APT groups share target lists, session
credentials, and C2 infrastructure. This leaves coordinated multi-agent red tactics completely
unexplored.

---

### D3: SD state exploits at 75% despite confirmed decoy presence [DESIGN GAP]
**File:** `FiniteStateRedAgent.py:543`

In SD state (decoy detected), the probability vector assigns 75% to `ExploitRemoteService` and
only 25% to `DiscoverDeception`. A real red operator who has confirmed a decoy on a host would
re-probe before committing, not exploit with 75% probability. The ratio should arguably be
reversed. `PIDSelectiveExploitActionSelector` mitigates this slightly but does not solve it.

---

### D4: 95% exploit detection rate is unrealistically high [QUESTIONABLE]
**File:** `ExploitAction.py:33`

Real host-based detection of network exploits (especially fileless or in-memory techniques) runs
10–40% in enterprise environments without dedicated EDR. A 95% rate unfairly advantages blue and
reduces the adversarial challenge. This parameter should be tunable to explore harder detection
scenarios.

---

## Design Choices (Correct or Acceptable)

| # | Finding | Verdict |
|---|---------|---------|
| DC1 | 1% phishing error rate (vs. real-world 3-5%) | Conservative but defensible |
| DC2 | `_clone_services` drops `_percent_reliable`; correct since backup taken at ep start | Fragile but currently correct |
| DC3 | GreenLocalWork FP dict lacks `pid` — alert fires but Remove cannot target it | Intentional: FP should not trigger Remove on real PID |
| DC4 | Routing enforced at simulator level, not FSM | Correct architectural separation |
| DC5 | Impact requires root session (SYSTEM/root check at Impact.py:65) | Realistic |
| DC6 | APT kill-chain modeled without C2/staging (single ExploitRemoteService step) | Known simplification, acceptable for simulation |

---

## Consolidated Findings Table

| ID | Finding | Label | Key File:Line |
|----|---------|-------|---------------|
| C1 | PhishingEmail calls `check_routable()` not `blocking_host()` — bypasses all firewalls | BUG | `PhishingEmail.py:77,90` |
| C2 | Fallback red-agent loop never shrinks candidate list — infinite loop risk + cross-zone phishing | BUG | `PhishingEmail.py:83-91` |
| C3 | `_choose_action` typo crashes FSM when all hosts in KD; KD missing from probability matrix | BUG | `FiniteStateRedAgent.py:315` |
| B1 | Remove comment claims file deletion; no file removal code exists | BUG | `Remove.py:70-71` |
| B2 | `host.restore()` clears `old_process_creation` — destroys pending detection events | BUG | `Host.py:340-343` |
| B3 | Multi-hop red sessions orphaned after Restore — KeyError on next action | BUG | `RestoreFromBackup.py:10-18` |
| I1 | `check_routable()` models physical topology, not email — wrong abstraction | QUESTIONABLE | `PhishingEmail.py`, `State.py:328` |
| I2 | Network-path detection for exploits commented out — reduces blue visibility | INCONSISTENCY | `ExploitAction.py:68-70` |
| I3 | DiscoverDeception TP/FP branches not mutually exclusive by decoy type | INCONSISTENCY | `DiscoverDeception.py:89-100` |
| I4 | `impact_count` not reset by Restore — fast Restore does not undo Impact penalty | INCONSISTENCY | `Host.py:363`, `Impact.py:87` |
| D1 | No persistence after eviction — unrealistic for real APT dwell | DESIGN GAP | `FiniteStateRedAgent.py` |
| D2 | No inter-red coordination despite messaging API existing | DESIGN GAP | `FiniteStateRedAgent.py:178,197` |
| D3 | SD state: 75% exploit despite confirmed decoy — too aggressive | DESIGN GAP | `FiniteStateRedAgent.py:543` |
| D4 | 95% exploit detection rate — unrealistically high | QUESTIONABLE | `ExploitAction.py:33` |

---

## Implications for Blue Agent Strategy

| Bug | Effect on blue | Our v6 mitigation |
|-----|---------------|-------------------|
| C1 (PhishingEmail bypasses blocks) | Firewall blocks useless for Channel B | Malfile_flag catches PrivEscalate after phishing entry |
| B1 (Remove does not delete files) | Malfile_flag persists after Remove | Correct: v6 Restores on malfile, never just Remove |
| B2 (Restore wipes pending events) | Rare 1-step detection blind spot during Restore | Malfile_flag is event-independent; not affected |
| I4 (impact_count not reset) | Fast Restore does not prevent penalty from completed Impact | No fix possible; underscores need to detect PrivEscalate early (malfile) |
| C3 (FSM crash on KD) | Red agent crashes in rare scenarios — reduces pressure on blue | No mitigation needed; benefits blue |

---

## Recommended Priority Fixes

1. **[Critical]** `PhishingEmail.py` — replace `check_routable()` with `blocking_host()` **or**
   document the SMTP out-of-band design intent with a comment and ADR.
2. **[Critical]** `FiniteStateRedAgent.py:315` — fix `self._choose_action` typo to
   `self._choose_host_and_action` and add KD to `state_transitions_probability` (e.g., same
   distribution as K state).
3. **[High]** `PhishingEmail.py:83-91` — add `red_agents.remove(r_agent)` inside the while loop
   to prevent infinite iteration.
4. **[Medium]** `Remove.py:70-71` — either implement file removal or remove the misleading comment.
5. **[Medium]** `Host.py:340-343` — preserve `old_process_creation` across Restore, or document
   the intentional event-wipe behavior.
6. **[Low]** `DiscoverDeception.py:89-100` — make TP/FP branches mutually exclusive per process.
