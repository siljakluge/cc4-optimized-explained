# V11 Optimization Plan - Prerequisite Diagnostics

**Generated**: 2026-04-08
**Baseline**: V10b agent (-814.0 mean reward)
**Scripts**: `scripts/trace_comms_policy.py`, `scripts/trace_action_labels.py`, `scripts/trace_green_access.py`

---

## Question 1: Which subnet pairs does comms_policy ALREADY block in each phase?

The comms_policy is defined in `BlueFlatWrapper._build_comms_policy_network()` as a NetworkX graph.
Edges = ALLOWED connections. Missing edges = BLOCKED.

### Phase 0 (Preplanning) — Steps 0-166

**ALLOWED connections (graph edges):**

| From | To | Allowed? |
|------|-----|----------|
| ADMIN | CONTR, OFFIC, PAZ, RZ_A, RZ_B, INTRN | Yes |
| CONTR | RZ_A, RZ_B, INTRN, PAZ, ADMIN, OFFIC | Yes |
| OFFIC | CONTR, PAZ, RZ_A, RZ_B, INTRN, ADMIN | Yes |
| PAZ | CONTR, RZ_A, RZ_B, INTRN, ADMIN, OFFIC | Yes |
| RZ_A | OZ_A, RZ_B, CONTR, ADMIN, OFFIC, PAZ, INTRN | Yes |
| RZ_B | OZ_B, RZ_A, CONTR, ADMIN, OFFIC, PAZ, INTRN | Yes |
| OZ_A | RZ_A only | Yes |
| OZ_B | RZ_B only | Yes |

**BLOCKED in Phase 0:**
- OZ_A is blocked from ALL subnets EXCEPT RZ_A
- OZ_B is blocked from ALL subnets EXCEPT RZ_B
- OZ_A <-> OZ_B: blocked (no direct connection ever)
- INTRN has no self-loop or any special restriction; it connects to everything except OZ_A and OZ_B

### Phase 1 (Mission A) — Steps 167-333

**Changes from Phase 0:**
- REMOVED (newly blocked): RZ_A <-> OZ_A, RZ_A <-> CONTR, RZ_A <-> RZ_B, RZ_A <-> INTRN
- **OZ_A becomes COMPLETELY ISOLATED** (zero edges in the graph)
- OZ_B remains connected to RZ_B only (unchanged)

**BLOCKED in Phase 1:**
- OZ_A: blocked from ALL subnets (completely isolated)
- OZ_B: blocked from all except RZ_B
- RZ_A: additionally blocked from CONTR, INTRN, OZ_A, RZ_B

### Phase 2 (Mission B) — Steps 334-499

**Changes from Phase 0:**
- REMOVED (newly blocked): RZ_B <-> OZ_B, RZ_B <-> CONTR, RZ_B <-> RZ_A, RZ_B <-> INTRN
- **OZ_B becomes COMPLETELY ISOLATED** (zero edges in the graph)
- OZ_A reconnects to RZ_A (same as Phase 0 for OZ_A)

**BLOCKED in Phase 2:**
- OZ_B: blocked from ALL subnets (completely isolated)
- OZ_A: blocked from all except RZ_A
- RZ_B: additionally blocked from CONTR, INTRN, OZ_B, RZ_A

### Key Finding: OZ Isolation Pattern

| Phase | OZ_A Connected To | OZ_B Connected To |
|-------|-------------------|-------------------|
| 0 (Preplanning) | RZ_A only | RZ_B only |
| 1 (Mission A) | **NONE (isolated)** | RZ_B only |
| 2 (Mission B) | RZ_A only | **NONE (isolated)** |

**CRITICAL**: The "active" OZ (the one being used for the mission) is the one that gets ISOLATED.
- Mission A isolates OZ_A (the one running the mission)
- Mission B isolates OZ_B

---

## Question 2: Which BlockTrafficZone actions are available to each agent?

Each agent can ONLY issue BlockTrafficZone/AllowTrafficZone targeting their OWN subnets (as `to_subnet`), from ANY other subnet (as `from_subnet`).

### Agent Action Summary

| Agent | Zone | BlockTrafficZone Target | # Block Actions | # Allow Actions |
|-------|------|-------------------------|-----------------|-----------------|
| blue_agent_0 | RZ_A | restricted_zone_a_subnet only | 8 | 8 |
| blue_agent_1 | OZ_A | operational_zone_a_subnet only | 8 | 8 |
| blue_agent_2 | RZ_B | restricted_zone_b_subnet only | 8 | 8 |
| blue_agent_3 | OZ_B | operational_zone_b_subnet only | 8 | 8 |
| blue_agent_4 | HQ | admin, office, PAZ (3 subnets) | 24 | 24 |

### Key Finding: Agent can block ALL 8 source subnets to its own subnet

Each agent (0-3) has exactly 8 BlockTrafficZone actions, one for each of the other 8 subnets. They can block traffic from any subnet to their controlled subnet.

**YES, blue_agent_1 (OZA) CAN issue BlockTrafficZone from ANY source subnet to OZA.**

Available BlockTrafficZone actions for blue_agent_1:
- Block admin_network_subnet -> operational_zone_a_subnet
- Block contractor_network_subnet -> operational_zone_a_subnet
- Block internet_subnet -> operational_zone_a_subnet
- Block office_network_subnet -> operational_zone_a_subnet
- Block operational_zone_b_subnet -> operational_zone_a_subnet
- Block public_access_zone_subnet -> operational_zone_a_subnet
- Block restricted_zone_a_subnet -> operational_zone_a_subnet
- Block restricted_zone_b_subnet -> operational_zone_a_subnet

**IMPORTANT LIMITATION**: An agent CANNOT block traffic to a subnet it doesn't control. blue_agent_1 (OZA) cannot block traffic to RZA or any other subnet. Only blue_agent_0 can block traffic to RZA.

---

## Question 3: How often does green access OZ from outside OZ?

### Source Code Analysis (Definitive)

Green agents use `GreenAccessService.random_reachable_ip()` which selects servers from `allowed_subnets`. The `allowed_subnets` are updated per mission phase by `SimulationController._update_agents_allowed_subnets()`.

**Green agent cross-subnet access to OZ (from source code):**

| Phase | Green Location | Can Access OZ_A Servers? | Can Access OZ_B Servers? |
|-------|---------------|--------------------------|--------------------------|
| Phase 0 | OZ_A | Yes (also RZ_A servers) | No |
| Phase 0 | OZ_B | No | Yes (also RZ_B servers) |
| Phase 0 | RZ_A | Yes (via allowed_subnets) | No |
| Phase 0 | RZ_B | No | Yes (via allowed_subnets) |
| Phase 0 | HQ (PAZ/ADMIN/OFFIC) | No | No |
| Phase 0 | CONTR | No | No |
| Phase 1 | OZ_A | **LOCAL ONLY (isolated)** | No |
| Phase 1 | RZ_A | **No (RZ_A isolated from OZ_A)** | No |
| Phase 1 | OZ_B | No | Yes (also RZ_B servers) |
| Phase 1 | RZ_B | No | Yes (via allowed_subnets) |
| Phase 2 | OZ_A | Yes (also RZ_A servers) | No |
| Phase 2 | RZ_A | Yes (via allowed_subnets) | No |
| Phase 2 | OZ_B | **LOCAL ONLY (isolated)** | No |
| Phase 2 | RZ_B | **No (RZ_B isolated from OZ_B)** | No |

### Key Findings

1. **Green agents NEVER access OZ from HQ/CONTR/INTRN subnets.** The comms_policy only connects OZ to its paired RZ.
2. **Cross-subnet green access to OZ comes ONLY from RZ** (and OZ green agents access RZ servers).
3. **During active mission phases, OZ green agents are LOCAL ONLY** — they cannot reach any server outside their subnet.
4. **The "important" cross-subnet traffic to OZ is RZ -> OZ**, which only exists in Phase 0 (both) and the non-active phase.

### Green Access Pattern Summary

- Phase 0: OZ_A green can reach RZ_A servers; RZ_A green can reach OZ_A servers
- Phase 1: NO green access to/from OZ_A at all (isolated); OZ_B/RZ_B still connected
- Phase 2: NO green access to/from OZ_B at all (isolated); OZ_A/RZ_A still connected

**This means blocking RZ_A -> OZ_A during Phase 1 has ZERO green impact** (already isolated by comms_policy). Similarly for RZ_B -> OZ_B during Phase 2.

---

## Question 4: What is the duration of BlockTrafficZone?

### Source Code Analysis

From `CybORG/Simulator/Actions/Action.py`:
```python
DEFAULT_DURATION = 1

class Action(CybORGLogger):
    def __init__(self):
        self.duration = DEFAULT_DURATION  # = 1
```

From `CybORG/Simulator/Actions/ConcreteActions/ControlTraffic.py`:
```python
class ControlTraffic(LocalAction):
    def __init__(self, session, agent):
        super().__init__(session, agent)
        self.priority = 1  # Higher priority (lower number = higher)

class BlockTrafficZone(ControlTraffic):
    # Does NOT set self.duration — inherits DEFAULT_DURATION = 1
```

### Findings

| Property | Value | Notes |
|----------|-------|-------|
| `BlockTrafficZone.duration` | **1 step** | Inherited from Action base class |
| `BlockTrafficZone.priority` | **1** | Set by ControlTraffic (higher than default 99) |
| Persistence | **Until AllowTrafficZone reverses it** | Block is added to `state.blocks` dict and stays |
| Execution time | **1 step** | The action takes 1 step to execute |

**CRITICAL**: BlockTrafficZone is a **persistent firewall rule**. Once issued, the block stays in `state.blocks` indefinitely until explicitly reversed by AllowTrafficZone. The `duration=1` only means the action itself takes 1 step to execute, not that the block expires after 1 step.

This is confirmed by `BlockTrafficZone.execute_control_traffic()`:
```python
state.blocks.setdefault(self.to_subnet, []).append(self.from_subnet)
```
It just appends to the blocks dict. There is no TTL or expiration mechanism.

---

## Question 5: Are there any paths to OZA/OZB that comms_policy leaves OPEN in Phases 1/2?

### Phase 1 (Mission A) — OZ_A paths

**OZ_A is COMPLETELY ISOLATED by comms_policy.** Zero edges in the policy graph.
- No subnet can reach OZ_A
- No path exists from any subnet to OZ_A
- Green agents on OZ_A are LOCAL ONLY

**OZ_B has ONE open path:** RZ_B <-> OZ_B

| Source | Destination | Path | Status |
|--------|-------------|------|--------|
| Any HQ subnet | OZ_A | None | BLOCKED by comms_policy |
| CONTR | OZ_A | None | BLOCKED by comms_policy |
| INTRN | OZ_A | None | BLOCKED by comms_policy |
| RZ_A | OZ_A | None | BLOCKED by comms_policy (edge removed in Phase 1) |
| RZ_B | OZ_A | None | BLOCKED by comms_policy |
| OZ_B | OZ_A | None | BLOCKED by comms_policy |
| Any subnet | OZ_B | Via RZ_B | **OPEN** (HQ->RZ_B->OZ_B) |

### Phase 2 (Mission B) — OZ_B paths

**OZ_B is COMPLETELY ISOLATED by comms_policy.** Zero edges in the policy graph.

**OZ_A has ONE open path:** RZ_A <-> OZ_A

| Source | Destination | Path | Status |
|--------|-------------|------|--------|
| Any HQ subnet | OZ_B | None | BLOCKED by comms_policy |
| CONTR | OZ_B | None | BLOCKED by comms_policy |
| INTRN | OZ_B | None | BLOCKED by comms_policy |
| RZ_B | OZ_B | None | BLOCKED by comms_policy (edge removed in Phase 2) |
| RZ_A | OZ_B | None | BLOCKED by comms_policy |
| OZ_A | OZ_B | None | BLOCKED by comms_policy |
| Any subnet | OZ_A | Via RZ_A | **OPEN** (HQ->RZ_A->OZ_A) |

### Phase 0 (Preplanning) — Both OZ paths open

| Destination | Open Path | Via |
|-------------|-----------|-----|
| OZ_A | Yes | Any HQ/CONTR/INTRN -> RZ_A -> OZ_A |
| OZ_B | Yes | Any HQ/CONTR/INTRN -> RZ_B -> OZ_B |

---

## Summary of V11 Optimization Implications

### 1. Blocking RZ -> OZ during active phase is FREE

During Phase 1, comms_policy already isolates OZ_A. Issuing `BlockTrafficZone(RZ_A -> OZ_A)` is redundant.
During Phase 2, comms_policy already isolates OZ_B. Issuing `BlockTrafficZone(RZ_B -> OZ_B)` is redundant.

**V11 implication**: If the plan involves blocking RZ->OZ during the active mission phase, this is already done by the environment. No action needed.

### 2. Blocking during Phase 0 and the NON-active phase matters

Phase 0: Both OZ_A and OZ_B are reachable via their paired RZ.
Phase 1: OZ_B is reachable via RZ_B (NON-active OZ).
Phase 2: OZ_A is reachable via RZ_A (NON-active OZ).

**V11 implication**: Blue agents should consider blocking RZ_A -> OZ_A during Phase 0 (pre-mission) and Phase 2 (when OZ_A is not the active mission but still reachable). Similarly for RZ_B -> OZ_B during Phase 0 and Phase 1.

### 3. Each agent can block to its own subnet only

- blue_agent_1 (OZA) can block any subnet -> OZA (8 possible blocks)
- blue_agent_0 (RZA) can block any subnet -> RZA (8 possible blocks)
- To block the RZ_A -> OZ_A path, blue_agent_1 must issue `BlockTrafficZone(from=RZ_A, to=OZ_A)`
- Blue_agent_0 CANNOT block traffic to OZ_A (only to RZ_A)

### 4. BlockTrafficZone is persistent and costs 1 step

The block persists until reversed. The agent uses 1 step to issue the block (cannot do Restore/Analyse that step). This is a meaningful trade-off: 1 step of lost defensive action for permanent traffic blocking.

### 5. Green cross-subnet access to OZ is minimal and phase-dependent

Green agents on OZ subnets can only access servers on their paired RZ (and vice versa). This is a small volume of cross-subnet traffic. Blocking RZ -> OZ will prevent this green traffic, but the impact is minimal since:
- During active phases, green on the active OZ is already isolated
- During non-active phases, the green traffic is legitimate and not harmful

### 6. The comms_policy observation vector correctly reflects blocking

Verified via observation vector parsing: the comms_policy bits in the observation accurately encode which subnet pairs are blocked per phase. The agent can read `comms_policy_vec` at offset 18-26 (per subnet) to determine current blocking status.

---

## Appendix: Comms Policy Observation Encoding

The observation vector for each agent's subnet contains:
```
offset 0:     mission_phase (0, 1, or 2)
Per subnet:
  offset 1-9:   subnet one-hot (9 bits)
  offset 10-18: blocked_subnets (9 bits) — actual state.blocks
  offset 19-27: comms_policy (9 bits) — 1 = should NOT be allowed
  offset 28+:   proc_flags, conn_flags per host
After all subnets:
  messages (32 bits)
  malfile_flags (V2 extension)
```

`comms_policy_vec[i] = 1` means traffic between this subnet and subnet i is NOT in the comms_policy graph (should be blocked). This is the NEGATED adjacency matrix.

`blocked_subnets[i] = 1` means traffic from subnet i to this subnet is ACTUALLY blocked in `state.blocks` (either by blue agent action or by comms_policy enforcement).

Note: `comms_policy` is the INTENDED policy. `blocked_subnets` is the ACTUAL firewall state. They may differ if blue has issued additional BlockTrafficZone/AllowTrafficZone actions.
