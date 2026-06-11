# Formal Verification Properties for CAGE Challenge 4

## Abstract

This document identifies and formally states properties of the CAGE Challenge 4 (CC4) cybersecurity simulation environment and the V11a heuristic blue-team agent that are amenable to formal verification. Properties are categorized by domain (red FSM, blue policy, environment invariants, stochastic dynamics, compositionality) and annotated with appropriate verification methods (model checking, theorem proving, probabilistic model checking, statistical verification).

All notation follows standard conventions: universally quantified variables use $\forall$, existential quantification uses $\exists$, probability is denoted $\mathbb{P}[\cdot]$, and expectation $\mathbb{E}[\cdot]$. State variables are drawn from the finite state space of the CC4 simulator.

---

## 1. Definitions and Notation

### 1.1 State Space

Let $\mathcal{H}$ denote the set of all hosts in the network. Each host $h \in \mathcal{H}$ has a red-state variable $\sigma(h) \in \mathcal{S}$ where:

$$\mathcal{S} = \{K, KD, S, SD, U, UD, R, RD, F\}$$

Define a threat-level ordering $\tau : \mathcal{S} \to \{0, 1, 2, 3, 4, 5, 6, 7, 8\}$:

$$\tau(K) = 0,\; \tau(KD) = 1,\; \tau(S) = 2,\; \tau(SD) = 3,\; \tau(U) = 4,\; \tau(UD) = 5,\; \tau(R) = 6,\; \tau(RD) = 7,\; \tau(F) = 8$$

The "D" (deception-aware) variants encode red's knowledge of decoys. $F$ denotes a foreign (out-of-scope) host.

### 1.2 Action Space

Red actions are indexed $a \in \{0, 1, \ldots, 8\}$ corresponding to:

| Index | Action | Duration (steps) |
|-------|--------|-------------------|
| 0 | DiscoverRemoteSystems (DRS) | 1 |
| 1 | AggressiveServiceDiscovery (ASD) | 1 |
| 2 | StealthServiceDiscovery (SSD) | 3 |
| 3 | DiscoverDeception (DD) | 2 |
| 4 | ExploitRemoteService (ERS) | 4 |
| 5 | PrivilegeEscalate (PE) | 2 |
| 6 | Impact (IMP) | 2 |
| 7 | DegradeServices (DEG) | 2 |
| 8 | Withdraw (W) | 1 |

### 1.3 Transition Matrices

Let $T^+_{s,a}$ denote the successor state on success, $T^-_{s,a}$ the successor state on failure, and $\Pi_{s,a}$ the probability of selecting action $a$ in state $s$. These are given by the code-defined matrices `state_transitions_success`, `state_transitions_failure`, and `state_transitions_probability` respectively. A value of `None` indicates the action is not available in that state.

### 1.4 Episode Structure

An episode consists of $N = 500$ steps partitioned into three phases:
- Phase 0 (Preplanning): steps $[0, 166]$, $|P_0| = 167$
- Phase 1 (Mission A): steps $[167, 333]$, $|P_1| = 167$
- Phase 2 (Mission B): steps $[334, 499]$, $|P_2| = 166$

---

## 2. Red FSM Properties

### P1: Monotonicity of Red Progression (Absent Blue Intervention)

**Statement.** Without blue intervention (no Remove, no Restore), and excluding Withdraw (which has probability 0 in all reachable states), the red FSM is monotonically non-decreasing in threat level.

$$\forall h \in \mathcal{H},\; \forall a \in \{0,\ldots,7\},\; \forall s \in \mathcal{S} \setminus \{F\}:$$
$$\Pi_{s,a} > 0 \implies \tau(T^+_{s,a}) \geq \tau(s) \;\wedge\; \tau(T^-_{s,a}) \geq \tau(s)$$

**Proof sketch.** By exhaustive inspection of the success and failure transition matrices:

- From $K$: success on ASD/SSD yields $S$ ($\tau = 2 > 0$); DRS yields $KD$ ($\tau = 1 > 0$). Failure stays in $K$.
- From $S$: success on ERS yields $U$ ($\tau = 4 > 2$); DD stays $S$; DRS yields $SD$. Failure stays in $S$.
- From $U$: success on PE yields $R$ ($\tau = 6 > 4$); DRS yields $UD$. Failure stays in $U$.
- From $R$: DRS yields $RD$; Impact/Degrade stay $R$/$RD$. Failure stays in $R$.
- D-variants follow the same monotonicity with $\tau(XD) > \tau(X)$.

Withdraw ($a = 8$) has $\Pi_{s,8} = 0.0$ for all reachable states $s \in \{U, UD, R, RD\}$ where it would cause regression. $\square$

**Verification method.** Model checking (SPIN/TLA+): enumerate all $(s, a)$ pairs where $\Pi_{s,a} > 0$ and verify $\tau(T^{\pm}_{s,a}) \geq \tau(s)$.

---

### P2: Guaranteed Escalation from UD

**Statement.** From state $UD$, PrivilegeEscalate is selected with probability 1.0 and always succeeds. Therefore, any host reaching $UD$ transitions to $RD$ within exactly $d_{PE} = 2$ steps with probability 1.

$$\sigma(h) = UD \implies \Pi_{UD, 5} = 1.0$$
$$T^+_{UD, 5} = RD$$
$$\mathbb{P}[\text{PE succeeds}] = 1.0$$
$$\therefore\; \mathbb{P}[\sigma_{t+2}(h) = RD \mid \sigma_t(h) = UD] = 1.0$$

**Verification method.** Direct inspection of `state_transitions_probability['UD']` = `[None, None, None, None, None, 1.0, None, None, 0.0]` and `state_transitions_success['UD'][5]` = `'RD'`. PrivilegeEscalate success rate = 100% is an environment constant.

---

### P3: Minimum Time to Impact

**Statement.** Define $d_{\min}(s, R)$ as the minimum number of steps from state $s$ to reaching state $R$ or $RD$ (prerequisite for Impact).

From state $U$ on a host within the agent's subnet:
$$d_{\min}(U, R) = d_{PE} = 2 \text{ steps}$$
$$d_{\min}(U, \text{Impact complete}) = d_{PE} + d_{IMP} = 2 + 2 = 4 \text{ steps}$$

From state $K$ on a remote host (minimum-duration path: ASD then ERS then PE then IMP):
$$d_{\min}(K, \text{Impact complete}) = d_{ASD} + d_{ERS} + d_{PE} + d_{IMP} = 1 + 4 + 2 + 2 = 9 \text{ steps}$$

This is a hard lower bound. The expected time is much longer due to:
- ERS success probability $\approx 24.9\%$ per attempt
- Action selection probabilities: $\Pi_{K,1} = 0.25$, $\Pi_{S,4} = 0.50$

**Verification method.** Shortest-path computation on the FSM directed graph with edge weights equal to action durations, restricted to positive-probability transitions. Probabilistic model checking (PRISM) for expected-time computation.

---

### P4: Session Removal State Reset

**Statement.** When blue removes a red session from host $h$ (via Remove or Restore), the red agent's internal state for $h$ transitions to $KD$:

$$\text{blue\_removes\_session}(h) \implies \sigma'(h) = KD$$

This holds regardless of the prior state. From $KD$, red must re-execute service discovery and re-exploit.

**Verification method.** Code-level inspection of `_session_removal_state_change()` (line 188): `self.host_states[ip]['state'] = 'KD'`. Verifiable by symbolic execution or unit test assertion.

---

### P5: Impact Boundedness

**Statement.** Let $\mathcal{H}_{OT} \subseteq \mathcal{H}$ denote hosts with OTService. By construction:

$$\mathcal{H}_{OT} = \{\texttt{operational\_zone\_a\_subnet/server\_host\_0},\; \texttt{operational\_zone\_b\_subnet/server\_host\_0}\}$$

Impact can only succeed on hosts with OTService:

$$\forall h \in \mathcal{H}:\; \text{Impact}(h) \text{ succeeds} \implies h \in \mathcal{H}_{OT}$$

$$\therefore\; |\{h : \text{Impact}(h) \text{ possible}\}| \leq 2$$

**Verification method.** Static analysis of scenario generator configuration. Model checking over host-service assignments.

---

### P6: DegradeServices Reliability Degradation

**Statement.** Let $\rho_t(h) \in [0, 100]$ denote the service reliability of host $h$ at step $t$. Initial condition: $\rho_0(h) = 100$ for all $h$. Each successful DegradeServices call reduces reliability by exactly 20:

$$\rho_{t+d_{DEG}}(h) = \max(0, \rho_t(h) - 20)$$

After $n$ successful DegradeServices calls without intervening Restore:

$$\rho(h) = \max(0, 100 - 20n)$$

At $n = 5$: $\rho(h) = 0$, causing deterministic failure of all GreenLocalWork on $h$. DegradeServices is permanent; only Restore resets $\rho(h)$ to 100 (host reimaging).

**Verification method.** Bounded model checking. The `Service.degrade_service_reliability()` method (with default `value=20`) and the `max(0, ...)` floor are directly verifiable from the `Service` class. The invariant $\rho(h) \in [0, 100]$ is maintained by construction.

---

## 3. Blue Agent Properties (V11a)

### P7: False Positive Filtering via Flag-Age Thresholding

**Statement.** Define $\text{flag\_age}(h, t) = t - t_{\text{first}}(h)$ where $t_{\text{first}}(h)$ is the step when `proc_flag(h)` first appeared. Green false-positive process events persist for exactly 1 step (cleared by Monitor, visible via `old_process_creation` for 1 additional step).

With threshold $\theta = 1$:

$$\mathbb{P}[\text{flag\_age}(h,t) \geq 1 \mid \text{green FP}] = 0$$

because green FP flags are cleared before they can persist to the next observation cycle. For real red presence:

$$\mathbb{P}[\text{flag\_age}(h,t) \geq 1 \mid \text{real red session on } h] > 0.99$$

because a red session maintains the process across steps.

**Corollary.** The V11a policy with $\theta = 1$ achieves:
- False positive rate (unnecessary Restore on green FP): 0%
- False negative rate (missed real red with persistence): < 1%

**Verification method.** Theorem proving (Lean/Coq): formalize the Monitor observation lifecycle and prove flag persistence properties. Statistical verification: Monte Carlo sampling to bound the FP/FN rates.

---

### P8: Restore Dominance over Remove for Escalated Hosts

**Statement.** Given:
- $d_{PE} = 2$ (PrivilegeEscalate duration)
- $d_{Remove} = 3$ (Remove duration)
- $\mathbb{P}[\text{PE succeeds}] = 1.0$
- Remove eliminates user sessions only; root sessions survive

Let $t_0$ be the step when red begins PrivilegeEscalate on host $h$ (state $U$ or $UD$). At $t_0 + 2$, red has root access. Remove, initiated at any $t \geq t_0$, completes at $t + 3$. Since Remove cannot evict root sessions:

$$\forall t \geq t_0:\; t + d_{Remove} = t + 3 > t_0 + 2$$

is not sufficient; the issue is that even when Remove completes, it fails to remove root sessions. Therefore:

$$\sigma(h) \in \{U, UD, R, RD\} \implies \text{Remove}(h) \text{ is weakly dominated by } \text{Restore}(h)$$

Restore ($d_{Restore} = 5$) clears ALL sessions including root. Despite higher duration cost, it is strictly dominant for any host where red may have escalated.

**Verification method.** Theorem proving. The dominance relation follows from: (1) Remove's inability to clear root sessions, and (2) PrivilegeEscalate's guaranteed success.

---

### P9: Zero-Cost Preemptive Blocking in Active Phases

**Statement.** Let $\text{ASF}(s, p)$ denote the AccessServiceFails penalty for subnet $s$ in phase $p$. From the reward table:

$$\text{ASF}(\texttt{operational\_zone\_a\_subnet}, 1) = 0$$
$$\text{ASF}(\texttt{operational\_zone\_b\_subnet}, 2) = 0$$

Therefore, BlockTrafficZone targeting the active operational zone during its mission phase incurs exactly zero ASF penalty. The only costs are:
1. The action slot (1 step of agent time)
2. Any ASF during Phase 0 where $\text{ASF}(\texttt{OZA}, 0) = -1$

**Formal statement.** Let $B(s_1, s_2, t)$ denote the action of blocking traffic from $s_1$ to $s_2$ at step $t$, and $C(B)$ the immediate reward cost:

$$t \in P_1 \implies C(B(\cdot, \texttt{OZA}, t)) = 0$$
$$t \in P_2 \implies C(B(\cdot, \texttt{OZB}, t)) = 0$$

**Verification method.** Direct inspection of the reward table in `BlueRewardMachine.get_phase_rewards()`. Model checking with reward annotations.

---

### P10: Decoy Coverage Probability Bound

**Statement.** With $k = $ MAX_DECOYS $ = 3$ decoy services deployed on a host, covering 3 exploit PIDs, and a total exploit pool of size $|E| \approx 5$:

The `DefaultExploitActionSelector` removes the highest-weight exploit and uniformly samples from the remaining $|E| - 1$ exploits. The `PIDSelectiveExploitActionSelector` (used by red after DiscoverDeception) excludes known decoy PIDs.

For blind red (no deception discovery):

$$\mathbb{P}[\text{hit decoy} \mid \text{blind exploit}] = \frac{k}{|E| - 1} = \frac{3}{4} = 0.75$$

For informed red (after successful DiscoverDeception on all $k$ decoys):

$$\mathbb{P}[\text{hit decoy} \mid \text{informed exploit}] = 0$$

**Verification method.** Probabilistic analysis of `DefaultExploitActionSelector` and `PIDSelectiveExploitActionSelector`. Statistical verification via Monte Carlo sampling of exploit selection.

---

### P11: Phase Transition Vulnerability Window

**Statement.** At phase boundaries, comms_policy changes dictate new blocking requirements. Without preemptive blocking:

$$T_{\text{block}}^{\text{reactive}} \in [167, 167 + d_{Restore}] = [167, 172]$$

where the delay arises from the blue agent potentially being busy with a Restore action when the phase transition occurs.

With V11a preemptive blocking (blocking 10 steps before the transition):

$$T_{\text{block}}^{\text{V11a}} \leq 157$$

Define the vulnerability window $W = T_{\text{block}} - 167$ (positive means blocks are late):

$$W^{\text{V11a}} \leq 157 - 167 = -10 < 0 \quad \text{(always closed before transition)}$$
$$W^{\text{reactive}} \in [0, 5] \quad \text{(potentially open for up to 5 steps)}$$

During an open window, red can exploit the path from $\texttt{RZA} \to \texttt{OZA}$ via ExploitRemoteService, bypassing the intended network isolation.

**Verification method.** Bounded model checking (TLA+): model the agent's action queue and verify that for all reachable states at step 167, the required blocks are in place. Statistical verification: measure $T_{\text{block}}$ distribution over episodes.

---

## 4. Environment Invariants

### P12: Reward Non-Positivity

**Statement.** The reward function $r : \mathcal{S}^{|\mathcal{H}|} \times \mathcal{A} \to \mathbb{R}$ satisfies:

$$\forall t \in [0, N-1]:\; r_t \leq 0$$

No positive rewards exist in the default reward mode. The maximum achievable episode return is:

$$R^* = \sum_{t=0}^{N-1} r_t = 0$$

achieved only when no green action fails, no red Impact succeeds, and no blue Restore is executed unnecessarily.

**Exception.** The `reward_blue_actions` flag, when enabled, awards $+0.1$ for blue Analyse actions, but this is non-default.

**Verification method.** Exhaustive enumeration of reward table entries in `BlueRewardMachine`. All LWF, ASF, RIA values are $\leq 0$ in the default mode.

---

### P13: Comms Policy Network Isolation Completeness

**Statement.** In Phase 1, the comms_policy mandates blocking all 8 directed paths to $\texttt{OZA}$ from non-OZA subnets. Let $\mathcal{N}(\texttt{OZA})$ be the set of subnets with routable paths to OZA.

$$|\mathcal{N}(\texttt{OZA})| = 8$$

If all 8 blocks are active:

$$\forall s \in \mathcal{N}(\texttt{OZA}):\; \text{blocked}(s, \texttt{OZA}) = \text{true}$$
$$\implies \forall h_s \in s,\; h_d \in \texttt{OZA}:\; \text{ExploitRemoteService}(h_s \to h_d) \text{ fails}$$

because `ExploitRemoteService.execute()` checks `blocking_host()` and returns failure if the path is blocked.

**Residual attack vector.** PhishingEmail bypasses blocks (see P14).

**Verification method.** Model checking: enumerate comms_policy vectors for each phase and verify completeness of blocking requirements.

---

### P14: Phishing Bypass of Firewall Blocks

**Statement.** `PhishingEmail` uses `check_routable()` (link-layer connectivity graph) rather than `blocking_host()` (firewall state). Therefore:

$$\forall s_1, s_2 \in \mathcal{S}_{\text{subnets}}:\; \text{blocked}(s_1, s_2) = \text{true} \;\not\!\!\!\implies \text{PhishingEmail}(s_1 \to s_2) \text{ fails}$$

Formally, PhishingEmail succeeds if and only if $s_1$ and $s_2$ are in the same connected component of the physical network topology (which is always true in CC4, as no physical links are severed):

$$\text{PhishingEmail}(h_s \to h_d) \text{ succeeds} \iff \text{check\_routable}(h_s, h_d) = \text{true}$$

This is independent of BlockTrafficZone state.

**Verification method.** Code-level theorem proving or symbolic execution of `PhishingEmail.execute()`, verifying the absence of any `blocking_host()` call.

---

### P15: Monitor Idempotency

**Statement.** Monitor executes automatically at the end of every step for every blue agent, independent of the agent's chosen action. Therefore, explicitly choosing Monitor as an action provides no additional information:

$$\text{obs}(\text{Monitor}) = \text{obs}(\text{any other action})$$

The Monitor action is strictly dominated by every other action:

$$\forall a \in \mathcal{A}_{\text{blue}} \setminus \{\text{Monitor}\}:\; a \succeq \text{Monitor}$$

with strict dominance for any $a$ that has positive utility (e.g., blocking, restoring).

**Verification method.** Code-level analysis of `SimulationController._execute_step()` confirming Monitor runs unconditionally post-action.

---

### P16: Block Persistence

**Statement.** `BlockTrafficZone` has `duration = 1` but its effects persist indefinitely. Once a subnet pair $(s_1, s_2)$ is blocked:

$$\text{blocked}(s_1, s_2, t) = \text{true} \implies \forall t' > t:\; \text{blocked}(s_1, s_2, t') = \text{true}$$

until explicitly reversed by $\text{AllowTrafficZone}(s_1, s_2)$.

This is a state-modifying action, not a time-limited effect. The `duration = 1` parameter refers to agent busy time, not effect duration.

**Verification method.** Model checking: verify that the `blocked` state variable is only modified by `AllowTrafficZone`. Code-level analysis of the firewall state transition.

---

## 5. Stochastic Properties

### P17: Expected Time to First Exploit (DTMC Analysis)

**Statement.** The time from state $K$ to state $U$ (first successful exploit) can be modeled as a discrete-time Markov chain (DTMC) with states $\{K, KD, S, SD, U\}$ where $U$ is absorbing.

Transition probabilities per action selection round (accounting for multi-step durations):

From $K$ ($\Pi_K = [0.5, 0.25, 0.25, -, -, -, -, -, -]$):
- DRS (1 step): stays $K$ or $KD$ (success reveals hosts, DRS on K→KD means decoy-aware)
- ASD (1 step, success $\to S$, failure $\to K$)
- SSD (3 steps, success $\to S$, failure $\to K$)

From $S$ ($\Pi_S = [0.25, -, -, 0.25, 0.50, -, -, -, -]$):
- DRS (1 step): $\to SD$
- DD (2 steps): stays $S$ regardless of success/failure
- ERS (4 steps, 24.9% success $\to U$, 75.1% failure $\to S$)

This yields a system of linear equations for expected hitting time $\mathbb{E}[T_{K \to U}]$. Let $E_K, E_S$ be the expected time-to-$U$ from $K$ and $S$ respectively.

$$E_K = \underbrace{0.5 \cdot (1 + E_K)}_{\text{DRS}} + \underbrace{0.25 \cdot (1 + p_{ASD} \cdot E_S + (1-p_{ASD}) \cdot E_K)}_{\text{ASD}} + \underbrace{0.25 \cdot (3 + p_{SSD} \cdot E_S + (1-p_{SSD}) \cdot E_K)}_{\text{SSD}}$$

$$E_S = \underbrace{0.25 \cdot (1 + E_S)}_{\text{DRS}} + \underbrace{0.25 \cdot (2 + E_S)}_{\text{DD}} + \underbrace{0.50 \cdot (4 + 0.249 \cdot 0 + 0.751 \cdot E_S)}_{\text{ERS}}$$

where $p_{ASD}, p_{SSD}$ are the success probabilities of service discovery (environment-dependent, typically high for ASD, lower for SSD).

**Verification method.** Probabilistic model checking (PRISM, Storm): encode the DTMC and compute $\mathbb{E}[T_{K \to U}]$ exactly. Monte Carlo validation.

---

### P18: Deception Discovery Trade-off

**Statement.** Red's decision between DiscoverDeception (DD, 2 steps) and immediate ExploitRemoteService (ERS, 4 steps) from state $S$ admits a formal comparison.

**Path A (exploit directly from $S$):**
$$\mathbb{P}[\text{ERS success} \mid S] = 0.249 \cdot (1 - P_{\text{decoy}})$$

where $P_{\text{decoy}} = k / (|E| - 1)$ is the probability of hitting a decoy.

$$\mathbb{E}[T_{S \to U} \mid \text{exploit directly}] = \frac{4}{0.249 \cdot (1 - P_{\text{decoy}})} \cdot \Pi_{S,4}^{-1}$$

**Path B (discover deception first, then exploit from $SD$):**
- DD success (50% TP rate): transitions to $S$ (decoy identified, excluded from future exploits)
- DD failure (50%): stays $S$
- From $SD$: $\Pi_{SD,4} = 0.75$, exploits avoid known decoys

$$\mathbb{E}[T_{S \to U} \mid \text{DD first}] = 2 + 0.5 \cdot \mathbb{E}[T_{SD \to U}] + 0.5 \cdot \mathbb{E}[T_{S \to U}]$$

The trade-off depends on the number of decoys $k$ and pool size $|E|$. For $k = 3, |E| = 5$:

$$P_{\text{decoy}} = 0.75 \text{ (blind)}, \quad P_{\text{decoy}}^{\text{informed}} = 0$$

The net effective ERS success rate improves from $0.249 \times 0.25 = 0.062$ (blind) to $0.249$ (informed), a 4x improvement at the cost of 2 steps per DD attempt (50% success rate).

**Verification method.** Probabilistic model checking (PRISM). Closed-form comparison via renewal theory.

---

### P19: Restore Cost-Benefit Threshold

**Statement.** For host $h$ in subnet $s$ during phase $p$, define:

- $C_{\text{restore}}(s, p)$: cost of executing Restore (action slot cost + green failures during 5-step duration)
- $B_{\text{restore}}(s, p, t)$: benefit of restoring (avoided future degradation/impact over remaining steps)

$$C_{\text{restore}} = -1 + \text{LWF}(s, p) \times d_{Restore} \times n_{\text{green}}(s)$$

where $n_{\text{green}}(s)$ is the number of green agents performing GreenLocalWork on subnet $s$ each step.

$$B_{\text{restore}} = \text{LWF}(s, p) \times \mathbb{E}[\text{degraded steps remaining}] + \text{RIA}(s, p) \times \mathbb{P}[\text{impact}]$$

For $\texttt{OZA}$ during Phase 1: $\text{LWF} = -10$, $\text{RIA} = -10$.

$$C_{\text{restore}}(\texttt{OZA}, 1) = -1 + (-10)(5) = -51$$

Restore is beneficial when:

$$|B_{\text{restore}}| > |C_{\text{restore}}|$$
$$\iff \mathbb{E}[\text{degraded steps}] > \frac{51}{|\text{LWF}|} = 5.1 \text{ steps}$$

With $\sim 160$ steps remaining in Phase 1, even a small probability of continued degradation makes Restore overwhelmingly beneficial.

**Verification method.** Analytical computation. Statistical verification via episode sampling.

---

## 6. Compositional Properties

### P20: Agent Independence and Monotone Improvability

**Statement.** The five blue agents operate independently with no shared mutable state. The joint policy $\pi = \pi_0 \times \pi_1 \times \pi_2 \times \pi_3 \times \pi_4$ is a product of independent policies.

**Monotonicity property:** Improving any single agent's policy (weakly) improves the joint return.

$$\forall i \in \{0,\ldots,4\},\; \forall \pi_i' \succeq \pi_i:\; V(\pi_0, \ldots, \pi_i', \ldots, \pi_4) \geq V(\pi_0, \ldots, \pi_i, \ldots, \pi_4)$$

where $\pi_i' \succeq \pi_i$ means $\pi_i'$ achieves weakly higher expected return than $\pi_i$ on agent $i$'s controlled subnets, holding other agents fixed.

**Coordination gap bound.** Agents share information only via 8-bit messages. The per-step information transfer is bounded:

$$I_{\text{coord}} \leq 8 \times 4 = 32 \text{ bits per agent per step}$$

The gap between the optimal joint policy and the product of optimal independent policies is bounded by the mutual information achievable through this channel.

**Verification method.** Theorem proving (Lean/Coq): formalize independence and monotonicity. Information-theoretic bounds via channel capacity analysis.

---

### P21: Phase Decomposability and Temporal Dependence

**Statement.** Let $R_p$ denote the cumulative reward during phase $p$. The total return decomposes:

$$R = R_0 + R_1 + R_2$$

However, $R_1$ depends on the state at the end of Phase 0 (red footholds, decoy deployments, block states):

$$\mathbb{E}[R_1] = f(\mathcal{S}_{167})$$

where $\mathcal{S}_{167}$ is the full system state at step 167. This temporal dependence prevents independent optimization:

$$\arg\max_{\pi} \mathbb{E}[R] \neq \arg\max_{\pi_0} \mathbb{E}[R_0] \times \arg\max_{\pi_1} \mathbb{E}[R_1] \times \arg\max_{\pi_2} \mathbb{E}[R_2]$$

**Verification method.** Counterexample construction: a policy that sacrifices Phase 0 performance (e.g., spending actions on preemptive blocking) but improves Phase 1 performance by eliminating the vulnerability window (P11).

---

## 7. Verification Methodology Summary

| Property | Class | Method | Tool |
|----------|-------|--------|------|
| P1 | FSM invariant | Model checking | TLA+, SPIN |
| P2 | Deterministic reachability | Model checking | TLA+, SPIN |
| P3 | Shortest path | Graph analysis | PRISM (weighted DTMC) |
| P4 | State reset | Code verification | Unit tests, symbolic execution |
| P5 | Structural constraint | Static analysis | Scenario config enumeration |
| P6 | Bounded degradation | Bounded model checking | SPIN, CBMC |
| P7 | FP/FN rates | Theorem proving + statistics | Lean/Coq + Monte Carlo |
| P8 | Strategy dominance | Theorem proving | Lean/Coq |
| P9 | Reward structure | Direct verification | Reward table inspection |
| P10 | Probabilistic coverage | Probabilistic analysis | PRISM, statistical testing |
| P11 | Temporal safety | Bounded model checking | TLA+ |
| P12 | Reward invariant | Exhaustive enumeration | Automated table scan |
| P13 | Network isolation | Model checking | TLA+ |
| P14 | Bypass property | Code-level verification | Symbolic execution |
| P15 | Action dominance | Theorem proving | Lean/Coq |
| P16 | State persistence | Model checking | SPIN, TLA+ |
| P17 | Expected hitting time | Probabilistic model checking | PRISM, Storm |
| P18 | Strategy comparison | Probabilistic analysis | PRISM, renewal theory |
| P19 | Cost-benefit threshold | Analytical + statistical | Closed-form + Monte Carlo |
| P20 | Compositionality | Theorem proving | Lean/Coq |
| P21 | Temporal dependence | Counterexample | Constructive proof |

---

## 8. Open Questions for Future Verification

1. **Optimal decoy placement**: Given $k = 3$ decoy slots per host and heterogeneous exploit pools, what placement maximizes $\mathbb{E}[\text{red steps wasted}]$?

2. **Message protocol optimality**: Is the 8-bit v9 message encoding Pareto-optimal with respect to coordination gain and channel capacity?

3. **Multi-agent equilibrium**: Does the CC4 game (5 blue vs. 6 red) have a Nash equilibrium in finite-memory strategies? What is the value of the game?

4. **Phase 0 action budget allocation**: Given the temporal dependence (P21), what is the optimal split of Phase 0 actions between decoy deployment, preemptive blocking, and red response?

5. **Robustness to red policy variation**: Properties P1-P3 assume the base FSM policy. How do the verification results change under variant red agents (e.g., `FiniteStateRedAgentMeander`, `FiniteStateRedAgentSleep`)?
