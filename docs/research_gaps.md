# Research Gaps and Open Questions in CAGE Challenge 4 Blue-Team Defense

## Abstract

This document identifies unsolved problems, open research questions, and opportunities for improvement in the CAGE Challenge 4 (CC4) autonomous cyber defense task. The current best heuristic agent (V11a) achieves a mean reward of -717.0 +/- 202.9 over 100 episodes, capturing 97.6% of the gap between a non-defending SleepAgent (-30,579) and the estimated theoretical floor (~-300). The remaining ~417 reward-point gap, while small in relative terms, represents a scientifically meaningful optimization frontier. We identify six concrete research gaps, formulate them as tractable problems, and assess the potential of Deep Reinforcement Learning (DRL) to address each. We also document fundamental limits of the environment that constrain any defense strategy.

---

## 1. Known Performance Bounds

### 1.1 Empirical Bounds

| Agent | Mean Reward | % of Gap Captured | Description |
|-------|-------------|-------------------|-------------|
| SleepAgent | -30,579 | 0.0% | No defensive actions; establishes the worst-case baseline |
| V11a Heuristic | -717.0 +/- 202.9 | 97.6% | Best known agent; reactive defense with preemptive OZ blocking |
| Theoretical Floor | ~-300 | 99.0% | Estimated lower bound on achievable penalty under perfect play |

The total defensible gap is approximately 30,279 reward points (from SleepAgent to theoretical floor). V11a closes 29,862 of these points, leaving approximately 417 points unrecovered.

### 1.2 Decomposition of the Remaining Gap

The residual penalty of ~717 points per episode can be attributed to the following irreducible or partially-reducible sources:

| Source | Contribution | Reducible? |
|--------|-------------|------------|
| DegradeServices damage (invisible to agents) | ~83.3% of LWF penalties | Partially, via indirect inference |
| Green agent false positives (FP rate = 0.776%) | ~3-5% of unnecessary Restores | Partially, via persistence tracking |
| Red exploit successes (24.9% success rate) | Variable | Not directly controllable |
| Action budget constraint (5 agents x 1 action/step) | Structural | Irreducible |

### 1.3 Experimental Exhaustion of Single-Parameter Improvements

A systematic sweep of 22 single-parameter modifications to V10b found zero statistically significant improvements (p < 0.05). The sole architectural change that yielded improvement was preemptive Operational Zone (OZ) blocking, which produced a +11.9% gain (from V10b to V11a). A messaging protocol redesign (V11b) yielded no measurable improvement (p = 0.80). This exhaustion of the single-parameter search space suggests that further gains require either multi-parameter interactions, fundamentally new mechanisms, or learned policies.

---

## 2. Open Research Gaps

### 2.1 Gap 1: DegradeServices Invisibility

#### Problem Statement

The red agent's `DegradeServices` action accounts for 83.3% of total damage via Loss-of-Work-Function (LWF) penalties. Unlike `ExploitRemoteService` and `PrivilegeEscalate`, `DegradeServices` produces zero observable signals in the blue agent's observation space: no `conn_flag`, no `proc_flag`, no `malfile` indicator. The action has a 100% success rate, requires root access, has a duration of 2 steps, and reduces service reliability by 20% per invocation. This creates a fundamental information asymmetry: the dominant damage source is entirely invisible to the defending agents.

#### Observable Indirect Evidence

Despite direct invisibility, two categories of indirect evidence exist:

1. **Precursor chain observability.** The red attack kill-chain requires `ExploitRemoteService` (generates `conn_flag` with high probability) followed by `PrivilegeEscalate` (generates `malfile` indicator) before `DegradeServices` can execute. A host that has been exploited and privilege-escalated is therefore a candidate for active degradation.

2. **Service reliability decay.** `GreenLocalWork` failure rates increase as services are degraded. While individual failures are noisy (green agents have intrinsic failure rates), a sustained increase in failure rate on a specific host constitutes statistical evidence of degradation.

#### Research Questions

**RQ 1.1:** Can shadow state tracking -- maintaining a posterior probability of degradation based on observed precursor events -- improve defensive response timing and reduce LWF penalties?

**RQ 1.2:** What is the optimal re-Restore frequency for hosts with confirmed prior compromise? Uniform proactive Restore at N=30 step intervals was tested and proved catastrophic (-461.2 worse than baseline), but this does not rule out targeted, gated re-Restore strategies conditioned on compromise history.

**RQ 1.3:** Is there a Bayesian estimator for the number of active `DegradeServices` calls on a host, given the sequence of observed `conn_flag` and `malfile` events and the known red action model?

#### Formal Problem

Let $h_t$ denote the hidden degradation state of a host at time $t$, and let $o_{1:t}$ denote the sequence of observations. The problem is to estimate $P(h_t | o_{1:t})$ and derive an optimal Restore policy $\pi^*(a_t | P(h_t | o_{1:t}))$ that minimizes expected cumulative LWF penalty subject to the constraint that Restore incurs its own service disruption cost.

---

### 2.2 Gap 2: Inter-Agent Coordination Failure

#### Problem Statement

The CC4 environment deploys 5 independent blue agents, one per subnet. The agents share no observations and can only communicate through an explicit messaging protocol. Two messaging protocol designs (V9 messaging, V11b redesign) were tested against no-messaging baselines; neither produced statistically significant improvement (p = 0.80 for the most recent test).

#### Root Cause Analysis

Three structural factors explain the coordination failure:

1. **comms_policy substitution.** The environment's `comms_policy` mechanism already provides implicit coordination by blocking network paths. When agents enforce comms_policy, they achieve the primary coordination benefit (attack path disruption) without requiring inter-agent communication.

2. **Sparse upstream relationships.** Analysis shows that 86.7% of (phase, agent) combinations have no upstream dependency. Only 2 of 15 possible (phase, agent) pairs have upstream mappings, meaning most agents have no actionable information to share.

3. **Action latency.** Messages arrive one step after sending. Given that red actions also take 1-2 steps, the coordination window is too narrow for reactive messaging to provide value.

#### Research Questions

**RQ 2.1:** Does there exist a messaging protocol that provides defensive value strictly beyond what `comms_policy` enforcement achieves? A formal proof of the value (or non-value) of communication in this POMDP would resolve this question.

**RQ 2.2:** Can agents coordinate Restore schedules to avoid simultaneous downtime on dependent hosts? If Host A and Host B are both compromised, restoring both simultaneously may cause correlated service disruption, whereas staggering Restores may reduce peak penalty.

**RQ 2.3:** What is the theoretical value of information (VOI) for perfect inter-agent observation sharing, compared to independent action with comms_policy?

#### Formal Problem

Model the 5-agent system as a Decentralized POMDP (Dec-POMDP). Let $V^*_{comm}$ denote the optimal value under communication and $V^*_{ind}$ denote the optimal value under independent policies with comms_policy. The open question is to bound $V^*_{comm} - V^*_{ind}$ and determine whether the bound is practically significant given the action budget constraint.

---

### 2.3 Gap 3: Phase 0 Optimization

#### Problem Statement

Phase 0 spans steps 0-166 (33% of the 500-step episode). During Phase 0, the Available Service Factor (ASF) penalty for blocking is -1 (non-free), compared to Phase 1-2 where blocking follows comms_policy at zero cost. The current V11a strategy during Phase 0 consists of deploying decoys, enforcing comms_policy, and responding reactively to alerts. No preemptive blocking is performed in Phase 0 due to the ASF cost.

Analysis of V9.1 penalty decomposition shows that Phase 0 accounts for 58% of total episode penalty, despite comprising only 33% of episode length. This disproportionate contribution suggests Phase 0 is under-defended.

#### Research Questions

**RQ 3.1:** Is blocking the contractor-to-HQ path during Phase 0 a net positive despite the ASF = -1 cost? This requires estimating the expected red damage prevented versus the blocking cost over 167 steps.

**RQ 3.2:** Can Phase 0 be used to pre-position defenses (e.g., strategic decoy placement, early Restore of high-value hosts) that reduce damage in Phases 1 and 2?

**RQ 3.3:** What is the optimal Phase 0 strategy given that red is building footholds for later exploitation? Is there a quantifiable tradeoff between early intervention cost and later damage reduction?

#### Formal Problem

Let $C_{block}(t)$ denote the per-step cost of blocking a path at time $t$ in Phase 0 (ASF = -1), and let $D_{prevent}(t)$ denote the expected damage prevented by that block from time $t$ through the end of the episode. The question is whether $\sum_{t=0}^{166} C_{block}(t) < \sum_{t=0}^{166} D_{prevent}(t)$ for any subset of blockable paths.

---

### 2.4 Gap 4: High Variance and Tail Risk

#### Problem Statement

V11a episodes range from -270 (best observed) to -1525 (worst observed), with a standard deviation of 202.9. This 5.6x range between best and worst episodes indicates that agent performance is highly sensitive to stochastic factors, likely including red agent initial targeting and early exploitation success.

#### Research Questions

**RQ 4.1:** What observable features at steps 0-50 are predictive of eventual episode quality? If early red success in the Operational Zone is the primary driver of catastrophic episodes, early detection of OZ compromise may enable preemptive countermeasures.

**RQ 4.2:** Can tail risk be reduced without sacrificing mean performance? A policy that achieves -750 mean but with standard deviation of 100 may be preferable to -717 mean with standard deviation of 203, depending on the evaluation metric.

**RQ 4.3:** Is there a Pareto frontier between mean reward and variance? If so, what is its shape, and where does V11a sit on it?

#### Formal Problem

Define the risk-adjusted objective as $J(\pi) = E[R(\pi)] - \lambda \cdot \text{CVaR}_\alpha[R(\pi)]$ where $\text{CVaR}_\alpha$ is the Conditional Value at Risk at quantile $\alpha$. The problem is to find $\pi^*$ that maximizes $J(\pi)$ for a given risk-aversion parameter $\lambda$.

---

### 2.5 Gap 5: Persistent conn_flag Tracking

#### Problem Statement

Green agent false positives (FP rate = 0.776%) generate `conn_flag` events that are observationally identical to red exploit attempts. However, green FPs clear within 1 step, while red-generated `conn_flag` events persist for 2+ steps. A `conn_flag` that persists across consecutive observations is therefore almost certainly a true positive.

The current V11a agent uses a fixed `flag_age >= 1` threshold for all signals, which does not exploit this persistence distinction.

#### Research Questions

**RQ 5.1:** Does implementing a threshold of 0 (immediate response) for persistent `conn_flag` (observed in 2+ consecutive steps) reduce response latency to real attacks without increasing false positive responses?

**RQ 5.2:** What is the expected improvement in reward from eliminating the 1-step response delay for persistent signals, given that red completes its kill-chain in 3-4 steps?

#### Formal Problem

Let $FPR_{persist}$ denote the false positive rate for `conn_flag` events persisting 2+ steps. If $FPR_{persist} \approx 0$ (which follows from the green FP clearance model), then the optimal threshold for persistent signals is $\theta_{persist} = 0$, yielding a 1-step reduction in response latency for true positives. The expected reward gain is proportional to the fraction of red exploits that would be caught 1 step earlier, multiplied by the per-step damage rate.

---

### 2.6 Gap 6: Decoy Depletion Awareness

#### Problem Statement

Each host can support at most 3 decoy services. When a red agent exploits a decoy, a `proc_flag` is generated. However, once all 3 decoys on a host have been consumed by red exploits, subsequent `proc_flag` events on that host cannot originate from decoy interactions and are therefore 100% attributable to real red activity.

The current V11a agent does not track decoy depletion state and applies the same detection threshold regardless of remaining decoy capacity.

#### Research Questions

**RQ 6.1:** Does tracking decoy depletion per host and lowering the response threshold on depleted hosts reduce detection latency for real attacks?

**RQ 6.2:** What is the optimal decoy redeployment strategy? Should depleted hosts be replenished immediately (consuming an action), or is the increased detection certainty on depleted hosts more valuable than the decoy's early-warning function?

---

## 3. Deep Reinforcement Learning Opportunities

### 3.1 Problems Well-Suited to DRL

#### 3.1.1 Multi-Agent Coordination via MARL

Multi-Agent Reinforcement Learning (MARL) could learn implicit coordination between the 5 blue agents without requiring explicit messaging protocols. Approaches such as QMIX, MAPPO, or MADDPG can learn decentralized policies that are trained with centralized critics, potentially discovering coordination strategies that hand-designed protocols miss.

**Formulation:**
- State space: Joint observation across all 5 agents (centralized training)
- Action space: Per-agent action selection (decentralized execution)  
- Reward: Shared team reward (total episode penalty)
- Horizon: T = 500 steps
- Baseline: V11a independent agents (no coordination value observed)

**Expected difficulty:** High. The Dec-POMDP is NEXP-hard in general. The observation that comms_policy already provides the primary coordination benefit suggests the marginal value of learned coordination may be small.

#### 3.1.2 Phase-Aware Dynamic Thresholds

Rather than fixed detection thresholds (`flag_age >= 1`), a learned policy could condition thresholds on phase, subnet identity, recent alert history, and estimated red pressure. The optimal threshold likely varies across the episode as red capabilities and blue information evolve.

**Formulation:**
- State: (phase, step_in_phase, subnet_id, recent_alert_count, decoy_status, host_compromise_history)
- Action: threshold in {0, 1, 2} for each signal type
- Reward: per-step penalty (sum of LWF, ASF costs, unnecessary Restore costs)
- Horizon: T = 500
- Baseline: V11a fixed threshold (flag_age >= 1)

**Expected difficulty:** Moderate. The state space is tractable and the action space is small. The challenge is credit assignment: threshold changes affect future outcomes non-linearly.

#### 3.1.3 DegradeServices Inference via Recurrent Policies

A recurrent neural network (LSTM or Transformer) could learn to infer the hidden degradation state from the temporal sequence of visible signals (past exploits, Restore events, reward trends). This directly addresses Gap 1 by learning a function approximation for $P(h_t | o_{1:t})$.

**Formulation:**
- Input: Sequence of per-host observations $(o_1, o_2, \ldots, o_t)$
- Output: Estimated probability of active degradation per host
- Training signal: Reward difference between Restore and no-Restore decisions
- Architecture: LSTM or Transformer encoder with per-host attention

**Expected difficulty:** High. The signal is extremely weak (DegradeServices produces no direct observation), and the reward signal is delayed and confounded by other damage sources.

#### 3.1.4 Optimal Blocking Policy

The current blocking policy is rule-based (follow comms_policy + preemptive OZ blocks). A learned policy could discover blocking strategies that account for predicted attack flows, trading short-term ASF cost for long-term damage prevention.

**Formulation:**
- State: (phase, current_blocks, observed_red_activity_per_subnet, comms_policy)
- Action: Block or Allow for each subnet pair
- Reward: Net penalty (LWF damage prevented minus ASF cost incurred)
- Baseline: V11a rule-based blocking

**Expected difficulty:** Moderate. The action space is manageable (number of subnet pairs is bounded), and the comms_policy provides a strong prior.

#### 3.1.5 Adaptive Restore Scheduling

DRL could learn when to proactively Restore high-value hosts based on compromise history, rather than waiting for observable re-infection. The heuristic experiment showed that uniform proactive Restore is catastrophic, but a learned, selective policy may extract value from targeted re-Restoration.

**Formulation:**
- State: (host_id, steps_since_last_restore, prior_compromise_count, current_phase, observed_red_pressure)
- Action: Restore now or defer
- Reward: LWF penalty reduction minus Restore disruption cost
- Baseline: V11a reactive-only Restore

**Expected difficulty:** Moderate to high. The key challenge is learning the appropriate gate condition: which hosts benefit from proactive Restore, and when.

### 3.2 Problems Where DRL Will Struggle

#### 3.2.1 Credit Assignment over Long Horizons

Each episode contains 5 agents x 500 steps = 2,500 agent-actions. The episode reward is a single scalar aggregating all penalties. Attributing reward changes to specific actions at specific timesteps is an extreme credit assignment problem. Temporal difference methods will propagate credit slowly, and Monte Carlo methods will suffer from high variance.

**Quantitative concern:** With a reward standard deviation of 202.9 and 2,500 actions per episode, the per-action signal-to-noise ratio is approximately 202.9 / sqrt(2500) = 4.06, before accounting for action correlations and delayed effects.

#### 3.2.2 Fundamental Partial Observability

DegradeServices invisibility means the observation space provably does not contain sufficient information for optimal decision-making. No learning algorithm can overcome a fundamental information deficit. At best, DRL can learn the optimal policy given incomplete information, but this optimal-under-partial-observability policy may still incur substantial unavoidable penalty.

#### 3.2.3 Sample Efficiency

Empirically measured simulation throughput is approximately 55 steps/second. At 500 steps per episode, this yields approximately 0.11 episodes/second, or 396 episodes/hour. Modern RL algorithms typically require 10^6 to 10^8 environment interactions for complex tasks. At 396 episodes/hour:

| Training episodes | Wall-clock time |
|-------------------|----------------|
| 100,000 | 10.5 days |
| 1,000,000 | 105 days |
| 10,000,000 | 2.9 years |

Even with parallelization (e.g., 8 environments), training 1M episodes would require approximately 13 days. This is feasible but expensive, and hyperparameter tuning multiplies the cost.

#### 3.2.4 Diminishing Returns Against V11a

V11a already captures 97.6% of the total defensible gap. Any DRL agent must surpass -717.0 to be considered an improvement. If DRL achieves the theoretical floor of -300, the absolute gain is 417 points -- meaningful, but the engineering cost of developing, training, and validating a DRL system may not justify the marginal improvement, depending on the research objectives.

#### 3.2.5 Combinatorial Joint Action Space

Each agent has approximately 100+ possible actions per step (Sleep, Remove per host, Restore per host, Block/Allow per subnet pair, DeployDecoy per host). The joint action space across 5 agents is $|A|^5 \approx 10^{10}$, far too large for tabular methods. Function approximation is required, but the sparse reward signal makes gradient estimation noisy.

### 3.3 Recommended Hybrid Approach: DRL + Heuristic

The most promising research direction combines the strengths of V11a's heuristic with learned components:

1. **Heuristic as base policy.** Use V11a as the default policy for all agent decisions.
2. **Learned override.** Train a DRL agent to output a binary "override" signal and an alternative action. The DRL agent learns WHEN the heuristic is suboptimal, not how to solve the entire task from scratch.
3. **Focused scope.** Restrict DRL learning to the specific gaps identified above: DegradeServices response timing, Phase 0 strategy, and threshold adaptation.
4. **Curriculum learning.** Begin training in a simplified environment (e.g., single subnet, deterministic red) and gradually increase complexity to the full CC4 environment.
5. **Reward shaping.** Decompose the episode reward into per-host, per-step components to accelerate credit assignment. Use the known damage model (DegradeServices = -0.2 reliability per call) to construct auxiliary reward signals.

#### Concrete DRL Experiment Plan

| Experiment | State Space | Action Space | Expected Gain | Training Cost |
|-----------|-------------|-------------|---------------|---------------|
| Threshold adaptation | ~50 features | 3 thresholds x 5 agents | 20-50 points | ~100K episodes |
| Selective proactive Restore | ~30 features per host | Binary per host | 50-150 points | ~500K episodes |
| Phase 0 blocking | ~20 features | Block/Allow per path | 30-80 points | ~200K episodes |
| Full policy (end-to-end) | ~500 features | ~100 actions per agent | 100-400 points | ~5M episodes |

---

## 4. Promising Micro-Optimizations Not Yet Tested

The following modifications are low-cost to implement and evaluate but have not yet been tested:

### 4.1 conn_flag Persistence Tracking

Track `conn_flag` duration per host across consecutive steps. Apply immediate response (threshold = 0) when a `conn_flag` persists for 2+ steps, as green FPs clear within 1 step. Expected impact: 1-step reduction in response latency for a subset of true positives.

### 4.2 Decoy-Depletion-Aware Thresholds

Maintain a counter of consumed decoys per host. When all 3 decoys are depleted, lower the `proc_flag` response threshold to 0, since all subsequent `proc_flag` events are necessarily real red activity. Expected impact: improved detection precision on heavily-attacked hosts.

### 4.3 Phase 0 Contractor-to-HQ Blocking Cost-Benefit

Implement contractor-to-HQ blocking during Phase 0 and measure the net reward effect. The cost is ASF = -1 per step (167 steps total = -167 ASF penalty). The benefit is preventing the primary red lateral movement path. A break-even analysis should determine whether the LWF damage prevented exceeds the 167-point ASF cost.

### 4.4 Post-Restore Re-Infection Monitoring

After Restoring a host, apply a heightened alert state (lower threshold) for the subsequent N steps. The rationale is that red may immediately re-exploit a freshly Restored host, and the prior probability of red activity is elevated in the post-Restore window.

### 4.5 Adaptive Decoy Redeployment Priority

Prioritize decoy redeployment on hosts where decoys have been consumed, as these are hosts actively being targeted by red. Currently, decoy deployment follows a fixed priority order. Adapting priority based on observed red targeting may improve early-warning coverage.

---

## 5. Fundamental Limits

The following constraints are properties of the CC4 environment and cannot be overcome by any defensive strategy:

### 5.1 DegradeServices Invisibility Is Architectural

The `DegradeServices` action produces no entries in the observation space by design. This is not a limitation of the observation processing pipeline; it is a fundamental property of the red action model. No observation-based agent can detect `DegradeServices` directly. The only avenue is indirect inference from precursor actions and downstream effects.

### 5.2 Green False Positive Rate Is Fixed

The green agent false positive rate (0.776%) is a fixed parameter of the environment. Every defensive action taken in response to a green FP is wasted. At approximately 2,000 green actions per episode, this produces approximately 15.5 false alerts per episode. No blue agent modification can reduce this rate.

### 5.3 Red Exploit Success Rate Is Stochastic

Red exploit attempts succeed with a fixed probability (24.9% empirically). Blue agents cannot reduce this probability; they can only respond to successful exploits after the fact (or preemptively block the network path). The stochastic nature of red success is a primary driver of inter-episode variance.

### 5.4 Action Budget Is Hard-Constrained

Each of the 5 blue agents can execute exactly 1 action per step. With 500 steps per episode, the total action budget is 2,500 agent-actions. Given that the environment contains approximately 30 hosts across 5 subnets, each agent can attend to each host in its subnet approximately once every 6 steps on average. This limits the temporal resolution of any defense strategy.

### 5.5 No Positive Rewards Exist

The reward function is strictly non-positive. There is no mechanism to earn positive reward through good defense; the best achievable outcome is minimizing the magnitude of penalties. This means the optimal strategy is purely loss-minimizing, which has implications for RL reward shaping (there is no natural "success" signal, only "less failure").

### 5.6 Observation Delay Is Structural

Blue agents observe the state after red and green have acted. This 1-step observation delay means that by the time an exploit is observed, the red agent has already moved to its next action (potentially `PrivilegeEscalate` or `DegradeServices`). Combined with the 1-action-per-step constraint, the minimum response time to a detected exploit is 1 step, during which red can advance its kill-chain.

---

## 6. Summary and Prioritization

### Ranked by Expected Impact

| Priority | Gap | Expected Reward Gain | Difficulty | Approach |
|----------|-----|---------------------|------------|----------|
| 1 | DegradeServices inference (Gap 1) | 50-200 points | High | DRL / Bayesian estimation |
| 2 | Phase 0 optimization (Gap 3) | 30-100 points | Medium | Heuristic experiment / DRL |
| 3 | conn_flag persistence (Gap 5) | 10-30 points | Low | Heuristic implementation |
| 4 | Decoy depletion tracking (Gap 6) | 5-20 points | Low | Heuristic implementation |
| 5 | Variance reduction (Gap 4) | 0-50 points (tail) | Medium | Risk-sensitive RL |
| 6 | Inter-agent coordination (Gap 2) | 0-30 points | High | MARL / formal analysis |

### Recommended Research Agenda

**Short-term (heuristic experiments):** Implement and evaluate micro-optimizations 4.1 through 4.5. These require minimal engineering effort and provide immediate empirical data on the remaining optimization surface.

**Medium-term (hybrid DRL):** Implement the hybrid heuristic + DRL override approach (Section 3.3), focusing on threshold adaptation and selective proactive Restore. Use V11a as the base policy and curriculum learning to manage sample complexity.

**Long-term (theoretical analysis):** Formally characterize the Dec-POMDP structure of CC4, derive bounds on the value of communication (Gap 2), and establish whether the theoretical floor of -300 is achievable under partial observability.

---

## References

- CC4 Environment: CAGE Challenge 4, Cyber Autonomy Gym for Experimentation
- V11a Agent: `CybORG/Agents/SimpleAgents/EnterpriseHeuristicAgent.py`
- Optimization Analysis: `docs/optimality_analysis.md`, `docs/v11_optimization_plan.md`
- Attack Chain Analysis: `docs/attack_chain_analysis.md`
- Simulation Audit: `docs/simulation_audit_report.md`
