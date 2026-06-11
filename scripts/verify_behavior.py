"""Behavioral reproducibility and dtype safety verification script.

Run with:
    python scripts/verify_behavior.py
"""
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Step 1: Reproducibility check
# ---------------------------------------------------------------------------

def run_episode(seed, n_steps=50):
    from CybORG import CybORG
    from CybORG.Agents.SimpleAgents.ConstantAgent import SleepAgent
    from CybORG.Agents.Wrappers.BlueEnterpriseWrapper import BlueEnterpriseWrapper
    from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, red_agent_class=SleepAgent)
    cyborg = CybORG(scenario_generator=sg)
    env = BlueEnterpriseWrapper(cyborg)
    obs, _ = env.reset(seed=seed)
    rewards_total = 0.0
    first_obs = {k: v.copy() for k, v in obs.items()}
    for _ in range(n_steps):
        actions = {agent: 0 for agent in env.agents}
        obs, rewards, term, trunc, info = env.step(actions)
        rewards_total += sum(rewards.values())
    return rewards_total, first_obs


print("=" * 60)
print("REPRODUCIBILITY CHECK (seed=42, 50 steps)")
print("=" * 60)

r1, o1 = run_episode(42)
r2, o2 = run_episode(42)

reward_match = r1 == r2
obs_match = all(np.allclose(o1[a], o2[a]) for a in o1)

if reward_match and obs_match:
    print("REPRODUCIBILITY CHECK PASSED")
    print(f"  Seed=42 reward (50 steps): {r1:.4f}")
else:
    if not reward_match:
        print(f"FAIL: Reward mismatch — run 1: {r1}, run 2: {r2}")
    for agent in o1:
        if not np.allclose(o1[agent], o2[agent]):
            diff = np.abs(o1[agent] - o2[agent])
            print(f"FAIL: Obs mismatch for {agent} — max diff: {diff.max():.6f}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2: dtype check
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("DTYPE CHECK (float32 observation vectors)")
print("=" * 60)

from CybORG import CybORG
from CybORG.Agents.SimpleAgents.ConstantAgent import SleepAgent
from CybORG.Agents.Wrappers.BlueEnterpriseWrapper import BlueEnterpriseWrapper
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import EnterpriseScenarioGenerator

sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, red_agent_class=SleepAgent)
cyborg = CybORG(scenario_generator=sg)
env = BlueEnterpriseWrapper(cyborg)
obs, _ = env.reset(seed=42)

dtype_ok = True
for agent, vec in obs.items():
    if vec.dtype != np.float32:
        print(f"FAIL: {agent} has dtype {vec.dtype}, expected float32")
        dtype_ok = False

if dtype_ok:
    sample = list(obs.values())[0]
    print(f"DTYPE CHECK PASSED — all observations are float32 (sample shape: {sample.shape})")
else:
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 3: Heuristic agent check
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("HEURISTIC AGENT CHECK (100 steps)")
print("=" * 60)

from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import EnterpriseHeuristicAgent

sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, red_agent_class=SleepAgent)
env = BlueEnterpriseWrapper(CybORG(scenario_generator=sg))
obs, _ = env.reset()
agents = {a: EnterpriseHeuristicAgent() for a in env.agents}
total = 0.0
for _ in range(100):
    # get_action returns (action_idx, message) tuple
    action_messages = {a: agents[a].get_action(obs[a]) for a in env.agents if a in obs}
    actions = {a: v[0] for a, v in action_messages.items()}
    messages = {a: v[1] for a, v in action_messages.items()}
    obs, rew, term, trunc, info = env.step(actions, messages=messages)
    total += sum(rew.values())

print(f"Heuristic 100-step reward: {total:.4f}")
if total > -5000:
    print("HEURISTIC CHECK PASSED")
else:
    print(f"FAIL: Reward too low ({total:.4f} <= -5000)")
    sys.exit(1)

print()
print("=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
