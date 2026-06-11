"""
CybORG speed benchmark — measures per-step and per-episode timing.
Run from repo root: python scripts/benchmark_speed.py
"""
import time
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_EPISODES = 20
N_STEPS = 500
SEED = 42


def make_env(seed=SEED, steps=N_STEPS):
    from CybORG import CybORG
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent

    sg = EnterpriseScenarioGenerator(
        steps=steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    return BlueFlatWrapper(env=cyborg)


def run_benchmark():
    import numpy as np
    from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import make_heuristic_agents

    print(f"Benchmarking {N_EPISODES} episodes x {N_STEPS} steps (seed={SEED})...")

    episode_times = []
    step_times = []
    reset_times = []
    total_rewards = []

    env = make_env(seed=SEED)

    # Warm-up reset then create agents
    obs_dict, _ = env.reset()
    agents = make_heuristic_agents(env)
    agent_names = env.possible_agents

    for ep in range(N_EPISODES):
        t_reset_start = time.perf_counter()
        obs_dict, _ = env.reset()
        t_reset_end = time.perf_counter()
        reset_times.append(t_reset_end - t_reset_start)

        # Re-init agent action catalogues after each reset
        subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
        for name, ag in agents.items():
            ag.reset()
            ag.set_action_info(
                env.action_labels(name),
                env.action_mask(name),
                subnet_hosts,
            )

        ep_reward = 0.0
        ep_start = time.perf_counter()

        for step in range(N_STEPS):
            actions = {}
            for name, ag in agents.items():
                raw_obs = obs_dict.get(name, np.zeros(1))
                mask = env.action_mask(name)
                action_idx, _msg = ag.get_action(raw_obs, np.array(mask, dtype=bool))
                actions[name] = action_idx

            t_step_start = time.perf_counter()
            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions)
            t_step_end = time.perf_counter()

            step_times.append(t_step_end - t_step_start)
            ep_reward += sum(rew_dict.values())

            if all(
                term_dict.get(n, False) or trunc_dict.get(n, False)
                for n in agent_names
            ):
                break

        ep_end = time.perf_counter()
        episode_times.append(ep_end - ep_start)
        total_rewards.append(ep_reward)

        if (ep + 1) % 5 == 0:
            print(
                f"  Episode {ep+1}/{N_EPISODES}: "
                f"reward={ep_reward:.1f}, "
                f"time={episode_times[-1]:.2f}s"
            )

    return {
        "episode_times": episode_times,
        "step_times": step_times,
        "reset_times": reset_times,
        "total_rewards": total_rewards,
    }


if __name__ == "__main__":
    results = run_benchmark()

    et = results["episode_times"]
    st = results["step_times"]
    rt = results["reset_times"]
    rw = results["total_rewards"]

    print("\n=== RESULTS ===")
    print(f"Episodes/second:    {N_EPISODES / sum(et):.2f}")
    print(f"Steps/second:       {len(st) / sum(st):.1f}")
    print(f"Mean episode time:  {statistics.mean(et)*1000:.1f} ms  (std: {statistics.stdev(et)*1000:.1f} ms)")
    print(f"Mean step time:     {statistics.mean(st)*1000:.2f} ms  (std: {statistics.stdev(st)*1000:.2f} ms)")
    print(f"Mean reset time:    {statistics.mean(rt)*1000:.1f} ms  (std: {statistics.stdev(rt)*1000:.1f} ms)")
    print(f"Mean reward:        {statistics.mean(rw):.1f}  (std: {statistics.stdev(rw):.1f})")
    print(f"Min/Max reward:     {min(rw):.1f} / {max(rw):.1f}")
