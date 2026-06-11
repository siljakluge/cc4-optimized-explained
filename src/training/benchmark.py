"""Environment throughput benchmarking for CybORG.

Measures steps/second to identify performance bottlenecks before training.
"""
from __future__ import annotations

import time
import numpy as np


def benchmark_env(num_envs: int = 1, num_steps: int = 10_000, seed: int = 42) -> dict:
    """Run the env for num_steps and measure wall-clock throughput."""
    from src.envs.vectorized_env import SingleCybORGEnv, CybORGVecEnv

    if num_envs == 1:
        env = SingleCybORGEnv(max_steps=500, seed=seed)
        obs, _ = env.reset()
        t0 = time.perf_counter()
        actual_steps = 0
        episodes = 0
        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            actual_steps += 1
            if terminated or truncated:
                obs, _ = env.reset()
                episodes += 1
        elapsed = time.perf_counter() - t0
        env.close()
    else:
        env = CybORGVecEnv(num_envs=num_envs, max_steps=500, seed=seed)
        obs_batch, _ = env.reset()
        t0 = time.perf_counter()
        actual_steps = 0
        for _ in range(num_steps):
            actions = np.array([env.action_space.sample() for _ in range(num_envs)])
            obs_batch, rews, terms, truncs, infos = env.step(actions)
            actual_steps += num_envs
            # Auto-reset done envs
            for i in range(num_envs):
                if terms[i] or truncs[i]:
                    # Worker handles reset internally on next step
                    pass
        elapsed = time.perf_counter() - t0
        episodes = 0
        env.close()

    sps = actual_steps / elapsed
    return {
        "num_envs": num_envs,
        "total_steps": actual_steps,
        "elapsed_s": round(elapsed, 2),
        "steps_per_second": round(sps, 1),
    }


def compare_configs() -> None:
    """Benchmark num_envs in [1, 2, 4, 8] and print comparison table."""
    configs = [1, 2, 4, 8]
    steps_per_config = 2_000  # keep it short for a quick comparison

    print(f"\n{'num_envs':>10} {'steps':>10} {'time (s)':>10} {'steps/s':>12}")
    print("-" * 46)

    results = []
    for n in configs:
        print(f"  Benchmarking num_envs={n} ...", end=" ", flush=True)
        try:
            r = benchmark_env(num_envs=n, num_steps=steps_per_config)
            results.append(r)
            print(f"{r['steps_per_second']:.1f} steps/s")
            print(f"{n:>10} {r['total_steps']:>10} {r['elapsed_s']:>10} {r['steps_per_second']:>12.1f}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if results:
        baseline = results[0]["steps_per_second"]
        print("\nSpeedup vs single env:")
        for r in results:
            speedup = r["steps_per_second"] / baseline
            print(f"  num_envs={r['num_envs']}: {speedup:.2f}x")


if __name__ == "__main__":
    compare_configs()
