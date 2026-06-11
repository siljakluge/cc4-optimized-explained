#!/usr/bin/env python3
"""Benchmark CybORG environment throughput (steps/second).

Usage:
    python scripts/benchmark.py [--envs 1,2,4,8] [--steps 2000]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.benchmark import benchmark_env, compare_configs


def main():
    parser = argparse.ArgumentParser(description="Benchmark CybORG env throughput")
    parser.add_argument("--steps", type=int, default=2000, help="Steps per config")
    parser.add_argument("--single", action="store_true", help="Only benchmark single env")
    args = parser.parse_args()

    if args.single:
        result = benchmark_env(num_envs=1, num_steps=args.steps)
        print(f"\nSingle env: {result['steps_per_second']:.1f} steps/s "
              f"({result['total_steps']} steps in {result['elapsed_s']}s)")
    else:
        compare_configs()


if __name__ == "__main__":
    main()
