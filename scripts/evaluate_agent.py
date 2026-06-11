#!/usr/bin/env python3
"""Evaluate a trained DRL model on CybORG.

Usage:
    python scripts/evaluate_agent.py --model data/models/final_model.zip [--episodes 100]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import CybORGTrainer


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved CybORG model")
    parser.add_argument("--model", required=True, help="Path to saved model (.zip)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    trainer = CybORGTrainer(seed=args.seed, max_steps=args.max_steps)
    results = trainer.evaluate(args.model, n_episodes=args.episodes)

    print("\n=== Evaluation Results ===")
    print(f"  Mean reward    : {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Win rate       : {results['win_rate']:.1%}")
    print(f"  Mean ep length : {results['mean_length']:.1f} steps")
    print(f"  Episodes run   : {results['n_episodes']}")


if __name__ == "__main__":
    main()
