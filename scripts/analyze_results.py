#!/usr/bin/env python3
"""Analyze training results stored in the SQLite database.

Usage:
    python scripts/analyze_results.py [--run-id RUN_ID] [--compare] [--export]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.analysis import RunAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Analyze CybORG training results")
    parser.add_argument("--run-id", help="Specific run ID to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare all runs")
    parser.add_argument("--export", action="store_true", help="Export run to CSV")
    parser.add_argument("--db", default="data/training_runs.db")
    args = parser.parse_args()

    analyzer = RunAnalyzer(db_path=args.db)

    if args.compare or not args.run_id:
        runs = analyzer.get_runs()
        if not runs:
            print("No runs found in database.")
            return
        run_ids = [r["run_id"] for r in runs]
        print(f"\nFound {len(runs)} run(s):")
        comparison = analyzer.compare_runs(run_ids)
        print(f"\n{'run_id':>12} {'episodes':>10} {'mean_reward':>12} {'win_rate':>10} {'mean_length':>12}")
        print("-" * 60)
        for r in comparison:
            print(f"{r['run_id']:>12} {r['episodes']:>10} {r['final_mean_reward']:>12.3f} "
                  f"{r['final_win_rate']:>9.1%} {r['mean_ep_length']:>12.1f}")

    if args.run_id:
        print(f"\nPlotting learning curve for run {args.run_id}...")
        analyzer.plot_learning_curve(args.run_id)
        analyzer.plot_win_rate(args.run_id)

        if args.export:
            path = analyzer.export_to_csv(args.run_id)
            print(f"Exported to: {path}")


if __name__ == "__main__":
    main()
