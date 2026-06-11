#!/usr/bin/env python3
"""Live swarm monitoring dashboard — polls training logs and prints status.

Usage:
    python scripts/monitor_swarm.py --run-id RUN_ID [--log-dir data/logs/] [--interval 5]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def tail_jsonl(path: str, last_n: int = 20) -> list[dict]:
    """Read the last N lines from a JSONL file."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.readlines()
    entries = []
    for line in lines[-last_n:]:
        try:
            entries.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            pass
    return entries


def print_status(run_id: str, log_dir: str) -> None:
    log_path = os.path.join(log_dir, f"{run_id}.jsonl")
    entries = tail_jsonl(log_path, last_n=1)

    if not entries:
        print(f"[{run_id}] No data yet — waiting for training to start...")
        return

    latest = entries[-1]
    os.system("cls" if os.name == "nt" else "clear")
    print(f"=== CybORG Training Monitor — run {run_id} ===")
    print(f"  Timestamp       : {time.strftime('%H:%M:%S', time.localtime(latest.get('timestamp', 0)))}")
    print(f"  Total steps     : {latest.get('total_steps', 0):,}")
    print(f"  Steps/second    : {latest.get('steps_per_second', 0):.1f}")
    print(f"  Mean reward     : {latest.get('mean_reward_100ep', 0):.3f}")
    print(f"  Win rate        : {latest.get('win_rate_100ep', 0):.1%}")
    print(f"  Elapsed time    : {latest.get('elapsed_time', 0):.0f}s")

    # Agent monitor dashboard
    monitor_path = os.path.join(log_dir, "agent_monitor.json")
    if os.path.exists(monitor_path):
        with open(monitor_path) as f:
            try:
                dashboard = json.load(f)
                summary = dashboard.get("summary", {})
                print(f"\n  Swarm: {dashboard.get('swarm_id', '?')} — "
                      f"{summary.get('running', 0)} running / "
                      f"{summary.get('completed', 0)} completed / "
                      f"{summary.get('failed', 0)} failed")
            except json.JSONDecodeError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Monitor CybORG training run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-dir", default="data/logs/")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")
    args = parser.parse_args()

    print(f"Monitoring run {args.run_id} — refreshing every {args.interval}s (Ctrl+C to stop)")
    try:
        while True:
            print_status(args.run_id, args.log_dir)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
