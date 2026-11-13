#!/usr/bin/env python3
"""
Export CAGE4 training logs (*.pt) to CSV.

Expected log format (current train.py):
  log.append((avg_reward, e, sum(last_losses)/N_AGENTS))
and torch.save(log, 'logs/<name>.pt')

This script:
- loads each log
- prints top 10 updates by avg_reward (descending)
- writes a CSV per run and a combined CSV

Usage:
  python create_csv_logs.py \
      -- logs/contractoractive.pt \
      --outdir exported_csv
"""

from __future__ import annotations
import argparse
import os
from typing import Any, Dict, List, Tuple

import torch
import csv


def load_pt_log(path: str) -> List[Dict[str, Any]]:
    """
    Supports:
    - list of tuples: (avg_reward, e, avg_loss)
    - list of dicts (if you later change logging)
    """
    obj = torch.load(path, map_location="cpu")

    rows: List[Dict[str, Any]] = []

    if isinstance(obj, list):
        for idx, entry in enumerate(obj):
            if isinstance(entry, (tuple, list)) and len(entry) >= 3:
                avg_reward, e, avg_loss = entry[0], entry[1], entry[2]
                rows.append(
                    {
                        "idx": idx,
                        "step_e": int(e),
                        "avg_reward": float(avg_reward),
                        "avg_loss": float(avg_loss),
                    }
                )
            elif isinstance(entry, dict):
                # best-effort for future-proofing
                rows.append(
                    {
                        "idx": idx,
                        "step_e": int(entry.get("step_e", entry.get("e", idx))),
                        "avg_reward": float(entry.get("avg_reward", entry.get("reward", float("nan")))),
                        "avg_loss": float(entry.get("avg_loss", entry.get("loss", float("nan")))),
                        # keep any extras
                        **{k: v for k, v in entry.items() if k not in {"step_e", "e", "avg_reward", "reward", "avg_loss", "loss"}},
                    }
                )
            else:
                raise ValueError(
                    f"Unsupported entry type in {path}: idx={idx}, type={type(entry)} value={repr(entry)[:200]}"
                )
    else:
        raise ValueError(f"Unsupported top-level log type in {path}: {type(obj)}")

    return rows


def print_top10(rows: List[Dict[str, Any]], run_name: str) -> None:
    top = sorted(rows, key=lambda r: r.get("avg_reward", float("-inf")), reverse=True)[:10]
    print("=" * 80)
    print(f"TOP 10 by avg_reward: {run_name}")
    print("=" * 80)
    for r in top:
        print(
            f"idx={r['idx']:4d}  step_e={r['step_e']:7d}  "
            f"avg_reward={r['avg_reward']:12.6f}  avg_loss={r['avg_loss']:12.6f}"
        )
    print()


def write_csv(path: str, rows: List[Dict[str, Any]], extra_cols: List[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    base_cols = ["run", "idx", "step_e", "avg_reward", "avg_loss"]
    extra_cols = extra_cols or []
    cols = base_cols + [c for c in extra_cols if c not in base_cols]

    # gather any additional columns present in rows
    dynamic_cols = set()
    for r in rows:
        dynamic_cols.update(r.keys())
    for c in sorted(dynamic_cols):
        if c not in cols:
            cols.append(c)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logs",
        nargs="+",
        default=["logs/default_curriculum.pt"],
        help="Paths to .pt log files, e.g. logs/contractoractive.pt",
    )

    ap.add_argument("--outdir", default="exported_csv", help="Output directory for CSV files")
    args = ap.parse_args()

    combined: List[Dict[str, Any]] = []

    for log_path in args.logs:
        run_name = os.path.splitext(os.path.basename(log_path))[0]
        rows = load_pt_log(log_path)

        # add run column
        for r in rows:
            r["run"] = run_name

        print_top10(rows, run_name)

        out_csv = os.path.join(args.outdir, f"{run_name}.csv")
        write_csv(out_csv, rows)
        print(f"Wrote: {out_csv}")

        combined.extend(rows)

    combined_csv = os.path.join(args.outdir, "combined_all_runs.csv")
    write_csv(combined_csv, combined)
    print(f"Wrote: {combined_csv}")


if __name__ == "__main__":
    main()
