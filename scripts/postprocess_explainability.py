#!/usr/bin/env python3
"""
Post-process explain.py outputs for CAGE Challenge 4 explainability runs.

This script is intentionally separate from explain.py so it can be used as a
conflict-free integration check while explain.py and wrappers are still being
iterated on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.plot_actions import (  # noqa: E402
    DEFAULT_BLUE_AGENTS,
    plot_action_frequencies_all_attackers_agents,
)
from plotting.plot_rew_decomp import (  # noqa: E402
    ATTACKER_ORDER,
    build_attacker_subnet_table,
    plot_model_attacker_lines,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def _discover_profiles(run_dir: Path) -> list[str]:
    profiles = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "reward_log.jsonl").exists() or (child / "actions.jsonl").exists():
            profiles.append(child.name)
    return profiles


def _parse_csv_list(value: str | None, fallback: Iterable[str]) -> list[str]:
    if not value:
        return list(fallback)
    return [x.strip() for x in value.split(",") if x.strip()]


def validate_profile_logs(run_dir: Path, profiles: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for profile in profiles:
        profile_dir = run_dir / profile
        reward_path = profile_dir / "reward_log.jsonl"
        scalar_path = profile_dir / "scalar_rewards.jsonl"
        actions_path = profile_dir / "actions.jsonl"

        reward_rows = _read_jsonl(reward_path)
        scalar_rows = _read_jsonl(scalar_path)
        action_rows = _read_jsonl(actions_path)

        bad_reward = [
            i for i, row in enumerate(reward_rows, start=1)
            if not {"phase", "reward_list", "total", "step"}.issubset(row)
        ]
        bad_scalar = [
            i for i, row in enumerate(scalar_rows, start=1)
            if "step" not in row or not (("rewards" in row) or ("total_scalar" in row))
        ]
        bad_actions = [
            i for i, row in enumerate(action_rows, start=1)
            if "step" not in row or not isinstance(row.get("actions"), dict)
        ]

        rows.append({
            "profile": profile,
            "reward_rows": len(reward_rows),
            "scalar_rows": len(scalar_rows),
            "action_rows": len(action_rows),
            "bad_reward_rows": ",".join(map(str, bad_reward[:10])),
            "bad_scalar_rows": ",".join(map(str, bad_scalar[:10])),
            "bad_action_rows": ",".join(map(str, bad_actions[:10])),
            "has_reward_decomposition": bool(reward_rows) and not bad_reward,
            "has_scalar_rewards": bool(scalar_rows) and not bad_scalar,
            "has_action_statistics": bool(action_rows) and not bad_actions,
        })

    return pd.DataFrame(rows)


def write_reward_tables(run_dir: Path, out_dir: Path, profiles: list[str], avg: bool) -> Path:
    tables = []
    for profile in profiles:
        reward_path = run_dir / profile / "reward_log.jsonl"
        scalar_path = run_dir / profile / "scalar_rewards.jsonl"
        table = build_attacker_subnet_table(reward_path, scalar_path, avg=avg)
        if table.empty:
            continue
        table = table.copy()
        table["attacker"] = profile
        tables.append(table)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"reward_decomposition_{'avg' if avg else 'sum'}.csv"
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(out_csv, index=False)
    else:
        pd.DataFrame().to_csv(out_csv, index=False)
    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate explain.py logs and generate reward/action explainability artifacts."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory created by explain.py, e.g. Results/<agent_ts>")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for generated CSVs/plots")
    parser.add_argument("--profiles", default="", help="Comma-separated profiles; default: discover from run_dir")
    parser.add_argument("--top-k-actions", type=int, default=25, help="-1 keeps all action labels")
    parser.add_argument("--sum", action="store_true", help="Use cumulative instead of per-episode average rewards")
    parser.add_argument("--no-plots", action="store_true", help="Only validate logs and write summary CSVs")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    profiles = _parse_csv_list(args.profiles, _discover_profiles(run_dir))
    if not profiles:
        raise ValueError(f"No profile log directories found under {run_dir}")

    out_dir = (args.out_dir or (run_dir / "Postprocessed")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_profile_logs(run_dir, profiles)
    validation_csv = out_dir / "explainability_log_validation.csv"
    validation.to_csv(validation_csv, index=False)

    reward_csv = write_reward_tables(run_dir, out_dir, profiles, avg=not args.sum)

    generated: list[Path] = [validation_csv, reward_csv]
    if not args.no_plots:
        action_out = out_dir / "ActionStatistics"
        top_k = None if args.top_k_actions == -1 else args.top_k_actions
        action_artifacts = plot_action_frequencies_all_attackers_agents(
            base_dir=run_dir,
            out_dir=action_out,
            attackers=profiles,
            blue_agents=DEFAULT_BLUE_AGENTS,
            normalize=True,
            top_k=top_k,
        )
        if action_artifacts:
            generated.extend(Path(p) for p in action_artifacts.values())

        reward_out = out_dir / "RewardDecomposition"
        reward_plot = plot_model_attacker_lines(
            model_name=run_dir.name,
            model_dir=run_dir,
            out_dir=reward_out,
            attackers=[p for p in ATTACKER_ORDER if p in profiles] or profiles,
            avg=not args.sum,
            show_legend=True,
        )
        if reward_plot is not None:
            generated.append(Path(reward_plot))

    print("Validated profiles:", ", ".join(profiles))
    print("Generated artifacts:")
    for path in generated:
        print(f" - {path}")

    bad = validation[
        ~(
            validation["has_reward_decomposition"]
            & validation["has_scalar_rewards"]
            & validation["has_action_statistics"]
        )
    ]
    if not bad.empty:
        print("\nProfiles with missing or malformed logs:")
        print(bad.to_string(index=False))
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
