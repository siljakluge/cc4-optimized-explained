#!/usr/bin/env python3
"""CC4 Blue-Team Evaluation Workflow.

Master orchestrator that runs the full evaluation pipeline end-to-end:
  1. Collect telemetry for each requested agent variant
  2. Run analysis (scripts/analyze_telemetry.py)
  3. Print a comparison table with all agent scores
  4. Write a markdown workflow report

Usage:
    py scripts/run_evaluation_workflow.py [--agents AGENT ...] [--episodes N]
                                          [--steps N] [--seed N]
                                          [--skip-collect] [--no-analyse]
                                          [--out PATH]
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
DOCS = ROOT / "docs"

DB_MAIN = DATA / "cc4_telemetry.db"       # sleep, heuristic, v5
DB_RANDOM = DATA / "random_agent.db"
DB_V4_ANALYSE = DATA / "v4_analyse.db"

ANALYSIS_REPORT = DOCS / "telemetry_analysis_report.md"
COMPREHENSIVE_REPORT = DOCS / "comprehensive_cc4_report.md"

EVAL_SCRIPT = ROOT / "CybORG" / "Evaluation" / "evaluation.py"
SUBMISSION_PATH = ROOT / "CybORG" / "Evaluation" / "submission"
EVAL_OUTPUT_DIR = DATA / "official_eval"

# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

VALID_AGENTS = ["sleep", "heuristic", "random", "v4_analyse", "v5"]

# Map agent key -> (script_path, extra_args_template, db_path, db_agent_type)
# extra_args_template uses {episodes}, {steps}, {seed} placeholders.
AGENT_CONFIG: dict[str, dict] = {
    "sleep": {
        "script": SCRIPTS / "log_env_data.py",
        "extra_args": ["--agent", "sleep"],
        "db": DB_MAIN,
        "db_agent_type": "sleep",
        "display_name": "SleepAgent",
    },
    "heuristic": {
        "script": SCRIPTS / "log_env_data.py",
        "extra_args": ["--agent", "heuristic"],
        "db": DB_MAIN,
        "db_agent_type": "heuristic",
        "display_name": "v4/v5 Heuristic",
    },
    "v5": {
        "script": SCRIPTS / "log_env_data.py",
        "extra_args": ["--agent", "heuristic"],
        "db": DB_MAIN,
        "db_agent_type": "heuristic",
        "display_name": "v5 Heuristic",
    },
    "random": {
        "script": SCRIPTS / "log_random_agent.py",
        "extra_args": [],
        "db": DB_RANDOM,
        "db_agent_type": "random",
        "display_name": "Random",
    },
    "v4_analyse": {
        "script": SCRIPTS / "log_v4_analyse_agent.py",
        "extra_args": [],
        "db": DB_V4_ANALYSE,
        "db_agent_type": "v4_analyse",
        "display_name": "v4+Analyse",
    },
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_existing_data(agent_key: str) -> tuple[bool, int]:
    """Return (has_data, n_episodes) by querying the appropriate DB."""
    cfg = AGENT_CONFIG[agent_key]
    db_path: Path = cfg["db"]
    agent_type: str = cfg["db_agent_type"]

    if not db_path.exists():
        return False, 0

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE agent_type = ?",
                (agent_type,),
            ).fetchone()
            n = int(row[0]) if row else 0
            return n > 0, n
        finally:
            conn.close()
    except Exception:
        return False, 0


def query_agent_stats(agent_key: str) -> dict | None:
    """Return dict with mean, std, best, worst, n_eps or None if no data."""
    cfg = AGENT_CONFIG[agent_key]
    db_path: Path = cfg["db"]
    agent_type: str = cfg["db_agent_type"]

    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS n_eps,
                    AVG(total_reward) AS mean_rew,
                    -- SQLite has no STDDEV; compute manually as sqrt(avg(sq) - avg^2)
                    SQRT(MAX(AVG(total_reward * total_reward) - AVG(total_reward) * AVG(total_reward), 0)) AS std_rew,
                    MAX(total_reward) AS best_rew,
                    MIN(total_reward) AS worst_rew
                FROM episodes
                WHERE agent_type = ? AND total_reward IS NOT NULL
                """,
                (agent_type,),
            ).fetchone()
        finally:
            conn.close()

        if not row or row[0] == 0:
            return None

        return {
            "n_eps": int(row[0]),
            "mean": float(row[1]),
            "std": float(row[2]),
            "best": float(row[3]),
            "worst": float(row[4]),
        }
    except Exception:
        return None


def format_reward(r: float | None) -> str:
    """Format reward as signed integer string with thousands separator."""
    if r is None:
        return "N/A"
    return f"{r:,.0f}"


def format_std(r: float | None) -> str:
    """Format standard deviation with +- prefix."""
    if r is None:
        return "N/A"
    return f"+-{r:,.0f}"


# ---------------------------------------------------------------------------
# Step 1: Official evaluation
# ---------------------------------------------------------------------------

def run_official_evaluation(episodes: int, seed: int) -> dict | None:
    """Run the official CybORG evaluation.py and return summary dict or None."""
    print("\n[1/4] Official evaluation (CybORG/Evaluation/evaluation.py)")

    if not EVAL_SCRIPT.exists():
        print("  [WARN] evaluation.py not found -- skipping")
        return None
    if not SUBMISSION_PATH.exists():
        print(f"  [WARN] Submission not found at {SUBMISSION_PATH} -- skipping")
        return None

    # Create timestamped output directory
    import time as _time
    ts = _time.strftime("%Y%m%d_%H%M%S")
    out_dir = EVAL_OUTPUT_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        str(SUBMISSION_PATH),
        str(out_dir) + "/",
        "--max-eps", str(episodes),
        "--seed", str(seed),
    ]

    print(f"  Submission: {SUBMISSION_PATH.relative_to(ROOT)}")
    print(f"  Output:     {out_dir.relative_to(ROOT)}")
    print(f"  Episodes:   {episodes}  Seed: {seed}")

    # Ensure CybORG is importable in the subprocess by adding project root to PYTHONPATH
    import os as _os
    env = _os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + _os.pathsep + env.get("PYTHONPATH", "")

    t0 = _time.perf_counter()
    try:
        subprocess.run(cmd, check=True, env=env)
        elapsed = _time.perf_counter() - t0
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] evaluation.py failed (exit code {exc.returncode})")
        return None
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return None

    # Read summary.json
    import json as _json
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                data = _json.load(f)
            mean_rew = data["reward"]["mean"]
            std_rew = data["reward"]["stdev"]
            print(f"  Result: mean={mean_rew:.1f} +- {std_rew:.1f}  (elapsed {elapsed:.1f}s)")
            return {"mean": mean_rew, "std": std_rew, "n_eps": episodes, "output_dir": str(out_dir)}
        except Exception as exc:
            print(f"  [WARN] Could not read summary.json: {exc}")
    else:
        # Try scores.txt fallback
        scores_path = out_dir / "scores.txt"
        if scores_path.exists():
            try:
                lines = scores_path.read_text().splitlines()
                mean_rew = float(lines[0].split(": ")[1])
                std_rew = float(lines[1].split(": ")[1])
                print(f"  Result: mean={mean_rew:.1f} +- {std_rew:.1f}  (elapsed {elapsed:.1f}s)")
                return {"mean": mean_rew, "std": std_rew, "n_eps": episodes, "output_dir": str(out_dir)}
            except Exception:
                pass

    print(f"  [WARN] No results found in {out_dir}")
    return None


# ---------------------------------------------------------------------------
# Step 2: Collection
# ---------------------------------------------------------------------------

def run_collection(
    agents: list[str],
    episodes: int,
    steps: int,
    seed: int,
) -> dict[str, float]:
    """Collect telemetry for each agent. Returns wall-time per agent."""
    print("\n[2/4] Collecting telemetry")

    wall_times: dict[str, float] = {}

    for agent_key in agents:
        cfg = AGENT_CONFIG[agent_key]
        display = cfg["display_name"]
        label = f"  {agent_key:<14}"

        # Special note for v5
        if agent_key == "v5":
            has_data, n_eps = check_existing_data("heuristic")
            if has_data:
                print(
                    f"{label} [SKIP] Found {n_eps} existing episodes"
                    f" (same DB as heuristic, in-place v5)"
                )
                wall_times[agent_key] = 0.0
                continue
        else:
            has_data, n_eps = check_existing_data(agent_key)
            if has_data:
                print(
                    f"{label} [SKIP] Found {n_eps} existing episodes"
                    f" in {cfg['db'].relative_to(ROOT)}"
                )
                wall_times[agent_key] = 0.0
                continue

        # Build subprocess command
        cmd = [
            sys.executable,
            str(cfg["script"]),
            "--episodes", str(episodes),
            "--steps", str(steps),
            "--seed", str(seed),
        ] + [str(a) for a in cfg["extra_args"]]

        print(
            f"{label} Collecting {episodes} eps x {steps} steps,"
            f" seed={seed} ..."
        )

        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.perf_counter() - t0
            wall_times[agent_key] = elapsed
            print(f"{label} Done in {elapsed:.1f}s")
        except subprocess.CalledProcessError as exc:
            elapsed = time.perf_counter() - t0
            wall_times[agent_key] = elapsed
            print(f"{label} ERROR (exit code {exc.returncode}) -- continuing")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            wall_times[agent_key] = elapsed
            print(f"{label} ERROR: {exc} -- continuing")

    return wall_times


# ---------------------------------------------------------------------------
# Step 3: Analysis
# ---------------------------------------------------------------------------

def run_analysis() -> str | None:
    """Run analyze_telemetry.py and return report path, or None on failure."""
    print("\n[3/4] Running analysis...")

    analyse_script = SCRIPTS / "analyze_telemetry.py"
    if not analyse_script.exists():
        print("  [WARN] analyze_telemetry.py not found -- skipping")
        return None

    cmd = [sys.executable, str(analyse_script)]
    try:
        subprocess.run(cmd, check=True)
        report_path = str(ANALYSIS_REPORT)
        print(f"  Report: {ANALYSIS_REPORT.relative_to(ROOT)}")
        return report_path
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] analyze_telemetry.py failed (exit code {exc.returncode})")
        return None
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 4: Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(agents: list[str], official_result: dict | None = None) -> list[dict]:
    """Query DBs and print a formatted comparison table. Returns rows for report."""
    print("\n[4/4] Results comparison:")

    col_agent = 18
    col_n = 6
    col_rew = 11
    col_std = 11
    col_best = 11
    col_worst = 11

    header = (
        f"  {'Agent':<{col_agent}} {'N':>{col_n}}   "
        f"{'Mean':>{col_rew}}   {'Std':>{col_std}}   "
        f"{'Best':>{col_best}}   {'Worst':>{col_worst}}"
    )
    separator = "  " + "-" * (col_agent + col_n + col_rew + col_std + col_best + col_worst + 20)

    print(header)
    print(separator)

    rows: list[dict] = []

    # Show official evaluation result first (if available)
    if official_result:
        n_str = str(official_result.get("n_eps", "?"))
        mean_str = format_reward(official_result.get("mean"))
        std_str = format_std(official_result.get("std"))
        print(
            f"  {'v5 (official eval)':<{col_agent}} {n_str:>{col_n}}   "
            f"{mean_str:>{col_rew}}   {std_str:>{col_std}}   "
            f"{'N/A':>{col_best}}   {'N/A':>{col_worst}}"
        )
        rows.append({
            "agent": "v5 (official eval)",
            "n_eps": n_str,
            "mean": mean_str,
            "std": std_str,
            "best": "N/A",
            "worst": "N/A",
        })

    # Deduplicate: v5 and heuristic share a DB; show combined row if both selected
    seen_heuristic_db = False
    for agent_key in agents:
        if agent_key in ("heuristic", "v5"):
            if seen_heuristic_db:
                continue
            seen_heuristic_db = True
            stats = query_agent_stats("heuristic")
            display_name = "v4/v5 Heuristic"
        else:
            stats = query_agent_stats(agent_key)
            display_name = AGENT_CONFIG[agent_key]["display_name"]

        if stats:
            n_str = str(stats["n_eps"])
            mean_str = format_reward(stats["mean"])
            std_str = format_std(stats["std"])
            best_str = format_reward(stats["best"])
            worst_str = format_reward(stats["worst"])
        else:
            n_str = "???"
            mean_str = "???"
            std_str = "???"
            best_str = "???"
            worst_str = "???"

        print(
            f"  {display_name:<{col_agent}} {n_str:>{col_n}}   "
            f"{mean_str:>{col_rew}}   {std_str:>{col_std}}   "
            f"{best_str:>{col_best}}   {worst_str:>{col_worst}}"
        )

        rows.append({
            "agent": display_name,
            "n_eps": n_str,
            "mean": mean_str,
            "std": std_str,
            "best": best_str,
            "worst": worst_str,
        })

    print(separator)
    return rows


# ---------------------------------------------------------------------------
# Step 4: Write workflow report
# ---------------------------------------------------------------------------

def write_workflow_report(
    out_path: Path,
    agents: list[str],
    episodes: int,
    steps: int,
    seed: int,
    table_rows: list[dict],
    total_elapsed: float,
    official_result: dict | None = None,
) -> None:
    """Write a brief markdown workflow report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build markdown table
    md_table_lines = [
        "| Agent | N | Mean Reward | Std Dev | Best | Worst |",
        "|-------|---|-------------|---------|------|-------|",
    ]
    for row in table_rows:
        md_table_lines.append(
            f"| {row['agent']} | {row['n_eps']} | {row['mean']} "
            f"| {row['std']} | {row['best']} | {row['worst']} |"
        )

    lines = [
        "# CC4 Blue-Team Evaluation Workflow Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Total elapsed:** {total_elapsed:.1f}s",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Agents    | {', '.join(agents)} |",
        f"| Episodes  | {episodes} |",
        f"| Steps     | {steps} |",
        f"| Seed      | {seed} |",
        "",
        "## Results",
        "",
    ] + md_table_lines + [
        "",
        "## Reports",
        "",
        f"- Full analysis: [{ANALYSIS_REPORT.name}]({ANALYSIS_REPORT.relative_to(ROOT).as_posix()})",
        f"- Comprehensive report: [{COMPREHENSIVE_REPORT.name}]({COMPREHENSIVE_REPORT.relative_to(ROOT).as_posix()})",
        "",
        "> Note: v5 uses the current `EnterpriseHeuristicAgent` (updated in-place).",
        "> v5 results are stored under agent_type='heuristic' in `data/cc4_telemetry.db`.",
        "",
        "## Mission Phase Coverage",
        "",
        "Each evaluation episode runs for **500 steps**, covering all three mission phases:",
        "",
        "| Phase | Steps | Critical Zone | Penalty |",
        "|-------|-------|---------------|---------|",
        "| Phase 0 (Preplanning) | 0 -- 166 | None | -1 to -5/step |",
        "| Phase 1 (MissionA) | 167 -- 333 | operational_zone_a | -10/step RIA+LWF |",
        "| Phase 2 (MissionB) | 334 -- 499 | operational_zone_b | -10/step RIA+LWF |",
        "",
        "All evaluations use `EnterpriseScenarioGenerator(steps=500)` via the official",
        "`CybORG/Evaluation/evaluation.py` script to ensure consistent phase coverage.",
    ] + (
        [
            "",
            "## Official Evaluation Output",
            "",
            f"Results saved to: `{official_result.get('output_dir', 'N/A')}`",
            "Files: `summary.json`, `summary.txt`, `scores.txt`, `actions.txt`, `full.txt`",
        ]
        if official_result is not None
        else []
    ) + [
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CC4 Blue-Team Evaluation Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["sleep", "heuristic", "v5"],
        choices=VALID_AGENTS,
        metavar="AGENT",
        help=(
            f"Agent keys to evaluate (default: sleep heuristic v5). "
            f"Choices: {', '.join(VALID_AGENTS)}"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        metavar="N",
        help="Episodes per agent (default: 20)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        metavar="N",
        help="Max steps per episode (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        default=False,
        help="Skip data collection, only run analysis on existing DBs",
    )
    parser.add_argument(
        "--no-analyse",
        action="store_true",
        default=False,
        help="Skip analyze_telemetry.py step",
    )
    parser.add_argument(
        "--no-official-eval",
        action="store_true",
        default=False,
        help="Skip official CybORG evaluation.py step",
    )
    parser.add_argument(
        "--out",
        default=str(DOCS / "workflow_report.md"),
        metavar="PATH",
        help="Output report path (default: docs/workflow_report.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    agents: list[str] = args.agents

    t_start = time.perf_counter()

    print("=" * 60)
    print("CC4 Blue-Team Evaluation Workflow")
    print("=" * 60)
    print(
        f"Config: agents=[{', '.join(agents)}]  "
        f"episodes={args.episodes}  steps={args.steps}  seed={args.seed}"
    )

    # Note about v5
    if "v5" in agents:
        print(
            "\n  Note: v5 uses the current EnterpriseHeuristicAgent (updated in-place)."
            "\n        Results stored under agent_type='heuristic' in data/cc4_telemetry.db."
        )

    # Step 1: Official evaluation
    official_result: dict | None = None
    if not args.no_official_eval:
        official_result = run_official_evaluation(
            episodes=args.episodes,
            seed=args.seed,
        )
    else:
        print("\n[1/4] Official evaluation skipped (--no-official-eval)")

    # Step 2: Collection
    if not args.skip_collect:
        run_collection(
            agents=agents,
            episodes=args.episodes,
            steps=args.steps,
            seed=args.seed,
        )
    else:
        print("\n[2/4] Collection skipped (--skip-collect)")

    # Step 3: Analysis
    if not args.no_analyse:
        run_analysis()
    else:
        print("\n[3/4] Analysis skipped (--no-analyse)")

    # Step 4: Comparison table
    table_rows = print_comparison_table(agents, official_result=official_result)

    # Step 5: Write report
    total_elapsed = time.perf_counter() - t_start
    write_workflow_report(
        out_path=out_path,
        agents=agents,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        table_rows=table_rows,
        total_elapsed=total_elapsed,
        official_result=official_result,
    )

    print(f"\nWorkflow report: {out_path}")
    print(f"Total elapsed: {total_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
