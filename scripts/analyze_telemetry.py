#!/usr/bin/env python3
"""Analyze CC4 telemetry SQLite DB and produce a Markdown report.

Usage:
    python scripts/analyze_telemetry.py [--db data/cc4_telemetry.db] [--out docs/telemetry_analysis_report.md]
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DB_DEFAULT = Path(__file__).parent.parent / "data" / "cc4_telemetry.db"
OUT_DEFAULT = Path(__file__).parent.parent / "docs" / "telemetry_analysis_report.md"


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def q(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    return conn.execute(sql, params).fetchall()


def scalar(conn: sqlite3.Connection, sql: str, params: tuple = (), default=None):
    rows = q(conn, sql, params)
    return rows[0][0] if rows else default


# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------

def section_episode_summary(conn: sqlite3.Connection) -> str:
    rows = q(conn, """
        SELECT agent_type,
               COUNT(*) AS n_eps,
               AVG(total_reward) AS mean_rew,
               MIN(total_reward) AS min_rew,
               MAX(total_reward) AS max_rew,
               AVG(n_steps) AS mean_steps
        FROM episodes
        GROUP BY agent_type
        ORDER BY agent_type
    """)
    lines = ["## Episode Summary\n",
             "| Agent | Episodes | Mean Reward | Min Reward | Max Reward | Mean Steps |",
             "|-------|----------|-------------|------------|------------|------------|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.0f} | {r[3]:.0f} | {r[4]:.0f} | {r[5]:.0f} |")
    return "\n".join(lines) + "\n"


def section_phase_timing(conn: sqlite3.Connection) -> str:
    """Average step at which each phase begins per agent type."""
    rows = q(conn, """
        SELECT e.agent_type, s.phase, MIN(s.step) AS first_step
        FROM steps s
        JOIN episodes e ON e.episode_id = s.episode_id
        WHERE s.phase IS NOT NULL
        GROUP BY e.episode_id, s.phase
    """)
    # Aggregate per (agent_type, phase)
    from collections import defaultdict
    buckets: dict[tuple, list] = defaultdict(list)
    for agent_type, phase, first_step in rows:
        buckets[(agent_type, phase)].append(first_step)

    lines = ["## Phase Timing (avg first step)\n",
             "| Agent | Phase 0 Start | Phase 1 Start | Phase 2 Start |",
             "|-------|--------------|--------------|--------------|"]
    agent_types = sorted(set(k[0] for k in buckets))
    for at in agent_types:
        p0 = sum(buckets.get((at, 0), [0])) / max(len(buckets.get((at, 0), [1])), 1)
        p1 = sum(buckets.get((at, 1), [0])) / max(len(buckets.get((at, 1), [1])), 1)
        p2 = sum(buckets.get((at, 2), [0])) / max(len(buckets.get((at, 2), [1])), 1)
        lines.append(f"| {at} | {p0:.0f} | {p1:.0f} | {p2:.0f} |")
    return "\n".join(lines) + "\n"


def section_red_penetration(conn: sqlite3.Connection) -> str:
    """When does red first reach each subnet zone?"""
    rows = q(conn, """
        SELECT e.agent_type,
               first_entries.subnet,
               AVG(first_step) AS avg_first_step,
               MIN(first_step) AS min_first_step,
               COUNT(*) AS n_episodes
        FROM (
            SELECT episode_id, subnet, MIN(step) AS first_step
            FROM red_penetration
            WHERE subnet != ''
            GROUP BY episode_id, subnet
        ) AS first_entries
        JOIN episodes e ON e.episode_id = first_entries.episode_id
        GROUP BY e.agent_type, first_entries.subnet
        ORDER BY e.agent_type, avg_first_step
    """)
    lines = ["## Red Lateral Movement — First Reach per Subnet\n",
             "| Agent | Subnet | Avg First Step | Min First Step | Episodes Reached |",
             "|-------|--------|---------------|---------------|-----------------|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.0f} | {r[3]} | {r[4]} |")
    return "\n".join(lines) + "\n"


def section_red_privileged(conn: sqlite3.Connection) -> str:
    """How often does red achieve privileged access per subnet?"""
    rows = q(conn, """
        SELECT e.agent_type, rp.subnet,
               SUM(rp.is_privileged) AS priv_count,
               COUNT(*) AS total_count,
               ROUND(100.0 * SUM(rp.is_privileged) / COUNT(*), 1) AS priv_pct
        FROM red_penetration rp
        JOIN episodes e ON e.episode_id = rp.episode_id
        WHERE rp.subnet != ''
        GROUP BY e.agent_type, rp.subnet
        ORDER BY e.agent_type, priv_pct DESC
    """)
    lines = ["## Red Privileged Access Rate per Subnet\n",
             "| Agent | Subnet | Privileged Steps | Total Steps | Priv% |",
             "|-------|--------|-----------------|-------------|-------|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]}% |")
    return "\n".join(lines) + "\n"


def section_alerts(conn: sqlite3.Connection) -> str:
    """False positive rate: process/connection flags when no red session present."""
    rows = q(conn, """
        SELECT e.agent_type,
               ha.subnet,
               SUM(ha.has_process_flag) AS proc_alerts,
               SUM(ha.has_connection_flag) AS conn_alerts,
               COUNT(*) AS total_host_steps
        FROM host_alerts ha
        JOIN episodes e ON e.episode_id = ha.episode_id
        WHERE ha.subnet != ''
        GROUP BY e.agent_type, ha.subnet
        ORDER BY e.agent_type, conn_alerts DESC
    """)
    lines = ["## Host Alert Frequency per Subnet\n",
             "| Agent | Subnet | Process Alerts | Connection Alerts | Total Host-Steps |",
             "|-------|--------|---------------|------------------|-----------------|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    return "\n".join(lines) + "\n"


def section_service_degradation(conn: sqlite3.Connection) -> str:
    """Track OTService reliability degradation on operational zone servers."""
    rows = q(conn, """
        SELECT e.agent_type,
               ss.hostname,
               ss.service_name,
               AVG(ss.reliability_pct) AS avg_reliability,
               MIN(ss.reliability_pct) AS min_reliability,
               SUM(CASE WHEN ss.reliability_pct < 100 THEN 1 ELSE 0 END) AS degraded_steps,
               COUNT(*) AS total_steps
        FROM service_status ss
        JOIN episodes e ON e.episode_id = ss.episode_id
        WHERE (ss.hostname LIKE '%operational_zone%' OR ss.hostname LIKE '%restricted_zone%')
          AND ss.service_name LIKE '%OT%'
        GROUP BY e.agent_type, ss.hostname, ss.service_name
        ORDER BY e.agent_type, avg_reliability ASC
    """)

    if not rows:
        # Try without OTService filter
        rows = q(conn, """
            SELECT e.agent_type,
                   ss.hostname,
                   ss.service_name,
                   AVG(ss.reliability_pct) AS avg_reliability,
                   MIN(ss.reliability_pct) AS min_reliability,
                   SUM(CASE WHEN ss.reliability_pct < 100 THEN 1 ELSE 0 END) AS degraded_steps,
                   COUNT(*) AS total_steps
            FROM service_status ss
            JOIN episodes e ON e.episode_id = ss.episode_id
            GROUP BY e.agent_type, ss.hostname, ss.service_name
            ORDER BY e.agent_type, avg_reliability ASC
            LIMIT 40
        """)

    lines = ["## Service Reliability (Critical Hosts)\n",
             "| Agent | Hostname | Service | Avg Reliability | Min Reliability | Degraded Steps | Total Steps |",
             "|-------|----------|---------|----------------|----------------|---------------|------------|"]
    for r in rows:
        lines.append(
            f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.1f}% | {r[4]}% | {r[5]} | {r[6]} |"
        )
    return "\n".join(lines) + "\n"


def section_reward_by_phase(conn: sqlite3.Connection) -> str:
    """Cumulative reward accumulated per phase per agent type."""
    rows = q(conn, """
        SELECT e.agent_type,
               deltas.phase,
               SUM(delta_reward) AS total_reward,
               AVG(delta_reward) AS avg_per_step,
               COUNT(*) AS n_steps
        FROM (
            SELECT episode_id, phase, total_reward - LAG(total_reward, 1, 0) OVER (
                PARTITION BY episode_id ORDER BY step
            ) AS delta_reward
            FROM steps
            WHERE phase IS NOT NULL
        ) AS deltas
        JOIN episodes e ON e.episode_id = deltas.episode_id
        GROUP BY e.agent_type, deltas.phase
        ORDER BY e.agent_type, deltas.phase
    """)
    lines = ["## Reward Accumulation by Phase\n",
             "| Agent | Phase | Total Reward | Avg Reward/Step | Steps |",
             "|-------|-------|-------------|----------------|-------|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.0f} | {r[3]:.3f} | {r[4]} |")
    return "\n".join(lines) + "\n"


def section_worst_episodes(conn: sqlite3.Connection) -> str:
    """Top-10 worst episodes per agent type to understand catastrophic failures."""
    rows = q(conn, """
        SELECT agent_type, episode_id, total_reward, n_steps, seed
        FROM episodes
        ORDER BY total_reward ASC
        LIMIT 20
    """)
    lines = ["## Worst Episodes (potential catastrophic red successes)\n",
             "| Agent | Episode ID | Total Reward | Steps | Seed |",
             "|-------|-----------|-------------|-------|------|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.0f} | {r[3]} | {r[4]} |")
    return "\n".join(lines) + "\n"


def section_block_coverage(conn: sqlite3.Connection) -> str:
    """How many steps is contractor_network blocked per agent type?"""
    rows = q(conn, """
        SELECT e.agent_type,
               bs.to_subnet,
               COUNT(*) AS blocked_steps
        FROM blocks_status bs
        JOIN episodes e ON e.episode_id = bs.episode_id
        WHERE bs.from_subnets_json LIKE '%contractor%'
        GROUP BY e.agent_type, bs.to_subnet
        ORDER BY e.agent_type, blocked_steps DESC
        LIMIT 30
    """)
    lines = ["## Firewall: contractor_network Blocked FROM Subnets (step-count)\n",
             "| Agent | To Subnet | Steps Blocked |",
             "|-------|-----------|--------------|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    return "\n".join(lines) + "\n"


def section_alert_fp_analysis(conn: sqlite3.Connection) -> str:
    """Estimate false positive alert rate: alerts without red presence on same host."""
    rows = q(conn, """
        SELECT e.agent_type,
               ha.subnet,
               SUM(ha.has_process_flag) AS proc_alerts,
               SUM(ha.has_connection_flag) AS conn_alerts,
               (SELECT COUNT(*) FROM red_penetration rp
                WHERE rp.episode_id = ha.episode_id
                  AND rp.step = ha.step
                  AND rp.hostname = ha.hostname) AS red_present_steps
        FROM host_alerts ha
        JOIN episodes e ON e.episode_id = ha.episode_id
        WHERE ha.has_process_flag = 1 OR ha.has_connection_flag = 1
        GROUP BY e.agent_type, ha.subnet
        ORDER BY e.agent_type, proc_alerts DESC
        LIMIT 40
    """)
    lines = ["## Alert Analysis: Flags vs Red Session Coincidence\n",
             "_Note: 'red_present_steps' only counts rows where red session was on exact hostname/step_\n",
             "| Agent | Subnet | Proc Alerts | Conn Alerts |",
             "|-------|--------|------------|------------|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.out)

    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}. Run log_env_data.py first.")
        return

    conn = sqlite3.connect(str(db_path))

    sections = [
        "# CC4 Telemetry Analysis Report\n\n_Generated from `data/cc4_telemetry.db`_\n",
        section_episode_summary(conn),
        section_phase_timing(conn),
        section_reward_by_phase(conn),
        section_red_penetration(conn),
        section_red_privileged(conn),
        section_alerts(conn),
        section_alert_fp_analysis(conn),
        section_service_degradation(conn),
        section_block_coverage(conn),
        section_worst_episodes(conn),
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = "\n---\n\n".join(sections)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")

    conn.close()


if __name__ == "__main__":
    main()
