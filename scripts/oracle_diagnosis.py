#!/usr/bin/env python3
"""Diagnose oracle blue agent V1 failures in CAGE Challenge 4.

Runs the oracle agent instrumented to log every decision, tracking:
- What action the oracle chose and what red sessions existed
- Steps between red getting root and oracle responding
- How often oracle issues Remove when Restore would be faster
- How often red re-exploits during Remove's 3-step window
- Steps red has Impact running before oracle Restores
- Phase transition handling timing

Outputs results to data/oracle_diagnosis.db (SQLite).

Usage:
    python scripts/oracle_diagnosis.py [--episodes 30] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_db(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with diagnosis tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        DROP TABLE IF EXISTS step_log;
        DROP TABLE IF EXISTS episode_summary;
        DROP TABLE IF EXISTS remove_vs_restore;
        DROP TABLE IF EXISTS reexploit_events;
        DROP TABLE IF EXISTS impact_exposure;
        DROP TABLE IF EXISTS response_latency;

        CREATE TABLE step_log (
            episode INTEGER,
            step INTEGER,
            agent TEXT,
            action_label TEXT,
            phase INTEGER,
            red_root_hosts TEXT,
            red_user_hosts TEXT,
            red_any_hosts TEXT,
            step_reward REAL,
            cumulative_reward REAL
        );

        CREATE TABLE episode_summary (
            episode INTEGER PRIMARY KEY,
            total_reward REAL,
            phase0_reward REAL,
            phase1_reward REAL,
            phase2_reward REAL,
            total_removes INTEGER,
            total_restores INTEGER,
            total_blocks INTEGER,
            total_sleeps INTEGER,
            total_decoys INTEGER,
            total_red_root_steps INTEGER,
            total_red_user_steps INTEGER,
            max_red_hosts_simultaneous INTEGER
        );

        CREATE TABLE remove_vs_restore (
            episode INTEGER,
            step INTEGER,
            agent TEXT,
            hostname TEXT,
            action_type TEXT,
            had_root INTEGER,
            had_user INTEGER,
            red_reappeared_within_window INTEGER
        );

        CREATE TABLE reexploit_events (
            episode INTEGER,
            step INTEGER,
            hostname TEXT,
            agent TEXT,
            remove_step INTEGER,
            steps_since_remove INTEGER
        );

        CREATE TABLE impact_exposure (
            episode INTEGER,
            hostname TEXT,
            phase INTEGER,
            root_start_step INTEGER,
            restore_step INTEGER,
            exposure_steps INTEGER,
            is_ot_server INTEGER
        );

        CREATE TABLE response_latency (
            episode INTEGER,
            hostname TEXT,
            event_type TEXT,
            first_seen_step INTEGER,
            response_step INTEGER,
            latency_steps INTEGER,
            response_action TEXT
        );
    """)
    conn.commit()
    return conn


def run_diagnosis(n_episodes: int = 30, seed: int = 42):
    from CybORG import CybORG
    from CybORG.Agents.Wrappers import BlueFlatWrapperV2
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent
    from CybORG.Agents.SimpleAgents.OracleBlueAgent import make_oracle_agents

    db_path = Path(__file__).parent.parent / "data" / "oracle_diagnosis.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = create_db(db_path)

    sg = EnterpriseScenarioGenerator(
        steps=500,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapperV2(env=cyborg)

    obs_dict, _ = env.reset()
    agents = make_oracle_agents(env)
    agent_names = env.possible_agents

    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
        for name, ag in agents.items():
            ag.reset()
            ag.set_action_info(
                env.action_labels(name), env.action_mask(name), subnet_hosts
            )

        ep_reward = 0.0
        step_rewards = []

        # Per-episode tracking
        action_counts = {
            "removes": 0, "restores": 0, "blocks": 0,
            "sleeps": 0, "decoys": 0, "allows": 0,
        }
        red_root_step_count = 0
        red_user_step_count = 0
        max_simultaneous_red = 0

        # Track when red first gets root on each host (for latency measurement)
        red_root_first_seen: dict[str, int] = {}
        red_user_first_seen: dict[str, int] = {}
        # Track active removes for re-exploit detection
        active_removes: dict[str, int] = {}  # hostname -> step issued

        for step in range(500):
            state = env.env.environment_controller.state
            phase = state.mission_phase

            # Read red session state before action
            red_root: set[str] = set()
            red_user: set[str] = set()
            red_any: set[str] = set()

            for ag_name, sessions in state.sessions.items():
                if "red" not in ag_name:
                    continue
                for sid, sess in sessions.items():
                    if not sess.active:
                        continue
                    h = sess.hostname
                    red_any.add(h)
                    if sess.has_privileged_access():
                        red_root.add(h)
                    else:
                        red_user.add(h)

            red_root_step_count += len(red_root)
            red_user_step_count += len(red_user)
            max_simultaneous_red = max(max_simultaneous_red, len(red_any))

            # Track first-seen for latency
            for h in red_root:
                if h not in red_root_first_seen:
                    red_root_first_seen[h] = step
            for h in red_user:
                if h not in red_user_first_seen:
                    red_user_first_seen[h] = step

            # Check for re-exploits during active Remove windows
            for h, remove_step in list(active_removes.items()):
                if step <= remove_step + 2:  # Within 3-step Remove window
                    if h in red_any:
                        conn.execute(
                            "INSERT INTO reexploit_events VALUES (?,?,?,?,?,?)",
                            (ep, step, h, "", remove_step, step - remove_step),
                        )
                else:
                    del active_removes[h]

            # Get actions from all agents
            actions: dict[str, int] = {}
            messages: dict[str, np.ndarray] = {}
            for name, ag in agents.items():
                raw_obs = obs_dict.get(name, np.zeros(1))
                amask = env.action_mask(name)
                action_idx, msg_out = ag.get_action(
                    raw_obs, np.array(amask, dtype=bool)
                )
                actions[name] = action_idx
                messages[name] = msg_out

                # Classify action
                labels = env.action_labels(name)
                label = labels[action_idx] if action_idx < len(labels) else "Sleep"

                if label.startswith("Remove"):
                    action_counts["removes"] += 1
                    # Parse hostname
                    import re
                    m = re.match(r"Remove\s+(\S+)", label)
                    if m:
                        hname = m.group(1)
                        active_removes[hname] = step
                        had_root = 1 if hname in red_root else 0
                        had_user = 1 if hname in red_user else 0
                        conn.execute(
                            "INSERT INTO remove_vs_restore VALUES (?,?,?,?,?,?,?,?)",
                            (ep, step, name, hname, "Remove",
                             had_root, had_user, 0),
                        )
                        # Log latency
                        fs = red_user_first_seen.get(hname, step)
                        conn.execute(
                            "INSERT INTO response_latency VALUES (?,?,?,?,?,?,?)",
                            (ep, hname, "user_session", fs, step,
                             step - fs, "Remove"),
                        )
                elif label.startswith("Restore"):
                    action_counts["restores"] += 1
                    m = re.match(r"Restore\s+(\S+)", label)
                    if m:
                        hname = m.group(1)
                        had_root = 1 if hname in red_root else 0
                        had_user = 1 if hname in red_user else 0
                        conn.execute(
                            "INSERT INTO remove_vs_restore VALUES (?,?,?,?,?,?,?,?)",
                            (ep, step, name, hname, "Restore",
                             had_root, had_user, 0),
                        )
                        # Log latency
                        event_type = "root_session" if had_root else "user_session"
                        fs_dict = red_root_first_seen if had_root else red_user_first_seen
                        fs = fs_dict.get(hname, step)
                        conn.execute(
                            "INSERT INTO response_latency VALUES (?,?,?,?,?,?,?)",
                            (ep, hname, event_type, fs, step,
                             step - fs, "Restore"),
                        )
                        # Clear first-seen tracking (host will be restored)
                        red_root_first_seen.pop(hname, None)
                        red_user_first_seen.pop(hname, None)
                elif label.startswith("Block"):
                    action_counts["blocks"] += 1
                elif label.startswith("Allow"):
                    action_counts["allows"] += 1
                elif label.startswith("DeployDecoy"):
                    action_counts["decoys"] += 1
                elif label == "Sleep":
                    action_counts["sleeps"] += 1

                # Log step
                conn.execute(
                    "INSERT INTO step_log VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (ep, step, name, label, phase,
                     json.dumps(sorted(red_root)),
                     json.dumps(sorted(red_user)),
                     json.dumps(sorted(red_any)),
                     0.0, 0.0),
                )

            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(
                actions, messages=messages
            )
            step_rew = sum(rew_dict.values())
            ep_reward += step_rew
            step_rewards.append(step_rew)

            if all(
                term_dict.get(n, False) or trunc_dict.get(n, False)
                for n in agent_names
            ):
                break

        # Track impact exposure (root on OT servers)
        for h, first_step in red_root_first_seen.items():
            is_ot = 1 if "server_host_0" in h and "operational_zone" in h else 0
            conn.execute(
                "INSERT INTO impact_exposure VALUES (?,?,?,?,?,?,?)",
                (ep, h, -1, first_step, -1, 500 - first_step, is_ot),
            )

        # Episode summary
        while len(step_rewards) < 500:
            step_rewards.append(0.0)
        conn.execute(
            "INSERT INTO episode_summary VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ep, ep_reward,
             sum(step_rewards[0:167]),
             sum(step_rewards[167:334]),
             sum(step_rewards[334:]),
             action_counts["removes"],
             action_counts["restores"],
             action_counts["blocks"],
             action_counts["sleeps"],
             action_counts["decoys"],
             red_root_step_count,
             red_user_step_count,
             max_simultaneous_red),
        )
        conn.commit()

        print(f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:9.1f}  "
              f"removes={action_counts['removes']:3d}  "
              f"restores={action_counts['restores']:3d}  "
              f"red_root_steps={red_root_step_count}")

    elapsed = time.perf_counter() - t0
    conn.commit()

    # Print summary statistics
    print(f"\n{'='*70}")
    print("  Oracle V1 Diagnosis Summary")
    print(f"{'='*70}")

    cur = conn.cursor()

    # Overall reward
    row = cur.execute(
        "SELECT AVG(total_reward), MIN(total_reward), MAX(total_reward) "
        "FROM episode_summary"
    ).fetchone()
    print(f"  Mean reward: {row[0]:.1f}  (min={row[1]:.1f}, max={row[2]:.1f})")

    # Remove vs Restore counts
    row = cur.execute(
        "SELECT AVG(total_removes), AVG(total_restores), AVG(total_sleeps) "
        "FROM episode_summary"
    ).fetchone()
    print(f"  Avg removes/ep: {row[0]:.1f}  restores/ep: {row[1]:.1f}  "
          f"sleeps/ep: {row[2]:.1f}")

    # Remove on user sessions (the key flaw)
    row = cur.execute(
        "SELECT COUNT(*) FROM remove_vs_restore "
        "WHERE action_type='Remove' AND had_user=1 AND had_root=0"
    ).fetchone()
    print(f"  Remove on user-only sessions: {row[0]} total")

    # Re-exploit events
    row = cur.execute("SELECT COUNT(*) FROM reexploit_events").fetchone()
    print(f"  Re-exploit during Remove window: {row[0]} events")

    # Response latency
    row = cur.execute(
        "SELECT AVG(latency_steps), MAX(latency_steps) "
        "FROM response_latency WHERE event_type='root_session'"
    ).fetchone()
    if row[0] is not None:
        print(f"  Root session response latency: avg={row[0]:.1f} max={row[1]}")

    row = cur.execute(
        "SELECT AVG(latency_steps), MAX(latency_steps) "
        "FROM response_latency WHERE event_type='user_session'"
    ).fetchone()
    if row[0] is not None:
        print(f"  User session response latency: avg={row[0]:.1f} max={row[1]}")

    # Impact exposure
    row = cur.execute(
        "SELECT COUNT(*), AVG(exposure_steps) "
        "FROM impact_exposure WHERE is_ot_server=1"
    ).fetchone()
    if row[0]:
        print(f"  OT server root exposure: {row[0]} events, "
              f"avg {row[1]:.1f} steps")

    # Phase breakdown
    for p in range(3):
        col = f"phase{p}_reward"
        row = cur.execute(
            f"SELECT AVG({col}) FROM episode_summary"
        ).fetchone()
        print(f"  Phase {p} avg reward: {row[0]:.1f}")

    print(f"\n  Wall time: {elapsed:.1f}s")
    print(f"  Database: {db_path}")
    print(f"{'='*70}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose Oracle V1 failures")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_diagnosis(args.episodes, args.seed)
