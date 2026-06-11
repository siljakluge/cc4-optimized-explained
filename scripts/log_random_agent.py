#!/usr/bin/env python3
"""Random agent logger for CAGE Challenge 4.

Runs N episodes with a random blue agent (uniform over valid actions) and logs
extended state information to a SQLite database.

Usage:
    python scripts/log_random_agent.py [--episodes 20] [--steps 500] [--seed 42]

DB written to: data/random_agent.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# DB path
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "data" / "random_agent.db"

CREATE_TABLES_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS episodes (
    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_type TEXT DEFAULT 'random',
    total_reward REAL, n_steps INTEGER, seed INTEGER
);

CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER, phase INTEGER,
    cumulative_reward REAL, step_reward REAL
);

CREATE TABLE IF NOT EXISTS actions_taken (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER, agent_name TEXT,
    action_idx INTEGER, action_label TEXT, action_category TEXT,
    agent_reward REAL
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER, agent_name TEXT,
    obs_json TEXT
);

CREATE TABLE IF NOT EXISTS red_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER,
    agent_name TEXT, hostname TEXT, subnet TEXT, is_privileged INTEGER
);

CREATE TABLE IF NOT EXISTS host_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER,
    hostname TEXT, subnet TEXT,
    has_process_flag INTEGER, has_connection_flag INTEGER,
    n_process_events INTEGER, n_connection_events INTEGER
);

CREATE TABLE IF NOT EXISTS service_reliability (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER,
    hostname TEXT, service_name TEXT, reliability_pct INTEGER
);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(CREATE_TABLES_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------

def random_action(obs, mask) -> int:
    valid = np.where(np.array(mask, dtype=bool))[0]
    if len(valid) == 0:
        return 0
    return int(np.random.choice(valid))


# ---------------------------------------------------------------------------
# Action category
# ---------------------------------------------------------------------------

def categorize_action(label: str) -> str:
    if label == "Sleep":
        return "sleep"
    if label == "Monitor":
        return "monitor"
    if label.startswith("Restore"):
        return "restore"
    if label.startswith("Remove"):
        return "remove"
    if label.startswith("Analyse"):
        return "analyse"
    if label.startswith("DeployDecoy"):
        return "decoy"
    if label.startswith("BlockTrafficZone"):
        return "block"
    if label.startswith("AllowTrafficZone"):
        return "allow"
    return "other"


# ---------------------------------------------------------------------------
# State extraction helpers
# ---------------------------------------------------------------------------

def get_state(env):
    try:
        return env.env.environment_controller.state
    except AttributeError:
        pass
    try:
        return env.environment_controller.state
    except AttributeError:
        pass
    return None


def get_hostname_subnet(state, hostname: str) -> str:
    try:
        subnet_enum = state.hostname_subnet_map.get(hostname)
        if subnet_enum is None:
            return ""
        return str(subnet_enum).lower() if subnet_enum else ""
    except Exception:
        return ""


def collect_red_sessions(state, episode_id: int, step: int) -> list[tuple]:
    rows = []
    try:
        for agent_name, sessions_dict in state.sessions.items():
            if "red" not in agent_name.lower():
                continue
            for _sid, session in sessions_dict.items():
                try:
                    hostname = str(getattr(session, "hostname", None) or "")
                    subnet = get_hostname_subnet(state, hostname)
                    is_priv = 1 if session.has_privileged_access() else 0
                    rows.append((episode_id, step, str(agent_name), hostname, subnet, is_priv))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


def collect_host_events(state, episode_id: int, step: int) -> list[tuple]:
    rows = []
    try:
        for hostname, host in state.hosts.items():
            try:
                subnet = get_hostname_subnet(state, hostname)
                proc_events = (
                    list(getattr(host.events, "process_creation", []))
                    + list(getattr(host.events, "old_process_creation", []))
                )
                conn_events = (
                    list(getattr(host.events, "network_connections", []))
                    + list(getattr(host.events, "old_network_connections", []))
                )
                has_proc = 1 if proc_events else 0
                has_conn = 1 if conn_events else 0
                rows.append((
                    episode_id, step, str(hostname), subnet,
                    has_proc, has_conn,
                    len(proc_events), len(conn_events),
                ))
            except Exception:
                pass
    except Exception:
        pass
    return rows


def collect_service_reliability(state, episode_id: int, step: int) -> list[tuple]:
    rows = []
    try:
        for hostname, host in state.hosts.items():
            try:
                for svc_name, svc in host.services.items():
                    try:
                        reliability = getattr(svc, "_percent_reliable", None)
                        if reliability is None:
                            reliability = svc.get_service_reliability()
                        rows.append((episode_id, step, str(hostname), str(svc_name), int(reliability)))
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    return rows


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(
    n_episodes: int,
    max_steps: int,
    seed: int,
    conn: sqlite3.Connection,
) -> None:
    from CybORG import CybORG
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent

    np.random.seed(seed)

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapper(env=cyborg)

    # Initial reset to get agent names
    obs_dict, _ = env.reset()
    agent_names = env.possible_agents

    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()

        ep_reward = 0.0
        cumulative_reward = 0.0

        cursor = conn.execute(
            "INSERT INTO episodes (agent_type, total_reward, n_steps, seed) VALUES (?, ?, ?, ?)",
            ("random", None, None, seed),
        )
        episode_id = cursor.lastrowid
        conn.commit()

        rows_steps: list[tuple] = []
        rows_actions: list[tuple] = []
        rows_obs: list[tuple] = []
        rows_red: list[tuple] = []
        rows_host: list[tuple] = []
        rows_svc: list[tuple] = []

        final_step = 0

        for step in range(max_steps):
            # Build actions for all blue agents
            actions: dict[str, int] = {}
            action_labels_this_step: dict[str, str] = {}

            for name in agent_names:
                raw_obs = obs_dict.get(name, np.zeros(1))
                mask = env.action_mask(name)
                idx = random_action(raw_obs, mask)
                actions[name] = idx
                labels = env.action_labels(name)
                label = labels[idx] if idx < len(labels) else str(idx)
                action_labels_this_step[name] = label

            obs_dict, rew_dict, term_dict, trunc_dict, _info = env.step(actions)
            step_reward = sum(rew_dict.values())
            ep_reward += step_reward
            cumulative_reward += step_reward

            # Internal state
            state = get_state(env)
            phase = None
            if state is not None:
                try:
                    phase = int(state.mission_phase)
                except Exception:
                    phase = None

            rows_steps.append((episode_id, step, phase, cumulative_reward, step_reward))

            # Per-agent actions
            for name in agent_names:
                idx = actions[name]
                label = action_labels_this_step[name]
                category = categorize_action(label)
                agent_rew = float(rew_dict.get(name, 0.0))
                rows_actions.append((episode_id, step, str(name), idx, label, category, agent_rew))

            # Observations — only every 10 steps
            if step % 10 == 0:
                for name in agent_names:
                    raw_obs = obs_dict.get(name)
                    if raw_obs is not None:
                        try:
                            obs_list = raw_obs.tolist() if hasattr(raw_obs, "tolist") else list(raw_obs)
                            rows_obs.append((episode_id, step, str(name), json.dumps(obs_list)))
                        except Exception:
                            pass

            # State-based tables
            if state is not None:
                rows_red.extend(collect_red_sessions(state, episode_id, step))
                rows_host.extend(collect_host_events(state, episode_id, step))
                rows_svc.extend(collect_service_reliability(state, episode_id, step))

            final_step = step

            if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in agent_names):
                break

        # Bulk inserts
        conn.executemany(
            "INSERT INTO steps (episode_id, step, phase, cumulative_reward, step_reward) "
            "VALUES (?,?,?,?,?)",
            rows_steps,
        )
        conn.executemany(
            "INSERT INTO actions_taken "
            "(episode_id, step, agent_name, action_idx, action_label, action_category, agent_reward) "
            "VALUES (?,?,?,?,?,?,?)",
            rows_actions,
        )
        conn.executemany(
            "INSERT INTO observations (episode_id, step, agent_name, obs_json) VALUES (?,?,?,?)",
            rows_obs,
        )
        conn.executemany(
            "INSERT INTO red_sessions "
            "(episode_id, step, agent_name, hostname, subnet, is_privileged) "
            "VALUES (?,?,?,?,?,?)",
            rows_red,
        )
        conn.executemany(
            "INSERT INTO host_events "
            "(episode_id, step, hostname, subnet, has_process_flag, has_connection_flag, "
            "n_process_events, n_connection_events) VALUES (?,?,?,?,?,?,?,?)",
            rows_host,
        )
        conn.executemany(
            "INSERT INTO service_reliability (episode_id, step, hostname, service_name, reliability_pct) "
            "VALUES (?,?,?,?,?)",
            rows_svc,
        )

        conn.execute(
            "UPDATE episodes SET total_reward=?, n_steps=? WHERE episode_id=?",
            (ep_reward, final_step + 1, episode_id),
        )
        conn.commit()

        elapsed = time.perf_counter() - t0
        print(
            f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:9.1f}  "
            f"steps={final_step+1}  elapsed={elapsed:.1f}s"
        )

    total_elapsed = time.perf_counter() - t0
    print(f"\nDone. {n_episodes} episodes in {total_elapsed:.1f}s — DB: {DB_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Log CybORG CC4 random-agent telemetry to SQLite (data/random_agent.db)"
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes (default 20)")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode (default 500)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    args = parser.parse_args()

    print(
        f"\nRandom agent logger — episodes={args.episodes}  "
        f"steps={args.steps}  seed={args.seed}"
    )
    print(f"Writing to: {DB_PATH}\n")

    conn = open_db(DB_PATH)
    try:
        run_episodes(
            n_episodes=args.episodes,
            max_steps=args.steps,
            seed=args.seed,
            conn=conn,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
