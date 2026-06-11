#!/usr/bin/env python3
"""Telemetry logging script for CAGE Challenge 4.

Runs N episodes and logs all available state data to a SQLite database.

Usage:
    python scripts/log_env_data.py [--agent sleep|heuristic] [--episodes 30] [--steps 500] [--seed 42]

DB written to: data/cc4_telemetry.db
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
# DB helpers
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "data" / "cc4_telemetry.db"

CREATE_TABLES_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS episodes (
    episode_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_type   TEXT    NOT NULL,
    total_reward REAL,
    n_steps      INTEGER,
    seed         INTEGER
);

CREATE TABLE IF NOT EXISTS steps (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id   INTEGER NOT NULL,
    step         INTEGER NOT NULL,
    phase        INTEGER,
    total_reward REAL,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS zone_rewards (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id   INTEGER NOT NULL,
    step         INTEGER NOT NULL,
    zone         TEXT,
    lwf_reward   REAL,
    asf_reward   REAL,
    ria_reward   REAL,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS red_penetration (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id   INTEGER NOT NULL,
    step         INTEGER NOT NULL,
    agent_name   TEXT,
    hostname     TEXT,
    subnet       TEXT,
    is_privileged INTEGER,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS host_alerts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id          INTEGER NOT NULL,
    step                INTEGER NOT NULL,
    hostname            TEXT,
    subnet              TEXT,
    has_process_flag    INTEGER,
    has_connection_flag INTEGER,
    n_process_events    INTEGER,
    n_connection_events INTEGER,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS service_status (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL,
    step            INTEGER NOT NULL,
    hostname        TEXT,
    service_name    TEXT,
    reliability_pct INTEGER,
    is_active       INTEGER,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS blocks_status (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id          INTEGER NOT NULL,
    step                INTEGER NOT NULL,
    to_subnet           TEXT,
    from_subnets_json   TEXT,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(CREATE_TABLES_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Reward estimation helpers
# ---------------------------------------------------------------------------

# Known reward weights per zone for LWF/ASF/RIA (estimated from CC4 docs).
# Zones: contractor_network, internet, admin_network, office_network,
#        public_access_zone, restricted_zone_a, restricted_zone_b,
#        operational_zone_a, operational_zone_b
# These are rough per-step cost estimates used to apportion the scalar reward.
ZONE_REWARD_TABLE = {
    "contractor_network_subnet":  {"lwf": -0.1, "asf": -0.1, "ria": 0.0},
    "internet_subnet":            {"lwf": -0.1, "asf": -0.1, "ria": 0.0},
    "admin_network_subnet":       {"lwf": -0.1, "asf": -0.1, "ria": 0.0},
    "office_network_subnet":      {"lwf": -0.1, "asf": -0.1, "ria": 0.0},
    "public_access_zone_subnet":  {"lwf": -0.1, "asf": -0.5, "ria": 0.0},
    "restricted_zone_a_subnet":   {"lwf": -1.0, "asf": -1.0, "ria": 0.0},
    "restricted_zone_b_subnet":   {"lwf": -1.0, "asf": -1.0, "ria": 0.0},
    "operational_zone_a_subnet":  {"lwf": -1.0, "asf": -1.0, "ria": -1.0},
    "operational_zone_b_subnet":  {"lwf": -1.0, "asf": -1.0, "ria": -1.0},
}


def estimate_zone_rewards(step_reward: float, zone: str) -> tuple[float, float, float]:
    """Return a rough (lwf, asf, ria) split for a given zone based on known table."""
    entry = ZONE_REWARD_TABLE.get(zone, {"lwf": 0.0, "asf": 0.0, "ria": 0.0})
    total = entry["lwf"] + entry["asf"] + entry["ria"]
    if total == 0:
        return 0.0, 0.0, 0.0
    scale = step_reward / total if total != 0 else 0.0
    return entry["lwf"] * scale, entry["asf"] * scale, entry["ria"] * scale


# ---------------------------------------------------------------------------
# State extraction helpers
# ---------------------------------------------------------------------------

def get_state(env):
    """Return the internal CybORG State object, trying multiple access paths."""
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
    """Return subnet name for a hostname, or empty string if unknown."""
    try:
        subnet_enum = state.hostname_subnet_map.get(hostname)
        if subnet_enum is None:
            return ""
        # subnet_enum may be a string or an Enum — normalise to string
        return str(subnet_enum).lower() if subnet_enum else ""
    except Exception:
        return ""


def collect_red_penetration(state, episode_id: int, step: int) -> list[tuple]:
    """Return rows for red_penetration table."""
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


def collect_host_alerts(state, episode_id: int, step: int) -> list[tuple]:
    """Return rows for host_alerts table."""
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


def collect_service_status(state, episode_id: int, step: int) -> list[tuple]:
    """Return rows for service_status table."""
    rows = []
    try:
        for hostname, host in state.hosts.items():
            try:
                for svc_name, svc in host.services.items():
                    try:
                        reliability = getattr(svc, "_percent_reliable", None)
                        if reliability is None:
                            reliability = svc.get_service_reliability()
                        is_active = 1 if getattr(svc, "active", True) else 0
                        rows.append((episode_id, step, str(hostname), str(svc_name), int(reliability), is_active))
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    return rows


def collect_blocks_status(state, episode_id: int, step: int) -> list[tuple]:
    """Return rows for blocks_status table."""
    rows = []
    try:
        for to_subnet, from_list in state.blocks.items():
            try:
                rows.append((episode_id, step, str(to_subnet), json.dumps(list(from_list))))
            except Exception:
                pass
    except Exception:
        pass
    return rows


def collect_zone_rewards(episode_id: int, step: int, step_reward: float) -> list[tuple]:
    """Return estimated zone reward rows."""
    rows = []
    for zone in ZONE_REWARD_TABLE:
        lwf, asf, ria = estimate_zone_rewards(step_reward, zone)
        rows.append((episode_id, step, zone, lwf, asf, ria))
    return rows


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(
    agent_type: str,
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
    from CybORG.Agents import SleepAgent

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapper(env=cyborg)

    # Initial reset to get agent names and init heuristic agents if needed
    obs_dict, _ = env.reset()
    agent_names = env.possible_agents

    if agent_type == "heuristic":
        from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import make_heuristic_agents
        agents = make_heuristic_agents(env)
    else:
        # SleepAgent mode: action index 0 for all blue agents
        agents = {name: SleepAgent() for name in agent_names}

    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()

        if agent_type == "heuristic":
            subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
            for name, ag in agents.items():
                ag.reset()
                ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)

        ep_reward = 0.0
        cumulative_reward = 0.0

        # Insert episode row (update total_reward + n_steps at end of episode)
        cursor = conn.execute(
            "INSERT INTO episodes (agent_type, total_reward, n_steps, seed) VALUES (?, ?, ?, ?)",
            (agent_type, None, None, seed),
        )
        episode_id = cursor.lastrowid
        conn.commit()

        step_rows_steps = []
        step_rows_zone = []
        step_rows_red = []
        step_rows_alerts = []
        step_rows_services = []
        step_rows_blocks = []

        final_step = 0

        for step in range(max_steps):
            actions: dict[str, int] = {}

            if agent_type == "heuristic":
                for name, ag in agents.items():
                    raw_obs = obs_dict.get(name, np.zeros(1))
                    mask = env.action_mask(name)
                    action_idx, _msg = ag.get_action(raw_obs, np.array(mask, dtype=bool))
                    actions[name] = action_idx
            else:
                # Sleep: action index 0 for every agent
                for name in agent_names:
                    actions[name] = 0

            obs_dict, rew_dict, term_dict, trunc_dict, _info = env.step(actions)
            step_reward = sum(rew_dict.values())
            ep_reward += step_reward
            cumulative_reward += step_reward

            # --- Access internal state ---
            state = get_state(env)
            phase = None
            if state is not None:
                try:
                    phase = int(state.mission_phase)
                except Exception:
                    phase = None

            # steps table
            step_rows_steps.append((episode_id, step, phase, cumulative_reward))

            # zone_rewards table (estimated split)
            step_rows_zone.extend(collect_zone_rewards(episode_id, step, step_reward))

            if state is not None:
                # red_penetration
                step_rows_red.extend(collect_red_penetration(state, episode_id, step))
                # host_alerts
                step_rows_alerts.extend(collect_host_alerts(state, episode_id, step))
                # service_status
                step_rows_services.extend(collect_service_status(state, episode_id, step))
                # blocks_status
                step_rows_blocks.extend(collect_blocks_status(state, episode_id, step))

            final_step = step

            if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in agent_names):
                break

        # Bulk-insert all collected step data
        conn.executemany(
            "INSERT INTO steps (episode_id, step, phase, total_reward) VALUES (?,?,?,?)",
            step_rows_steps,
        )
        conn.executemany(
            "INSERT INTO zone_rewards (episode_id, step, zone, lwf_reward, asf_reward, ria_reward) "
            "VALUES (?,?,?,?,?,?)",
            step_rows_zone,
        )
        conn.executemany(
            "INSERT INTO red_penetration (episode_id, step, agent_name, hostname, subnet, is_privileged) "
            "VALUES (?,?,?,?,?,?)",
            step_rows_red,
        )
        conn.executemany(
            "INSERT INTO host_alerts "
            "(episode_id, step, hostname, subnet, has_process_flag, has_connection_flag, "
            "n_process_events, n_connection_events) VALUES (?,?,?,?,?,?,?,?)",
            step_rows_alerts,
        )
        conn.executemany(
            "INSERT INTO service_status (episode_id, step, hostname, service_name, reliability_pct, is_active) "
            "VALUES (?,?,?,?,?,?)",
            step_rows_services,
        )
        conn.executemany(
            "INSERT INTO blocks_status (episode_id, step, to_subnet, from_subnets_json) VALUES (?,?,?,?)",
            step_rows_blocks,
        )

        # Update episode totals
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
        description="Log CybORG CC4 telemetry to SQLite (data/cc4_telemetry.db)"
    )
    parser.add_argument(
        "--agent",
        choices=["sleep", "heuristic"],
        default="sleep",
        help="Blue agent type: 'sleep' (SleepAgent, action=0) or 'heuristic' (EnterpriseHeuristicAgent)",
    )
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes (default 30)")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode (default 500)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    args = parser.parse_args()

    print(
        f"\nTelemetry logger — agent={args.agent}  episodes={args.episodes}  "
        f"steps={args.steps}  seed={args.seed}"
    )
    print(f"Writing to: {DB_PATH}\n")

    conn = open_db(DB_PATH)
    try:
        run_episodes(
            agent_type=args.agent,
            n_episodes=args.episodes,
            max_steps=args.steps,
            seed=args.seed,
            conn=conn,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
