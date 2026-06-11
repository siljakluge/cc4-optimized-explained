"""Deep telemetry collector for CybORG CC4 environment state.

Captures full environment state at every step for 3D visualization dashboard.
Uses WAL mode, thread-local connections, and batch inserts for performance.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

_J = json.dumps  # shorthand


class TelemetryDB:
    """Thread-safe SQLite store for full CC4 environment telemetry."""

    def __init__(self, db_path: str = "data/cc4_graph_telemetry.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _ex(self, sql: str, params: tuple = ()) -> None:
        """Execute + commit (single row insert / update helper)."""
        self._conn().execute(sql, params)
        self._conn().commit()

    def _exm(self, sql: str, rows: list[tuple]) -> None:
        """Executemany + commit (batch insert helper)."""
        self._conn().executemany(sql, rows)
        self._conn().commit()

    def _init_schema(self) -> None:
        self._conn().executescript(_SCHEMA_SQL)
        self._conn().commit()

    # -- Episode lifecycle ------------------------------------------------- #

    def start_episode(self, episode_id: str, seed: int, agent_type: str,
                      red_agent_type: str, max_steps: int) -> None:
        self._ex(
            "INSERT INTO episodes (episode_id,seed,agent_type,red_agent_type,"
            "max_steps,start_time,status) VALUES (?,?,?,?,?,?,?)",
            (episode_id, seed, agent_type, red_agent_type, max_steps, time.time(), "running"))

    def end_episode(self, episode_id: str, total_reward: float,
                    status: str = "completed") -> None:
        self._ex("UPDATE episodes SET end_time=?,total_reward=?,status=? WHERE episode_id=?",
                 (time.time(), total_reward, status, episode_id))

    # -- Topology (once per episode) --------------------------------------- #

    def log_topology(self, episode_id: str, topology_list: list[dict]) -> None:
        self._exm(
            "INSERT INTO network_topology (episode_id,subnet_name,host_name,"
            "host_type,ip_address,os_type,services_json) VALUES (?,?,?,?,?,?,?)",
            [(episode_id, t["subnet_name"], t["host_name"],
              t.get("host_type", "unknown"), t.get("ip_address"),
              t.get("os_type"), _J(t.get("services", []))) for t in topology_list])

    # -- Per-step logging -------------------------------------------------- #

    def log_step(self, episode_id: str, step: int, mission_phase: int,
                 reward: float, total_reward: float) -> None:
        self._ex(
            "INSERT INTO step_snapshots (episode_id,step,mission_phase,"
            "total_reward_so_far,step_reward,timestamp) VALUES (?,?,?,?,?,?)",
            (episode_id, step, mission_phase, total_reward, reward, time.time()))

    def log_action(self, episode_id: str, step: int, agent_name: str, team: str,
                   action_type: str, action_label: str, target_host: Optional[str],
                   target_subnet: Optional[str], action_idx: Optional[int],
                   success: bool, duration: float = 0.0,
                   reasoning: Optional[dict] = None) -> None:
        self._ex(
            "INSERT INTO agent_actions (episode_id,step,agent_name,agent_team,"
            "action_type,action_label,target_host,target_subnet,action_idx,"
            "success,duration,reasoning_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (episode_id, step, agent_name, team, action_type, action_label,
             target_host, target_subnet, action_idx, int(success), duration,
             _J(reasoning) if reasoning else None))

    def log_host_state(self, episode_id: str, step: int, host_name: str,
                       subnet_name: str, state_dict: dict[str, Any]) -> None:
        self._ex(
            "INSERT INTO host_states (episode_id,step,host_name,subnet_name,"
            "compromised_level,has_red_session,has_blue_session,num_processes,"
            "num_connections,has_malware,decoy_count,is_restoring,"
            "is_being_removed,service_reliability_pct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _host_row(episode_id, step, state_dict, host_name, subnet_name))

    def log_session(self, episode_id: str, step: int, session_id: int,
                    agent_name: str, host_name: str, session_type: str,
                    privilege: str, is_active: bool,
                    parent_id: Optional[int] = None) -> None:
        self._ex(
            "INSERT INTO sessions (episode_id,step,session_id,agent_name,"
            "host_name,session_type,privilege,is_active,parent_session_id) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (episode_id, step, session_id, agent_name, host_name,
             session_type, privilege, int(is_active), parent_id))

    def log_traffic(self, episode_id: str, step: int, source_subnet: str,
                    dest_subnet: str, is_blocked: bool,
                    should_be_blocked: bool) -> None:
        self._ex(
            "INSERT INTO network_traffic (episode_id,step,source_subnet,"
            "dest_subnet,is_blocked,should_be_blocked) VALUES (?,?,?,?,?,?)",
            (episode_id, step, source_subnet, dest_subnet,
             int(is_blocked), int(should_be_blocked)))

    def log_belief(self, episode_id: str, step: int, agent_name: str,
                   belief_type: str, target_host: str, confidence: float,
                   evidence: Optional[dict] = None) -> None:
        self._ex(
            "INSERT INTO agent_beliefs (episode_id,step,agent_name,belief_type,"
            "target_host,confidence,evidence_json) VALUES (?,?,?,?,?,?,?)",
            (episode_id, step, agent_name, belief_type, target_host,
             confidence, _J(evidence) if evidence else None))

    def log_reward_breakdown(self, episode_id: str, step: int, agent_name: str,
                             component: str, value: float) -> None:
        self._ex(
            "INSERT INTO reward_breakdown (episode_id,step,agent_name,component,value) "
            "VALUES (?,?,?,?,?)",
            (episode_id, step, agent_name, component, value))

    def log_message(self, episode_id: str, step: int, sender: str,
                    message_bits: int, decoded_meaning: Optional[dict] = None) -> None:
        self._ex(
            "INSERT INTO messages (episode_id,step,sender_agent,message_bits,"
            "decoded_meaning_json) VALUES (?,?,?,?,?)",
            (episode_id, step, sender, message_bits,
             _J(decoded_meaning) if decoded_meaning else None))

    # -- Batch inserts ----------------------------------------------------- #

    def log_host_states_batch(self, episode_id: str, step: int,
                              states: list[dict[str, Any]]) -> None:
        """Insert multiple host states in one transaction."""
        self._exm(
            "INSERT INTO host_states (episode_id,step,host_name,subnet_name,"
            "compromised_level,has_red_session,has_blue_session,num_processes,"
            "num_connections,has_malware,decoy_count,is_restoring,"
            "is_being_removed,service_reliability_pct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [_host_row(episode_id, step, s) for s in states])

    def log_traffic_batch(self, episode_id: str, step: int,
                          traffic: list[dict]) -> None:
        """Insert multiple traffic entries in one transaction."""
        self._exm(
            "INSERT INTO network_traffic (episode_id,step,source_subnet,"
            "dest_subnet,is_blocked,should_be_blocked) VALUES (?,?,?,?,?,?)",
            [(episode_id, step, t["source_subnet"], t["dest_subnet"],
              int(t["is_blocked"]), int(t["should_be_blocked"])) for t in traffic])

    # -- Query methods ----------------------------------------------------- #

    def get_snapshot(self, episode_id: str, step: int) -> dict[str, Any]:
        """Return full environment state at a given step."""
        conn = self._conn()
        _q = lambda tbl: [dict(r) for r in conn.execute(
            f"SELECT * FROM {tbl} WHERE episode_id=? AND step=?",
            (episode_id, step)).fetchall()]
        snap = conn.execute(
            "SELECT * FROM step_snapshots WHERE episode_id=? AND step=?",
            (episode_id, step)).fetchone()
        return {
            "step_info": dict(snap) if snap else None,
            "actions": _q("agent_actions"),
            "host_states": _q("host_states"),
            "sessions": _q("sessions"),
            "network_traffic": _q("network_traffic"),
            "beliefs": _q("agent_beliefs"),
            "reward_breakdown": _q("reward_breakdown"),
            "messages": _q("messages"),
        }

    def get_timeline(self, episode_id: str) -> list[dict[str, Any]]:
        """Return all steps with summary info for an episode."""
        rows = self._conn().execute(
            "SELECT s.*, "
            "(SELECT COUNT(*) FROM agent_actions a "
            " WHERE a.episode_id=s.episode_id AND a.step=s.step) AS action_count, "
            "(SELECT COUNT(*) FROM host_states h "
            " WHERE h.episode_id=s.episode_id AND h.step=s.step "
            " AND h.compromised_level != 'none') AS compromised_hosts "
            "FROM step_snapshots s WHERE s.episode_id=? ORDER BY s.step",
            (episode_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_agent_history(self, episode_id: str,
                          agent_name: str) -> list[dict[str, Any]]:
        """Return all actions and beliefs for one agent across the episode."""
        conn = self._conn()
        actions = [dict(r) for r in conn.execute(
            "SELECT 'action' AS record_type,step,action_type,action_label,"
            "target_host,target_subnet,success,reasoning_json "
            "FROM agent_actions WHERE episode_id=? AND agent_name=? ORDER BY step",
            (episode_id, agent_name)).fetchall()]
        beliefs = [dict(r) for r in conn.execute(
            "SELECT 'belief' AS record_type,step,belief_type,target_host,"
            "confidence,evidence_json "
            "FROM agent_beliefs WHERE episode_id=? AND agent_name=? ORDER BY step",
            (episode_id, agent_name)).fetchall()]
        combined = actions + beliefs
        combined.sort(key=lambda x: x["step"])
        return combined

    # -- Cleanup ----------------------------------------------------------- #

    def close(self) -> None:
        if conn := getattr(self._local, "conn", None):
            conn.close()
            self._local.conn = None


def _host_row(episode_id: str, step: int, s: dict[str, Any],
              host_name: Optional[str] = None,
              subnet_name: Optional[str] = None) -> tuple:
    """Build a host_states row tuple from a state dict."""
    return (
        episode_id, step,
        host_name or s["host_name"], subnet_name or s["subnet_name"],
        s.get("compromised_level", "none"),
        int(s.get("has_red_session", False)),
        int(s.get("has_blue_session", False)),
        s.get("num_processes", 0),
        s.get("num_connections", 0),
        int(s.get("has_malware", False)),
        s.get("decoy_count", 0),
        int(s.get("is_restoring", False)),
        int(s.get("is_being_removed", False)),
        s.get("service_reliability_pct", 100.0))


# -- Schema SQL ------------------------------------------------------------ #

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY, seed INTEGER, agent_type TEXT NOT NULL,
    red_agent_type TEXT, max_steps INTEGER, start_time REAL, end_time REAL,
    total_reward REAL, status TEXT DEFAULT 'running');
CREATE TABLE IF NOT EXISTS network_topology (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
    subnet_name TEXT NOT NULL, host_name TEXT NOT NULL,
    host_type TEXT DEFAULT 'unknown', ip_address TEXT, os_type TEXT,
    services_json TEXT, FOREIGN KEY (episode_id) REFERENCES episodes(episode_id));
CREATE INDEX IF NOT EXISTS idx_topo_episode ON network_topology(episode_id);
CREATE TABLE IF NOT EXISTS step_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
    step INTEGER NOT NULL, mission_phase INTEGER, total_reward_so_far REAL,
    step_reward REAL, timestamp REAL,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id));
CREATE UNIQUE INDEX IF NOT EXISTS idx_snap_ep_step ON step_snapshots(episode_id, step);
CREATE TABLE IF NOT EXISTS agent_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
    step INTEGER NOT NULL, agent_name TEXT NOT NULL, agent_team TEXT NOT NULL,
    action_type TEXT NOT NULL, action_label TEXT, target_host TEXT,
    target_subnet TEXT, action_idx INTEGER, success INTEGER,
    duration REAL DEFAULT 0.0, reasoning_json TEXT,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id));
CREATE INDEX IF NOT EXISTS idx_actions_ep_step ON agent_actions(episode_id, step);
CREATE INDEX IF NOT EXISTS idx_actions_agent ON agent_actions(episode_id, agent_name);
CREATE TABLE IF NOT EXISTS host_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
    step INTEGER NOT NULL, host_name TEXT NOT NULL, subnet_name TEXT NOT NULL,
    compromised_level TEXT DEFAULT 'none', has_red_session INTEGER DEFAULT 0,
    has_blue_session INTEGER DEFAULT 0, num_processes INTEGER DEFAULT 0,
    num_connections INTEGER DEFAULT 0, has_malware INTEGER DEFAULT 0,
    decoy_count INTEGER DEFAULT 0, is_restoring INTEGER DEFAULT 0,
    is_being_removed INTEGER DEFAULT 0, service_reliability_pct REAL DEFAULT 100.0,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id));
CREATE INDEX IF NOT EXISTS idx_hosts_ep_step ON host_states(episode_id, step);
CREATE INDEX IF NOT EXISTS idx_hosts_name ON host_states(episode_id, host_name);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
    step INTEGER NOT NULL, session_id INTEGER NOT NULL, agent_name TEXT NOT NULL,
    host_name TEXT NOT NULL, session_type TEXT, privilege TEXT,
    is_active INTEGER DEFAULT 1, parent_session_id INTEGER,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id));
CREATE INDEX IF NOT EXISTS idx_sess_ep_step ON sessions(episode_id, step);
CREATE TABLE IF NOT EXISTS network_traffic (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
    step INTEGER NOT NULL, source_subnet TEXT NOT NULL, dest_subnet TEXT NOT NULL,
    is_blocked INTEGER DEFAULT 0, should_be_blocked INTEGER DEFAULT 0,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id));
CREATE INDEX IF NOT EXISTS idx_traffic_ep_step ON network_traffic(episode_id, step);
CREATE TABLE IF NOT EXISTS agent_beliefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
    step INTEGER NOT NULL, agent_name TEXT NOT NULL, belief_type TEXT NOT NULL,
    target_host TEXT, confidence REAL DEFAULT 1.0, evidence_json TEXT,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id));
CREATE INDEX IF NOT EXISTS idx_beliefs_ep_step ON agent_beliefs(episode_id, step);
CREATE INDEX IF NOT EXISTS idx_beliefs_agent ON agent_beliefs(episode_id, agent_name);
CREATE TABLE IF NOT EXISTS reward_breakdown (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
    step INTEGER NOT NULL, agent_name TEXT NOT NULL, component TEXT NOT NULL,
    value REAL, FOREIGN KEY (episode_id) REFERENCES episodes(episode_id));
CREATE INDEX IF NOT EXISTS idx_rewards_ep_step ON reward_breakdown(episode_id, step);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
    step INTEGER NOT NULL, sender_agent TEXT NOT NULL, message_bits INTEGER,
    decoded_meaning_json TEXT,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id));
CREATE INDEX IF NOT EXISTS idx_msgs_ep_step ON messages(episode_id, step);
"""
