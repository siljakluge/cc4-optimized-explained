"""SQLite-backed metrics collector for CybORG training runs.

Uses WAL mode for fast concurrent writes and thread-local connections for safety.
Steps are sampled 1-in-100 to keep the DB size manageable.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


_STEP_SAMPLE_RATE = 100  # log 1 in every N steps


class MetricsDB:
    """Thread-safe SQLite metrics store."""

    def __init__(self, db_path: str = "data/training_runs.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._step_counter: dict[str, int] = {}
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                algo TEXT,
                num_envs INTEGER,
                seed INTEGER,
                total_timesteps INTEGER,
                start_time REAL,
                end_time REAL,
                status TEXT DEFAULT 'running'
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                episode INTEGER,
                agent TEXT,
                total_reward REAL,
                episode_length INTEGER,
                win INTEGER,
                timestamp REAL
            );
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step INTEGER,
                agent TEXT,
                reward REAL,
                action INTEGER,
                done INTEGER,
                timestamp REAL
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id);
            CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id);
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                episode INTEGER,
                step INTEGER,
                agent TEXT,
                obs_json TEXT,
                action INTEGER,
                reward REAL,
                timestamp REAL
            );
            CREATE INDEX IF NOT EXISTS idx_obs_run ON observations(run_id, episode);
            CREATE TABLE IF NOT EXISTS action_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                agent TEXT,
                action_idx INTEGER,
                count INTEGER DEFAULT 1,
                UNIQUE(run_id, agent, action_idx)
            );
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_id: str,
        algo: str,
        num_envs: int,
        seed: int,
        total_timesteps: int,
    ) -> None:
        self._conn().execute(
            "INSERT INTO runs (run_id, algo, num_envs, seed, total_timesteps, start_time) VALUES (?,?,?,?,?,?)",
            (run_id, algo, num_envs, seed, total_timesteps, time.time()),
        )
        self._conn().commit()

    def log_episode(
        self,
        run_id: str,
        episode: int,
        agent: str,
        total_reward: float,
        episode_length: int,
        win: bool,
    ) -> None:
        self._conn().execute(
            "INSERT INTO episodes (run_id, episode, agent, total_reward, episode_length, win, timestamp) VALUES (?,?,?,?,?,?,?)",
            (run_id, episode, agent, total_reward, episode_length, int(win), time.time()),
        )
        self._conn().commit()

    def log_step(
        self,
        run_id: str,
        step: int,
        agent: str,
        reward: float,
        action: int,
        done: bool,
    ) -> None:
        counter = self._step_counter.get(run_id, 0) + 1
        self._step_counter[run_id] = counter
        if counter % _STEP_SAMPLE_RATE != 0:
            return
        self._conn().execute(
            "INSERT INTO steps (run_id, step, agent, reward, action, done, timestamp) VALUES (?,?,?,?,?,?,?)",
            (run_id, step, agent, reward, action, int(done), time.time()),
        )
        self._conn().commit()

    def log_observation(
        self,
        run_id: str,
        episode: int,
        step: int,
        agent: str,
        obs,
        action: int,
        reward: float,
    ) -> None:
        """Log an observation snapshot every 50 steps to keep DB size manageable."""
        if step % 50 != 0:
            return
        import json as _json
        try:
            obs_list = obs.tolist() if hasattr(obs, "tolist") else list(obs)
            obs_json = _json.dumps(obs_list)
        except Exception:
            obs_json = str(obs)
        conn = self._conn()
        conn.execute(
            "INSERT INTO observations (run_id, episode, step, agent, obs_json, action, reward, timestamp)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (run_id, episode, step, agent, obs_json, action, reward, time.time()),
        )
        conn.execute(
            "INSERT INTO action_logs (run_id, agent, action_idx, count) VALUES (?,?,?,1)"
            " ON CONFLICT(run_id, agent, action_idx) DO UPDATE SET count=count+1",
            (run_id, agent, action),
        )
        conn.commit()

    def get_action_distribution(self, run_id: str, agent: str = None) -> list[dict]:
        """Return action counts ordered by frequency, limit 20."""
        conn = self._conn()
        conn.row_factory = sqlite3.Row
        if agent is not None:
            rows = conn.execute(
                "SELECT action_idx, count FROM action_logs WHERE run_id=? AND agent=?"
                " ORDER BY count DESC LIMIT 20",
                (run_id, agent),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT action_idx, SUM(count) AS count FROM action_logs WHERE run_id=?"
                " GROUP BY action_idx ORDER BY count DESC LIMIT 20",
                (run_id,),
            ).fetchall()
        conn.row_factory = None
        return [dict(r) for r in rows]

    def get_run_summary(self, run_id: str) -> dict:
        """Return aggregated summary for a single run."""
        conn = self._conn()
        conn.row_factory = sqlite3.Row
        run_row = conn.execute(
            "SELECT run_id, algo, start_time, end_time, status FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            return {}
        result = dict(run_row)
        agg = conn.execute(
            "SELECT COUNT(*) AS episodes_count, AVG(total_reward) AS mean_reward,"
            " SUM(CASE WHEN win=1 THEN 1 ELSE 0 END)*1.0/COUNT(*) AS win_rate,"
            " AVG(episode_length) AS mean_length"
            " FROM episodes WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if agg:
            result.update({
                "episodes_count": agg["episodes_count"],
                "mean_reward": agg["mean_reward"],
                "std_reward": None,
                "win_rate": agg["win_rate"],
                "mean_length": agg["mean_length"],
            })
        conn.row_factory = None
        return result

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        self._conn().execute(
            "UPDATE runs SET end_time=?, status=? WHERE run_id=?",
            (time.time(), status, run_id),
        )
        self._conn().commit()

    def close(self) -> None:
        if conn := getattr(self._local, "conn", None):
            conn.close()
            self._local.conn = None
