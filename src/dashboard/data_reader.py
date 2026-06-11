"""Reads training JSONL logs and agent_monitor.json for the dashboard."""
import json, os, time
from pathlib import Path
from collections import deque
from typing import Optional

LOG_DIR = Path("data/logs")

class DashboardDataReader:
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self._episode_cache: dict[str, deque] = {}  # run_id -> deque of episode dicts (maxlen=500)
        self._step_cache: dict[str, deque] = {}     # run_id -> deque of step dicts (maxlen=200)
        self._file_positions: dict[str, int] = {}   # path -> last read byte position

    def get_runs(self) -> list[str]:
        """Return list of run IDs that have log files."""
        if not self.log_dir.exists():
            return []
        return [f.stem for f in self.log_dir.glob("*.jsonl")
                if f.stem not in ("swarm", "hooks", "swarm_events")]

    def read_latest_metrics(self, run_id: str) -> Optional[dict]:
        """Return the most recent metrics entry for a run."""
        path = self.log_dir / f"{run_id}.jsonl"
        if not path.exists():
            return None
        lines = path.read_text().strip().splitlines()
        if not lines:
            return None
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None

    def read_all_metrics(self, run_id: str, max_entries: int = 500) -> list[dict]:
        """Return all metric entries for a run (for charting)."""
        path = self.log_dir / f"{run_id}.jsonl"
        if not path.exists():
            return []
        entries = []
        for line in path.read_text().strip().splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return entries[-max_entries:]

    def read_agent_monitor(self) -> dict:
        """Return the agent monitor dashboard JSON."""
        path = self.log_dir / "agent_monitor.json"
        if not path.exists():
            return {"agents": {}, "summary": {}}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"agents": {}, "summary": {}}

    def read_swarm_events(self, max_events: int = 50) -> list[dict]:
        """Return recent swarm events from swarm.jsonl."""
        path = self.log_dir / "swarm.jsonl"
        if not path.exists():
            return []
        events = []
        for line in path.read_text().strip().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return events[-max_events:]

    def get_db_runs(self) -> list[dict]:
        """Return all runs from SQLite training_runs.db."""
        import sqlite3
        db_path = Path("data/training_runs.db")
        if not db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT run_id, algo, num_envs, seed, total_timesteps, start_time, end_time, status "
                "FROM runs ORDER BY start_time DESC"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_db_episodes(self, run_id: str, limit: int = 200) -> list[dict]:
        """Return episode data for a run from SQLite."""
        import sqlite3
        db_path = Path("data/training_runs.db")
        if not db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT episode, agent, total_reward, episode_length, win, timestamp "
                "FROM episodes WHERE run_id=? ORDER BY episode DESC LIMIT ?",
                (run_id, limit),
            ).fetchall()
            conn.close()
            return list(reversed([dict(r) for r in rows]))
        except Exception:
            return []

    def get_dashboard_state(self) -> dict:
        """Assemble full dashboard state for WebSocket push."""
        runs = self.get_runs()
        active_run = runs[0] if runs else None

        metrics_history = []
        latest_metrics = {}
        if active_run:
            metrics_history = self.read_all_metrics(active_run)
            latest_metrics = self.read_latest_metrics(active_run) or {}

        return {
            "timestamp": time.time(),
            "active_run": active_run,
            "runs": runs,
            "latest_metrics": latest_metrics,
            "metrics_history": metrics_history,
            "agent_monitor": self.read_agent_monitor(),
            "swarm_events": self.read_swarm_events(20),
            "runner_state": {},
        }
