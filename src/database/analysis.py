"""Data analysis utilities for CybORG training runs stored in SQLite."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class RunAnalyzer:
    """Query and visualise training run data from the MetricsDB."""

    def __init__(self, db_path: str = "data/training_runs.db"):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get_runs(self):
        """Return a list of dicts describing all runs."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM runs ORDER BY start_time DESC").fetchall()
        return [dict(r) for r in rows]

    def get_episode_stats(self, run_id: str) -> list[dict]:
        """Return episode-level stats for a run."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT episode, total_reward, episode_length, win FROM episodes WHERE run_id=? ORDER BY episode",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def plot_learning_curve(self, run_id: str, save_path: str | None = None) -> None:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib required: pip install matplotlib")
            return

        episodes = self.get_episode_stats(run_id)
        if not episodes:
            print(f"No episode data for run {run_id}")
            return

        rewards = [e["total_reward"] for e in episodes]
        xs = list(range(len(rewards)))
        window = min(50, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, rewards, alpha=0.3, label="raw reward")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"rolling mean ({window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"Learning Curve — run {run_id}")
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_win_rate(self, run_id: str, window: int = 100, save_path: str | None = None) -> None:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib required: pip install matplotlib")
            return

        episodes = self.get_episode_stats(run_id)
        if not episodes:
            return

        wins = [float(e["win"]) for e in episodes]
        roll = np.convolve(wins, np.ones(window) / window, mode="valid")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(window - 1, len(wins)), roll)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Win Rate")
        ax.set_title(f"Win Rate (rolling {window}) — run {run_id}")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close(fig)

    def compare_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        """Return a comparison table of final performance across runs."""
        results = []
        for run_id in run_ids:
            eps = self.get_episode_stats(run_id)
            if not eps:
                continue
            last = eps[-100:] if len(eps) >= 100 else eps
            results.append({
                "run_id": run_id,
                "episodes": len(eps),
                "final_mean_reward": sum(e["total_reward"] for e in last) / len(last),
                "final_win_rate": sum(e["win"] for e in last) / len(last),
                "mean_ep_length": sum(e["episode_length"] for e in last) / len(last),
            })
        return results

    def get_action_distribution(self, run_id: str, agent: str = None) -> list[dict]:
        """Return action counts from action_logs ordered by frequency, limit 20."""
        try:
            with self._conn() as conn:
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
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_all_run_summaries(self) -> list[dict]:
        """Return a summary row per run joining runs + episodes."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT r.run_id, r.algo, r.status, r.start_time, r.end_time,"
                " COUNT(e.id) AS episode_count,"
                " AVG(e.total_reward) AS mean_reward,"
                " SUM(CASE WHEN e.win=1 THEN 1 ELSE 0 END)*1.0/NULLIF(COUNT(e.id),0) AS win_rate"
                " FROM runs r LEFT JOIN episodes e ON r.run_id=e.run_id"
                " GROUP BY r.run_id ORDER BY r.start_time DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_observations(self, run_id: str, episode: int = None, limit: int = 100) -> list[dict]:
        """Return stored observation snapshots for a run, optionally filtered by episode."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                if episode is not None:
                    rows = conn.execute(
                        "SELECT * FROM observations WHERE run_id=? AND episode=?"
                        " ORDER BY step LIMIT ?",
                        (run_id, episode, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM observations WHERE run_id=? ORDER BY episode, step LIMIT ?",
                        (run_id, limit),
                    ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def export_to_csv(self, run_id: str, output_dir: str = "data/exports/") -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(output_dir) / f"{run_id}_episodes.csv"
        eps = self.get_episode_stats(run_id)
        if not eps:
            print(f"No data for run {run_id}")
            return ""

        header = ",".join(eps[0].keys())
        lines = [header] + [",".join(str(v) for v in e.values()) for e in eps]
        out_path.write_text("\n".join(lines))
        print(f"Exported {len(eps)} episodes to {out_path}")
        return str(out_path)


if __name__ == "__main__":
    analyzer = RunAnalyzer()
    runs = analyzer.get_runs()
    if not runs:
        print("No runs in database yet.")
    else:
        print(f"\n{'run_id':>12} {'algo':>6} {'envs':>5} {'timesteps':>12} {'status':>12}")
        print("-" * 55)
        for r in runs:
            print(f"{r['run_id']:>12} {r.get('algo','?'):>6} {r.get('num_envs',0):>5} {r.get('total_timesteps',0):>12,} {r.get('status','?'):>12}")
