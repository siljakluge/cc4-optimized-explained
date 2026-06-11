"""Real-time training metrics collection."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path


class TrainingMonitor:
    """Tracks per-step metrics with rolling stats and JSONL logging."""

    def __init__(self, run_id: str, log_dir: str = "data/logs/", update_interval: int = 100):
        self.run_id = run_id
        self.log_dir = log_dir
        self.update_interval = update_interval

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._log_path = Path(log_dir) / f"{run_id}.jsonl"

        self._rewards: deque[float] = deque(maxlen=100)
        self._lengths: deque[int] = deque(maxlen=100)
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._total_steps = 0
        self._t_start = time.perf_counter()
        self._last_log_step = 0
        self._last_reward = 0.0
        self._last_ep_length = 0

    def update(self, step: int, reward: float, done: bool, info: dict) -> None:
        self._ep_reward += reward
        self._ep_steps += 1
        self._total_steps = step

        if done:
            self._rewards.append(self._ep_reward)
            self._lengths.append(self._ep_steps)
            self._last_reward = self._ep_reward
            self._last_ep_length = self._ep_steps
            self._ep_reward = 0.0
            self._ep_steps = 0

        if (step - self._last_log_step) >= self.update_interval:
            self.log_to_file()
            self._last_log_step = step

    def get_stats(self) -> dict:
        elapsed = time.perf_counter() - self._t_start
        sps = self._total_steps / elapsed if elapsed > 0 else 0.0
        mean_rew = sum(self._rewards) / len(self._rewards) if self._rewards else 0.0
        win_rate = sum(1 for r in self._rewards if r > 0) / len(self._rewards) if self._rewards else 0.0

        return {
            "steps_per_second": round(sps, 1),
            "mean_reward_100ep": round(mean_rew, 3),
            "win_rate_100ep": round(win_rate, 3),
            "elapsed_time": round(elapsed, 1),
            "total_steps": self._total_steps,
            "last_reward": self._last_reward,
            "last_ep_length": self._last_ep_length,
        }

    def log_to_file(self) -> None:
        entry = {"timestamp": time.time(), "run_id": self.run_id, "step": self._total_steps, **self.get_stats()}
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


class SwarmMonitor:
    """Tracks status of all agents in a RuFlo swarm."""

    def __init__(self, swarm_id: str, log_dir: str = "data/logs/"):
        self.swarm_id = swarm_id
        self._agents: dict[str, dict] = {}
        self._log_path = Path(log_dir) / "swarm.jsonl"
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def register_agent(self, agent_id: str, agent_type: str) -> None:
        self._agents[agent_id] = {
            "type": agent_type,
            "status": "registered",
            "progress": 0.0,
            "metrics": {},
            "registered_at": time.time(),
        }
        self.log_event("register", f"Agent {agent_id} ({agent_type}) registered")

    def update_agent(self, agent_id: str, status: str, progress: float, metrics: dict | None = None) -> None:
        if agent_id in self._agents:
            self._agents[agent_id].update({
                "status": status,
                "progress": progress,
                "metrics": metrics or {},
                "last_update": time.time(),
            })

    def get_swarm_status(self) -> dict:
        statuses = [a["status"] for a in self._agents.values()]
        return {
            "swarm_id": self.swarm_id,
            "total_agents": len(self._agents),
            "running": statuses.count("running"),
            "completed": statuses.count("completed"),
            "failed": statuses.count("failed"),
            "agents": self._agents,
        }

    def print_status_table(self) -> None:
        status = self.get_swarm_status()
        print(f"\n=== Swarm {self.swarm_id} ===")
        print(f"{'agent_id':>20} {'type':>15} {'status':>12} {'progress':>10}")
        print("-" * 62)
        for aid, info in status["agents"].items():
            print(f"{aid:>20} {info['type']:>15} {info['status']:>12} {info['progress']:>9.1%}")
        print(f"\nRunning: {status['running']}  Completed: {status['completed']}  Failed: {status['failed']}")

    def log_event(self, event_type: str, message: str) -> None:
        entry = {"timestamp": time.time(), "swarm_id": self.swarm_id, "event": event_type, "message": message}
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
