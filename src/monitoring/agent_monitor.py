"""Agent monitor dashboard — tracks RuFlo swarm agent status.

Integrates with Claude-Code-Agent-Monitor patterns.
Writes JSON dashboard for external monitoring tools to consume.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class AgentStatus:
    agent_id: str
    agent_type: str
    status: str = "registered"
    task: str = ""
    progress: float = 0.0
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)
    result: Any = None


class AgentMonitorDashboard:
    """Tracks all swarm agents and writes a live JSON dashboard."""

    def __init__(self, swarm_id: str, output_file: str = "data/logs/agent_monitor.json"):
        self.swarm_id = swarm_id
        self.output_file = output_file
        self._agents: dict[str, AgentStatus] = {}
        self._lock = threading.Lock()
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    def register(self, agent_id: str, agent_type: str, task: str = "") -> None:
        with self._lock:
            self._agents[agent_id] = AgentStatus(
                agent_id=agent_id, agent_type=agent_type, task=task
            )
        self.save()

    def update(
        self,
        agent_id: str,
        status: str,
        progress: float,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if agent_id not in self._agents:
                return
            agent = self._agents[agent_id]
            agent.status = status
            agent.progress = progress
            agent.last_update = time.time()
            if result is not None:
                agent.result = result
            if error:
                agent.errors.append(error)
        self.save()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "swarm_id": self.swarm_id,
                "timestamp": time.time(),
                "summary": self.get_summary(),
                "agents": {aid: asdict(a) for aid, a in self._agents.items()},
            }

    def save(self) -> None:
        """Atomic write via temp file + rename."""
        data = json.dumps(self.to_dict(), indent=2, default=str)
        dir_ = os.path.dirname(self.output_file) or "."
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
            f.write(data)
            tmp_path = f.name
        os.replace(tmp_path, self.output_file)

    def print_dashboard(self) -> None:
        summary = self.get_summary()
        print(f"\n=== Agent Monitor — Swarm {self.swarm_id} ===")
        print(f"{'agent_id':>25} {'type':>15} {'status':>12} {'progress':>10} {'task'}")
        print("-" * 80)
        with self._lock:
            for agent in self._agents.values():
                print(
                    f"{agent.agent_id:>25} {agent.agent_type:>15} "
                    f"{agent.status:>12} {agent.progress:>9.1%}  {agent.task[:30]}"
                )
        print(
            f"\nTotal: {summary['total']}  Running: {summary['running']}  "
            f"Completed: {summary['completed']}  Failed: {summary['failed']}  "
            f"Progress: {summary['progress_pct']:.1%}"
        )

    def get_summary(self) -> dict:
        with self._lock:
            statuses = [a.status for a in self._agents.values()]
            progresses = [a.progress for a in self._agents.values()]
        total = len(statuses)
        return {
            "total": total,
            "running": statuses.count("running"),
            "completed": statuses.count("completed"),
            "failed": statuses.count("failed"),
            "registered": statuses.count("registered"),
            "progress_pct": sum(progresses) / total if total else 0.0,
        }


class monitor_training_run:
    """Context manager: registers training run as a monitored agent and updates progress."""

    def __init__(self, run_id: str, trainer, update_interval_steps: int = 10_000):
        self.run_id = run_id
        self.trainer = trainer
        self.update_interval = update_interval_steps
        self._dashboard = AgentMonitorDashboard(
            swarm_id="swarm-1775416296001-dw8i37"
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def __enter__(self):
        self._dashboard.register(self.run_id, "training", task=f"DRL training run {self.run_id}")
        self._dashboard.update(self.run_id, "running", 0.0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if exc_type:
            self._dashboard.update(self.run_id, "failed", 0.0, error=str(exc_val))
        else:
            self._dashboard.update(self.run_id, "completed", 1.0)
