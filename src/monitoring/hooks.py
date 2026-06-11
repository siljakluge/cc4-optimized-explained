"""RuFlo-compatible hooks for the CybORG training pipeline."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def pre_training_hook(config: dict) -> None:
    """Validate config and create required directories before training starts."""
    required_keys = ["training", "environment", "database", "monitoring"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    for path_key in ["database.path", "monitoring.log_dir"]:
        section, key = path_key.split(".")
        p = config.get(section, {}).get(key)
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    Path("data/models").mkdir(parents=True, exist_ok=True)
    logger.info("pre_training_hook: directories ready, config validated")
    _append_event("pre_training", {"config_keys": list(config.keys())})


def post_episode_hook(episode: int, reward: float, win: bool, info: dict) -> None:
    """Log episode summary."""
    if episode % 100 == 0:
        logger.info("Episode %d: reward=%.3f win=%s", episode, reward, win)
    _append_event("post_episode", {"episode": episode, "reward": reward, "win": win})


def post_training_hook(run_id: str, final_stats: dict) -> None:
    """Log final stats after training completes."""
    logger.info("Training complete — run_id=%s stats=%s", run_id, final_stats)
    _append_event("post_training", {"run_id": run_id, "stats": final_stats})


def on_checkpoint_hook(step: int, model_path: str) -> None:
    """Log checkpoint save."""
    logger.info("Checkpoint saved at step %d → %s", step, model_path)
    _append_event("checkpoint", {"step": step, "model_path": model_path})


def _append_event(event_type: str, data: dict) -> None:
    log_path = Path("data/logs/hooks.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": time.time(), "event": event_type, **data}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
