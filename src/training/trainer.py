"""DRL training harness for CybORG using stable-baselines3 (SB3).

Uses MaskablePPO from sb3_contrib to respect the CybORG action mask.
Logs metrics to SQLite via src.database.collector.MetricsDB.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.database.collector import MetricsDB
from src.monitoring.metrics import TrainingMonitor


def _make_single_env(agent_name: str, max_steps: int, seed: int, pad_spaces: bool = True):
    """Create a single-agent gymnasium.Env compatible with SB3.

    Uses SingleCybORGEnv which wraps the multi-agent PettingZoo env into
    a standard single-agent gymnasium interface.
    """
    from src.envs.vectorized_env import SingleCybORGEnv
    return SingleCybORGEnv(agent_name=agent_name, max_steps=max_steps, seed=seed)


class CybORGCallback:
    """SB3 BaseCallback subclass for metrics logging and periodic checkpointing."""

    def __init__(self, run_id: str, db: MetricsDB, monitor: TrainingMonitor, save_dir: str, save_freq: int = 100_000):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Inner(BaseCallback):
            def __init__(inner_self):
                super().__init__(verbose=0)
                inner_self._episode = 0
                inner_self._last_save = 0

            def _on_step(inner_self) -> bool:
                rewards = inner_self.locals.get("rewards", [0.0])
                dones = inner_self.locals.get("dones", [False])
                infos = inner_self.locals.get("infos", [{}])

                reward = float(rewards[0]) if len(rewards) > 0 else 0.0
                done = bool(dones[0]) if len(dones) > 0 else False
                info = infos[0] if len(infos) > 0 else {}

                monitor.update(inner_self.num_timesteps, reward, done, info)

                if done:
                    stats = monitor.get_stats()
                    win = reward > 0
                    db.log_episode(
                        run_id, inner_self._episode,
                        "blue", stats.get("last_reward", reward),
                        stats.get("last_ep_length", 0), win,
                    )
                    inner_self._episode += 1

                if (inner_self.num_timesteps - inner_self._last_save) >= save_freq:
                    ckpt = os.path.join(save_dir, f"checkpoint_{inner_self.num_timesteps}")
                    inner_self.model.save(ckpt)
                    inner_self._last_save = inner_self.num_timesteps

                return True

            def _on_training_end(inner_self) -> None:
                inner_self.model.save(os.path.join(save_dir, "final_model"))

        self._callback = _Inner()

    @property
    def sb3_callback(self):
        """Return the SB3-compatible BaseCallback instance."""
        return self._callback

    def on_training_end(self, model=None) -> None:
        # Kept for backward-compatibility; SB3 will call _on_training_end automatically.
        if model is not None:
            model.save(os.path.join(self._callback.logger.dir if hasattr(self._callback, "logger") and self._callback.logger else ".", "final_model"))


class CybORGTrainer:
    """Train a DRL agent on CybORG using SB3."""

    def __init__(
        self,
        algo: str = "PPO",
        num_envs: int = 4,
        total_timesteps: int = 1_000_000,
        seed: int = 42,
        max_steps: int = 500,
        agent_name: str = "blue_agent_0",
        db_path: str = "data/training_runs.db",
        log_dir: str = "data/logs/",
        model_dir: str = "data/models/",
    ):
        self.algo = algo.upper()
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        self.seed = seed
        self.max_steps = max_steps
        self.agent_name = agent_name
        self.db_path = db_path
        self.log_dir = log_dir
        self.model_dir = model_dir

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    def _make_vec_env(self):
        """Create an SB3-compatible vectorized environment with action masks."""
        try:
            from sb3_contrib import MaskablePPO  # noqa: F401 — just check availability
            from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

            def _env_fn(seed_offset: int):
                def _inner():
                    return _make_single_env(self.agent_name, self.max_steps, self.seed + seed_offset)
                return _inner

            fns = [_env_fn(i) for i in range(self.num_envs)]
            VecEnvCls = SubprocVecEnv if self.num_envs > 1 else DummyVecEnv
            return VecEnvCls(fns)
        except ImportError as exc:
            raise ImportError(
                "Install stable-baselines3 and sb3-contrib: "
                "pip install stable-baselines3 sb3-contrib"
            ) from exc

    def train(self) -> dict[str, Any]:
        """Run training. Returns final evaluation stats."""
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import VecMonitor

        run_id = str(uuid.uuid4())[:8]
        db = MetricsDB(self.db_path)
        db.start_run(run_id, self.algo, self.num_envs, self.seed, self.total_timesteps)
        monitor = TrainingMonitor(run_id, self.log_dir)

        print(f"[CybORGTrainer] run_id={run_id}  algo={self.algo}  num_envs={self.num_envs}  timesteps={self.total_timesteps:,}")

        vec_env = VecMonitor(self._make_vec_env())

        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            seed=self.seed,
            verbose=1,
            tensorboard_log=self.log_dir,
        )

        callback = CybORGCallback(run_id, db, monitor, self.model_dir)
        t0 = time.perf_counter()

        try:
            model.learn(total_timesteps=self.total_timesteps, callback=callback.sb3_callback)
        finally:
            elapsed = time.perf_counter() - t0
            stats = monitor.get_stats()
            db.finish_run(run_id, "completed")
            vec_env.close()
            print(f"[CybORGTrainer] Done in {elapsed:.1f}s — {stats}")

        return stats

    def evaluate(self, model_path: str, n_episodes: int = 100) -> dict[str, Any]:
        """Load a saved model and evaluate it."""
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(model_path)
        env = _make_single_env(self.agent_name, self.max_steps, self.seed)

        rewards, lengths, wins = [], [], []
        for _ in range(n_episodes):
            obs_dict, info_dict = env.reset()
            obs = obs_dict.get(self.agent_name, np.zeros(1))
            total_reward, ep_len, done = 0.0, 0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(
                    {self.agent_name: int(action)}
                )
                obs = obs_dict.get(self.agent_name, obs)
                reward = float(rew_dict.get(self.agent_name, 0.0))
                total_reward += reward
                ep_len += 1
                done = term_dict.get(self.agent_name, False) or trunc_dict.get(self.agent_name, False)
            rewards.append(total_reward)
            lengths.append(ep_len)
            wins.append(int(total_reward > 0))

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "win_rate": float(np.mean(wins)),
            "mean_length": float(np.mean(lengths)),
            "n_episodes": n_episodes,
        }


def run_training(
    algo: str = "PPO",
    num_envs: int = 4,
    total_timesteps: int = 1_000_000,
    seed: int = 42,
) -> None:
    trainer = CybORGTrainer(algo=algo, num_envs=num_envs, total_timesteps=total_timesteps, seed=seed)
    trainer.train()
