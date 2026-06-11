"""Vectorized CybORG environment wrappers for fast DRL training.

Key design decisions based on codebase analysis:
- CybORG uses legacy gym internally; wrappers use gymnasium — handled via shim
- BlueFlatWrapper returns MultiDiscrete observations; BlueFixedActionWrapper gives Discrete actions
- SubprocVecEnv pattern: each worker is an independent process with its own CybORG instance
- BlueFlatWrapper.observation_space() is sized for MAX_HOSTS per subnet, but
  observation_change() builds vectors for the actual (randomised) host count,
  which can be smaller.  Observations are zero-padded to the declared max shape
  so SB3 always sees a consistent space.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

import numpy as np
import gymnasium


def _worker(
    conn: mp.connection.Connection,
    agent_name: str,
    max_steps: int,
    seed: int,
) -> None:
    """Worker process: owns one CybORG + BlueFlatWrapper instance."""
    # Import inside worker to avoid pickling issues
    from CybORG import CybORG
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(steps=max_steps)
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapper(env=cyborg)

    # Determine max obs size from the declared space (MAX_HOSTS-based)
    _raw_space = env.observation_space(agent_name)
    _max_size = int(np.prod(_raw_space.shape)) if hasattr(_raw_space, "shape") else 1
    _max_shape = _raw_space.shape if hasattr(_raw_space, "shape") else (1,)

    def _pad(raw) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32).ravel()
        if arr.size < _max_size:
            arr = np.pad(arr, (0, _max_size - arr.size))
        elif arr.size > _max_size:
            arr = arr[:_max_size]
        return arr.reshape(_max_shape)

    while True:
        cmd, data = conn.recv()

        if cmd == "reset":
            obs_dict, info_dict = env.reset()
            obs = _pad(obs_dict.get(agent_name, np.zeros(1)))
            info = info_dict.get(agent_name, {})
            conn.send((obs, info))

        elif cmd == "step":
            action = data
            obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(
                {agent_name: action}
            )
            obs = _pad(obs_dict.get(agent_name, np.zeros(1)))
            reward = float(rew_dict.get(agent_name, 0.0))
            terminated = bool(term_dict.get(agent_name, False))
            truncated = bool(trunc_dict.get(agent_name, False))
            info = info_dict.get(agent_name, {})
            conn.send((obs, reward, terminated, truncated, info))

        elif cmd == "get_spaces":
            obs_space = env.observation_space(agent_name)
            act_space = env.action_space(agent_name)
            conn.send((obs_space, act_space))

        elif cmd == "close":
            conn.close()
            break


class SingleCybORGEnv(gymnasium.Env):
    """Single-instance gymnasium wrapper for CybORG + BlueFlatWrapper.

    The observation_space is updated lazily after the first reset() call
    because BlueFlatWrapper.observation_space() is sized for the maximum
    possible host count while observation_change() builds vectors for the
    actual (randomised) host count.  Using a pre-allocated buffer sized
    from the declared space would cause a ValueError on np.copyto when the
    actual observation is shorter than the declared shape.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_name: str = "blue_agent_0",
        max_steps: int = 500,
        seed: int = 42,
    ):
        super().__init__()
        self.agent_name = agent_name
        self.max_steps = max_steps
        self._seed = seed
        self.step_count = 0

        from CybORG import CybORG
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        sg = EnterpriseScenarioGenerator(steps=max_steps)
        cyborg = CybORG(scenario_generator=sg, seed=seed)
        self._env = BlueFlatWrapper(env=cyborg)

        # Build observation_space as a Box with the declared max shape.
        # BlueFlatWrapper may return a MultiDiscrete; we convert it to Box
        # so SB3 (which expects Box for MlpPolicy) gets a consistent space.
        _raw_obs_space = self._env.observation_space(agent_name)
        _max_shape = _raw_obs_space.shape if hasattr(_raw_obs_space, "shape") else (1,)
        _high = float(_raw_obs_space.nvec.max()) if hasattr(_raw_obs_space, "nvec") else 1.0
        self.observation_space: gymnasium.Space = gymnasium.spaces.Box(
            low=0.0, high=_high, shape=_max_shape, dtype=np.float32
        )
        self._obs_max_size: int = int(np.prod(_max_shape))
        self.action_space: gymnasium.Space = self._env.action_space(agent_name)

    def _pad_obs(self, raw_obs) -> np.ndarray:
        """Zero-pad raw obs to the declared max shape so SB3 sees a fixed-size space."""
        obs = np.asarray(raw_obs, dtype=np.float32).ravel()
        if obs.size < self._obs_max_size:
            obs = np.pad(obs, (0, self._obs_max_size - obs.size))
        elif obs.size > self._obs_max_size:
            obs = obs[:self._obs_max_size]
        return obs.reshape(self.observation_space.shape)

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        obs_dict, info_dict = self._env.reset()
        obs = self._pad_obs(obs_dict.get(self.agent_name, np.zeros(1)))
        return obs, info_dict.get(self.agent_name, {})

    def step(self, action: int):
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(
            {self.agent_name: action}
        )
        obs = self._pad_obs(obs_dict.get(self.agent_name, np.zeros(1)))

        reward = float(rew_dict.get(self.agent_name, 0.0))
        terminated = bool(term_dict.get(self.agent_name, False))
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        return (
            obs,
            reward,
            terminated,
            truncated,
            info_dict.get(self.agent_name, {}),
        )

    def action_masks(self) -> np.ndarray:
        """Return a boolean action mask required by MaskablePPO.

        Delegates to BlueFixedActionWrapper.action_mask() and converts the
        list[bool] to a numpy bool array expected by sb3_contrib.
        """
        mask = self._env.action_mask(self.agent_name)
        return np.array(mask, dtype=bool)

    def render(self):
        pass

    def close(self):
        pass


class CybORGVecEnv:
    """Vectorized CybORG environment using subprocess workers (SubprocVecEnv pattern).

    Runs num_envs independent CybORG instances in separate processes and
    collects results via multiprocessing Pipes.

    The _obs_batch array is lazily initialised on the first reset() call so
    that its shape reflects the actual (post-reset) observation size rather
    than the declared space size, which may be larger due to host-count
    randomisation in the scenario generator.
    """

    def __init__(
        self,
        num_envs: int = 4,
        agent_name: str = "blue_agent_0",
        max_steps: int = 500,
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.agent_name = agent_name
        self.max_steps = max_steps

        ctx = mp.get_context("spawn")
        self._parent_conns: list[mp.connection.Connection] = []
        self._procs: list[mp.Process] = []

        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child_conn, agent_name, max_steps, seed + i),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent doesn't need child end
            self._parent_conns.append(parent_conn)
            self._procs.append(proc)

        # Get spaces from first worker — workers pad to declared max shape
        self._parent_conns[0].send(("get_spaces", None))
        obs_space, act_space = self._parent_conns[0].recv()
        self.observation_space = obs_space
        self.action_space = act_space

        obs_shape = obs_space.shape if hasattr(obs_space, "shape") else (1,)
        self._obs_batch = np.zeros((num_envs, *obs_shape), dtype=np.float32)
        self._rew_batch = np.zeros(num_envs, dtype=np.float32)
        self._term_batch = np.zeros(num_envs, dtype=bool)
        self._trunc_batch = np.zeros(num_envs, dtype=bool)

    def reset(self):
        for conn in self._parent_conns:
            conn.send(("reset", None))

        infos = []
        for i, conn in enumerate(self._parent_conns):
            obs, info = conn.recv()
            self._obs_batch[i] = obs
            infos.append(info)

        return self._obs_batch.copy(), infos

    def step(self, actions: np.ndarray):
        for conn, action in zip(self._parent_conns, actions):
            conn.send(("step", int(action)))

        infos = []
        for i, conn in enumerate(self._parent_conns):
            obs, rew, term, trunc, info = conn.recv()
            self._obs_batch[i] = obs
            self._rew_batch[i] = rew
            self._term_batch[i] = term
            self._trunc_batch[i] = trunc
            infos.append(info)

        return (
            self._obs_batch.copy(),
            self._rew_batch.copy(),
            self._term_batch.copy(),
            self._trunc_batch.copy(),
            infos,
        )

    def close(self):
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    def __del__(self):
        self.close()


def make_cyborg_env(
    num_envs: int = 4,
    seed: int = 42,
    max_steps: int = 500,
    agent_name: str = "blue_agent_0",
) -> CybORGVecEnv:
    """Factory: create a vectorized CybORG environment."""
    return CybORGVecEnv(
        num_envs=num_envs,
        agent_name=agent_name,
        max_steps=max_steps,
        seed=seed,
    )
