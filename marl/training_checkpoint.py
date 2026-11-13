"""Crash-resistant joint PPO training checkpoints (not inference exports)."""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1


def atomic_torch_save(value, path, *, min_free_gb=2.0):
    """Replace destination only after a complete, synced temporary write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    free = shutil.disk_usage(path.parent).free
    if free < min_free_gb * (1024 ** 3):
        raise OSError(f"Not enough free disk space for checkpoint: {free / 1024**3:.2f} GiB")
    try:
        with temporary.open('wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Validate serialized structure before committing it to the live checkpoint.
        with temporary.open('rb') as handle:
            torch.load(handle, map_location='cpu', weights_only=False)
        os.replace(temporary, path)
        # Ensure rename is persisted as far as the platform allows.
        if hasattr(os, 'O_DIRECTORY'):
            fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_state():
    return {
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_training_checkpoint(path, agents, next_update, config, log,
                             per_profile_log=None, *, min_free_gb=2.0):
    record = {
        'format_version': FORMAT_VERSION,
        'next_update': int(next_update),
        'config': config,
        'log': list(log),
        'per_profile_log': list(per_profile_log or []),
        'agents': [
            {
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_optimizer': agent.actor.opt.state_dict(),
                'critic_optimizer': agent.critic.opt.state_dict(),
            }
            for agent in agents
        ],
        **_runtime_state(),
    }
    atomic_torch_save(record, path, min_free_gb=min_free_gb)


def _optimizer_to_device(optimizer, device):
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)


def load_training_checkpoint(path, agents, config):
    """Restore networks/optimizers/RNG and return next update + saved logs.

    Checkpoint files must come from trusted sources because PyTorch loads pickle.
    """
    record = torch.load(path, map_location='cpu', weights_only=False)
    if record.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported training checkpoint version')
    if record['config'] != config:
        raise ValueError(f"Training configuration differs from checkpoint. Saved: {record['config']}; current: {config}")
    if len(record['agents']) != len(agents):
        raise ValueError('Unexpected agent count in training checkpoint')
    # Preflight all network tensor keys and shapes before loading any state.
    for agent, saved in zip(agents, record['agents']):
        for network_name in ('actor', 'critic'):
            current = getattr(agent, network_name).state_dict()
            incoming = saved[network_name]
            if set(current) != set(incoming) or any(current[k].shape != incoming[k].shape for k in current):
                raise ValueError(f'Incompatible {network_name} weights')
    for agent, saved in zip(agents, record['agents']):
        agent.actor.load_state_dict(saved['actor'])
        agent.critic.load_state_dict(saved['critic'])
        agent.actor.opt.load_state_dict(saved['actor_optimizer'])
        agent.critic.opt.load_state_dict(saved['critic_optimizer'])
        _optimizer_to_device(agent.actor.opt, agent.device)
        _optimizer_to_device(agent.critic.opt, agent.device)
    random.setstate(record['python_rng'])
    np.random.set_state(record['numpy_rng'])
    torch.set_rng_state(record['torch_rng'])
    if record['cuda_rng'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG state present but CUDA unavailable')
        torch.cuda.set_rng_state_all(record['cuda_rng'])
    return record
