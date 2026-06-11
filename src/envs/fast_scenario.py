"""Fast caching wrapper around EnterpriseScenarioGenerator.

Pre-builds a small pool of scenarios at init time; subsequent calls
return deep-copies of pooled scenarios rather than rebuilding the full
graph, which is the bottleneck during environment resets.
"""
from __future__ import annotations

import copy

import numpy as np

from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import (
    EnterpriseScenarioGenerator,
)


class FastEnterpriseScenarioGenerator(EnterpriseScenarioGenerator):
    """Caching scenario generator that pre-warms a pool of scenarios.

    Parameters
    ----------
    pool_size : int
        Number of distinct scenarios to pre-build and cache.  Each reset
        rotates to the next slot and returns a deep-copy, avoiding the
        expensive graph rebuild on every episode.
    **kwargs
        All remaining keyword arguments are forwarded verbatim to
        :class:`EnterpriseScenarioGenerator`.
    """

    def __init__(self, *args, pool_size: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool_size = pool_size
        self._scenario_pool: list = [None] * pool_size
        self._pool_idx: int = 0

        # Pre-warm: build every slot once with a deterministic RNG so that
        # the pool is fully populated before training starts.
        seed_rng = np.random.default_rng(seed=0)
        for i in range(pool_size):
            slot_rng = np.random.default_rng(seed=i)
            self._scenario_pool[i] = copy.deepcopy(
                super().create_scenario(slot_rng)
            )

    # ------------------------------------------------------------------
    def create_scenario(self, np_random):  # noqa: D102
        idx = self._pool_idx % self._pool_size
        self._pool_idx += 1

        if self._scenario_pool[idx] is None:
            # Fallback: slot not yet filled (shouldn't happen after pre-warm)
            self._scenario_pool[idx] = copy.deepcopy(
                super().create_scenario(np_random)
            )

        return copy.deepcopy(self._scenario_pool[idx])
