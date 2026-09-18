"""Simple Analyse-before-Remove heuristic for duration experiments.

Author: Coda
"""
from __future__ import annotations

import re

import numpy as np

from CybORG import CybORG
from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import EnterpriseHeuristicAgent
from CybORG.Evaluation.submission.submission import HeuristicEnv, HeuristicSubmissionAgent


class AnalyseProbeAgent(HeuristicSubmissionAgent):
    """Run Analyse once before the first Remove attempted on each host."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(agent_name)
        self._inner = EnterpriseHeuristicAgent(agent_name=agent_name)
        self._analysed_hosts: set[str] = set()

    def get_action(self, observation, action_space=None):
        if self._inner._step == 0:
            self._analysed_hosts.clear()

        action_idx = super().get_action(observation, action_space)
        if self._env is None:
            return action_idx

        labels = self._env.action_labels(self.agent_name)
        selected = labels[int(action_idx)]
        match = re.match(r"Remove\s+(\S+)", selected)
        if match is None:
            return action_idx

        hostname = match.group(1)
        if hostname in self._analysed_hosts:
            return action_idx

        mask = np.asarray(self._env.action_mask(self.agent_name), dtype=bool)
        prefix = f"Analyse {hostname}"
        for idx, label in enumerate(labels):
            if label.startswith(prefix) and idx < len(mask) and mask[idx]:
                self._analysed_hosts.add(hostname)
                self._info[-1]["ActionClass"] = "Analyse"
                return idx
        return action_idx


class Submission:
    NAME = "EnterpriseHeuristicAgent Analyse Probe"
    TEAM = "CC4-Optimized"
    TECHNIQUE = "v9.1 heuristic with a one-shot Analyse before the first Remove per host."

    AGENTS = {
        f"blue_agent_{i}": AnalyseProbeAgent(f"blue_agent_{i}")
        for i in range(5)
    }

    @staticmethod
    def wrap(env: CybORG) -> HeuristicEnv:
        return HeuristicEnv(env=env, agents=Submission.AGENTS)
