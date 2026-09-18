"""CC4 v9.1 heuristic submission used for Remove-duration experiments.

Author: Coda
"""
from __future__ import annotations

from CybORG import CybORG
from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import EnterpriseHeuristicAgent
from CybORG.Evaluation.submission.submission import HeuristicEnv, HeuristicSubmissionAgent


class RemoveAwareSubmissionAgent(HeuristicSubmissionAgent):
    def __init__(self, agent_name: str) -> None:
        super().__init__(agent_name)
        self._inner = EnterpriseHeuristicAgent(agent_name=agent_name)


class Submission:
    NAME = "EnterpriseHeuristicAgent v9.1"
    TEAM = "CC4-Optimized"
    TECHNIQUE = "Remove-aware rule-based heuristic used for paired action-duration evaluation."

    AGENTS = {
        f"blue_agent_{i}": RemoveAwareSubmissionAgent(f"blue_agent_{i}")
        for i in range(5)
    }

    @staticmethod
    def wrap(env: CybORG) -> HeuristicEnv:
        return HeuristicEnv(env=env, agents=Submission.AGENTS)
