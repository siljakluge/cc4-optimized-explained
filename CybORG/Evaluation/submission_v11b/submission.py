"""Official CC4 submission adapter for EnterpriseHeuristicAgent v11b."""
from __future__ import annotations

import numpy as np

from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Agents.Wrappers import BlueFlatWrapperV2
from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11b import EnterpriseHeuristicAgentV11b


class HeuristicSubmissionAgent(BaseAgent):
    """Adapter for the evaluation/explainability interface."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._inner = EnterpriseHeuristicAgentV11b(agent_name=agent_name)
        self._env: "HeuristicEnv | None" = None
        self._last_message: "np.ndarray | None" = None
        self._info: list[dict] = []

    @property
    def info(self):
        return self._info

    def get_action(self, observation, action_space=None):
        mask = None
        if self._env is not None:
            try:
                mask = np.array(self._env.action_mask(self.agent_name), dtype=bool)
            except Exception:
                pass
        action_idx, msg = self._inner.get_action(observation, mask)
        self._last_message = msg
        self._info.append({
            "Predicates": self._observation_features(observation),
            "ActionClass": self._action_class(action_idx),
        })
        return action_idx

    def _action_class(self, action_idx: int) -> str:
        if self._env is None:
            return str(action_idx)
        try:
            label = str(self._env.action_labels(self.agent_name)[int(action_idx)])
        except Exception:
            return str(action_idx)
        return label.split()[0] if label else str(action_idx)

    @staticmethod
    def _observation_features(observation) -> dict[str, float]:
        obs = np.asarray(observation, dtype=np.float32).ravel()
        return {f"obs_{i:03d}": float(v) for i, v in enumerate(obs)}

    def train(self, *args, **kwargs):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self, *args, **kwargs):
        pass


class HeuristicEnv(BlueFlatWrapperV2):
    """BlueFlatWrapperV2 with v11b agent reset and message forwarding."""

    def __init__(self, env: CybORG, agents: dict[str, HeuristicSubmissionAgent]) -> None:
        super().__init__(env=env)
        self._heuristic_agents = agents
        for agent in agents.values():
            agent._env = self

    def reset(self, **kwargs):
        obs_dict, info = super().reset(**kwargs)
        subnet_hosts = getattr(self, "_cached_subnet_hosts", {})
        for agent_name, agent in self._heuristic_agents.items():
            agent._inner.reset()
            agent._info.clear()
            agent._last_message = None
            try:
                agent._inner.set_action_info(
                    self.action_labels(agent_name),
                    self.action_mask(agent_name),
                    subnet_hosts,
                )
            except Exception:
                pass
        return obs_dict, info

    def step(self, actions=None, messages=None, **kwargs):
        if messages is None:
            messages = {}
        for agent_name, agent in self._heuristic_agents.items():
            if agent_name not in messages and agent._last_message is not None:
                messages[agent_name] = agent._last_message
        return super().step(actions=actions, messages=messages, **kwargs)


class Submission:
    NAME: str = "EnterpriseHeuristicAgent v11b"
    TEAM: str = "CC4-Optimized"
    TECHNIQUE: str = "Rule-based priority heuristic with v11b coordinated messaging and blocking."

    AGENTS: dict[str, HeuristicSubmissionAgent] = {
        f"blue_agent_{i}": HeuristicSubmissionAgent(f"blue_agent_{i}") for i in range(5)
    }

    @staticmethod
    def wrap(env: CybORG) -> HeuristicEnv:
        return HeuristicEnv(env=env, agents=Submission.AGENTS)
