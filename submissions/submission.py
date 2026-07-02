"""Official CC4 submission -- EnterpriseHeuristicAgent v11a (Preemptive OZ Blocking).

Compatible with CybORG/Evaluation/evaluation.py interface.
Uses BlueFlatWrapperV2 for observations (adds malicious-file flags) and
action masks.

v11a improvements over v10b:
  - All v10b features retained: Restore-only, flag_age >= 1, MAX_DECOYS=3
  - NEW: Preemptive OZ blocking at phase transitions (P_BLOCK_OZ priority)
    Blocks RZA->OZA before Phase 1, RZB->OZB before Phase 2 (ASF=0 = free)
  - Eliminates vulnerability window at phase transitions where comms_policy
    switches but agents are busy Restoring
  - Priority elevation: OZ blocking runs above P2 Allow during active phases

Performance: -700.0 ± 160.5 (seed 42, 30 eps) = +14.0% over v10b baseline.
Cross-validated: -695.2 ± 174.7 (seed 123, 30 eps). 35% lower variance.
Beats Oracle V3 (-893.5) due to decoy prevention + preemptive blocking.

Note on evaluation.py compatibility: evaluation.py does not pass messages to step().
HeuristicEnv.step() intercepts each call and injects stored outgoing messages so
inter-agent communication works end-to-end within the official evaluation harness.
"""
from __future__ import annotations

import numpy as np

from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Agents.Wrappers import BlueFlatWrapperV2
from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11b import EnterpriseHeuristicAgentV11b
from CybORG.Evaluation.submission_v11b.submission import extract_message_matrix, extract_shap_features


class HeuristicSubmissionAgent(BaseAgent):
    """Adapter: wraps EnterpriseHeuristicAgentV11a for the evaluation.py interface.

    The evaluation calls get_action(obs, action_space).  We ignore the
    action_space argument and instead fetch the boolean action mask directly
    from the BlueFlatWrapperV2 stored in self._env.
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._inner = EnterpriseHeuristicAgentV11b(agent_name=agent_name)
        self._env: "HeuristicEnv | None" = None
        self._last_message: "np.ndarray | None" = None
        self._info: list[dict] = []

    @property
    def info(self):
        return self._info

    # Called by evaluation.py each step
    def get_action(self, observation, action_space=None):
        mask = None
        if self._env is not None:
            try:
                mask = np.array(self._env.action_mask(self.agent_name), dtype=bool)
            except Exception:
                pass
        action_idx, msg = self._inner.get_action(observation, mask)
        self._last_message = msg  # stored for HeuristicEnv.step() to collect
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
    """BlueFlatWrapperV2 that reinitialises heuristic agents on each reset().

    Intercepts reset() to call agent.reset() and agent.set_action_info()
    so that per-episode state (decoy tracking, remove/restore timers, etc.)
    is cleared at the start of every episode.
    """

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
        """Intercept step() to inject stored outgoing messages.

        evaluation.py does not pass messages, so we collect the _last_message
        stored by each HeuristicSubmissionAgent.get_action() call and forward
        them to the parent step() — enabling full v9 inter-agent messaging.
        """
        if messages is None:
            messages = {}
        for agent_name, agent in self._heuristic_agents.items():
            if agent_name not in messages and agent._last_message is not None:
                messages[agent_name] = agent._last_message

        shap_info = {}
        actions = actions or {}
        for agent_name, agent in self._heuristic_agents.items():
            try:
                dict_obs = dict(self.env.environment_controller.get_last_observation(agent_name).data)
                msg = extract_message_matrix(dict_obs)
                shap_info[agent_name] = {
                    "chosen_action_type": agent._action_class(actions.get(agent_name)),
                    "chosen_action_str": str(actions.get(agent_name)),
                    "shap_features": extract_shap_features(
                        dict_obs,
                        msg,
                        mission_phase=getattr(self.env.environment_controller.state, "mission_phase", None),
                    ),
                }
            except Exception:
                continue

        obs, rewards, term, trunc, info = super().step(actions=actions, messages=messages, **kwargs)
        for agent_name, agent_info in shap_info.items():
            if agent_name not in info or info[agent_name] is None:
                info[agent_name] = {}
            info[agent_name].update(agent_info)
        return obs, rewards, term, trunc, info


class Submission:
    # -- Required metadata ----------------------------------------------------
    NAME: str = "EnterpriseHeuristicAgent v11b"
    TEAM: str = "CC4-Optimized"
    TECHNIQUE: str = (
        "Rule-based priority heuristic with Restore-only threat response and preemptive OZ "
        "blocking. Restore-only (Remove eliminated — PrivEsc 100% in 2 steps beats Remove 3). "
        "flag_age >= 1 filters green FPs; OZ server_host_0 gets immediate Restore. "
        "MAX_DECOYS=3 (75% blind exploit failure). Preemptive blocking: blocks traffic to "
        "active OZ subnet at phase transitions (ASF=0 = zero cost). P_BLOCK_OZ priority "
        "ensures active OZ isolation before Allow actions. 10-step preemptive window before "
        "phase transitions eliminates vulnerability gap. Comms-policy-driven firewall management."
    )

    # One agent per blue team member (blue_agent_0 through blue_agent_4)
    AGENTS: dict[str, HeuristicSubmissionAgent] = {
        f"blue_agent_{i}": HeuristicSubmissionAgent(f"blue_agent_{i}") for i in range(5)
    }

    @staticmethod
    def wrap(env: CybORG) -> HeuristicEnv:
        """Wrap CybORG with BlueFlatWrapperV2 + heuristic agent reset logic."""
        return HeuristicEnv(env=env, agents=Submission.AGENTS)
