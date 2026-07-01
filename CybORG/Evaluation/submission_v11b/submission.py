"""Official CC4 submission adapter for EnterpriseHeuristicAgent v11b."""
from __future__ import annotations

import numpy as np
from enum import Enum

from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Agents.Wrappers import BlueFlatWrapperV2
from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11b import EnterpriseHeuristicAgentV11b


def _action_type(a) -> str:
    if a is None:
        return "None"
    s = str(a).strip()
    return s.split()[0] if s else "Unknown"


def _ternary_to_int(x) -> int:
    if isinstance(x, Enum):
        return int(x.value)
    try:
        return int(x)
    except Exception:
        return -1


def _iter_strings(obj):
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj.lower()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k).lower()
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)
    else:
        yield str(obj).lower()


def extract_shap_features(dict_obs: dict, msg_matrix: np.ndarray | None = None) -> dict:
    feats = {
        "prev_action_success": _ternary_to_int(dict_obs.get("success")),
        "prev_action": _action_type(dict_obs.get("action")),
    }

    strings = list(_iter_strings(dict_obs))

    def has_kw(*kws):
        kws = [kw.lower() for kw in kws]
        return int(any(any(kw in s for kw in kws) for s in strings))

    feats["has_cmd_sh"] = has_kw("cmd.sh", "cmd_sh", "cmdsh")
    feats["has_escalate"] = has_kw("escalate", "privilege escalation", "privesc", "sudo")
    feats["has_decoy_exploit"] = has_kw("decoy", "exploit", "rfi")

    if msg_matrix is not None:
        m = np.asarray(msg_matrix)
        if m.ndim == 2 and m.shape[1] >= 3:
            scanned = m[:, 0]
            compromised = m[:, 1]
            received = m[:, 2]
            feats["msg_any_received"] = int(np.any(received > 0))
            feats["msg_n_received"] = int(np.sum(received > 0))
            feats["msg_any_scanned"] = int(np.any(scanned > 0))
            feats["msg_any_compromised"] = int(np.any(compromised > 0))
            feats["msg_n_scanned_flags"] = int(np.sum(scanned > 0))
            feats["msg_n_compromised_flags"] = int(np.sum(compromised > 0))
            return feats

    feats["msg_any_received"] = 0
    feats["msg_n_received"] = 0
    feats["msg_any_scanned"] = 0
    feats["msg_any_compromised"] = 0
    feats["msg_n_scanned_flags"] = 0
    feats["msg_n_compromised_flags"] = 0
    return feats


def extract_message_matrix(dict_obs: dict) -> np.ndarray | None:
    msg = dict_obs.get("message")
    if msg is None:
        return None
    try:
        msg = np.stack(msg, axis=0)
    except Exception:
        return None

    if msg.ndim != 2 or msg.shape[1] < 1:
        return None

    received_msg = msg[:, -1:]
    if msg.shape[0] > 1:
        if msg.shape[0] >= 9:
            msg_small = msg[:-1, :2]
            msg_big = msg[-1, :6].reshape(3, 2)
            msg = np.concatenate([msg_small, msg_big], axis=0)
            received_msg = np.concatenate([received_msg, np.zeros((2, 1))], axis=0)
            received_msg[-2:] = received_msg[-3]
        else:
            msg = msg[:, :2]
    else:
        msg = msg[:, :2]

    try:
        return np.concatenate([msg, received_msg], axis=1)
    except Exception:
        return None


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

        shap_info = {}
        actions = actions or {}
        for agent_name, agent in self._heuristic_agents.items():
            try:
                dict_obs = dict(self.env.environment_controller.get_last_observation(agent_name).data)
                msg = extract_message_matrix(dict_obs)
                shap_info[agent_name] = {
                    "chosen_action_type": agent._action_class(actions.get(agent_name)),
                    "chosen_action_str": str(actions.get(agent_name)),
                    "shap_features": extract_shap_features(dict_obs, msg),
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
    NAME: str = "EnterpriseHeuristicAgent v11b"
    TEAM: str = "CC4-Optimized"
    TECHNIQUE: str = "Rule-based priority heuristic with v11b coordinated messaging and blocking."

    AGENTS: dict[str, HeuristicSubmissionAgent] = {
        f"blue_agent_{i}": HeuristicSubmissionAgent(f"blue_agent_{i}") for i in range(5)
    }

    @staticmethod
    def wrap(env: CybORG) -> HeuristicEnv:
        return HeuristicEnv(env=env, agents=Submission.AGENTS)
