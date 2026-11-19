#!/usr/bin/env python3
"""Shared coalition-evaluation harness for the explainability pipeline.

Ported from ExplainabilityCC4/analysis and adapted to cc4-optimized-explained.
Differences from the original repo are concentrated here so the analysis
scripts (shapley, interactions, ablation) stay repo-agnostic.

Key adaptations vs. ExplainabilityCC4:
  * Heuristic agent is EnterpriseHeuristicAgentV11a reached through the
    official Submission wrapper (BlueFlatWrapperV2 + v9 messaging), not
    Silja's H_Agent/HeuristicWrapper.
  * Actions are flat integer indices (not raw Action objects); an absent
    agent in a coalition submits its per-agent Sleep index.
  * The environment is built with the plain EnterpriseScenarioGenerator, NOT
    FastEnterpriseScenarioGenerator -- the fast pool freezes the topology to
    numpy seeds 0..pool_size-1, which would make every coalition share the
    same few networks. The plain generator varies the network per seed.
  * v(S) is the scalar team return (sum of blue per-step rewards). Per-reward-
    component decomposition (LWF/ASF/RIA/...) needs the reward machine to log
    its summary, which this repo does not do out of the box; left as a TODO.

The environment here is the FIXED one (reward bug fixed, Remove duration 2,
Monitor removed from the policy action space). Numbers therefore will NOT match
the ICMCIS/MILCOM paper, which used ExplainabilityCC4's original environment.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from CybORG import CybORG                                              # noqa: E402
from CybORG.Agents import (                                           # noqa: E402
    SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator     # noqa: E402

AGENT_NAMES = [f"blue_agent_{i}" for i in range(5)]
N_AGENTS = len(AGENT_NAMES)
MSG_LEN = 8                       # bits per inter-agent message


# --------------------------------------------------------------------------- #
# Red behaviour profiles (the ablation / robustness sweep).
# --------------------------------------------------------------------------- #
def red_registry() -> dict:
    from evaluate import PROFILE_REGISTRY
    return dict(PROFILE_REGISTRY)


# --------------------------------------------------------------------------- #
# Heuristic .
# --------------------------------------------------------------------------- #
def _load_submission():
    """Load the official heuristic Submission module by file path."""
    path = REPO / "CybORG" / "Evaluation" / "submission_v11a" / "submission.py"
    spec = importlib.util.spec_from_file_location("_cc4_submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ablate_message(msg: np.ndarray, mode: str) -> np.ndarray:
    """Transform an outgoing 8-bit message for the communication ablation.

    If everyone sends an ablated message, everyone receives one -- equivalent
    to ablating incoming messages, without touching the wrapper internals.
        'content' : zero content bits, keep the final validity/check bit.
        'empty'   : zero the whole message (CybORG EMPTY_MESSAGE).
    """
    if msg is None:
        return msg
    m = np.asarray(msg).copy()
    if mode == "empty":
        m[:] = 0
    elif mode == "content":
        m[:-1] = 0
    return m


def run_episode_heuristic(active_idx, submission, seed, steps, red_cls,
                          ablate_comms=None):
    """Run one episode with a coalition of active heuristic agents.

    active_idx : iterable of agent indices that run the real policy; the rest
                 submit Sleep (present but passive).
    ablate_comms : None | 'content' | 'empty' -- message ablation (full team
                 only; meaningful because V11a reads peer messages).
    Returns the scalar team return v(S).
    """
    active = set(active_idx)
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent,
        red_agent_class=red_cls, steps=steps)
    env = submission.Submission.wrap(CybORG(sg, "sim", seed=seed))
    agents = submission.Submission.AGENTS
    obs, _ = env.reset()

    # per-agent Sleep index (varies: agent_4 has a larger action space)
    sleep_idx = {}
    for a in AGENT_NAMES:
        labels = env.action_labels(a)
        sleep_idx[a] = labels.index("Sleep")

    total = 0.0
    for _ in range(steps):
        live = [a for a in env.agents if a in obs]
        actions = {}
        for a in live:
            i = AGENT_NAMES.index(a)
            if i in active:
                actions[a] = agents[a].get_action(obs[a], env.action_space(a))
                if ablate_comms:
                    agents[a]._last_message = _ablate_message(
                        agents[a]._last_message, ablate_comms)
            else:
                actions[a] = sleep_idx[a]
                agents[a]._last_message = None     # passive: send nothing
        obs, rew, term, trunc, _ = env.step(actions)
        # CC4 gives the SAME shared team reward to every blue agent, so take one
        # value, not the sum (sum would be 5x the team reward).
        total += float(next(iter(rew.values()))) if rew else 0.0
        if (term and all(term.values())) or (trunc and all(trunc.values())):
            break
    return total


# --------------------------------------------------------------------------- #
# MARL.
# --------------------------------------------------------------------------- #
def run_episode_marl(active_idx, team, seed, steps, red_cls, ablate_comms=None):
    """Matched-seed coalition/communication evaluation with the fixed graph wrapper.

    Absent defenders use the wrapper's None -> Monitor fallback and send no
    messages. All defenders remain in the environment with their own sessions.
    """
    import random
    import torch

    if ablate_comms not in (None, "content", "empty"):
        raise ValueError(f"Unknown ablation mode: {ablate_comms}")
    active = set(active_idx)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent,
        red_agent_class=red_cls, steps=steps,
    )
    env = team.wrap(CybORG(sg, "sim", seed=seed))
    obs, _ = env.reset()
    total = 0.0
    for t in range(steps):
        actions = {}
        for i, name in enumerate(AGENT_NAMES):
            if name not in env.agents or name not in obs:
                continue
            if i in active:
                # Match policy RNG per agent and timestep, even across coalitions.
                torch.manual_seed((int(seed) * 1000003 + t * 5 + i) % (2**63))
                actions[name] = team.AGENTS[name].get_action(obs[name], env.action_space(name))
            else:
                actions[name] = None  # GraphWrapper translates None to Monitor
            if i not in active:
                env.msg[name] = np.zeros_like(env.msg[name])
            elif ablate_comms:
                env.msg[name] = _ablate_message(env.msg[name], ablate_comms)
        result = env.step(actions)
        obs, rew, term, trunc, _ = result[:5]
        total += float(rew["blue_agent_0"]) if rew else 0.0
        if term.get("__all__", False) or trunc.get("__all__", False):
            break
        if actions and all(term.get(a, False) or trunc.get(a, False) for a in actions):
            break
    return total


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def make_backend(kind: str):
    """Return (run_episode_fn, team_obj). team_obj is backend-specific."""
    os.environ.setdefault("CYBORG_REWARD_LOG_PATH", os.devnull)
    if kind == "heuristic":
        sub = _load_submission()
        return (lambda active, team, seed, steps, red, abl=None:
                run_episode_heuristic(active, sub, seed, steps, red, abl)), sub
    if kind == "marl":
        # Root submission is imported by explicit path, not the heuristic module.
        spec = importlib.util.spec_from_file_location("_cc4_marl_submission", REPO / "submission.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return run_episode_marl, module.Submission()
    raise ValueError(f"unknown backend: {kind}")
