"""
submission.py — TTCP CAGE Challenge 4 Agent Submission Interface

Defines the `Submission` class containing all necessary metadata and agent-loading functionality
for challenge evaluation. It uses GNN-based PPO agents and loads trained weights for each Blue agent.

Main Contents:
- NAME, TEAM, TECHNIQUE: Identifiers for the submitted solution
- AGENTS: Dictionary mapping agent names to trained agent instances (loaded from file)
- wrap(env): Optional wrapper to apply GraphWrapper to CybORG
"""

import os
import sys

# Ensure marl directory is on the path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from CybORG import CybORG
from CybORG.Agents import BaseAgent

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:
    MultiAgentEnv = None

### Import custom agents here ###
from models.cage4 import load
from wrappers.graph_wrapper import GraphWrapper

# Checkpoint prefix is configurable via KEEP_RUN env var; defaults to the
# debug run shipped in marl/checkpoints/. Point at a trained-run prefix
# for real evaluation (e.g. KEEP_RUN=my_run).
_CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
_CKPT_PREFIX = os.environ.get('KEEP_RUN', 'test_cpu')


class Submission:

    # Submission name
    NAME: str = "KEEP"

    # Name of your team
    TEAM: str = "Cybermonic"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "Graph-based PPO With Intra-agent Communication"

    AGENTS = {
        f"blue_agent_{i}": load(os.path.join(_CKPT_DIR, f'{_CKPT_PREFIX}-{i}_checkpoint.pt'))
        for i in range(5)
    }

    # Use this function to optionally wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG):
        return GraphWrapper(env)
