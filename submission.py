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

from CybORG import CybORG
from CybORG.Agents import BaseAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFixedActionWrapper
from wrappers.heuristic_wrapper import HeuristicWrapper
from wrappers.graph_wrapper import GraphWrapper

from ray.rllib.env.multi_agent_env import MultiAgentEnv
### Import custom agents here ###
from models.cage4 import load
from wrappers.graph_wrapper import GraphWrapper
"""
### Import custom agents here ###
from Heuristic.Heuristic_Agent_v0_3 import H_Agent

class Submission:

    # Submission name
    NAME: str = "heuristic"

    # Name of your team
    TEAM: str = "SIX"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "Heuristic Agent"

    # Use this function to define your agents.
    AGENTS = {
        f"blue_agent_{i}": H_Agent(agent_name=f"blue_agent_{i}")
        for i in range(5)
    }

    # Use this function to optionally wrap CybORG with your custom wrapper(s).
    @staticmethod
    def wrap(env: CybORG) -> MultiAgentEnv:
        return HeuristicWrapper(env)
        #return BlueFixedActionWrapper(env)


"""
class Submission:
    NAME = "GNN_marl_02"
    TEAM = "Team.KI"
    TECHNIQUE = "Graph-based PPO With Intra-agent Communication"

    def __init__(self):
        self.AGENTS = {
            f"blue_agent_{i}": load(
                f"{os.path.dirname(__file__)}/weights/marl_run_02/gnn_ppo-{i}.pt",
                map_location="cpu"
            )
            for i in range(5)
        }

    @staticmethod
    def wrap(env):
        return GraphWrapper(env)

"""
class Submission:
    NAME = "Sleep"
    TEAM = "Team.KI"
    TECHNIQUE = "Sleep Agent"

    def __init__(self):
        self.AGENTS = {
            f"blue_agent_{i}": SleepAgent(agent_name=f"blue_agent_{i}")
            for i in range(5)
        }

    # Use this function to optionally wrap CybORG with your custom wrapper(s).
    @staticmethod
    def wrap(env: CybORG) -> MultiAgentEnv:
        return BlueFixedActionWrapper(env)
        #return BlueFixedActionWrapper(env)
"""
