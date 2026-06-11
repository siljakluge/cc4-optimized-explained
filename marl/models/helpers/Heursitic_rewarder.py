from CybORG.Simulator.Actions import DeployDecoy, Restore, Monitor, Remove, Analyse, InvalidAction
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import AllowTrafficZone, BlockTrafficZone
import numpy as np

class BaseRewarder():
    def __init__(self):
        pass

    def get_reward(self, state, action, next_state):
        raise NotImplementedError("Subclasses should implement this method.")
    

class HeuristicRewarder(BaseRewarder):
    
    def __init__(self, h_agents:list,  weights: list = [5.0, 5.0, 0.0, 1]):
        super().__init__()
        self.h_agents = h_agents
        self.weights = weights


    def get_reward(self, state, action):
        rewards = [0] * len(self.h_agents)
        for agent in self.h_agents:
            idx = int(agent.agent_name.split("_")[-1])
            h_action = agent.get_action(state[agent.agent_name])
            if type(action[agent.agent_name]) == type(h_action):
                rewards[idx] = self.weights[0]
            else:
                rewards[idx] = -self.weights[1]
        return rewards
    

class EnterpriseHeuristicRewarder(BaseRewarder):
    def __init__(self, h_agents:list, env, weights: list = [1.0, 1.0, 1]):
        super().__init__()
        self.h_agents = h_agents
        self.weights = weights
        self.action_space = env._action_space
        self.env = env

    def get_reward(self, state, actions):
        # translate action
        # get Heutistic action
        h_actions = {}
        obs = {name: self.env.malfile_obs_change(name, state[name]) for name in self.h_agents}
        for name, ag in self.h_agents.items():
                raw_obs = obs.get(name, np.zeros(1))
                mask = self.env.action_mask(name)
                action_idx, _ = ag.get_action(raw_obs, np.array(mask, dtype=bool)) # no use for messages
                h_actions[name] = action_idx
        h_action_dict = {
            agent: self.action_space[agent]["actions"][action]
            for agent, action in h_actions.items()
        }
        
        rewards = [0] * len(self.h_agents)

        #Rewards
        for name,agent in self.h_agents.items():
            idx = int(agent.agent_name.split("_")[-1])
            h_action = h_action_dict.get(name)
            # Either needs to hit exacly same action or just same type
            if self.weights[-2] != 0:
                if type(actions[name]) == type(h_action):
                    rewards[idx] = self.weights[0]
                else:
                    rewards[idx] = -self.weights[1]
            else:
                if actions[name] == h_action:
                    rewards[idx] = self.weights[0]
                else:
                    rewards[idx] = -self.weights[1]
        return rewards



class RewardShaper(BaseRewarder):
    """
    This class defines a heursitic reward function.
    
    The reward function penalizes the agent for:
    - using allow traffic more often than necessary
    - having a unwanted file


    From observation I know:
    - what is blocked/unblocked
    
    Goal State: no detected threats, all connections open, everything is decoyed, and continuously analyzing.

    """
    def __init__(self, weights:list = [10.0, 5.0, 5.0, 10.0, 0, 0, 0, 0]):
        super().__init__()
        self.weights = weights
        self.penalties = {
            "Analyzing": 1,
            "AllowTrafficZone": -1,
            "BlockTrafficZone": -1,
            "ControlTrafficImbalance": 2,
            "UnwantedRootAccess": 1,
            "UnwantedShellAccess": 1,
            "NotAnalyzing": 1,
            "NotDecoyed": 1,
        }
        self.traffic_counter = [0] * 5 # -1 for allow, +1 for block
        

    def get_reward(self, action, state):
        """
        Add heuristic-based rewards based on specific conditions observed in the environment.
        This is a placeholder function and should be implemented with actual heuristic rules relevant to the environment and task.
        Args:
            reward: Current reward dictionary for each agent.
            obs: Current observations for each agent.
        Returns:
            Updated reward dictionary with heuristic rewards added.
        """
        # Example: If an agent successfully analyses a critical host, give an additional reward
        rewards = [0] * len(action)
        for agent in action:
            idx = int(agent.split("_")[-1])
            if isinstance(action[agent], Analyse):
                rewards[idx] += self.weights[0] * self.penalties["Analyzing"]  # reward for analysing a host
                print(f"rewards for {agent}: {rewards[idx]}")
            elif isinstance(action[agent], AllowTrafficZone):
                rewards[idx] +=self.weights[1] * self.penalties["AllowTrafficZone"]  # reward for allowing traffic
                self.traffic_counter[idx] -= 1
            elif isinstance(action[agent], BlockTrafficZone):
                rewards[idx] +=self.weights[2] * self.penalties["BlockTrafficZone"]  # reward for blocking traffic
                self.traffic_counter[idx] += 1
            # Penalize if there is an imbalance in traffic control actions
            if abs(self.traffic_counter[idx]) > self.penalties["ControlTrafficImbalance"]:
                rewards[idx] -= self.weights[3] * abs(self.traffic_counter[idx])
        print(self.traffic_counter)
                    
        return rewards