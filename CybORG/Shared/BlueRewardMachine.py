from CybORG.Shared.RewardCalculator import RewardCalculator
from CybORG.Simulator.State import State
from CybORG.Simulator.Actions.GreenActions import GreenAccessService, GreenLocalWork
from CybORG.Simulator.Actions.AbstractActions.Impact import Impact

try:
    from CybORG.Simulator.Actions.AbstractActions.PrivilegeEscalate import PrivilegeEscalate
except Exception:
    PrivilegeEscalate = tuple()
try:
    from CybORG.Simulator.Actions.AbstractActions.Analyse import Analyse
except Exception:
    Analyse = tuple()
try:
    from CybORG.Simulator.Actions.AbstractActions.ExploitRemoteService import ExploitRemoteService
except Exception:
    ExploitRemoteService = tuple()

try:
    from CybORG.Simulator.Actions.AbstractActions.DegradeServices import DegradeServices
except Exception:
    DegradeServices = tuple()
from CybORG.Simulator.Actions.Action import InvalidAction
import json
from pathlib import Path
from collections import defaultdict
import os
from numbers import Number
from datetime import datetime as _dt


class BlueRewardMachine(RewardCalculator):
    """The reward calculator for CC4

    Attributes
    ----------
    phase_rewards : Dict[str, Dict[str, int]]
        the reward mapping for the current mission phase
    """

    def __init__(
            self,
            *args,
            reward_green=False,
            reward_red=True,
            reward_blue_actions=False,
            phase_reward_mode="default",
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.phase_reward_mode = os.environ.get("CYBORG_PHASE_REWARD_MODE", "default")
        self.reward_blue_actions = os.environ.get("CYBORG_REWARD_BLUE", "0") == "1"

        self._episode = 0
        self._step = 0
        self._prev_time = None

    def get_phase_rewards(self, cur_mission_phase):
        """Gets the pre-set reward mapping for the current mission phase

        Rewards Key:
        - LWF = Local Work Fails
        - ASF = Access Service Fails
        - RIA = Red Impact/Access

        Parameters
        ----------
        cur_mission_phase : int
            the current mission phase of the episode

        Returns
        -------
        : Dict[str, Dict[str, int]]
            the phase reward mapping for the current mission phase
        """
        default_rewards = {
            0: {
                "public_access_zone_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
                "admin_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
                "office_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
                "contractor_network_subnet": {"LWF": 0, "ASF": -5, "RIA": -5},
                "restricted_zone_a_subnet": {"LWF": -1, "ASF": -3, "RIA": -1},
                "operational_zone_a_subnet": {"LWF": -1, "ASF": -1, "RIA": -1},
                "restricted_zone_b_subnet": {"LWF": -1, "ASF": -3, "RIA": -1},
                "operational_zone_b_subnet": {"LWF": -1, "ASF": -1, "RIA": -1},
                "internet_subnet": {"LWF": 0, "ASF": 0, "RIA": 0}},
            1: {
                "public_access_zone_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
                "admin_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
                "office_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
                "contractor_network_subnet": {"LWF": 0, "ASF": 0, "RIA": 0},
                "restricted_zone_a_subnet": {"LWF": -2, "ASF": -1, "RIA": -3},
                "operational_zone_a_subnet": {"LWF": -10, "ASF": 0, "RIA": -10},
                "restricted_zone_b_subnet": {"LWF": -1, "ASF": -1, "RIA": -1},
                "operational_zone_b_subnet": {"LWF": -1, "ASF": -1, "RIA": -1},
                "internet_subnet": {"LWF": 0, "ASF": 0, "RIA": 0}},
            2: {
                "public_access_zone_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
                "admin_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
                "office_network_subnet": {"LWF": -1, "ASF": -1, "RIA": -3},
                "contractor_network_subnet": {"LWF": 0, "ASF": 0, "RIA": 0},
                "restricted_zone_a_subnet": {"LWF": -1, "ASF": -3, "RIA": -3},
                "operational_zone_a_subnet": {"LWF": -1, "ASF": -1, "RIA": -1},
                "restricted_zone_b_subnet": {"LWF": -2, "ASF": -1, "RIA": -3},
                "operational_zone_b_subnet": {"LWF": -10, "ASF": 0, "RIA": -10},
                "internet_subnet": {"LWF": 0, "ASF": 0, "RIA": 0}}
        }

        contractor_off = {
            phase: {
                subnet: values.copy()
                for subnet, values in rewards.items()
            }
            for phase, rewards in default_rewards.items()
        }

        for phase in contractor_off:
            contractor_off[phase]["contractor_network_subnet"] = {"LWF": 0, "ASF": 0, "RIA": 0}

        red_only = {}

        for phase, subnets in default_rewards.items():
            red_only[phase] = {}

            for subnet, values in subnets.items():
                ria = values["RIA"]

                red_only[phase][subnet] = {
                    "LWF": 0,
                    "ASF": 0,
                    "RIA": ria,
                    "EXP": ria * 0.7,
                    "PRIV": ria * 0.8,
                    "DEG": ria * 0.6,
                }

        modes = {
            "default": default_rewards,
            "contractor_off": contractor_off,
            "red_only": red_only,
        }

        reward_table = modes[self.phase_reward_mode]
        return reward_table[cur_mission_phase]

    def calculate_reward(self, current_state: dict, action_dict: dict, agent_observations: dict, done: bool,
                         state: State):
        reward_list = []
        reward_summary = {
            "total": 0,
            "subnet_rewards": defaultdict(
                lambda: {"LWF": 0, "ASF": 0, "RIA": 0, "EXP": 0, "PRIV": 0, "DEG": 0, "ANALYSE": 0}
            ),
        }

        self.phase_rewards = self.get_phase_rewards(state.mission_phase)
        rew_mode = self.phase_reward_mode
        blue_rew = self.reward_blue_actions

        for agent_name, action in action_dict.items():
            if not action:
                continue

            action = action[0]
            r = 0
            hostname = None
            subnet_name = None

            # ----------------------------
            # Resolve hostname / subnet
            # ----------------------------
            try:
                if isinstance(action, (Impact, Analyse, PrivilegeEscalate, DegradeServices)):
                    hostname = action.hostname

                elif isinstance(action, (GreenAccessService, GreenLocalWork, ExploitRemoteService)):
                    hostname = state.ip_addresses[action.ip_address]

                # if hostname was found, resolve subnet
                if hostname is not None:
                    subnet_name = state.hostname_subnet_map[hostname].value

            except Exception as e:
                print(f"[reward] Could not resolve hostname/subnet for {agent_name}: {action} ({e})")
                hostname = None
                subnet_name = None

            sessions = state.sessions[agent_name].values()
            if len([session.ident for session in sessions if session.active]) == 0:
                continue

            try:
                success = agent_observations[agent_name].observations[0].data["success"]
            except Exception:
                success = False

            rewards_for_zone = self.phase_rewards.get(subnet_name, None) if subnet_name is not None else None

            # ----------------------------
            # Green rewards
            # ----------------------------
            if "green" in agent_name and success is False and rew_mode != "red_only" and rewards_for_zone is not None:
                if isinstance(action, GreenLocalWork):
                    r = rewards_for_zone["LWF"]
                    reward_summary["subnet_rewards"][subnet_name]["LWF"] += r

                elif isinstance(action, GreenAccessService):
                    r = rewards_for_zone["ASF"]
                    reward_summary["subnet_rewards"][subnet_name]["ASF"] += r

            # ----------------------------
            # Red rewards
            # ----------------------------
            elif "red" in agent_name and success and rewards_for_zone is not None:
                if isinstance(action, Impact):
                    r = rewards_for_zone["RIA"]
                    reward_summary["subnet_rewards"][subnet_name]["RIA"] += r

                elif rew_mode == "red_only" and isinstance(action, PrivilegeEscalate):
                    # print("Priv")
                    r = rewards_for_zone["PRIV"]
                    reward_summary["subnet_rewards"][subnet_name]["PRIV"] += r

                elif rew_mode == "red_only" and isinstance(action, ExploitRemoteService):
                    # print("Exploit")
                    r = rewards_for_zone["EXP"]
                    reward_summary["subnet_rewards"][subnet_name]["EXP"] += r

                elif rew_mode == "red_only" and isinstance(action, DegradeServices):
                    # print("Degrade")
                    r = rewards_for_zone["DEG"]
                    reward_summary["subnet_rewards"][subnet_name]["DEG"] += r

            # ----------------------------
            # Blue analyse reward
            # ----------------------------
            elif "blue" in agent_name and blue_rew and isinstance(action, Analyse):
                r = 0.1
                reward_summary["subnet_rewards"][subnet_name or "unknown"]["ANALYSE"] += r

            reward_list.append(r)
            reward_summary["total"] += r

        try:
            # --- determine "time" in a safe way ---
            cur_time = getattr(state, "time", None)

            def _is_number(x):
                return isinstance(x, Number) and not isinstance(x, bool)

            # Case A: numeric timestep available
            if _is_number(cur_time):
                cur_step = int(cur_time)
                # detect reset
                if self._prev_time is not None and _is_number(self._prev_time):
                    if cur_step < int(self._prev_time):
                        self._episode += 1
                self._step = cur_step
                self._prev_time = cur_time

            # Case B: datetime timestamps (or anything non-numeric)
            else:
                # We cannot infer step from datetime reliably, so we keep our own counters.
                # Episode increments when done=True; step increments every call.
                # Reset step at end of episode.
                pass

            # --- choose log path (configurable) ---
            log_path_str = os.environ.get("CYBORG_REWARD_LOG_PATH", "reward_log.jsonl")
            log_path = Path(log_path_str)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            profile = os.environ.get("CYBORG_ATTACK_PROFILE", None)
            run_tag = os.environ.get("CYBORG_RUN_TAG", None)

            # --- write entry ---
            log_entry = {
                "profile": profile,
                "run_tag": run_tag,
                "episode": int(self._episode),
                "step": int(self._step),
                "phase": int(state.mission_phase),
                "reward_list": reward_summary["subnet_rewards"],
                "total": reward_summary["total"],
            }

            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")

            # --- update fallback counters when time is non-numeric ---
            if not _is_number(cur_time):
                self._step += 1
                if done:
                    self._episode += 1
                    self._step = 0

        except Exception as e:
            print(f"[BlueRewardMachine] Warning: could not log the reward to file: {e}")

        return sum(reward_list)


  
        
        
 
     
        
    