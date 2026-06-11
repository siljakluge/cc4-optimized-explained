from CybORG.Shared import Observation
from CybORG.Simulator.Actions.ConcreteActions.TargetedLocalAction import TargetedLocalAction
from CybORG.Simulator.Host import Host
from CybORG.Simulator.State import State


class RestoreFromBackup(TargetedLocalAction):

    def execute_targeteted_local_action(self, state: State, target_host: Host) -> Observation:
        # Collect ALL sessions that reference this host from the authoritative state.sessions.
        # Using target_host.sessions alone misses sessions whose hostname matches but that were
        # not registered in host.sessions (e.g. pivoted-through sessions), leaving dangling
        # state.sessions references that raise KeyError on subsequent red actions.
        all_host_sessions = {}  # {agent: {session_id: session_obj}}
        for agent, session_dict in state.sessions.items():
            for sid, session_obj in list(session_dict.items()):
                if session_obj.hostname == target_host.hostname:
                    all_host_sessions.setdefault(agent, {})[sid] = state.sessions[agent].pop(sid)

        target_host.restore()

        # Re-inject only the sessions that survived the restore (present in restored host.sessions).
        # Red sessions and any out-of-sync references are discarded.
        for agent, sessions in target_host.sessions.items():
            for sid in sessions:
                if agent in all_host_sessions and sid in all_host_sessions[agent]:
                    state.sessions[agent][sid] = all_host_sessions[agent][sid]
        return Observation()
