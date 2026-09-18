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

        # Restore permanently removes every non-baseline session on the target
        # host.  Keep all redundant session indexes consistent and disconnect
        # descendants whose pivot/parent disappeared with the reimage.
        permanently_removed_by_agent = {}
        for agent, removed in all_host_sessions.items():
            surviving_ids = set(state.sessions[agent])
            permanently_removed = set(removed) - surviving_ids
            permanently_removed_by_agent[agent] = permanently_removed
            if permanently_removed:
                state.sessions_count[agent] -= len(permanently_removed)

        # A session reached through a removed pivot is no longer usable. Remove
        # these descendants transitively instead of leaving inactive sessions
        # in state.sessions (RedSessionCheck requires that index to contain only
        # live sessions).
        for agent, sessions in state.sessions.items():
            removed_ids = permanently_removed_by_agent.get(agent, set())
            while True:
                orphan_ids = {
                    sid for sid, session in sessions.items()
                    if session.parent in removed_ids
                }
                if not orphan_ids:
                    break
                for sid in orphan_ids:
                    orphan = sessions.pop(sid)
                    host_sessions = state.hosts[orphan.hostname].sessions.get(agent, [])
                    if sid in host_sessions:
                        host_sessions.remove(sid)
                state.sessions_count[agent] -= len(orphan_ids)
                removed_ids.update(orphan_ids)

        # Child mappings can survive agent reassignment and may therefore point
        # across agent dictionaries. Validate them by object identity against
        # the authoritative state rather than by id within the parent's agent.
        for agent, sessions in state.sessions.items():
            for session in sessions.values():
                for child_id, child in list(session.children.items()):
                    live_child = state.sessions.get(child.agent, {}).get(child.ident)
                    if live_child is not child:
                        session.dead_child(child_id)

        # Session removals invalidate State's cached pid-to-session lookup.
        state._pid_index_dirty = True
        return Observation()
