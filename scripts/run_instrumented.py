#!/usr/bin/env python3
"""Instrumented episode runner for 3D visualization dashboard telemetry.

Usage: python scripts/run_instrumented.py --episodes 3 --steps 100 --seed 42
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

# Ensure project root importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _get_state(env):
    """Return internal CybORG State via multiple wrapper paths."""
    for attr_chain in ("env.environment_controller.state",
                       "environment_controller.state"):
        obj = env
        try:
            for part in attr_chain.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


def _hostname_subnet(state, hostname: str) -> str:
    """Map hostname to subnet name string."""
    try:
        sn = state.hostname_subnet_map.get(hostname)
        return str(sn).lower() if sn else ""
    except Exception:
        return ""


_RE_BLOCK = re.compile(
    r"BlockTrafficZone\s+(\S+)\s+(\S+)\s*<-\s*(\S+)", re.IGNORECASE
)
_RE_ALLOW = re.compile(
    r"AllowTrafficZone\s+(\S+)\s+(\S+)\s*<-\s*(\S+)", re.IGNORECASE
)
_RE_HOST_ACTION = re.compile(
    r"(Remove|Restore|DeployDecoy)\s+(\S+)", re.IGNORECASE
)


def _parse_action_label(label: str):
    """Return (action_type, target_host, target_subnet)."""
    label = label.strip()
    if label == "Sleep":
        return "Sleep", None, None

    m = _RE_BLOCK.match(label)
    if m:
        return "BlockTrafficZone", m.group(2), m.group(1)

    m = _RE_ALLOW.match(label)
    if m:
        return "AllowTrafficZone", m.group(2), m.group(1)

    m = _RE_HOST_ACTION.match(label)
    if m:
        return m.group(1), m.group(2), None

    return label.split()[0] if label else "Unknown", None, None


def _build_reasoning(action_type: str, target_host, target_subnet, phase: int) -> str:
    """Human-readable reasoning string for the action."""
    if action_type == "Sleep":
        return "No threats detected, conserving resources"
    if action_type == "BlockTrafficZone":
        return f"Enforcing comms policy phase {phase}: isolating {target_subnet}"
    if action_type == "AllowTrafficZone":
        return f"Relaxing stale block per comms policy on {target_subnet}"
    if action_type == "Remove":
        return f"Removing suspected user-level compromise on {target_host}"
    if action_type == "Restore":
        return f"Restoring {target_host} -- root-level compromise or Remove failed"
    if action_type == "DeployDecoy":
        return f"Deploying decoy on {target_host} to trap red exploits"
    return f"Executing {action_type}"


def _collect_host_states(state):
    """Yield dicts for every host in the environment."""
    for hostname, host in state.hosts.items():
        subnet = _hostname_subnet(state, hostname)
        red_sessions = []
        blue_sessions = []
        for agent_name, sess_dict in state.sessions.items():
            for _sid, sess in sess_dict.items():
                if str(getattr(sess, "hostname", "")) != str(hostname):
                    continue
                if not getattr(sess, "active", False):
                    continue
                if "red" in str(agent_name).lower():
                    red_sessions.append(sess)
                elif "blue" in str(agent_name).lower():
                    blue_sessions.append(sess)

        has_root = any(s.has_privileged_access() for s in red_sessions)
        has_user = len(red_sessions) > 0 and not has_root
        if has_root:
            level = "root"
        elif has_user:
            level = "user"
        else:
            level = "none"

        num_procs = 0
        try:
            num_procs = len(list(host.processes.values())) if hasattr(host, "processes") else 0
        except Exception:
            pass

        num_conns = 0
        try:
            conns = getattr(host.events, "network_connections", [])
            num_conns = len(list(conns))
        except Exception:
            pass

        # Count decoy processes
        decoy_count = 0
        try:
            if hasattr(host, "processes") and isinstance(host.processes, dict):
                decoy_count = sum(1 for p in host.processes.values()
                                  if getattr(p, "decoy_type", None) is not None)
        except Exception:
            pass

        # Restore tracking: restore_count > 0 means host was recently restored
        restore_count = getattr(host, "restore_count", 0)
        impact_count = getattr(host, "impact_count", 0)

        # Service reliability approximation: 0% if impacted, else 100%
        reliability = 0.0 if impact_count > 0 else 100.0

        yield {
            "host_name": str(hostname),
            "subnet_name": subnet,
            "compromised_level": level,
            "has_red_session": len(red_sessions) > 0,
            "has_blue_session": len(blue_sessions) > 0,
            "num_processes": num_procs,
            "num_connections": num_conns,
            "has_malware": has_root,
            "decoy_count": decoy_count,
            "is_restoring": restore_count > 0,
            "is_being_removed": False,
            "service_reliability_pct": reliability,
        }


def _collect_sessions(state, step: int):
    """Yield tuples for all active sessions."""
    for agent_name, sess_dict in state.sessions.items():
        for sid, sess in sess_dict.items():
            hostname = str(getattr(sess, "hostname", ""))
            stype = str(getattr(sess, "session_type", "unknown"))
            priv = "root" if sess.has_privileged_access() else "user"
            active = bool(getattr(sess, "active", False))
            parent = getattr(sess, "parent", None)
            yield (sid, str(agent_name), hostname, stype, priv, active, parent)


def _collect_traffic(state):
    """Yield traffic block status for all subnet pairs.

    state.blocks is {subnet_name_str: [blocked_source_name_str, ...]}.
    state.subnets keys are IPv4Network objects (CIDR), not names.
    We use the hostname_subnet_map to get subnet name strings.
    """
    blocks = getattr(state, "blocks", {})
    # Collect all known subnet name strings from blocks + hostname_subnet_map
    subnet_names = set(blocks.keys())
    hsm = getattr(state, "hostname_subnet_map", {})
    for sn in hsm.values():
        subnet_names.add(str(sn))
    subnet_names = sorted(subnet_names)

    for dst in subnet_names:
        blocked_sources = blocks.get(dst, [])
        blocked_set = {str(b) for b in blocked_sources}
        for src in subnet_names:
            if src == dst:
                continue
            yield {
                "source_subnet": src,
                "dest_subnet": dst,
                "is_blocked": src in blocked_set,
                "should_be_blocked": False,
            }


def _infer_beliefs(obs, agent, agent_name):
    """Infer blue agent beliefs from observation flags."""
    beliefs = []
    if len(obs) < 2:
        return beliefs
    base = 1
    for sn in agent._subnets_in_obs:
        hosts = agent._subnet_host_list.get(sn, [])
        n_hosts = len(hosts)
        off_proc = 27
        off_conn = 27 + n_hosts

        proc_flags = obs[base + off_proc: base + off_proc + n_hosts]
        conn_flags = obs[base + off_conn: base + off_conn + n_hosts]

        for hi, hostname in enumerate(hosts):
            pf = proc_flags[hi] if hi < len(proc_flags) else 0
            cf = conn_flags[hi] if hi < len(conn_flags) else 0
            if pf and cf:
                beliefs.append((hostname, "suspected_compromised", 0.9,
                                {"process_flag": True, "connection_flag": True}))
            elif pf:
                beliefs.append((hostname, "suspected_compromised", 0.8,
                                {"process_flag": True}))
            elif cf:
                beliefs.append((hostname, "network_activity_detected", 0.6,
                                {"connection_flag": True}))
            else:
                beliefs.append((hostname, "presumed_clean", 0.5, None))
        base += 27 + 2 * n_hosts
    return beliefs


def _extract_topology(env, state):
    """Build topology list from env and state."""
    topo = []
    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
    if subnet_hosts:
        for sn, hosts in subnet_hosts.items():
            for h in hosts:
                topo.append({"subnet_name": str(sn), "host_name": str(h),
                             "host_type": "server" if "server" in str(h) else "user"})
    elif state and hasattr(state, "hosts"):
        for hostname in state.hosts:
            subnet = _hostname_subnet(state, str(hostname))
            topo.append({"subnet_name": subnet, "host_name": str(hostname),
                         "host_type": "server" if "server" in str(hostname) else "user"})
    return topo


def run_instrumented(n_episodes: int, max_steps: int, seed: int, agent_type: str):
    try:
        from CybORG import CybORG
        from CybORG.Agents.Wrappers import BlueFlatWrapper
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
        from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
        from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent
        from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import make_heuristic_agents
    except ImportError as e:
        print(f"ERROR: Could not import CybORG components: {e}")
        print("Ensure CybORG is installed and the project root is on PYTHONPATH.")
        sys.exit(1)

    from src.database.telemetry_collector import TelemetryDB

    db = TelemetryDB()
    print(f"Telemetry DB: {db.db_path}")
    print(f"Running {n_episodes} episodes, {max_steps} steps, seed={seed}, agent={agent_type}")
    print("-" * 60)

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg_raw = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapper(env=cyborg_raw)

    all_rewards = []
    t_start = time.perf_counter()

    for ep in range(n_episodes):
        episode_id = f"ep_{seed}_{ep}_{uuid.uuid4().hex[:8]}"
        ep_seed = seed + ep

        obs_dict, _ = env.reset()

        agents = make_heuristic_agents(env)
        agent_names = env.possible_agents
        subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
        for name, ag in agents.items():
            ag.reset()
            ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)

        db.start_episode(episode_id, ep_seed, agent_type, "FiniteStateRedAgent", max_steps)

        state = _get_state(env)
        topo = _extract_topology(env, state)
        if topo:
            db.log_topology(episode_id, topo)

        total_reward = 0.0
        key_events = defaultdict(int)

        for step in range(max_steps):
            state = _get_state(env)
            phase = int(state.mission_phase) if state else 0

            actions = {}
            messages = {}

            # --- Agent decisions ---
            for name, ag in agents.items():
                raw_obs = obs_dict.get(name, np.zeros(1))
                mask = env.action_mask(name)
                obs_arr = np.asarray(raw_obs, dtype=np.float32)

                action_idx, msg = ag.get_action(obs_arr, np.array(mask, dtype=bool))
                actions[name] = action_idx
                messages[name] = msg

                label = ag._labels[action_idx] if action_idx < len(ag._labels) else "Sleep"
                action_type, target_host, target_subnet = _parse_action_label(label)
                reasoning = _build_reasoning(action_type, target_host, target_subnet, phase)

                db.log_action(
                    episode_id, step, name, "blue",
                    action_type, label, target_host, target_subnet,
                    action_idx, success=True, duration=0.0,
                    reasoning={"text": reasoning},
                )

                if action_type != "Sleep":
                    key_events[action_type] += 1

                # Log message bits
                msg_int = int(sum(int(b) << i for i, b in enumerate(msg))) if msg is not None else 0
                if msg_int:
                    db.log_message(episode_id, step, name, msg_int)

                # Log beliefs from observation
                beliefs = _infer_beliefs(obs_arr, ag, name)
                for hostname, btype, conf, evidence in beliefs:
                    if btype != "presumed_clean":
                        db.log_belief(episode_id, step, name, btype, hostname, conf, evidence)

            # --- Step environment ---
            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions, messages=messages)
            step_reward = sum(rew_dict.values())
            total_reward += step_reward

            db.log_step(episode_id, step, phase, step_reward, total_reward)

            # --- Per-agent reward breakdown ---
            for name, r in rew_dict.items():
                if r != 0:
                    db.log_reward_breakdown(episode_id, step, name, "step_reward", r)

            # --- Red/Green agent actions from environment controller ---
            state = _get_state(env)
            if state:
                ec = getattr(env, "env", None)
                ec = getattr(ec, "environment_controller", None) if ec else None
                if ec and hasattr(ec, "action"):
                    for ag_name, act_list in ec.action.items():
                        ag_str = str(ag_name)
                        if "blue" in ag_str:
                            continue  # already logged above
                        team = "red" if "red" in ag_str else "green"
                        act_label = str(act_list[0]) if act_list else "Sleep"
                        act_type = act_label.split()[0] if act_label else "Sleep"
                        target = None
                        parts = act_label.split()
                        if len(parts) > 1:
                            target = parts[1]
                        db.log_action(
                            episode_id, step, ag_str, team,
                            act_type, act_label, target, None,
                            -1, success=True, duration=0.0,
                            reasoning={"text": f"{team} agent action"},
                        )

            # --- Ground truth: host states ---
            if state:
                host_states = list(_collect_host_states(state))
                if host_states:
                    db.log_host_states_batch(episode_id, step, host_states)

                # --- Sessions ---
                for sid, ag_name, hostname, stype, priv, active, parent in _collect_sessions(state, step):
                    db.log_session(episode_id, step, sid, ag_name, hostname, stype, priv, active, parent)

                # --- Traffic ---
                traffic = list(_collect_traffic(state))
                if traffic:
                    db.log_traffic_batch(episode_id, step, traffic)

            if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in agent_names):
                # Log the final step+1 state snapshot before breaking
                final_state = _get_state(env)
                if final_state:
                    final_phase = int(final_state.mission_phase) if final_state else phase
                    db.log_step(episode_id, step + 1, final_phase, 0.0, total_reward)
                    fhs = list(_collect_host_states(final_state))
                    if fhs:
                        db.log_host_states_batch(episode_id, step + 1, fhs)
                break

        db.end_episode(episode_id, total_reward, "completed")
        all_rewards.append(total_reward)

        events_str = ", ".join(f"{k}={v}" for k, v in sorted(key_events.items()))
        print(f"  Episode {ep + 1}/{n_episodes}  reward={total_reward:9.1f}  "
              f"steps={step + 1}  [{events_str}]")

    elapsed = time.perf_counter() - t_start
    mean_r = float(np.mean(all_rewards))
    std_r = float(np.std(all_rewards)) if len(all_rewards) > 1 else 0.0

    print("-" * 60)
    print(f"Completed {n_episodes} episodes in {elapsed:.1f}s")
    print(f"Mean reward: {mean_r:.1f} +/- {std_r:.1f}")
    print(f"Min/Max: {min(all_rewards):.1f} / {max(all_rewards):.1f}")
    print(f"Telemetry DB: {Path(db.db_path).resolve()}")

    db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run instrumented CC4 episodes for 3D visualization telemetry."
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--agent", type=str, default="heuristic",
                        choices=["heuristic"], help="Agent type")
    args = parser.parse_args()
    run_instrumented(args.episodes, args.steps, args.seed, args.agent)


if __name__ == "__main__":
    main()
