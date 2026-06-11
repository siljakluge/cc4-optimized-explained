#!/usr/bin/env python3
"""Experimental validation of DegradeServices dynamics in CAGE Challenge 4.

Four experiments testing degradation visibility, timing gaps, targeted proactive
Restore on OZ hosts, and flag_age=0 for active OZ hosts.

Usage:
    python scripts/degrade_experiment.py --experiment 1     # degradation visibility
    python scripts/degrade_experiment.py --experiment 2     # timing gap
    python scripts/degrade_experiment.py --experiment 3     # proactive OZ Restore
    python scripts/degrade_experiment.py --experiment 4     # flag_age=0 for OZ
    python scripts/degrade_experiment.py --experiment all   # run all
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PHASE_1_START = 167
PHASE_2_START = 334
MAX_STEPS = 500

_SORTED_SUBNETS = [
    "admin_network_subnet",
    "contractor_network_subnet",
    "internet_subnet",
    "office_network_subnet",
    "operational_zone_a_subnet",
    "operational_zone_b_subnet",
    "public_access_zone_subnet",
    "restricted_zone_a_subnet",
    "restricted_zone_b_subnet",
]

_OFF_PROC = 27  # offset within per-subnet obs block to process flags


def _get_phase(step: int) -> int:
    if step < PHASE_1_START:
        return 0
    elif step < PHASE_2_START:
        return 1
    return 2


def _create_env(seed: int = 42, max_steps: int = MAX_STEPS):
    """Create CybORG + BlueFlatWrapperV2 environment."""
    from CybORG import CybORG
    from CybORG.Agents.Wrappers import BlueFlatWrapperV2
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapperV2(env=cyborg)
    return cyborg, env


def _create_agents_v11a(env):
    """Create V11a agents."""
    from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11a import (
        make_heuristic_agents_v11a,
    )
    return make_heuristic_agents_v11a(env)


def _reset_episode(env, agents):
    """Reset environment and agents, return initial obs_dict."""
    obs_dict, _ = env.reset()
    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
    for name, ag in agents.items():
        ag.reset()
        ag.set_action_info(env.action_labels(name), env.action_mask(name), subnet_hosts)
    return obs_dict


def _step_env(env, agents, obs_dict):
    """Step one tick: gather actions from agents, step env, return new obs + reward."""
    actions = {}
    messages = {}
    for name, ag in agents.items():
        raw_obs = obs_dict.get(name, np.zeros(1))
        mask = env.action_mask(name)
        action_idx, msg = ag.get_action(raw_obs, np.array(mask, dtype=bool))
        actions[name] = action_idx
        messages[name] = msg
    obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions, messages=messages)
    return obs_dict, rew_dict, term_dict, trunc_dict, actions


def _get_state(env):
    """Get the internal simulation state from wrapped env."""
    return env.env.environment_controller.state


def _get_sim_controller(env):
    """Get the simulation controller from wrapped env."""
    return env.env.environment_controller


def _get_all_hostnames(state) -> list[str]:
    """Return all host names in the simulation."""
    return list(state.hosts.keys())


def _get_service_reliability(state, hostname: str) -> dict[str, int]:
    """Return {service_name: percent_reliable} for a host."""
    host = state.hosts.get(hostname)
    if host is None:
        return {}
    result = {}
    for sname, svc in host.services.items():
        result[sname] = svc._percent_reliable
    return result


def _red_has_root(state, hostname: str) -> bool:
    """Check if any red agent has privileged access on hostname."""
    for agent_name, sessions in state.sessions.items():
        if "red" not in agent_name.lower():
            continue
        for sid, sess in sessions.items():
            if sess.hostname == hostname and sess.has_privileged_access():
                return True
    return False


def _red_has_any_session(state, hostname: str) -> bool:
    """Check if any red agent has any session on hostname."""
    for agent_name, sessions in state.sessions.items():
        if "red" not in agent_name.lower():
            continue
        for sid, sess in sessions.items():
            if sess.hostname == hostname:
                return True
    return False


def _get_last_red_actions(sim_controller) -> list[str]:
    """Get string representations of last executed red actions."""
    results = []
    action_dict = getattr(sim_controller, "action", {})
    for agent_name, actions in action_dict.items():
        if "red" not in agent_name.lower():
            continue
        if isinstance(actions, list):
            for a in actions:
                results.append(f"{agent_name}: {a}")
        else:
            results.append(f"{agent_name}: {actions}")
    return results


def _get_blue_obs_flags(obs_dict, agents) -> dict[str, dict[str, tuple[bool, bool, bool]]]:
    """Parse blue observations to extract per-host (conn, proc, malfile) flags.

    Returns: {agent_name: {hostname: (conn_flag, proc_flag, malfile_flag)}}
    """
    result = {}
    for name, ag in agents.items():
        raw_obs = obs_dict.get(name, np.zeros(1))
        obs = np.asarray(raw_obs, dtype=np.float32)
        if len(obs) <= 1:
            continue

        host_flags = {}
        base = 1

        # Compute expected base length without malfile
        n_malfile_hosts = sum(
            len(ag._subnet_host_list.get(sn, []))
            for sn in ag._subnets_in_obs
        )
        base_subnet_len = sum(
            27 + 2 * len(ag._subnet_host_list.get(sn, []))
            for sn in ag._subnets_in_obs
        )
        expected_base_len = 1 + base_subnet_len + 32  # 32 = NUM_MSG_BITS
        has_malfile = (n_malfile_hosts > 0 and
                       len(obs) == expected_base_len + n_malfile_hosts)
        malfile_start = expected_base_len if has_malfile else len(obs)

        malfile_cursor = malfile_start
        for sn in ag._subnets_in_obs:
            hosts = ag._subnet_host_list.get(sn, [])
            n_hosts = len(hosts)
            off_conn = 27 + n_hosts

            proc_flags = obs[base + 27: base + 27 + n_hosts]
            conn_flags = obs[base + off_conn: base + off_conn + n_hosts]

            if has_malfile:
                malfile_vec = obs[malfile_cursor: malfile_cursor + n_hosts]
                malfile_cursor += n_hosts
            else:
                malfile_vec = np.zeros(n_hosts)

            for hi, hostname in enumerate(hosts):
                pf = bool(proc_flags[hi]) if hi < len(proc_flags) else False
                cf = bool(conn_flags[hi]) if hi < len(conn_flags) else False
                mf = bool(malfile_vec[hi]) if hi < len(malfile_vec) else False
                host_flags[hostname] = (cf, pf, mf)

            base += 27 + 2 * n_hosts

        result[name] = host_flags
    return result


def _any_blue_alert(obs_flags: dict[str, dict[str, tuple[bool, bool, bool]]],
                    hostname: str) -> tuple[bool, bool, bool]:
    """Return (any_conn, any_proc, any_malfile) across all agents for a host."""
    any_c, any_p, any_m = False, False, False
    for agent_name, hf in obs_flags.items():
        if hostname in hf:
            c, p, m = hf[hostname]
            any_c = any_c or c
            any_p = any_p or p
            any_m = any_m or m
    return any_c, any_p, any_m


def _get_blue_action_labels(actions: dict[str, int], agents) -> dict[str, str]:
    """Map blue agent action indices to labels."""
    result = {}
    for name, idx in actions.items():
        ag = agents.get(name)
        if ag and idx < len(ag._labels):
            result[name] = ag._labels[idx]
        else:
            result[name] = f"idx={idx}"
    return result


# ---------------------------------------------------------------------------
# Experiment 1: Measure actual degradation visibility
# ---------------------------------------------------------------------------

def experiment_1(n_episodes: int = 10, seed: int = 42):
    """Prove/disprove whether degradation produces ANY observable signal in blue's obs."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Degradation Visibility Analysis")
    print("=" * 70)

    cyborg, env = _create_env(seed=seed)
    agents = _create_agents_v11a(env)

    # Tracking
    total_degrade_events = 0
    degrade_with_any_alert = 0
    degrade_with_conn = 0
    degrade_with_proc = 0
    degrade_with_malfile = 0
    degrade_invisible = 0

    # Per-host tracking
    host_degrade_counts = defaultdict(int)
    host_degrade_invisible = defaultdict(int)

    # Correlation: track reliability drop vs observation signals
    reliability_drop_records = []  # list of dicts

    for ep in range(n_episodes):
        obs_dict = _reset_episode(env, agents)
        state = _get_state(env)
        sim = _get_sim_controller(env)
        all_hosts = _get_all_hostnames(state)

        # Track previous reliability to detect drops
        prev_reliability = {}
        for h in all_hosts:
            prev_reliability[h] = _get_service_reliability(state, h)

        ep_reward = 0.0
        for step in range(MAX_STEPS):
            obs_dict, rew_dict, term_dict, trunc_dict, blue_actions = _step_env(
                env, agents, obs_dict
            )
            ep_reward += sum(rew_dict.values())
            state = _get_state(env)

            # Get current reliability
            curr_reliability = {}
            for h in all_hosts:
                curr_reliability[h] = _get_service_reliability(state, h)

            # Get blue observation flags
            obs_flags = _get_blue_obs_flags(obs_dict, agents)

            # Get last red actions
            red_actions = _get_last_red_actions(sim)

            # Check for degradation events (reliability dropped)
            for h in all_hosts:
                prev_r = prev_reliability.get(h, {})
                curr_r = curr_reliability.get(h, {})
                for sname in curr_r:
                    prev_val = prev_r.get(sname, 100)
                    curr_val = curr_r[sname]
                    if curr_val < prev_val:
                        # Degradation occurred
                        total_degrade_events += 1
                        host_degrade_counts[h] += 1

                        any_c, any_p, any_m = _any_blue_alert(obs_flags, h)
                        has_root = _red_has_root(state, h)

                        if any_c or any_p or any_m:
                            degrade_with_any_alert += 1
                        else:
                            degrade_invisible += 1
                            host_degrade_invisible[h] += 1

                        if any_c:
                            degrade_with_conn += 1
                        if any_p:
                            degrade_with_proc += 1
                        if any_m:
                            degrade_with_malfile += 1

                        # Find which red action caused this
                        red_action_str = "; ".join(
                            a for a in red_actions if h in a
                        ) or "unknown"

                        reliability_drop_records.append({
                            "ep": ep,
                            "step": step,
                            "host": h,
                            "service": sname,
                            "prev_reliability": prev_val,
                            "curr_reliability": curr_val,
                            "conn_flag": any_c,
                            "proc_flag": any_p,
                            "malfile_flag": any_m,
                            "red_has_root": has_root,
                            "red_action": red_action_str,
                            "phase": _get_phase(step),
                        })

            prev_reliability = curr_reliability

            if all(
                term_dict.get(n, False) or trunc_dict.get(n, False)
                for n in env.possible_agents
            ):
                break

        print(f"  ep {ep + 1:3d}/{n_episodes}  reward={ep_reward:9.1f}  "
              f"degrade_events_so_far={total_degrade_events}")

    # Report
    print("\n" + "-" * 70)
    print("  RESULTS: Degradation Visibility")
    print("-" * 70)
    print(f"  Total degradation events:          {total_degrade_events}")
    print(f"  With ANY blue alert (conn/proc/mf): {degrade_with_any_alert} "
          f"({degrade_with_any_alert / max(total_degrade_events, 1) * 100:.1f}%)")
    print(f"    - with conn flag:                 {degrade_with_conn}")
    print(f"    - with proc flag:                 {degrade_with_proc}")
    print(f"    - with malfile flag:              {degrade_with_malfile}")
    print(f"  INVISIBLE (no alert at all):        {degrade_invisible} "
          f"({degrade_invisible / max(total_degrade_events, 1) * 100:.1f}%)")

    print(f"\n  Top 10 hosts by degradation count:")
    for h, cnt in sorted(host_degrade_counts.items(), key=lambda x: -x[1])[:10]:
        invis = host_degrade_invisible.get(h, 0)
        print(f"    {h:50s}  total={cnt:4d}  invisible={invis:4d} "
              f"({invis / max(cnt, 1) * 100:.0f}%)")

    # Show a sample of degradation records
    print(f"\n  Sample degradation events (first 20):")
    print(f"  {'ep':>3s} {'step':>4s} {'phase':>5s} {'host':40s} "
          f"{'rel':>7s} {'conn':>4s} {'proc':>4s} {'malf':>4s} {'root':>4s} {'red_action'}")
    for rec in reliability_drop_records[:20]:
        print(f"  {rec['ep']:3d} {rec['step']:4d} {rec['phase']:5d} "
              f"{rec['host']:40s} "
              f"{rec['prev_reliability']:3d}->{rec['curr_reliability']:3d} "
              f"{'Y' if rec['conn_flag'] else 'N':>4s} "
              f"{'Y' if rec['proc_flag'] else 'N':>4s} "
              f"{'Y' if rec['malfile_flag'] else 'N':>4s} "
              f"{'Y' if rec['red_has_root'] else 'N':>4s} "
              f"{rec['red_action'][:50]}")

    # Correlation analysis
    if reliability_drop_records:
        # How often does degradation happen with vs without prior conn/proc/malfile
        print(f"\n  CONCLUSION:")
        if degrade_invisible > total_degrade_events * 0.5:
            print(f"  >>> DEGRADATION IS MOSTLY INVISIBLE to blue agents.")
            print(f"  >>> {degrade_invisible}/{total_degrade_events} events had NO observable signal.")
            print(f"  >>> Blue cannot detect DegradeServices directly from observations.")
        elif degrade_invisible > 0:
            print(f"  >>> DEGRADATION IS PARTIALLY VISIBLE.")
            print(f"  >>> {degrade_with_any_alert}/{total_degrade_events} events had some alert.")
            print(f"  >>> But {degrade_invisible} events were fully invisible.")
        else:
            print(f"  >>> ALL degradation events had observable signals.")

    return reliability_drop_records


# ---------------------------------------------------------------------------
# Experiment 2: Measure the timing gap
# ---------------------------------------------------------------------------

def experiment_2(n_episodes: int = 10, seed: int = 42):
    """Quantify the exact timing gap between red root, degradation, and blue response."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Timing Gap Analysis")
    print("=" * 70)

    cyborg, env = _create_env(seed=seed)
    agents = _create_agents_v11a(env)

    # Per-host per-episode tracking
    # {(ep, host): {first_root, first_degrade, first_alert, first_restore,
    #               restore_complete, degrade_count_before_restore}}
    host_timelines = defaultdict(lambda: {
        "first_root": None,
        "first_degrade": None,
        "first_alert": None,
        "first_restore_issued": None,
        "restore_complete": None,
        "degrade_calls_before_restore": 0,
        "degrade_calls_total": 0,
        "min_reliability": 100,
    })

    all_gap_root_to_degrade = []
    all_gap_degrade_to_alert = []
    all_gap_alert_to_restore = []
    all_gap_root_to_restore = []
    all_degrade_before_restore = []

    for ep in range(n_episodes):
        obs_dict = _reset_episode(env, agents)
        state = _get_state(env)
        sim = _get_sim_controller(env)
        all_hosts = _get_all_hostnames(state)

        prev_reliability = {}
        for h in all_hosts:
            prev_reliability[h] = _get_service_reliability(state, h)

        # Track which hosts had root last step (to detect first root)
        prev_root = {h: False for h in all_hosts}

        ep_reward = 0.0
        for step in range(MAX_STEPS):
            obs_dict, rew_dict, term_dict, trunc_dict, blue_actions = _step_env(
                env, agents, obs_dict
            )
            ep_reward += sum(rew_dict.values())
            state = _get_state(env)

            curr_reliability = {}
            for h in all_hosts:
                curr_reliability[h] = _get_service_reliability(state, h)

            obs_flags = _get_blue_obs_flags(obs_dict, agents)
            blue_labels = _get_blue_action_labels(blue_actions, agents)

            for h in all_hosts:
                key = (ep, h)
                tl = host_timelines[key]

                # Check root
                has_root = _red_has_root(state, h)
                if has_root and tl["first_root"] is None:
                    tl["first_root"] = step

                # Check degradation
                prev_r = prev_reliability.get(h, {})
                curr_r = curr_reliability.get(h, {})
                for sname in curr_r:
                    if curr_r[sname] < prev_r.get(sname, 100):
                        tl["degrade_calls_total"] += 1
                        if tl["first_degrade"] is None:
                            tl["first_degrade"] = step
                        if tl["first_restore_issued"] is None:
                            tl["degrade_calls_before_restore"] += 1
                        tl["min_reliability"] = min(tl["min_reliability"],
                                                     curr_r[sname])

                # Check blue alerts
                any_c, any_p, any_m = _any_blue_alert(obs_flags, h)
                if (any_c or any_p or any_m) and tl["first_alert"] is None:
                    tl["first_alert"] = step

                # Check if blue issued Restore on this host
                for ag_name, label in blue_labels.items():
                    if label.startswith("Restore") and h in label:
                        if tl["first_restore_issued"] is None:
                            tl["first_restore_issued"] = step
                            tl["restore_complete"] = step + 5  # RESTORE_DUR=5

                prev_root[h] = has_root

            prev_reliability = curr_reliability

            if all(
                term_dict.get(n, False) or trunc_dict.get(n, False)
                for n in env.possible_agents
            ):
                break

        print(f"  ep {ep + 1:3d}/{n_episodes}  reward={ep_reward:9.1f}")

    # Aggregate timing gaps for hosts that experienced degradation
    degraded_hosts = []
    for key, tl in host_timelines.items():
        if tl["first_degrade"] is not None:
            degraded_hosts.append((key, tl))

            if tl["first_root"] is not None and tl["first_degrade"] is not None:
                all_gap_root_to_degrade.append(tl["first_degrade"] - tl["first_root"])

            if tl["first_degrade"] is not None and tl["first_alert"] is not None:
                gap = tl["first_alert"] - tl["first_degrade"]
                all_gap_degrade_to_alert.append(gap)

            if tl["first_alert"] is not None and tl["first_restore_issued"] is not None:
                all_gap_alert_to_restore.append(
                    tl["first_restore_issued"] - tl["first_alert"]
                )

            if tl["first_root"] is not None and tl["first_restore_issued"] is not None:
                all_gap_root_to_restore.append(
                    tl["first_restore_issued"] - tl["first_root"]
                )

            all_degrade_before_restore.append(tl["degrade_calls_before_restore"])

    # Report
    print("\n" + "-" * 70)
    print("  RESULTS: Timing Gap Analysis")
    print("-" * 70)
    print(f"  Total host-episodes with degradation: {len(degraded_hosts)}")

    def _stats(label, data):
        if not data:
            print(f"  {label}: no data")
            return
        arr = np.array(data)
        print(f"  {label}:")
        print(f"    mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
              f"std={arr.std():.1f}  min={arr.min():.0f}  max={arr.max():.0f}  n={len(arr)}")

    _stats("Gap: root -> first DegradeServices", all_gap_root_to_degrade)
    _stats("Gap: first degrade -> first blue alert", all_gap_degrade_to_alert)
    _stats("Gap: first alert -> first Restore issued", all_gap_alert_to_restore)
    _stats("Gap: root -> first Restore issued", all_gap_root_to_restore)
    _stats("DegradeServices calls before first Restore", all_degrade_before_restore)

    # Show worst-case examples
    print(f"\n  Worst-case host timelines (top 10 by degrade count):")
    worst = sorted(degraded_hosts, key=lambda x: -x[1]["degrade_calls_total"])[:10]
    print(f"  {'ep':>3s} {'host':40s} {'root':>5s} {'1st_deg':>7s} "
          f"{'1st_alrt':>8s} {'1st_rst':>7s} {'deg_tot':>7s} {'deg_b4r':>7s} "
          f"{'min_rel':>7s}")
    for (ep, h), tl in worst:
        print(f"  {ep:3d} {h:40s} "
              f"{str(tl['first_root']) if tl['first_root'] is not None else 'N/A':>5s} "
              f"{str(tl['first_degrade']) if tl['first_degrade'] is not None else 'N/A':>7s} "
              f"{str(tl['first_alert']) if tl['first_alert'] is not None else 'N/A':>8s} "
              f"{str(tl['first_restore_issued']) if tl['first_restore_issued'] is not None else 'N/A':>7s} "
              f"{tl['degrade_calls_total']:>7d} "
              f"{tl['degrade_calls_before_restore']:>7d} "
              f"{tl['min_reliability']:>7d}")

    # Hosts that were degraded but NEVER restored
    never_restored = [(k, tl) for k, tl in degraded_hosts
                      if tl["first_restore_issued"] is None]
    print(f"\n  Hosts degraded but NEVER restored: {len(never_restored)}/{len(degraded_hosts)}")
    if never_restored:
        for (ep, h), tl in sorted(never_restored,
                                   key=lambda x: -x[1]["degrade_calls_total"])[:5]:
            print(f"    ep={ep} {h:40s} degrade_total={tl['degrade_calls_total']} "
                  f"min_reliability={tl['min_reliability']}%")

    return host_timelines


# ---------------------------------------------------------------------------
# Experiment 3: Proactive Restore on OZ hosts after compromise
# ---------------------------------------------------------------------------

def _create_agents_v11a_proactive_oz(env, proactive_interval: int = 15):
    """Create V11a agents with proactive OZ Restore modification.

    After Restore completes on an active OZ host that was previously compromised,
    issue another Restore after `proactive_interval` steps even without new alerts.
    """
    from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11a import (
        EnterpriseHeuristicAgentV11a,
        _sorted_by_priority,
        _is_active_oz_server,
        RESTORE_DUR,
        MAX_DECOYS,
    )

    class ProactiveOZAgent(EnterpriseHeuristicAgentV11a):
        """V11a with proactive re-Restore on active OZ hosts."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._proactive_interval = proactive_interval
            # Track hosts that were compromised and restored
            self._oz_last_restore_complete: dict[str, int] = {}

        def reset(self):
            super().reset()
            self._oz_last_restore_complete.clear()

        def get_action(self, observation, action_mask=None):
            # Track when OZ restores complete
            for h, restore_step in list(self._restore_at.items()):
                if _is_active_oz_server(h, _get_phase(self._step)):
                    complete_step = restore_step + RESTORE_DUR
                    if self._step == complete_step:
                        self._oz_last_restore_complete[h] = self._step

            # Check if we should proactively re-Restore an OZ host
            phase = _get_phase(self._step + 1)  # +1 because _step incremented in parent
            for h, complete_step in list(self._oz_last_restore_complete.items()):
                if not _is_active_oz_server(h, phase):
                    continue
                if self._step >= complete_step + self._proactive_interval:
                    if not self._busy(h):
                        idx = self._restore.get(h)
                        if idx is not None and (action_mask is None or
                                                (idx < len(action_mask) and action_mask[idx])):
                            # Issue proactive restore
                            self._issue_restore(h)
                            self._oz_last_restore_complete[h] = (
                                self._step + RESTORE_DUR
                            )
                            msg = np.zeros(8, dtype=bool)
                            # Let parent handle messaging for accuracy;
                            # but we need to return immediately
                            return idx, msg

            return super().get_action(observation, action_mask)

    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
    agents = {}
    for agent_name in env.possible_agents:
        ag = ProactiveOZAgent(agent_name=agent_name)
        try:
            ag.set_action_info(
                env.action_labels(agent_name),
                env.action_mask(agent_name),
                subnet_hosts,
            )
        except Exception:
            pass
        agents[agent_name] = ag
    return agents


def experiment_3(n_episodes: int = 30, seed: int = 42):
    """Test targeted proactive Restore on OZ hosts."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Proactive OZ Restore (15-step interval)")
    print("=" * 70)

    # Run baseline (V11a)
    print("\n  --- Baseline V11a ---")
    baseline_rewards = _run_episodes(
        n_episodes, seed, agent_factory=_create_agents_v11a, label="baseline"
    )

    # Run modified (proactive OZ)
    print("\n  --- Modified: Proactive OZ Restore ---")
    modified_rewards = _run_episodes(
        n_episodes, seed,
        agent_factory=lambda env: _create_agents_v11a_proactive_oz(env, 15),
        label="proactive_oz",
    )

    _compare_results("V11a Baseline", baseline_rewards,
                     "Proactive OZ Restore", modified_rewards)


# ---------------------------------------------------------------------------
# Experiment 4: flag_age=0 for active OZ hosts only
# ---------------------------------------------------------------------------

def _create_agents_v11a_oz_flag_age_0(env):
    """Create V11a agents with flag_age=0 for active OZ hosts with LWF=-10."""
    from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11a import (
        EnterpriseHeuristicAgentV11a,
        _sorted_by_priority,
        _is_active_oz_server,
        RESTORE_DUR,
        _OFF_PROC,
    )

    class OZFlagAge0Agent(EnterpriseHeuristicAgentV11a):
        """V11a with flag_age=0 threshold for active OZ hosts."""

        def get_action(self, observation, action_mask=None):
            # We override the P4 logic by monkey-patching the threshold
            # The cleanest approach: override get_action entirely
            # But that duplicates too much code. Instead, we modify
            # _proc_flagged_step for active OZ hosts to make flag_age >= 0 always true.
            # This is a hack: we set their flagged_step to a very old value.
            #
            # Actually, let's just call parent and then check if we missed an OZ host.
            # The parent already handles is_critical_oz with threshold=0.
            # Wait -- checking the parent code:
            #   if peer_escalate_t3 or is_critical_oz:
            #       threshold = 0
            # So _is_active_oz_server hosts already get threshold=0!
            # But _is_active_oz_server only matches server_host_0.
            # Let's expand to ALL hosts in the active OZ subnet.

            return super().get_action(observation, action_mask)

    # Actually, re-reading the V11a code more carefully:
    # _is_active_oz_server checks for "server_host_0" specifically.
    # The experiment asks to use flag_age=0 for ALL hosts in active OZ (LWF=-10).
    # That means all hosts in operational_zone_a during phase 1, and
    # all hosts in operational_zone_b during phase 2.

    # We need to patch _is_active_oz_server at the module level.
    import CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11a as v11a_mod

    original_fn = v11a_mod._is_active_oz_server

    def _is_active_oz_host_all(hostname: str, phase: int) -> bool:
        """Expanded: ANY host in active OZ, not just server_host_0."""
        if phase == 1 and "operational_zone_a" in hostname:
            return True
        if phase == 2 and "operational_zone_b" in hostname:
            return True
        return False

    # Temporarily patch
    v11a_mod._is_active_oz_server = _is_active_oz_host_all

    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})
    agents = {}
    for agent_name in env.possible_agents:
        ag = OZFlagAge0Agent(agent_name=agent_name)
        try:
            ag.set_action_info(
                env.action_labels(agent_name),
                env.action_mask(agent_name),
                subnet_hosts,
            )
        except Exception:
            pass
        agents[agent_name] = ag

    # Restore original function
    v11a_mod._is_active_oz_server = original_fn

    return agents


def _create_agents_v11a_oz_flag_age_0_persistent(env):
    """Like above but keeps the patch active during episode execution."""
    import CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11a as v11a_mod

    def _is_active_oz_host_all(hostname: str, phase: int) -> bool:
        if phase == 1 and "operational_zone_a" in hostname:
            return True
        if phase == 2 and "operational_zone_b" in hostname:
            return True
        return False

    v11a_mod._is_active_oz_server = _is_active_oz_host_all

    agents = _create_agents_v11a(env)
    return agents


def experiment_4(n_episodes: int = 30, seed: int = 42):
    """Test flag_age=0 for all active OZ hosts (not just server_host_0)."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: flag_age=0 for ALL Active OZ Hosts")
    print("=" * 70)

    import CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgentV11a as v11a_mod
    original_fn = v11a_mod._is_active_oz_server

    # Run baseline first (V11a with original _is_active_oz_server)
    print("\n  --- Baseline V11a ---")
    v11a_mod._is_active_oz_server = original_fn
    baseline_rewards = _run_episodes(
        n_episodes, seed, agent_factory=_create_agents_v11a, label="baseline"
    )

    # Run modified
    print("\n  --- Modified: flag_age=0 for ALL active OZ hosts ---")

    def _patched_is_active_oz(hostname: str, phase: int) -> bool:
        if phase == 1 and "operational_zone_a" in hostname:
            return True
        if phase == 2 and "operational_zone_b" in hostname:
            return True
        return False

    v11a_mod._is_active_oz_server = _patched_is_active_oz
    modified_rewards = _run_episodes(
        n_episodes, seed, agent_factory=_create_agents_v11a, label="oz_flag_age_0"
    )

    # Restore original
    v11a_mod._is_active_oz_server = original_fn

    _compare_results("V11a Baseline", baseline_rewards,
                     "OZ flag_age=0 (all hosts)", modified_rewards)


# ---------------------------------------------------------------------------
# Shared evaluation runner and comparison
# ---------------------------------------------------------------------------

def _run_episodes(
    n_episodes: int,
    seed: int,
    agent_factory,
    label: str = "",
    max_steps: int = MAX_STEPS,
) -> list[float]:
    """Run n_episodes and return list of total rewards."""
    cyborg, env = _create_env(seed=seed, max_steps=max_steps)
    agents = agent_factory(env)

    episode_rewards = []
    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs_dict = _reset_episode(env, agents)
        ep_reward = 0.0

        for step in range(max_steps):
            obs_dict, rew_dict, term_dict, trunc_dict, _ = _step_env(
                env, agents, obs_dict
            )
            ep_reward += sum(rew_dict.values())

            if all(
                term_dict.get(n, False) or trunc_dict.get(n, False)
                for n in env.possible_agents
            ):
                break

        episode_rewards.append(ep_reward)
        if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
            print(f"    [{label}] ep {ep + 1:3d}/{n_episodes}  "
                  f"reward={ep_reward:9.1f}  "
                  f"running_mean={np.mean(episode_rewards):9.1f}")

    elapsed = time.perf_counter() - t0
    print(f"    [{label}] completed in {elapsed:.1f}s "
          f"({sum(1 for _ in episode_rewards) * max_steps / elapsed:.0f} steps/sec)")
    return episode_rewards


def _compare_results(
    label_a: str, rewards_a: list[float],
    label_b: str, rewards_b: list[float],
):
    """Compare two sets of episode rewards with t-test."""
    from scipy import stats as scipy_stats

    a = np.array(rewards_a)
    b = np.array(rewards_b)

    print("\n" + "-" * 70)
    print("  COMPARISON RESULTS")
    print("-" * 70)
    print(f"  {label_a:30s}: mean={a.mean():8.1f} +/- {a.std():6.1f}  "
          f"(n={len(a)}, min={a.min():.0f}, max={a.max():.0f})")
    print(f"  {label_b:30s}: mean={b.mean():8.1f} +/- {b.std():6.1f}  "
          f"(n={len(b)}, min={b.min():.0f}, max={b.max():.0f})")

    delta = b.mean() - a.mean()
    print(f"\n  Delta (modified - baseline): {delta:+.1f}")
    pct = delta / abs(a.mean()) * 100 if a.mean() != 0 else 0
    print(f"  Relative change:             {pct:+.1f}%")

    # Welch's t-test (does not assume equal variance)
    t_stat, p_value = scipy_stats.ttest_ind(a, b, equal_var=False)
    print(f"\n  Welch's t-test:")
    print(f"    t-statistic: {t_stat:.3f}")
    print(f"    p-value:     {p_value:.4f}")
    if p_value < 0.05:
        direction = "BETTER" if delta > 0 else "WORSE"
        print(f"    Result:      SIGNIFICANT at p<0.05 -- modification is {direction}")
    else:
        print(f"    Result:      NOT SIGNIFICANT at p<0.05 -- no conclusive difference")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    if pooled_std > 0:
        cohens_d = delta / pooled_std
        print(f"    Cohen's d:   {cohens_d:.3f} ", end="")
        if abs(cohens_d) < 0.2:
            print("(negligible)")
        elif abs(cohens_d) < 0.5:
            print("(small)")
        elif abs(cohens_d) < 0.8:
            print("(medium)")
        else:
            print("(large)")

    # 95% confidence interval for the difference
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    ci_lo = delta - 1.96 * se
    ci_hi = delta + 1.96 * se
    print(f"    95% CI:      [{ci_lo:.1f}, {ci_hi:.1f}]")
    print("-" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DegradeServices experimental validation for CC4"
    )
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Which experiment to run (1-4 or 'all')",
    )
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episode count (default varies by experiment)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    experiments = args.experiment
    if experiments == "all":
        experiments = ["1", "2", "3", "4"]
    else:
        experiments = [experiments]

    print(f"\nDegradeServices Experiment Suite")
    print(f"Experiments: {', '.join(experiments)}")
    print(f"Seed: {args.seed}")
    t0 = time.perf_counter()

    for exp in experiments:
        if exp == "1":
            n = args.episodes or 10
            experiment_1(n_episodes=n, seed=args.seed)
        elif exp == "2":
            n = args.episodes or 10
            experiment_2(n_episodes=n, seed=args.seed)
        elif exp == "3":
            n = args.episodes or 30
            experiment_3(n_episodes=n, seed=args.seed)
        elif exp == "4":
            n = args.episodes or 30
            experiment_4(n_episodes=n, seed=args.seed)

    elapsed = time.perf_counter() - t0
    print(f"\nAll experiments completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
