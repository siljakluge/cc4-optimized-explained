#!/usr/bin/env python3
"""Task 1: Instrument Green Agent Behavior via monkey-patching.

Patches GreenLocalWork.execute and GreenAccessService.execute to trace:
- Every green action, target, outcome
- Which actions create proc_flags and conn_flags (FP events)
- Phishing events and whether they create real red sessions
- FP rates per host and per action type

Logs to SQLite: data/green_trace.db

Usage:
    python scripts/green_agent_tracer.py [--episodes 30] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        DROP TABLE IF EXISTS green_actions;
        DROP TABLE IF EXISTS phishing_events;
        DROP TABLE IF EXISTS fp_analysis;
        DROP TABLE IF EXISTS summary_stats;

        CREATE TABLE green_actions (
            episode_id INTEGER,
            step INTEGER,
            green_agent TEXT,
            action_type TEXT,
            target_host TEXT,
            target_subnet TEXT,
            success INTEGER,
            created_proc_fp INTEGER,
            created_net_fp INTEGER,
            created_phishing INTEGER,
            blocked INTEGER,
            mission_phase INTEGER
        );

        CREATE TABLE phishing_events (
            episode_id INTEGER,
            step INTEGER,
            green_agent TEXT,
            target_host TEXT,
            target_subnet TEXT,
            created_red_session INTEGER,
            red_agent_assigned TEXT,
            red_already_present INTEGER,
            mission_phase INTEGER
        );

        CREATE TABLE fp_analysis (
            episode_id INTEGER,
            step INTEGER,
            hostname TEXT,
            subnet TEXT,
            flag_type TEXT,
            is_true_positive INTEGER,
            source_agent TEXT,
            source_action TEXT,
            mission_phase INTEGER
        );

        CREATE TABLE summary_stats (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
    return conn


# Global trace log for current step
_trace_log = []


def get_mission_phase(step: int, total_steps: int = 500) -> int:
    remainder = total_steps % 3
    base = total_steps // 3
    if remainder == 2:
        phases = (base + 1, base + 1, base)
    elif remainder == 1:
        phases = (base + 1, base, base)
    else:
        phases = (base, base, base)
    if step < phases[0]:
        return 0
    elif step < phases[0] + phases[1]:
        return 1
    else:
        return 2


def _success_to_int(val) -> int:
    """Convert success value (bool, TernaryEnum, or other) to int."""
    if val is None:
        return -1
    if isinstance(val, bool):
        return int(val)
    # TernaryEnum: TRUE=1, FALSE=3, IN_PROGRESS=4, UNKNOWN=2
    if hasattr(val, 'name'):
        return 1 if val.name == 'TRUE' else 0
    try:
        return int(bool(val))
    except (TypeError, ValueError):
        return -1


def patch_green_actions():
    """Monkey-patch GreenLocalWork and GreenAccessService to trace FP/phishing."""
    from CybORG.Simulator.Actions.GreenActions.GreenLocalWork import GreenLocalWork
    from CybORG.Simulator.Actions.GreenActions.GreenAccessService import GreenAccessService
    from CybORG.Simulator.Actions.ConcreteActions.PhishingEmail import PhishingEmail
    from CybORG.Shared import Observation

    # Patch GreenLocalWork.execute
    _orig_local_work_execute = GreenLocalWork.execute

    def traced_local_work_execute(self, state):
        hostname = None
        if self.session in state.sessions.get(self.agent, {}):
            session = state.sessions[self.agent][self.session]
            hostname = session.hostname

        # Count pre-existing events
        host = state.hosts.get(hostname) if hostname else None
        pre_proc_count = len(host.events.process_creation) if host else 0

        # Snapshot red sessions on this host
        pre_red_on_host = set()
        if hostname:
            for rname, rsessions in state.sessions.items():
                if 'red' not in rname:
                    continue
                for sid, sess in rsessions.items():
                    if sess.hostname == hostname and sess.active:
                        pre_red_on_host.add((rname, sid))

        obs = _orig_local_work_execute(self, state)

        # Check what happened
        post_proc_count = len(host.events.process_creation) if host else 0
        created_proc_fp = 1 if post_proc_count > pre_proc_count else 0

        # Check for new red sessions (phishing)
        created_phishing = 0
        phishing_red_agent = None
        red_already = 0
        if hostname:
            post_red_on_host = set()
            for rname, rsessions in state.sessions.items():
                if 'red' not in rname:
                    continue
                for sid, sess in rsessions.items():
                    if sess.hostname == hostname and sess.active:
                        post_red_on_host.add((rname, sid))
            new_red = post_red_on_host - pre_red_on_host
            if new_red:
                created_phishing = 1
                phishing_red_agent = list(new_red)[0][0]
            # Check if red was already present before phishing attempt
            if pre_red_on_host:
                red_already = 1

        raw_success = obs.data.get('success', True) if hasattr(obs, 'data') else True
        success = bool(raw_success) if not hasattr(raw_success, 'name') else (raw_success.name == 'TRUE')

        _trace_log.append({
            'type': 'local_work',
            'agent': self.agent,
            'hostname': hostname or 'unknown',
            'success': success,
            'created_proc_fp': created_proc_fp,
            'created_net_fp': 0,
            'created_phishing': created_phishing,
            'blocked': 0,
            'phishing_red_agent': phishing_red_agent,
            'red_already_present': red_already,
        })

        return obs

    GreenLocalWork.execute = traced_local_work_execute

    # Patch GreenAccessService.execute
    _orig_access_service_execute = GreenAccessService.execute

    def traced_access_service_execute(self, state):
        # We need to trace what happens inside execute
        # The execute sets self.dest_ip, so we save it after
        obs = Observation(False)

        self.dest_ip = self.random_reachable_ip(state)
        if self.dest_ip is None:
            _trace_log.append({
                'type': 'access_service',
                'agent': self.agent,
                'hostname': 'no_reachable',
                'success': False,
                'created_proc_fp': 0,
                'created_net_fp': 0,
                'created_phishing': 0,
                'blocked': 0,
                'phishing_red_agent': None,
                'red_already_present': 0,
            })
            return obs

        if not self.available_dest_service:
            pass  # This is a bug in the original code - it checks the method object, not calling it

        from_host = state.ip_addresses[self.dest_ip]
        from_host_obj = state.hosts[from_host]
        self.dest_port = from_host_obj.get_ephemeral_port()
        from_subnet = state.hostname_subnet_map[from_host].value

        to_host = state.ip_addresses[self.ip_address]
        to_subnet = state.hostname_subnet_map[to_host].value

        # Check blocking
        connection_failure_flag = False
        if to_subnet in state.blocks:
            if from_subnet in state.blocks[to_subnet]:
                connection_failure_flag = True
        if from_subnet in state.blocks:
            if to_subnet in state.blocks[from_subnet]:
                connection_failure_flag = True

        blocked = 0
        created_net_fp = 0

        if connection_failure_flag:
            from CybORG.Simulator.HostEvents import NetworkConnection
            event = NetworkConnection(
                local_address=state.hostname_ip_map[from_host],
                remote_address=state.hostname_ip_map[to_host],
                remote_port=8800)
            from_host_obj.events.network_connections.append(event)
            blocked = 1

            _trace_log.append({
                'type': 'access_service',
                'agent': self.agent,
                'hostname': from_host,
                'success': False,
                'created_proc_fp': 0,
                'created_net_fp': 0,  # blocked connection event is deterministic, not FP
                'created_phishing': 0,
                'blocked': 1,
                'phishing_red_agent': None,
                'red_already_present': 0,
            })
            return obs

        # FP detection
        if state.np_random.random() < self.fp_detection_rate:
            from CybORG.Simulator.HostEvents import NetworkConnection
            event = NetworkConnection(
                local_address=self.ip_address,
                remote_address=self.dest_ip,
                remote_port=self.dest_port
            )
            from_host_obj.events.network_connections.append(event)
            created_net_fp = 1

        obs.set_success(True)

        _trace_log.append({
            'type': 'access_service',
            'agent': self.agent,
            'hostname': from_host,
            'success': True,
            'created_proc_fp': 0,
            'created_net_fp': created_net_fp,
            'created_phishing': 0,
            'blocked': 0,
            'phishing_red_agent': None,
            'red_already_present': 0,
        })
        return obs

    GreenAccessService.execute = traced_access_service_execute


def run_tracer(n_episodes: int = 30, seed: int = 42, max_steps: int = 500):
    global _trace_log

    db_path = str(Path(__file__).parent.parent / "data" / "green_trace.db")
    conn = create_db(db_path)

    patch_green_actions()

    from CybORG import CybORG
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)

    # Accumulators
    counts = {
        'total_actions': 0,
        'local_work': 0, 'access_service': 0, 'sleep': 0,
        'local_work_success': 0, 'local_work_fail': 0,
        'access_success': 0, 'access_fail': 0, 'access_blocked': 0,
        'proc_fp': 0, 'net_fp': 0,
        'phishing_total': 0, 'phishing_new_session': 0,
        'phishing_red_already': 0,
    }
    fp_by_subnet = {}
    fp_by_host_type = {}  # server vs user_host
    phishing_by_subnet = {}
    phishing_by_phase = {0: 0, 1: 0, 2: 0}

    t0 = time.perf_counter()

    for ep in range(n_episodes):
        cyborg.reset()
        sim = cyborg.environment_controller

        for step in range(max_steps):
            phase = get_mission_phase(step, max_steps)
            _trace_log = []

            # Step the simulation
            sim.step()

            # Process trace log
            for entry in _trace_log:
                counts['total_actions'] += 1
                hostname = entry['hostname']
                subnet = sim.state.hostname_subnet_map.get(hostname)
                subnet_name = subnet.value if subnet else 'unknown'

                if entry['type'] == 'local_work':
                    counts['local_work'] += 1
                    if entry['success']:
                        counts['local_work_success'] += 1
                    else:
                        counts['local_work_fail'] += 1

                    if entry['created_proc_fp']:
                        counts['proc_fp'] += 1
                        fp_by_subnet[subnet_name] = fp_by_subnet.get(subnet_name, 0) + 1
                        host_type = 'server' if 'server' in hostname else 'user'
                        fp_by_host_type[host_type] = fp_by_host_type.get(host_type, 0) + 1

                        conn.execute(
                            "INSERT INTO fp_analysis VALUES (?,?,?,?,?,?,?,?,?)",
                            (ep, step, hostname, subnet_name, 'process_creation',
                             0, entry['agent'], 'GreenLocalWork', phase)
                        )

                    if entry['created_phishing']:
                        counts['phishing_total'] += 1
                        if entry['red_already_present']:
                            counts['phishing_red_already'] += 1
                        else:
                            counts['phishing_new_session'] += 1
                        phishing_by_subnet[subnet_name] = phishing_by_subnet.get(subnet_name, 0) + 1
                        phishing_by_phase[phase] += 1

                        conn.execute(
                            "INSERT INTO phishing_events VALUES (?,?,?,?,?,?,?,?,?)",
                            (ep, step, entry['agent'], hostname, subnet_name,
                             1 if not entry['red_already_present'] else 0,
                             entry['phishing_red_agent'] or 'unknown',
                             int(entry['red_already_present']),
                             phase)
                        )

                    conn.execute(
                        "INSERT INTO green_actions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (ep, step, entry['agent'], 'GreenLocalWork', hostname, subnet_name,
                         _success_to_int(entry['success']),
                         entry['created_proc_fp'], 0, entry['created_phishing'],
                         0, phase)
                    )

                elif entry['type'] == 'access_service':
                    counts['access_service'] += 1
                    if entry['success']:
                        counts['access_success'] += 1
                    elif entry['blocked']:
                        counts['access_blocked'] += 1
                        counts['access_fail'] += 1
                    else:
                        counts['access_fail'] += 1

                    if entry['created_net_fp']:
                        counts['net_fp'] += 1
                        fp_by_subnet[subnet_name] = fp_by_subnet.get(subnet_name, 0) + 1
                        host_type = 'server' if 'server' in hostname else 'user'
                        fp_by_host_type[host_type] = fp_by_host_type.get(host_type, 0) + 1

                        conn.execute(
                            "INSERT INTO fp_analysis VALUES (?,?,?,?,?,?,?,?,?)",
                            (ep, step, hostname, subnet_name, 'network_connection',
                             0, entry['agent'], 'GreenAccessService', phase)
                        )

                    conn.execute(
                        "INSERT INTO green_actions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (ep, step, entry['agent'], 'GreenAccessService', hostname, subnet_name,
                         _success_to_int(entry['success']),
                         0, entry['created_net_fp'], 0,
                         entry['blocked'], phase)
                    )

        if (ep + 1) % 5 == 0:
            conn.commit()
            elapsed = time.perf_counter() - t0
            print(f"  Episode {ep+1}/{n_episodes} done ({elapsed:.1f}s)")

    conn.commit()

    # Compute derived stats
    lw = counts['local_work']
    acs = counts['access_service']
    total = counts['total_actions']

    proc_fp_rate = counts['proc_fp'] / max(lw, 1) * 100
    net_fp_rate = counts['net_fp'] / max(acs, 1) * 100
    phishing_rate = counts['phishing_total'] / max(lw, 1) * 100
    phishing_new_rate = counts['phishing_new_session'] / max(lw, 1) * 100
    lw_fail_rate = counts['local_work_fail'] / max(lw, 1) * 100
    acs_fail_rate = counts['access_fail'] / max(acs, 1) * 100

    # Store summary
    summary = {
        **{k: str(v) for k, v in counts.items()},
        'proc_fp_rate_pct': f"{proc_fp_rate:.3f}",
        'net_fp_rate_pct': f"{net_fp_rate:.3f}",
        'phishing_rate_pct': f"{phishing_rate:.3f}",
        'phishing_new_rate_pct': f"{phishing_new_rate:.3f}",
        'lw_fail_rate_pct': f"{lw_fail_rate:.2f}",
        'acs_fail_rate_pct': f"{acs_fail_rate:.2f}",
        'fp_by_subnet': json.dumps(fp_by_subnet),
        'fp_by_host_type': json.dumps(fp_by_host_type),
        'phishing_by_subnet': json.dumps(phishing_by_subnet),
        'phishing_by_phase': json.dumps(phishing_by_phase),
        'n_episodes': str(n_episodes),
        'seed': str(seed),
        'green_agents_per_episode': str(lw + acs + counts.get('sleep', 0)) if n_episodes > 0 else '0',
    }
    for k, v in summary.items():
        conn.execute("INSERT OR REPLACE INTO summary_stats VALUES (?,?)", (k, str(v)))
    conn.commit()

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*70}")
    print(f"GREEN AGENT TRACER RESULTS: {n_episodes} episodes, seed={seed}")
    print(f"{'='*70}")
    print(f"\n--- Action Distribution ---")
    print(f"  Total green actions traced:  {total}")
    print(f"  GreenLocalWork:              {lw:>7} ({100*lw/max(total,1):.1f}%)")
    print(f"  GreenAccessService:          {acs:>7} ({100*acs/max(total,1):.1f}%)")
    print(f"  (Sleep actions not traced by monkey-patch)")
    print(f"\n--- False Positive Rates ---")
    print(f"  Process creation FPs:        {counts['proc_fp']:>5} ({proc_fp_rate:.3f}% of local work)")
    print(f"  Network connection FPs:      {counts['net_fp']:>5} ({net_fp_rate:.3f}% of access service)")
    print(f"  Expected rate:               1.000% (configured fp_detection_rate)")
    print(f"\n--- Phishing ---")
    print(f"  Total phishing events:       {counts['phishing_total']:>5} ({phishing_rate:.3f}% of local work)")
    print(f"  New red sessions created:    {counts['phishing_new_session']:>5} ({phishing_new_rate:.3f}%)")
    print(f"  Red already present:         {counts['phishing_red_already']:>5}")
    print(f"  Per episode:                 {counts['phishing_total']/n_episodes:.2f} total, {counts['phishing_new_session']/n_episodes:.2f} new sessions")
    print(f"  By phase:                    {json.dumps(phishing_by_phase)}")
    print(f"  By subnet:                   {json.dumps(phishing_by_subnet, indent=4)}")
    print(f"\n--- Green Action Failure Rates ---")
    print(f"  LocalWork failures:          {counts['local_work_fail']:>5} ({lw_fail_rate:.2f}%)")
    print(f"  AccessService failures:      {counts['access_fail']:>5} ({acs_fail_rate:.2f}%)")
    print(f"    of which blocked:          {counts['access_blocked']:>5}")
    print(f"\n--- FP by Subnet ---")
    for sn, cnt in sorted(fp_by_subnet.items(), key=lambda x: -x[1]):
        print(f"    {sn:40s} {cnt:>5}")
    print(f"\n--- FP by Host Type ---")
    for ht, cnt in sorted(fp_by_host_type.items(), key=lambda x: -x[1]):
        print(f"    {ht:40s} {cnt:>5}")
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Database: {db_path}")

    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Green Agent Tracer")
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_tracer(n_episodes=args.episodes, seed=args.seed)
