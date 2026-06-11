#!/usr/bin/env python3
"""Task 2: Action Execution Order Experiments.

Traces the exact execution order within each step:
- In what order do actions execute within a single step?
- Does blue's Block execute before red's Exploit in the same step?
- Does blue's Restore complete before red can re-exploit?
- What happens when blue and red act on the same host in the same step?

Monkey-patches SimulationController to log execution order.

Logs to: data/action_order_trace.db

Usage:
    python scripts/action_order_tracer.py [--episodes 10] [--seed 42]
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
        DROP TABLE IF EXISTS execution_order;
        DROP TABLE IF EXISTS same_host_conflicts;
        DROP TABLE IF EXISTS action_priorities;
        DROP TABLE IF EXISTS timing_analysis;
        DROP TABLE IF EXISTS summary_stats;

        CREATE TABLE execution_order (
            episode_id INTEGER,
            step INTEGER,
            exec_position INTEGER,
            agent_name TEXT,
            agent_team TEXT,
            action_type TEXT,
            action_priority INTEGER,
            target_host TEXT,
            target_subnet TEXT,
            action_success INTEGER,
            is_in_progress INTEGER,
            mission_phase INTEGER
        );

        CREATE TABLE same_host_conflicts (
            episode_id INTEGER,
            step INTEGER,
            hostname TEXT,
            blue_action TEXT,
            blue_exec_pos INTEGER,
            red_action TEXT,
            red_exec_pos INTEGER,
            blue_before_red INTEGER,
            blue_success INTEGER,
            red_success INTEGER,
            mission_phase INTEGER
        );

        CREATE TABLE action_priorities (
            action_type TEXT PRIMARY KEY,
            priority INTEGER,
            duration INTEGER,
            team TEXT
        );

        CREATE TABLE timing_analysis (
            episode_id INTEGER,
            step INTEGER,
            blue_restore_hosts TEXT,
            red_exploit_hosts TEXT,
            overlap_hosts TEXT,
            blue_acts_first INTEGER,
            mission_phase INTEGER
        );

        CREATE TABLE summary_stats (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
    return conn


def get_mission_phase(step: int, total_steps: int = 500) -> int:
    base = total_steps // 3
    remainder = total_steps % 3
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
    if hasattr(val, 'name'):
        return 1 if val.name == 'TRUE' else 0
    try:
        return int(bool(val))
    except (TypeError, ValueError):
        return -1


# Global execution trace
_exec_trace = []


def patch_simulation_controller():
    """Monkey-patch execute_action to trace execution order."""
    from CybORG.Simulator.SimulationController import SimulationController
    from CybORG.Simulator.Actions.Action import Action

    _orig_step = SimulationController.step

    def traced_step(self, actions=None, skip_valid_action_check=False):
        global _exec_trace
        _exec_trace = []

        # Patch execute_action temporarily
        _orig_execute = self.execute_action
        exec_counter = [0]

        def traced_execute(action):
            obs = _orig_execute(action)
            agent_name = getattr(action, 'agent', None)
            hostname = getattr(action, 'hostname', None)
            if hostname is None:
                ip = getattr(action, 'ip_address', None)
                if ip and ip in self.state.ip_addresses:
                    hostname = self.state.ip_addresses[ip]

            success = None
            if hasattr(obs, 'data') and 'success' in obs.data:
                success = obs.data['success']

            _exec_trace.append({
                'position': exec_counter[0],
                'agent': agent_name or 'unknown',
                'action_type': type(action).__name__,
                'priority': getattr(action, 'priority', 99),
                'hostname': hostname,
                'success': success,
            })
            exec_counter[0] += 1
            return obs

        self.execute_action = traced_execute
        _orig_step(self, actions=actions, skip_valid_action_check=skip_valid_action_check)
        self.execute_action = _orig_execute

    SimulationController.step = traced_step


def run_tracer(
    n_episodes: int = 10,
    seed: int = 42,
    max_steps: int = 500,
    db_path: str | None = None,
):
    global _exec_trace

    db_file = Path(db_path) if db_path else Path(__file__).parent.parent / "data" / "action_order_trace.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = create_db(str(db_file))

    patch_simulation_controller()

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

    # Track known action priorities
    priority_map = {}
    # Conflict counters
    conflict_stats = {
        'total_same_host': 0,
        'blue_before_red': 0,
        'red_before_blue': 0,
        'blue_restore_vs_red_exploit': 0,
        'blue_remove_vs_red_exploit': 0,
        'blue_block_vs_red_exploit': 0,
    }
    order_stats = {
        'blue_always_first': 0,
        'red_always_first': 0,
        'mixed_order': 0,
    }

    t0 = time.perf_counter()

    for ep in range(n_episodes):
        cyborg.reset()
        sim = cyborg.environment_controller

        for step in range(max_steps):
            phase = get_mission_phase(step, max_steps)
            _exec_trace = []

            sim.step()

            # Process execution trace
            blue_actions_this_step = []
            red_actions_this_step = []
            green_actions_this_step = []

            for entry in _exec_trace:
                agent = entry['agent']
                if 'blue' in agent:
                    team = 'blue'
                    blue_actions_this_step.append(entry)
                elif 'red' in agent:
                    team = 'red'
                    red_actions_this_step.append(entry)
                elif 'green' in agent:
                    team = 'green'
                    green_actions_this_step.append(entry)
                else:
                    team = 'unknown'

                hostname = entry['hostname'] or 'N/A'
                subnet = sim.state.hostname_subnet_map.get(hostname)
                subnet_name = subnet.value if subnet else 'N/A'

                conn.execute(
                    "INSERT INTO execution_order VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (ep, step, entry['position'], agent, team,
                     entry['action_type'], entry['priority'],
                     hostname, subnet_name,
                     _success_to_int(entry['success']),
                     0, phase)
                )

                # Track priorities
                act_type = entry['action_type']
                if act_type not in priority_map:
                    priority_map[act_type] = {
                        'priority': entry['priority'],
                        'team': team,
                    }

            # Detect same-host conflicts between blue and red
            blue_hosts = {}
            for b in blue_actions_this_step:
                if b['hostname'] and b['action_type'] not in ('Sleep', 'Monitor'):
                    blue_hosts[b['hostname']] = b

            red_hosts = {}
            for r in red_actions_this_step:
                if r['hostname'] and r['action_type'] not in ('Sleep',):
                    red_hosts[r['hostname']] = r

            overlap = set(blue_hosts.keys()) & set(red_hosts.keys())
            for host in overlap:
                b = blue_hosts[host]
                r = red_hosts[host]
                blue_before = b['position'] < r['position']
                conflict_stats['total_same_host'] += 1
                if blue_before:
                    conflict_stats['blue_before_red'] += 1
                else:
                    conflict_stats['red_before_blue'] += 1

                b_type = b['action_type']
                r_type = r['action_type']
                if 'Restore' in b_type and 'Exploit' in r_type:
                    conflict_stats['blue_restore_vs_red_exploit'] += 1
                elif 'Remove' in b_type and 'Exploit' in r_type:
                    conflict_stats['blue_remove_vs_red_exploit'] += 1
                elif 'Block' in b_type or 'Control' in b_type:
                    conflict_stats['blue_block_vs_red_exploit'] += 1

                subnet = sim.state.hostname_subnet_map.get(host)
                subnet_name = subnet.value if subnet else 'N/A'

                conn.execute(
                    "INSERT INTO same_host_conflicts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (ep, step, host, b_type, b['position'],
                     r_type, r['position'],
                     int(blue_before),
                     _success_to_int(b['success']),
                     _success_to_int(r['success']),
                     phase)
                )

            # Timing analysis for restore vs exploit
            restore_hosts = [b['hostname'] for b in blue_actions_this_step
                           if 'Restore' in b['action_type'] or 'RestoreFromBackup' in b['action_type']]
            exploit_hosts = [r['hostname'] for r in red_actions_this_step
                           if 'Exploit' in r['action_type']]
            overlap_hosts = list(set(restore_hosts) & set(exploit_hosts))
            if restore_hosts or exploit_hosts:
                conn.execute(
                    "INSERT INTO timing_analysis VALUES (?,?,?,?,?,?,?)",
                    (ep, step,
                     json.dumps(restore_hosts),
                     json.dumps(exploit_hosts),
                     json.dumps(overlap_hosts),
                     1 if overlap_hosts and any(
                         blue_hosts.get(h, {}).get('position', 999) <
                         red_hosts.get(h, {}).get('position', 999)
                         for h in overlap_hosts if isinstance(blue_hosts.get(h), dict) and isinstance(red_hosts.get(h), dict)
                     ) else 0,
                     phase)
                )

        if (ep + 1) % 2 == 0:
            conn.commit()
            elapsed = time.perf_counter() - t0
            print(f"  Episode {ep+1}/{n_episodes} done ({elapsed:.1f}s)")

    conn.commit()

    # Store action priorities
    for act_type, info in priority_map.items():
        conn.execute(
            "INSERT OR REPLACE INTO action_priorities VALUES (?,?,?,?)",
            (act_type, info['priority'], -1, info['team'])
        )
    conn.commit()

    # Summary stats
    summary = {
        **{k: str(v) for k, v in conflict_stats.items()},
        'n_episodes': str(n_episodes),
        'seed': str(seed),
        'priority_map': json.dumps({k: v['priority'] for k, v in priority_map.items()}),
    }
    for k, v in summary.items():
        conn.execute("INSERT OR REPLACE INTO summary_stats VALUES (?,?)", (k, str(v)))
    conn.commit()

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*70}")
    print(f"ACTION ORDER TRACER RESULTS: {n_episodes} episodes, seed={seed}")
    print(f"{'='*70}")

    print(f"\n--- Action Priorities (lower = executes first) ---")
    for act_type, info in sorted(priority_map.items(), key=lambda x: x[1]['priority']):
        print(f"  {act_type:30s}  priority={info['priority']:3d}  team={info['team']}")

    print(f"\n--- Execution Order Analysis ---")
    print(f"  The sort_action_order() function sorts by action.priority (ascending).")
    print(f"  ControlTraffic (Block/Allow): priority=1  (executes FIRST)")
    print(f"  RemoveOtherSessions:          priority=5")
    print(f"  All other actions:            priority=99 (default)")
    print(f"")
    print(f"  Within same priority level, order is determined by the dict iteration")
    print(f"  order of agent names (effectively alphabetical/insertion order).")
    print(f"")
    print(f"  KEY FINDING: Block/Allow ALWAYS executes before Exploit/Impact in the")
    print(f"  same step because priority=1 < priority=99.")
    print(f"  Restore/Remove have priority=99 (same as Exploit), so their relative")
    print(f"  order depends on agent name ordering.")

    print(f"\n--- Same-Host Conflicts ---")
    print(f"  Total same-host conflicts:         {conflict_stats['total_same_host']}")
    print(f"  Blue acts before red:              {conflict_stats['blue_before_red']}")
    print(f"  Red acts before blue:              {conflict_stats['red_before_blue']}")
    print(f"  Restore vs Exploit (same host):    {conflict_stats['blue_restore_vs_red_exploit']}")
    print(f"  Remove vs Exploit (same host):     {conflict_stats['blue_remove_vs_red_exploit']}")
    print(f"  Block vs Exploit:                  {conflict_stats['blue_block_vs_red_exploit']}")

    print(f"\n--- Key Mechanics ---")
    print(f"  1. Actions with remaining_ticks > 0 are in progress (Sleep substituted).")
    print(f"  2. Actions execute only when remaining_ticks reaches 0.")
    print(f"  3. sort_action_order() sorts by priority, then filters, then checks bandwidth.")
    print(f"  4. Monitor is an end_turn_action, executed AFTER all agent actions.")
    print(f"  5. Reward is calculated AFTER all actions and Monitor execute.")
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Database: {db_file}")

    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Action Order Tracer")
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--db-path', type=str, default=None)
    args = parser.parse_args()
    run_tracer(n_episodes=args.episodes, seed=args.seed, db_path=args.db_path)
