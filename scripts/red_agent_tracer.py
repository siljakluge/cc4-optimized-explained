#!/usr/bin/env python3
"""Red Agent FSM Tracer for CAGE Challenge 4.

Runs episodes while logging every red agent's FSM state, action, target host,
and outcome per step. Records all data to a SQLite database for analysis.

Usage:
    python scripts/red_agent_tracer.py [--episodes 30] [--steps 500] [--seed 42]
    python scripts/red_agent_tracer.py --analyze   # analyze existing data
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "red_trace.db"


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with all required tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        DROP TABLE IF EXISTS episodes;
        DROP TABLE IF EXISTS red_actions;
        DROP TABLE IF EXISTS red_sessions;
        DROP TABLE IF EXISTS attack_chains;

        CREATE TABLE episodes (
            episode_id INTEGER PRIMARY KEY,
            seed INTEGER,
            total_reward REAL
        );

        CREATE TABLE red_actions (
            episode_id INTEGER,
            step INTEGER,
            red_agent TEXT,
            fsm_state TEXT,
            action_type TEXT,
            target_host TEXT,
            target_ip TEXT,
            success TEXT,
            duration_remaining INTEGER,
            chosen_host_ip TEXT,
            decoy_hit INTEGER DEFAULT 0,
            FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
        );

        CREATE TABLE red_sessions (
            episode_id INTEGER,
            step INTEGER,
            red_agent TEXT,
            hostname TEXT,
            session_type TEXT,
            is_root INTEGER,
            FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
        );

        CREATE TABLE attack_chains (
            episode_id INTEGER,
            red_agent TEXT,
            chain_id INTEGER,
            start_step INTEGER,
            end_step INTEGER,
            start_host TEXT,
            end_host TEXT,
            reached_impact INTEGER,
            FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
        );

        CREATE INDEX idx_red_actions_ep ON red_actions(episode_id, step);
        CREATE INDEX idx_red_actions_agent ON red_actions(red_agent);
        CREATE INDEX idx_red_actions_state ON red_actions(fsm_state);
        CREATE INDEX idx_red_sessions_ep ON red_sessions(episode_id, step);
        CREATE INDEX idx_attack_chains_ep ON attack_chains(episode_id);
    """)
    conn.commit()
    return conn


def get_action_name(action) -> str:
    """Extract action class name from action object."""
    name = type(action).__name__
    return name


def get_action_target(action) -> tuple:
    """Extract target hostname and IP from action."""
    hostname = getattr(action, 'hostname', None)
    ip_addr = getattr(action, 'ip_address', None)
    subnet = getattr(action, 'subnet', None)

    target = hostname or (str(ip_addr) if ip_addr else None) or (str(subnet) if subnet else None)
    ip_str = str(ip_addr) if ip_addr else None

    return target, ip_str


def run_tracing(n_episodes: int = 30, max_steps: int = 500, seed: int = 42) -> Path:
    """Run episodes and trace all red agent activity."""
    from CybORG import CybORG
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent
    from CybORG.Simulator.Actions import Sleep

    conn = create_database(DB_PATH)
    cursor = conn.cursor()

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)

    t0 = time.perf_counter()

    for ep in range(n_episodes):
        cyborg.reset()
        sc = cyborg.environment_controller

        # Get the reward calculator
        ep_reward = 0.0

        # Track attack chains per red agent
        # chain = sequence of actions on a host from K -> Impact
        agent_chains = {}  # red_agent -> list of chain dicts
        agent_current_chain = {}  # red_agent -> current chain dict

        for step in range(max_steps):
            # --- BEFORE STEP: snapshot red agent states ---
            red_agent_states = {}
            red_agent_host_states = {}
            for agent_name, agent_iface in sc.agent_interfaces.items():
                if 'red' not in agent_name:
                    continue
                if not agent_iface.active:
                    continue
                red_agent = agent_iface.agent
                if hasattr(red_agent, 'host_states'):
                    # Deep copy the states before step
                    red_agent_states[agent_name] = {
                        ip: hs.copy() for ip, hs in red_agent.host_states.items()
                    }
                    red_agent_host_states[agent_name] = dict(red_agent.host_states)

            # --- EXECUTE STEP ---
            sc.step()
            sc_step = sc.step_count

            # --- COMPUTE REWARD ---
            step_reward = 0.0
            for team_name, team_calcs in sc.reward.items():
                if team_name == 'Blue':
                    for rname, rval in team_calcs.items():
                        if isinstance(rval, (int, float)):
                            step_reward += rval
                        elif isinstance(rval, dict):
                            step_reward += sum(rval.values())
            ep_reward += step_reward

            # --- AFTER STEP: log red agent actions ---
            for agent_name, agent_iface in sc.agent_interfaces.items():
                if 'red' not in agent_name:
                    continue
                if not agent_iface.active:
                    continue

                red_agent = agent_iface.agent
                if not hasattr(red_agent, 'host_states'):
                    continue

                # Get what action was executed this step
                actions_this_step = sc.action.get(agent_name, [])
                for action in actions_this_step:
                    action_name = get_action_name(action)
                    target, target_ip = get_action_target(action)

                    # Get FSM state for the targeted host (pre-step snapshot)
                    fsm_state = None
                    chosen_ip = None
                    pre_states = red_agent_states.get(agent_name, {})

                    if target_ip and target_ip in pre_states:
                        fsm_state = pre_states[target_ip].get('state', '?')
                        chosen_ip = target_ip
                    elif target:
                        # Find by hostname
                        for ip, hs in pre_states.items():
                            if hs.get('hostname') == target:
                                fsm_state = hs.get('state', '?')
                                chosen_ip = ip
                                break

                    if fsm_state is None and pre_states:
                        # For subnet-level actions (DRS), show the first host's state
                        fsm_state = 'multi'

                    # Get action success from observation
                    obs = sc.observation.get(agent_name)
                    success_str = '?'
                    if obs and obs.observations:
                        for o in obs.observations:
                            if hasattr(o, 'data') and 'success' in o.data:
                                s = o.data['success']
                                if hasattr(s, 'name'):
                                    success_str = s.name
                                else:
                                    success_str = str(s)
                                break

                    # Check duration remaining
                    aip = sc.actions_in_progress.get(agent_name)
                    dur_remaining = 0
                    if aip is not None:
                        dur_remaining = aip.get('remaining_ticks', 0)

                    # Check if this was a decoy hit
                    decoy_hit = 0
                    if action_name == 'ExploitRemoteService' and success_str in ('FALSE', 'False', 'False'):
                        # Check if the host had decoys
                        if hasattr(red_agent, 'host_service_decoy_status'):
                            host_key = target_ip or target
                            if host_key and host_key in red_agent.host_service_decoy_status:
                                decoy_hit = 1

                    cursor.execute(
                        "INSERT INTO red_actions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (ep, step, agent_name, fsm_state, action_name,
                         target, target_ip, success_str, dur_remaining,
                         chosen_ip, decoy_hit)
                    )

                # Log sessions
                for session_id, session in sc.state.sessions.get(agent_name, {}).items():
                    is_root = 1 if session.has_privileged_access() else 0
                    sess_type = str(session.session_type) if hasattr(session.session_type, 'name') else str(session.session_type)
                    cursor.execute(
                        "INSERT INTO red_sessions VALUES (?,?,?,?,?,?)",
                        (ep, step, agent_name, session.hostname,
                         sess_type, is_root)
                    )

        # End of episode
        cursor.execute("INSERT INTO episodes VALUES (?,?,?)", (ep, seed, ep_reward))

        # Build attack chains from action log
        _build_attack_chains(cursor, ep)

        conn.commit()
        print(f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:9.1f}")

    elapsed = time.perf_counter() - t0
    conn.commit()
    conn.close()

    print(f"\nTracing complete: {n_episodes} episodes in {elapsed:.1f}s")
    print(f"Database: {DB_PATH}")
    return DB_PATH


def _build_attack_chains(cursor: sqlite3.Cursor, episode_id: int):
    """Build attack chains from the action log for one episode.

    A chain is a sequence of actions by a single red agent on a host path
    from initial discovery through to Impact (or as far as it got).
    """
    cursor.execute("""
        SELECT step, red_agent, fsm_state, action_type, target_host, success
        FROM red_actions
        WHERE episode_id = ? AND action_type != 'Sleep'
        ORDER BY red_agent, step
    """, (episode_id,))

    rows = cursor.fetchall()
    if not rows:
        return

    # Group by red_agent
    agent_actions = defaultdict(list)
    for row in rows:
        step, agent, state, action, target, success = row
        agent_actions[agent].append({
            'step': step, 'state': state, 'action': action,
            'target': target, 'success': success
        })

    chain_id = 0
    for agent, actions in agent_actions.items():
        # Track host-level chains
        host_chains = {}  # target -> chain info

        for act in actions:
            target = act['target']
            if target is None:
                continue

            if target not in host_chains:
                host_chains[target] = {
                    'start_step': act['step'],
                    'end_step': act['step'],
                    'start_host': target,
                    'reached_impact': False
                }
            else:
                host_chains[target]['end_step'] = act['step']

            if act['action'] == 'Impact' and act['success'] in ('TRUE', 'True'):
                host_chains[target]['reached_impact'] = True

        for target, chain in host_chains.items():
            cursor.execute(
                "INSERT INTO attack_chains VALUES (?,?,?,?,?,?,?,?)",
                (episode_id, agent, chain_id,
                 chain['start_step'], chain['end_step'],
                 chain['start_host'], target,
                 1 if chain['reached_impact'] else 0)
            )
            chain_id += 1


def analyze_data(db_path: Path = DB_PATH):
    """Analyze the trace data and print comprehensive statistics."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    print("=" * 72)
    print("RED AGENT TRACE ANALYSIS")
    print("=" * 72)

    # --- 1. Episode summary ---
    eps = conn.execute("SELECT COUNT(*), AVG(total_reward), MIN(total_reward), MAX(total_reward) FROM episodes").fetchone()
    print(f"\nEpisodes: {eps[0]}, Mean reward: {eps[1]:.1f}, Min: {eps[2]:.1f}, Max: {eps[3]:.1f}")

    # --- 2. Steps from K to Impact per red agent ---
    print("\n" + "=" * 72)
    print("DISTRIBUTION: Steps from First Action to Impact per Red Agent")
    print("=" * 72)

    chains = conn.execute("""
        SELECT red_agent, start_step, end_step, reached_impact,
               start_host, end_host
        FROM attack_chains
        ORDER BY red_agent
    """).fetchall()

    agent_impact_times = defaultdict(list)
    agent_no_impact = defaultdict(int)
    for c in chains:
        if c['reached_impact']:
            duration = c['end_step'] - c['start_step']
            agent_impact_times[c['red_agent']].append(duration)
        else:
            agent_no_impact[c['red_agent']] += 1

    for agent in sorted(set(c['red_agent'] for c in chains)):
        times = agent_impact_times.get(agent, [])
        no_impact = agent_no_impact.get(agent, 0)
        if times:
            arr = np.array(times)
            print(f"\n  {agent}:")
            print(f"    Chains reaching Impact: {len(times)}")
            print(f"    Chains NOT reaching Impact: {no_impact}")
            print(f"    Steps to Impact: mean={arr.mean():.1f}, std={arr.std():.1f}, "
                  f"min={arr.min()}, max={arr.max()}, median={np.median(arr):.0f}")
            # Distribution buckets
            buckets = [0, 10, 20, 30, 50, 100, 200, 500]
            for i in range(len(buckets) - 1):
                count = np.sum((arr >= buckets[i]) & (arr < buckets[i+1]))
                if count > 0:
                    print(f"    [{buckets[i]:3d}-{buckets[i+1]:3d}): {count}")
        else:
            print(f"\n  {agent}: No Impact achieved ({no_impact} chains)")

    # --- 3. Probability of reaching root before blue responds ---
    print("\n" + "=" * 72)
    print("PRIVILEGE ESCALATION TIMING")
    print("=" * 72)

    privesc = conn.execute("""
        SELECT red_agent, step, success, target_host
        FROM red_actions
        WHERE action_type = 'PrivilegeEscalate'
        ORDER BY episode_id, red_agent, step
    """).fetchall()

    pe_success = sum(1 for p in privesc if p['success'] in ('TRUE', 'True'))
    pe_total = len(privesc)
    print(f"\n  Total PrivilegeEscalate attempts: {pe_total}")
    print(f"  Successful: {pe_success} ({100*pe_success/max(pe_total,1):.1f}%)")
    print(f"  Failed: {pe_total - pe_success} ({100*(pe_total-pe_success)/max(pe_total,1):.1f}%)")

    # --- 4. Decoy interaction statistics ---
    print("\n" + "=" * 72)
    print("DECOY INTERACTION STATISTICS")
    print("=" * 72)

    exploits = conn.execute("""
        SELECT action_type, success, decoy_hit, COUNT(*) as cnt
        FROM red_actions
        WHERE action_type = 'ExploitRemoteService'
        GROUP BY success, decoy_hit
    """).fetchall()

    total_exploits = sum(e['cnt'] for e in exploits)
    for e in exploits:
        label = f"success={e['success']}, decoy_hit={e['decoy_hit']}"
        print(f"  {label}: {e['cnt']} ({100*e['cnt']/max(total_exploits,1):.1f}%)")
    print(f"  Total exploit attempts: {total_exploits}")

    # --- 5. Most attacked hosts ---
    print("\n" + "=" * 72)
    print("HOST TARGETING FREQUENCY (non-Sleep, non-DRS actions)")
    print("=" * 72)

    host_freq = conn.execute("""
        SELECT target_host, COUNT(*) as cnt, action_type
        FROM red_actions
        WHERE action_type NOT IN ('Sleep', 'RedSessionCheck')
            AND target_host IS NOT NULL
        GROUP BY target_host
        ORDER BY cnt DESC
        LIMIT 30
    """).fetchall()

    for h in host_freq:
        print(f"  {h['target_host']:50s} {h['cnt']:5d}")

    # More detailed: actions per host
    print("\n  Top hosts by action type:")
    host_action_freq = conn.execute("""
        SELECT target_host, action_type, COUNT(*) as cnt
        FROM red_actions
        WHERE action_type NOT IN ('Sleep', 'RedSessionCheck')
            AND target_host IS NOT NULL
        GROUP BY target_host, action_type
        ORDER BY target_host, cnt DESC
    """).fetchall()

    host_actions = defaultdict(list)
    for h in host_action_freq:
        host_actions[h['target_host']].append((h['action_type'], h['cnt']))

    for host in sorted(host_actions.keys()):
        actions_str = ", ".join(f"{a}={c}" for a, c in host_actions[host][:5])
        print(f"    {host:45s} {actions_str}")

    # --- 6. Time spent in each FSM state ---
    print("\n" + "=" * 72)
    print("AVERAGE TIME SPENT IN EACH FSM STATE")
    print("=" * 72)

    state_time = conn.execute("""
        SELECT fsm_state, COUNT(*) as cnt, action_type
        FROM red_actions
        WHERE fsm_state IS NOT NULL
        GROUP BY fsm_state
        ORDER BY cnt DESC
    """).fetchall()

    total_state_actions = sum(s['cnt'] for s in state_time)
    for s in state_time:
        pct = 100 * s['cnt'] / max(total_state_actions, 1)
        print(f"  State {s['fsm_state']:5s}: {s['cnt']:6d} actions ({pct:5.1f}%)")

    # State-action breakdown
    print("\n  State -> Action distribution:")
    state_action = conn.execute("""
        SELECT fsm_state, action_type, COUNT(*) as cnt,
               SUM(CASE WHEN success IN ('TRUE','True') THEN 1 ELSE 0 END) as success_cnt
        FROM red_actions
        WHERE fsm_state IS NOT NULL AND action_type != 'Sleep'
        GROUP BY fsm_state, action_type
        ORDER BY fsm_state, cnt DESC
    """).fetchall()

    for sa in state_action:
        success_rate = 100 * sa['success_cnt'] / max(sa['cnt'], 1)
        print(f"    {sa['fsm_state']:5s} -> {sa['action_type']:30s}: {sa['cnt']:5d} "
              f"(success: {sa['success_cnt']}/{sa['cnt']} = {success_rate:.0f}%)")

    # --- 7. Re-exploitation speed after blue Remove vs Restore ---
    print("\n" + "=" * 72)
    print("RE-EXPLOITATION SPEED (time from KD back to U/R)")
    print("=" * 72)

    # Find sequences where a host goes from U/R -> KD -> U/R
    reexploit = conn.execute("""
        SELECT episode_id, red_agent, step, fsm_state, action_type, success, target_host
        FROM red_actions
        WHERE fsm_state IN ('KD', 'SD', 'S', 'U', 'UD', 'R', 'RD')
            AND action_type != 'Sleep'
        ORDER BY episode_id, red_agent, target_host, step
    """).fetchall()

    # Group by (episode, agent, host) and find KD->U transitions
    host_sequences = defaultdict(list)
    for r in reexploit:
        key = (r['episode_id'], r['red_agent'], r['target_host'])
        host_sequences[key].append((r['step'], r['fsm_state'], r['action_type'], r['success']))

    recovery_times = []
    for key, seq in host_sequences.items():
        in_kd = False
        kd_start = None
        for step, state, action, success in seq:
            if state == 'KD' and not in_kd:
                in_kd = True
                kd_start = step
            elif in_kd and state in ('U', 'UD', 'R', 'RD'):
                recovery_times.append(step - kd_start)
                in_kd = False
                kd_start = None

    if recovery_times:
        arr = np.array(recovery_times)
        print(f"  Recovery events (KD -> U/R): {len(arr)}")
        print(f"  Steps to re-exploit: mean={arr.mean():.1f}, std={arr.std():.1f}, "
              f"min={arr.min()}, max={arr.max()}, median={np.median(arr):.0f}")
    else:
        print("  No KD -> U/R recovery sequences found")

    # --- 8. Per-agent action distribution ---
    print("\n" + "=" * 72)
    print("PER-AGENT ACTION DISTRIBUTION")
    print("=" * 72)

    agent_dist = conn.execute("""
        SELECT red_agent, action_type, COUNT(*) as cnt
        FROM red_actions
        WHERE action_type != 'Sleep'
        GROUP BY red_agent, action_type
        ORDER BY red_agent, cnt DESC
    """).fetchall()

    current_agent = None
    for ad in agent_dist:
        if ad['red_agent'] != current_agent:
            current_agent = ad['red_agent']
            print(f"\n  {current_agent}:")
        print(f"    {ad['action_type']:35s}: {ad['cnt']:5d}")

    # --- 9. Impact timing distribution ---
    print("\n" + "=" * 72)
    print("IMPACT ACTION STATISTICS")
    print("=" * 72)

    impacts = conn.execute("""
        SELECT red_agent, step, target_host, success, episode_id
        FROM red_actions
        WHERE action_type = 'Impact'
        ORDER BY episode_id, step
    """).fetchall()

    impact_success = [i for i in impacts if i['success'] in ('TRUE', 'True')]
    impact_fail = [i for i in impacts if i['success'] not in ('TRUE', 'True')]

    print(f"  Total Impact attempts: {len(impacts)}")
    print(f"  Successful: {len(impact_success)} ({100*len(impact_success)/max(len(impacts),1):.1f}%)")
    print(f"  Failed: {len(impact_fail)} ({100*len(impact_fail)/max(len(impacts),1):.1f}%)")

    if impact_success:
        first_impacts = defaultdict(list)
        for i in impact_success:
            first_impacts[i['episode_id']].append(i['step'])
        first_steps = [min(steps) for steps in first_impacts.values()]
        arr = np.array(first_steps)
        print(f"\n  First successful Impact per episode:")
        print(f"    Mean step: {arr.mean():.1f}, Std: {arr.std():.1f}")
        print(f"    Min: {arr.min()}, Max: {arr.max()}, Median: {np.median(arr):.0f}")

    # Impact by host
    print("\n  Impact attempts by host:")
    impact_hosts = conn.execute("""
        SELECT target_host, COUNT(*) as total,
               SUM(CASE WHEN success IN ('TRUE','True') THEN 1 ELSE 0 END) as successes
        FROM red_actions
        WHERE action_type = 'Impact'
        GROUP BY target_host
        ORDER BY total DESC
    """).fetchall()

    for ih in impact_hosts:
        print(f"    {ih['target_host']:45s} total={ih['total']:4d}  success={ih['successes']:4d}")

    # --- 10. DegradeServices statistics ---
    print("\n" + "=" * 72)
    print("DEGRADE SERVICES STATISTICS")
    print("=" * 72)

    degrade = conn.execute("""
        SELECT target_host, COUNT(*) as total,
               SUM(CASE WHEN success IN ('TRUE','True') THEN 1 ELSE 0 END) as successes
        FROM red_actions
        WHERE action_type = 'DegradeServices'
        GROUP BY target_host
        ORDER BY total DESC
    """).fetchall()

    for d in degrade:
        print(f"  {d['target_host']:45s} total={d['total']:4d}  success={d['successes']:4d}")

    # --- 11. DiscoverDeception statistics ---
    print("\n" + "=" * 72)
    print("DISCOVER DECEPTION STATISTICS")
    print("=" * 72)

    dd = conn.execute("""
        SELECT red_agent, COUNT(*) as total,
               SUM(CASE WHEN success IN ('TRUE','True') THEN 1 ELSE 0 END) as successes
        FROM red_actions
        WHERE action_type = 'DiscoverDeception'
        GROUP BY red_agent
        ORDER BY total DESC
    """).fetchall()

    for d in dd:
        print(f"  {d['red_agent']:30s} total={d['total']:4d}  success={d['successes']:4d}")

    # --- 12. Per-phase analysis ---
    print("\n" + "=" * 72)
    print("PER-PHASE RED ACTIVITY")
    print("=" * 72)

    # For 500-step episodes: phase 0 = 0-166, phase 1 = 167-333, phase 2 = 334-499
    for phase_name, start, end in [("Phase 0 (0-166)", 0, 166),
                                    ("Phase 1 (167-333)", 167, 333),
                                    ("Phase 2 (334-499)", 334, 499)]:
        phase_actions = conn.execute("""
            SELECT action_type, COUNT(*) as cnt,
                   SUM(CASE WHEN success IN ('TRUE','True') THEN 1 ELSE 0 END) as successes
            FROM red_actions
            WHERE step >= ? AND step <= ? AND action_type != 'Sleep'
            GROUP BY action_type
            ORDER BY cnt DESC
        """, (start, end)).fetchall()

        print(f"\n  {phase_name}:")
        for pa in phase_actions:
            print(f"    {pa['action_type']:35s}: {pa['cnt']:5d} (success: {pa['successes']})")

    conn.close()
    print("\n" + "=" * 72)
    print("Analysis complete.")


def main():
    parser = argparse.ArgumentParser(description="Red Agent FSM Tracer for CC4")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing data instead of running new episodes")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to database file")
    args = parser.parse_args()

    if args.db:
        global DB_PATH
        DB_PATH = Path(args.db)

    if args.analyze:
        if not DB_PATH.exists():
            print(f"Database not found: {DB_PATH}")
            print("Run without --analyze first to generate data.")
            sys.exit(1)
        analyze_data(DB_PATH)
    else:
        run_tracing(args.episodes, args.steps, args.seed)
        print("\nRunning analysis on collected data...")
        analyze_data(DB_PATH)


if __name__ == "__main__":
    main()
