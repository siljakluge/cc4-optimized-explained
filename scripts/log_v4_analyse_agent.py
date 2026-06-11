#!/usr/bin/env python3
"""V4AnalyseAgent logger for CAGE Challenge 4.

Runs N episodes with V4AnalyseAgent (EnterpriseHeuristicAgent v4 + Priority 3.5 Analyse)
and logs extended state information to a SQLite database.

Usage:
    python scripts/log_v4_analyse_agent.py [--episodes 20] [--steps 500] [--seed 42]

DB written to: data/v4_analyse.db
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# DB path
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "data" / "v4_analyse.db"

CREATE_TABLES_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS episodes (
    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_type TEXT DEFAULT 'v4_analyse',
    total_reward REAL, n_steps INTEGER, seed INTEGER
);

CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER, phase INTEGER,
    cumulative_reward REAL, step_reward REAL
);

CREATE TABLE IF NOT EXISTS actions_taken (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER, agent_name TEXT,
    action_idx INTEGER, action_label TEXT, action_category TEXT,
    agent_reward REAL
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER, agent_name TEXT,
    obs_json TEXT
);

CREATE TABLE IF NOT EXISTS red_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER,
    agent_name TEXT, hostname TEXT, subnet TEXT, is_privileged INTEGER
);

CREATE TABLE IF NOT EXISTS host_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER,
    hostname TEXT, subnet TEXT,
    has_process_flag INTEGER, has_connection_flag INTEGER,
    n_process_events INTEGER, n_connection_events INTEGER
);

CREATE TABLE IF NOT EXISTS service_reliability (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER, step INTEGER,
    hostname TEXT, service_name TEXT, reliability_pct INTEGER
);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(CREATE_TABLES_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# V4AnalyseAgent
# ---------------------------------------------------------------------------

from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import (  # noqa: E402
    EnterpriseHeuristicAgent,
    REMOVE_DUR,
    RESTORE_DUR,
    _sorted_by_priority,
    _host_priority,
    _pair_priority,
    _SORTED_SUBNETS,
)

# Offsets mirrored from parent (needed for obs parsing in overridden get_action)
_NUM_SUBNETS = 9
_MAX_HOSTS   = 16
_OFF_BLOCKED = _NUM_SUBNETS          # 9
_OFF_COMMS   = _NUM_SUBNETS * 2      # 18
_OFF_PROC    = _NUM_SUBNETS * 3      # 27


class V4AnalyseAgent(EnterpriseHeuristicAgent):
    """EnterpriseHeuristicAgent v4 + Priority 3.5: Analyse actions."""

    def __init__(self, agent_name: str = "blue_agent_0"):
        super().__init__(agent_name)
        self._analyse: dict[str, int] = {}      # hostname → action_idx
        self._analysed_at: dict[str, int] = {}  # hostname → step last analysed

    def reset(self) -> None:
        super().reset()
        self._analysed_at.clear()

    def _parse_labels(self, labels: list[str], subnet_hosts: dict) -> None:
        super()._parse_labels(labels, subnet_hosts)
        self._analyse.clear()
        for idx, raw in enumerate(labels):
            label = raw.strip()
            if label.startswith("[Invalid]"):
                continue
            if label.startswith("Analyse"):
                m = re.match(r"Analyse\s+(\S+)", label)
                if m:
                    self._analyse[m.group(1)] = idx

    def get_action(
        self,
        observation: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> tuple[int, np.ndarray]:
        """Return (action_idx, 8-bit message).

        Priority order:
          1  Restore on connection flags
          2  Allow paths per comms_policy
          3  Block paths per comms_policy
          3.5 Analyse (post-Remove check + proactive operational scan)
          4  Remove / immediate Restore on process flags
          5  Restore persistent process flags
          6  Re-deploy decoys after Restore
          7  Deploy decoys (initial)
          8  Sleep
        """
        if not self._labels:
            return 0, np.zeros(8, dtype=bool)

        self._step += 1
        obs  = np.asarray(observation, dtype=np.float32)
        mask = action_mask

        # ── Parse per-subnet obs ─────────────────────────────────────────────
        phase = int(obs[0])
        conn_alerts  = {}
        proc_alerts  = {}
        blocked_now  = {}
        should_block = {}

        base = 1
        for si, sn in enumerate(self._subnets_in_obs):
            n_hosts  = len(self._subnet_host_list.get(sn, []))
            off_conn = _OFF_PROC + n_hosts

            blocked_vec      = obs[base + _OFF_BLOCKED : base + _OFF_COMMS]
            comms_policy_vec = obs[base + _OFF_COMMS   : base + _OFF_PROC]
            proc_flags       = obs[base + _OFF_PROC    : base + off_conn]
            conn_flags       = obs[base + off_conn     : base + off_conn + n_hosts]

            for i, src in enumerate(_SORTED_SUBNETS):
                if src == sn:
                    continue
                pair = (src, sn)
                blocked_now[pair]  = bool(blocked_vec[i])
                should_block[pair] = bool(comms_policy_vec[i])

            hosts = self._subnet_host_list.get(sn, [])
            for hi, hostname in enumerate(hosts):
                if conn_flags[hi]:
                    conn_alerts[hostname] = True
                if proc_flags[hi]:
                    proc_alerts[hostname] = True

            base += 27 + 2 * n_hosts

        # ── Update process-flag first-seen tracker ───────────────────────────
        for h in list(self._proc_flagged_step.keys()):
            if h not in proc_alerts:
                del self._proc_flagged_step[h]
        for h in proc_alerts:
            if h not in self._proc_flagged_step:
                self._proc_flagged_step[h] = self._step

        # ── Build outbound message ────────────────────────────────────────────
        msg = np.zeros(8, dtype=bool)
        if conn_alerts:
            msg[0] = True
        if not any(
            should_block[p] and not blocked_now.get(p, False)
            for p in should_block
        ):
            msg[1] = True

        from CybORG.Agents.SimpleAgents.EnterpriseHeuristicAgent import (
            _top_host, _subnet_of, _SUBNET_IDX,
        )

        top_alert_host = _top_host(list(conn_alerts) + list(proc_alerts), phase)
        if top_alert_host:
            sn = _subnet_of(top_alert_host)
            si = _SUBNET_IDX.get(sn, 0)
            for b in range(4):
                msg[2 + b] = bool((si >> b) & 1)

        for h, fs in self._proc_flagged_step.items():
            ra = self._remove_at.get(h, -1)
            if ra >= 0 and self._step > ra + REMOVE_DUR and h in proc_alerts:
                msg[6] = True
                break

        # ── Priority 1: Restore on connection flags ───────────────────────────
        for hostname in _sorted_by_priority(conn_alerts, phase):
            if self._busy(hostname):
                continue
            idx = self._restore.get(hostname)
            if idx is not None and self._valid(idx, mask):
                self._restore_at[hostname] = self._step
                self._decoy_deployed.pop(hostname, None)
                return idx, msg

        # ── Priority 2: Allow paths per comms_policy ─────────────────────────
        for pair, is_allowed in sorted(
            should_block.items(), key=lambda kv: _pair_priority(kv[0], phase)
        ):
            if not is_allowed and blocked_now.get(pair, False):
                idx = self._allow.get(pair)
                if idx is not None and self._valid(idx, mask):
                    return idx, msg

        # ── Priority 3: Block paths per comms_policy ──────────────────────────
        for pair, should_be_blocked in sorted(
            should_block.items(),
            key=lambda kv: _pair_priority(kv[0], phase),
            reverse=True,
        ):
            if should_be_blocked and not blocked_now.get(pair, False):
                idx = self._block.get(pair)
                if idx is not None and self._valid(idx, mask):
                    return idx, msg

        # ── Priority 3.5: Analyse — post-Remove + proactive scan ─────────────
        # Post-Remove check: host still flagged after Remove duration → Analyse
        # to confirm root session before escalating to Restore.
        for hostname in _sorted_by_priority(proc_alerts, phase):
            if self._busy(hostname):
                continue
            ra = self._remove_at.get(hostname, -1)
            if ra >= 0 and self._step > ra + REMOVE_DUR:
                if hostname in self._analyse and self._valid(self._analyse[hostname], mask):
                    if self._step - self._analysed_at.get(hostname, -999) > 1:
                        self._analysed_at[hostname] = self._step
                        return self._analyse[hostname], msg

        # Proactive operational zone scan (only when no alerts are pending).
        if not conn_alerts and not proc_alerts:
            target_zone = "operational_zone_a" if phase == 1 else "operational_zone_b"
            candidates = [
                h for h in self._analyse
                if target_zone in h and not self._busy(h)
            ]
            if candidates:
                h = min(candidates, key=lambda x: self._analysed_at.get(x, -999))
                if self._step - self._analysed_at.get(h, -999) > 10:
                    if self._valid(self._analyse[h], mask):
                        self._analysed_at[h] = self._step
                        return self._analyse[h], msg

        # ── Priority 4: Remove or immediate Restore on process flags ──────────
        for hostname in _sorted_by_priority(proc_alerts, phase):
            if self._busy(hostname):
                continue
            ra = self._remove_at.get(hostname, -1)
            if ra >= 0:
                if self._step > ra:
                    idx = self._restore.get(hostname)
                    if idx is not None and self._valid(idx, mask):
                        self._restore_at[hostname] = self._step
                        self._decoy_deployed.pop(hostname, None)
                        return idx, msg
                continue

            if _host_priority(hostname, phase) >= 100:
                idx = self._restore.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._restore_at[hostname] = self._step
                    self._decoy_deployed.pop(hostname, None)
                    return idx, msg
            else:
                idx = self._remove.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._remove_at[hostname] = self._step
                    return idx, msg

        # ── Priority 5: Restore persistent process flags ──────────────────────
        for hostname in _sorted_by_priority(proc_alerts, phase):
            if self._busy(hostname):
                continue
            if (
                self._remove_at.get(hostname, -1) < 0
                and self._restore_at.get(hostname, -1) < 0
            ):
                idx = self._restore.get(hostname)
                if idx is not None and self._valid(idx, mask):
                    self._restore_at[hostname] = self._step
                    self._decoy_deployed.pop(hostname, None)
                    return idx, msg

        # ── Priority 6: Re-deploy decoys after Restore ────────────────────────
        for hostname in self._server_hosts:
            rs = self._restore_at.get(hostname, -1)
            if rs >= 0 and self._step >= rs + RESTORE_DUR:
                if self._decoy_deployed.get(hostname, 0) < 1 and hostname in self._decoy:
                    idx = self._decoy[hostname]
                    if self._valid(idx, mask):
                        self._decoy_deployed[hostname] = self._decoy_deployed.get(hostname, 0) + 1
                        return idx, msg

        # ── Priority 7: Deploy decoys on server hosts (initial setup) ─────────
        for hostname in self._server_hosts:
            if self._decoy_deployed.get(hostname, 0) < 1 and hostname in self._decoy:
                idx = self._decoy[hostname]
                if self._valid(idx, mask):
                    self._decoy_deployed[hostname] = self._decoy_deployed.get(hostname, 0) + 1
                    return idx, msg

        # ── Fallback: Sleep ───────────────────────────────────────────────────
        return self._sleep_idx, msg


# ---------------------------------------------------------------------------
# Action category
# ---------------------------------------------------------------------------

def categorize_action(label: str) -> str:
    if label == "Sleep":
        return "sleep"
    if label == "Monitor":
        return "monitor"
    if label.startswith("Restore"):
        return "restore"
    if label.startswith("Remove"):
        return "remove"
    if label.startswith("Analyse"):
        return "analyse"
    if label.startswith("DeployDecoy"):
        return "decoy"
    if label.startswith("BlockTrafficZone"):
        return "block"
    if label.startswith("AllowTrafficZone"):
        return "allow"
    return "other"


# ---------------------------------------------------------------------------
# State extraction helpers
# ---------------------------------------------------------------------------

def get_state(env):
    try:
        return env.env.environment_controller.state
    except AttributeError:
        pass
    try:
        return env.environment_controller.state
    except AttributeError:
        pass
    return None


def get_hostname_subnet(state, hostname: str) -> str:
    try:
        subnet_enum = state.hostname_subnet_map.get(hostname)
        if subnet_enum is None:
            return ""
        return str(subnet_enum).lower() if subnet_enum else ""
    except Exception:
        return ""


def collect_red_sessions(state, episode_id: int, step: int) -> list[tuple]:
    rows = []
    try:
        for agent_name, sessions_dict in state.sessions.items():
            if "red" not in agent_name.lower():
                continue
            for _sid, session in sessions_dict.items():
                try:
                    hostname = str(getattr(session, "hostname", None) or "")
                    subnet = get_hostname_subnet(state, hostname)
                    is_priv = 1 if session.has_privileged_access() else 0
                    rows.append((episode_id, step, str(agent_name), hostname, subnet, is_priv))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


def collect_host_events(state, episode_id: int, step: int) -> list[tuple]:
    rows = []
    try:
        for hostname, host in state.hosts.items():
            try:
                subnet = get_hostname_subnet(state, hostname)
                proc_events = (
                    list(getattr(host.events, "process_creation", []))
                    + list(getattr(host.events, "old_process_creation", []))
                )
                conn_events = (
                    list(getattr(host.events, "network_connections", []))
                    + list(getattr(host.events, "old_network_connections", []))
                )
                has_proc = 1 if proc_events else 0
                has_conn = 1 if conn_events else 0
                rows.append((
                    episode_id, step, str(hostname), subnet,
                    has_proc, has_conn,
                    len(proc_events), len(conn_events),
                ))
            except Exception:
                pass
    except Exception:
        pass
    return rows


def collect_service_reliability(state, episode_id: int, step: int) -> list[tuple]:
    rows = []
    try:
        for hostname, host in state.hosts.items():
            try:
                for svc_name, svc in host.services.items():
                    try:
                        reliability = getattr(svc, "_percent_reliable", None)
                        if reliability is None:
                            reliability = svc.get_service_reliability()
                        rows.append((episode_id, step, str(hostname), str(svc_name), int(reliability)))
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    return rows


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(
    n_episodes: int,
    max_steps: int,
    seed: int,
    conn: sqlite3.Connection,
) -> None:
    from CybORG import CybORG
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
    from CybORG.Agents.SimpleAgents.EnterpriseGreenAgent import EnterpriseGreenAgent

    np.random.seed(seed)

    sg = EnterpriseScenarioGenerator(
        steps=max_steps,
        red_agent_class=FiniteStateRedAgent,
        green_agent_class=EnterpriseGreenAgent,
    )
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFlatWrapper(env=cyborg)

    # Initial reset to discover agent names and action catalogue
    obs_dict, _ = env.reset()
    agent_names = env.possible_agents
    subnet_hosts = getattr(env, "_cached_subnet_hosts", {})

    # Build agents
    agents: dict[str, V4AnalyseAgent] = {}
    for name in agent_names:
        ag = V4AnalyseAgent(agent_name=name)
        ag.set_action_info(
            env.action_labels(name),
            env.action_mask(name),
            subnet_hosts,
        )
        agents[name] = ag

    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        subnet_hosts = getattr(env, "_cached_subnet_hosts", {})

        for name, ag in agents.items():
            ag.reset()
            ag.set_action_info(
                env.action_labels(name),
                env.action_mask(name),
                subnet_hosts,
            )

        ep_reward = 0.0
        cumulative_reward = 0.0

        cursor = conn.execute(
            "INSERT INTO episodes (agent_type, total_reward, n_steps, seed) VALUES (?, ?, ?, ?)",
            ("v4_analyse", None, None, seed),
        )
        episode_id = cursor.lastrowid
        conn.commit()

        rows_steps: list[tuple] = []
        rows_actions: list[tuple] = []
        rows_obs: list[tuple] = []
        rows_red: list[tuple] = []
        rows_host: list[tuple] = []
        rows_svc: list[tuple] = []

        final_step = 0

        for step in range(max_steps):
            actions: dict[str, int] = {}
            action_labels_this_step: dict[str, str] = {}

            for name in agent_names:
                raw_obs = obs_dict.get(name, np.zeros(1))
                mask = np.array(env.action_mask(name), dtype=bool)
                action_idx, _msg = agents[name].get_action(raw_obs, mask)
                actions[name] = action_idx
                labels = env.action_labels(name)
                label = labels[action_idx] if action_idx < len(labels) else str(action_idx)
                action_labels_this_step[name] = label

            obs_dict, rew_dict, term_dict, trunc_dict, _info = env.step(actions)
            step_reward = sum(rew_dict.values())
            ep_reward += step_reward
            cumulative_reward += step_reward

            state = get_state(env)
            phase = None
            if state is not None:
                try:
                    phase = int(state.mission_phase)
                except Exception:
                    phase = None

            rows_steps.append((episode_id, step, phase, cumulative_reward, step_reward))

            for name in agent_names:
                idx = actions[name]
                label = action_labels_this_step[name]
                category = categorize_action(label)
                agent_rew = float(rew_dict.get(name, 0.0))
                rows_actions.append((episode_id, step, str(name), idx, label, category, agent_rew))

            if step % 10 == 0:
                for name in agent_names:
                    raw_obs = obs_dict.get(name)
                    if raw_obs is not None:
                        try:
                            obs_list = raw_obs.tolist() if hasattr(raw_obs, "tolist") else list(raw_obs)
                            rows_obs.append((episode_id, step, str(name), json.dumps(obs_list)))
                        except Exception:
                            pass

            if state is not None:
                rows_red.extend(collect_red_sessions(state, episode_id, step))
                rows_host.extend(collect_host_events(state, episode_id, step))
                rows_svc.extend(collect_service_reliability(state, episode_id, step))

            final_step = step

            if all(term_dict.get(n, False) or trunc_dict.get(n, False) for n in agent_names):
                break

        # Bulk inserts
        conn.executemany(
            "INSERT INTO steps (episode_id, step, phase, cumulative_reward, step_reward) "
            "VALUES (?,?,?,?,?)",
            rows_steps,
        )
        conn.executemany(
            "INSERT INTO actions_taken "
            "(episode_id, step, agent_name, action_idx, action_label, action_category, agent_reward) "
            "VALUES (?,?,?,?,?,?,?)",
            rows_actions,
        )
        conn.executemany(
            "INSERT INTO observations (episode_id, step, agent_name, obs_json) VALUES (?,?,?,?)",
            rows_obs,
        )
        conn.executemany(
            "INSERT INTO red_sessions "
            "(episode_id, step, agent_name, hostname, subnet, is_privileged) "
            "VALUES (?,?,?,?,?,?)",
            rows_red,
        )
        conn.executemany(
            "INSERT INTO host_events "
            "(episode_id, step, hostname, subnet, has_process_flag, has_connection_flag, "
            "n_process_events, n_connection_events) VALUES (?,?,?,?,?,?,?,?)",
            rows_host,
        )
        conn.executemany(
            "INSERT INTO service_reliability (episode_id, step, hostname, service_name, reliability_pct) "
            "VALUES (?,?,?,?,?)",
            rows_svc,
        )

        conn.execute(
            "UPDATE episodes SET total_reward=?, n_steps=? WHERE episode_id=?",
            (ep_reward, final_step + 1, episode_id),
        )
        conn.commit()

        elapsed = time.perf_counter() - t0
        print(
            f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:9.1f}  "
            f"steps={final_step+1}  elapsed={elapsed:.1f}s"
        )

    total_elapsed = time.perf_counter() - t0
    print(f"\nDone. {n_episodes} episodes in {total_elapsed:.1f}s — DB: {DB_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Log CybORG CC4 V4AnalyseAgent telemetry to SQLite (data/v4_analyse.db)"
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes (default 20)")
    parser.add_argument("--steps",    type=int, default=500, help="Max steps per episode (default 500)")
    parser.add_argument("--seed",     type=int, default=42,  help="RNG seed (default 42)")
    args = parser.parse_args()

    print(
        f"\nV4AnalyseAgent logger — episodes={args.episodes}  "
        f"steps={args.steps}  seed={args.seed}"
    )
    print(f"Writing to: {DB_PATH}\n")

    conn = open_db(DB_PATH)
    try:
        run_episodes(
            n_episodes=args.episodes,
            max_steps=args.steps,
            seed=args.seed,
            conn=conn,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
