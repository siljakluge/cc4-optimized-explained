"""
Initialize and populate the CC4 findings database.

Creates data/findings.db with tables for experiments, ablations, attack chains,
decision rules, findings, and performance comparisons. Populates from
docs/optimality_analysis_data.json and prior analysis documents.

Usage:
    python scripts/init_findings_db.py
"""

import json
import os
import sqlite3
import sys

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "findings.db")
DATA_JSON = os.path.join(
    os.path.dirname(__file__), "..", "docs", "optimality_analysis_data.json"
)


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables in the findings database."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT,
            description TEXT,
            episodes INTEGER,
            seed INTEGER,
            mean_reward REAL,
            std_reward REAL,
            ci95 REAL
        );

        CREATE TABLE IF NOT EXISTS ablations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            feature_disabled TEXT NOT NULL,
            mean_reward REAL,
            std_reward REAL,
            delta_vs_baseline REAL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );

        CREATE TABLE IF NOT EXISTS attack_chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            min_steps INTEGER,
            expected_steps INTEGER,
            blue_detection_step INTEGER,
            optimal_response TEXT
        );

        CREATE TABLE IF NOT EXISTS decision_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            priority INTEGER NOT NULL,
            condition TEXT NOT NULL,
            action TEXT NOT NULL,
            rationale TEXT,
            phase_applicability TEXT
        );

        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            finding TEXT NOT NULL,
            evidence TEXT,
            impact TEXT,
            actionable INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS performance_comparison (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            version TEXT,
            mean_reward REAL,
            std_reward REAL,
            episodes INTEGER,
            seed INTEGER
        );
        """
    )
    conn.commit()


def load_json_data() -> dict:
    """Load the optimality analysis JSON data."""
    with open(DATA_JSON, "r") as f:
        return json.load(f)


def populate_experiments(conn: sqlite3.Connection, data: dict) -> dict:
    """Populate experiments table. Returns mapping of name -> id."""
    experiments = [
        (
            "baseline_v9.1",
            "2026-04-08",
            "EnterpriseHeuristicAgent v9.1 full evaluation",
            100,
            42,
            data["baseline"]["mean"],
            data["baseline"]["std"],
            data["baseline"]["ci95"],
        ),
        (
            "sleep_baseline",
            "2026-04-08",
            "SleepAgent baseline (blue does nothing)",
            50,
            42,
            data["sleep_baseline"]["mean"],
            data["sleep_baseline"]["std"],
            None,
        ),
        (
            "ablation_study",
            "2026-04-08",
            "6-variant ablation study (n=30 each)",
            30,
            42,
            data["baseline"]["mean"],
            data["baseline"]["std"],
            None,
        ),
        (
            "action_budget",
            "2026-04-08",
            "Action allocation analysis across 30 episodes",
            30,
            42,
            data["baseline"]["mean"],
            data["baseline"]["std"],
            None,
        ),
        (
            "reward_curves",
            "2026-04-08",
            "Per-step temporal reward analysis",
            30,
            42,
            data["baseline"]["mean"],
            data["baseline"]["std"],
            None,
        ),
        (
            "seed_sensitivity",
            "2026-04-08",
            "Cross-seed robustness verification (5 seeds x 30 episodes)",
            150,
            None,
            -1044.7,
            50.4,
            None,
        ),
        (
            "reward_distribution",
            "2026-04-08",
            "Episode reward distribution shape analysis",
            100,
            42,
            data["baseline"]["mean"],
            data["baseline"]["std"],
            None,
        ),
    ]

    id_map = {}
    for exp in experiments:
        cursor = conn.execute(
            """INSERT INTO experiments
               (name, date, description, episodes, seed, mean_reward, std_reward, ci95)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            exp,
        )
        id_map[exp[0]] = cursor.lastrowid

    conn.commit()
    return id_map


def populate_ablations(
    conn: sqlite3.Connection, data: dict, exp_ids: dict
) -> None:
    """Populate ablations table from experimental data."""
    ablation_exp_id = exp_ids["ablation_study"]
    baseline_mean = -1037.2  # ablation baseline (n=30)

    ablation_rows = []
    for feature, values in data["ablations"].items():
        delta = values["mean"] - baseline_mean
        ablation_rows.append(
            (ablation_exp_id, feature, values["mean"], values["std"], delta)
        )

    conn.executemany(
        """INSERT INTO ablations
           (experiment_id, feature_disabled, mean_reward, std_reward, delta_vs_baseline)
           VALUES (?, ?, ?, ?, ?)""",
        ablation_rows,
    )
    conn.commit()


def populate_attack_chains(conn: sqlite3.Connection) -> None:
    """Populate attack chains from red agent analysis."""
    chains = [
        (
            "Starting host (U) to Impact: PrivEsc(2) + Impact(2)",
            4,
            4,
            None,
            "Restore immediately on any malfile-only signal (P1c)",
        ),
        (
            "New host fastest: DRS(1) + AggrSvc(1) + Exploit(4) + PrivEsc(2) + Impact(2)",
            10,
            10,
            5,
            "Restore on conn+malfile at exploit step (P1)",
        ),
        (
            "New host expected path with probabilistic action selection",
            10,
            23,
            8,
            "Decoys slow exploit phase; Restore on detection (P1/P4)",
        ),
        (
            "Post-Restore red recovery: KD -> re-scan -> re-exploit",
            11,
            15,
            8,
            "Redeploy decoys immediately after Restore (P6)",
        ),
        (
            "Decoy-aware path: K -> S -> DD -> SD -> Exploit(PID-selective) -> UD -> PrivEsc -> RD",
            12,
            20,
            8,
            "Upstream DECOYS_BYPASSED message triggers P1b Restore",
        ),
        (
            "DegradeServices: Root(R) -> Degrade(2 steps) -> silent reliability reduction",
            2,
            2,
            None,
            "No detection possible; only mitigated by fast Restore after root detection",
        ),
        (
            "Phishing entry: Green PhishingEmail -> red U session (bypasses blocks)",
            1,
            1,
            2,
            "Restore on subsequent proc_flag/conn_flag detection (P1/P4)",
        ),
        (
            "Lateral movement via server_host_0 chokepoint",
            10,
            18,
            5,
            "Block subnet paths (P3); Restore chokepoint server on detection (P1)",
        ),
    ]

    conn.executemany(
        """INSERT INTO attack_chains
           (description, min_steps, expected_steps, blue_detection_step, optimal_response)
           VALUES (?, ?, ?, ?, ?)""",
        chains,
    )
    conn.commit()


def populate_decision_rules(conn: sqlite3.Connection) -> None:
    """Populate the priority-based decision rules."""
    rules = [
        (
            1,
            "conn_flag=1 AND (malfile=1 OR proc_flag=1)",
            "Restore",
            "Confirmed red exploit: conn+malfile is 100% reliable. Remove cannot evict root sessions.",
            "All phases",
        ),
        (
            1,
            "conn_flag=1 AND malfile=0 AND proc_flag=0 AND no_decoy_deployed",
            "Restore",
            "Connection without decoy coverage: unknown source, assume real threat.",
            "All phases",
        ),
        (
            1,
            "conn_flag=1 AND malfile=0 AND proc_flag=0 AND decoy_deployed AND upstream_decoys_bypassed",
            "Restore",
            "Decoy-bypassed conn-only: red has PID intel from adjacent zone.",
            "Phase 1-2 (upstream mapping required)",
        ),
        (
            1,
            "conn_flag=1 AND malfile=0 AND proc_flag=0 AND decoy_deployed AND NOT upstream_decoys_bypassed",
            "Skip (Sleep)",
            "Decoy hit: red wasted 4 steps, no session created.",
            "All phases",
        ),
        (
            1,
            "malfile=1 AND conn_flag=0 AND proc_flag=0",
            "Restore",
            "Pure malfile = PrivilegeEscalate signature. Root session present. Only Restore clears root.",
            "All phases",
        ),
        (
            2,
            "Path blocked that comms_policy says should be open",
            "AllowTrafficZone",
            "Fix stale blocks from previous phase.",
            "Phase transitions",
        ),
        (
            3,
            "Path open that comms_policy says should be blocked",
            "BlockTrafficZone",
            "Enforce network isolation per mission requirements. Blocking is free (cost=0).",
            "Phase 1-2",
        ),
        (
            4,
            "proc_flag=1 AND flag_age=0 AND no prior Remove",
            "Remove",
            "First response to proc-only alert. 33-54% are green FP. Remove is free, 3 steps.",
            "All phases",
        ),
        (
            4,
            "proc_flag=1 AND prior Remove issued AND flag persists",
            "Restore",
            "Root session survived Remove. Escalate to Restore.",
            "All phases",
        ),
        (
            4,
            "proc_flag=1 AND upstream_red_count >= 3 (T3 escalation)",
            "Restore (immediate, skip Remove)",
            "Upstream zone saturated. No time for Remove; go straight to Restore.",
            "Phase 1-2 (upstream mapping required)",
        ),
        (
            4,
            "proc_flag=1 AND flag_age >= 1 AND threshold=1",
            "Restore",
            "Flag persisted 1+ step. Green FPs clear within 1 step, so this is real red.",
            "All phases",
        ),
        (
            6,
            "Host was Restored AND has fewer than MAX_DECOYS decoys AND not busy",
            "DeployDecoy",
            "Restore wipes all decoys. Redeploy to maintain 75% exploit interception rate.",
            "All phases",
        ),
        (
            7,
            "Host has < MAX_DECOYS decoys AND no prior Restore AND not busy",
            "DeployDecoy",
            "Initial decoy deployment in priority order (OZ targets first).",
            "Primarily Phase 0",
        ),
        (
            8,
            "No alerts, all decoys deployed, all blocks applied",
            "Sleep",
            "Nothing to do. Idle is optimal; spurious actions waste budget.",
            "All phases",
        ),
    ]

    conn.executemany(
        """INSERT INTO decision_rules
           (priority, condition, action, rationale, phase_applicability)
           VALUES (?, ?, ?, ?, ?)""",
        rules,
    )
    conn.commit()


def populate_findings(conn: sqlite3.Connection) -> None:
    """Populate key findings from all analyses."""
    findings = [
        # Performance findings
        (
            "performance",
            "v9.1 captures 97.6% of achievable improvement over SleepAgent",
            "baseline=-1039, sleep=-30579, oracle_floor~-300; capture=(29540/30279)=97.6%",
            "Near-optimal within heuristic framework",
            0,
        ),
        (
            "performance",
            "Best single episode is -545, within 1.8x of theoretical oracle floor (-300)",
            "Experiment 1: min=-545 across 100 episodes",
            "Demonstrates ceiling is structural, not strategic",
            0,
        ),
        (
            "performance",
            "Cross-seed CV is 4.8% across 5 seeds, indicating robust strategy",
            "Seeds 42/123/456/789/1337: means range [-1095, -952]",
            "Strategy is not overfit to specific random seed",
            0,
        ),
        # Phase analysis
        (
            "phase_analysis",
            "Phase 0 accounts for 50-58% of total losses despite lowest penalty multipliers",
            "Phase 0: -3.10/step vs Phase 1: -1.70/step vs Phase 2: -1.42/step",
            "Critical; no blocking available in Phase 0",
            0,
        ),
        (
            "phase_analysis",
            "Phase 1->2 transition (steps 334-336) produces worst per-step rewards",
            "Step 334: -16.83, Step 335: -12.83, Step 336: -11.67",
            "New blocking rules needed; red may have root from Phase 0/1",
            1,
        ),
        # Ablation findings
        (
            "ablation",
            "Restore is essential: removal causes 95.4% degradation (2073% worse)",
            "no_restore: -22559 vs baseline -1037",
            "Most critical feature; cannot evict root sessions without Restore",
            0,
        ),
        (
            "ablation",
            "Malfile detection is critical: removal causes 37.6% degradation (1077% worse)",
            "no_malfile: -12207 vs baseline -1037",
            "Validates BlueFlatWrapperV2 as high-impact design choice",
            0,
        ),
        (
            "ablation",
            "Decoys reduce damage by 81.6%: 75% blind-exploit failure rate",
            "no_decoys: -1883 vs baseline -1037",
            "3 decoys per host forces red to waste ~12 steps per successful exploit",
            0,
        ),
        (
            "ablation",
            "Remove is expendable: Restore-only (-932) slightly outperforms full agent (-1037)",
            "no_remove: -932 +/- 222 vs baseline -1037 +/- 610",
            "Within noise; Remove serves as hedge against green FPs",
            1,
        ),
        (
            "ablation",
            "Blocking provides 12.9% improvement",
            "no_blocking: -1172 vs baseline -1037",
            "Only affects Phases 1-2 (333 of 500 steps)",
            0,
        ),
        (
            "ablation",
            "Messaging impact is within noise (-982 vs -1037 baseline)",
            "no_messaging: -982 vs baseline -1037",
            "Upstream warnings affect only P4 escalation thresholds",
            0,
        ),
        # Red agent findings
        (
            "red_agent",
            "Red starts with user session in contractor_network; can reach Impact in 4 steps",
            "FSM: U -> PrivEsc(2) -> R -> Impact(2) = 4 steps minimum",
            "Extremely fast initial attack on starting host",
            0,
        ),
        (
            "red_agent",
            "Red prioritizes servers with 75% probability",
            "FiniteStateRedAgent._choose_host: prioritise_servers parameter",
            "Explains why server_host_0 dominates compromise counts",
            0,
        ),
        (
            "red_agent",
            "Decoy-aware red (UD state) attempts PrivEsc with 100% probability",
            "FSM probability matrix: UD row = [None,...,1.00,...] for PrivEsc",
            "Decoy knowledge makes red more dangerous, not less",
            0,
        ),
        (
            "red_agent",
            "Post-Restore red drops to KD state, needs ~11-15 steps to re-Impact",
            "_session_removal_state_change: host reset to KD",
            "Provides recovery window for blue to redeploy decoys",
            0,
        ),
        # Green agent findings
        (
            "green_agent",
            "Green FP rate: 1% per action per agent, producing 33-54% of all proc_flag alerts",
            "EnterpriseGreenAgent: fp_detection_rate=0.01",
            "Indistinguishable from real red activity without environment changes",
            0,
        ),
        (
            "green_agent",
            "Phishing rate: 1% per GreenLocalWork creates real red sessions that bypass all blocks",
            "PhishingEmail action bypasses BlockTrafficZone",
            "Creates ~50 new footholds per 500-step episode across 30 hosts",
            0,
        ),
        # Structural constraints
        (
            "structural",
            "Irreducible oracle floor estimated at ~-300 per episode",
            "Phase 0 uncontested: -200..400, phishing: -50..150, restore costs: -20, detection delay: -20..50",
            "Sets theoretical performance ceiling for any blue strategy",
            0,
        ),
        (
            "structural",
            "Single action per step prevents optimal multi-host response",
            "FM-2: When 2+ hosts flagged simultaneously, lower-priority host waits 5+ steps",
            "Fundamental architecture limitation",
            0,
        ),
        (
            "structural",
            "DegradeServices is permanent and completely invisible in observations",
            "No observation signal for service reliability degradation",
            "Only mitigated by faster Restore after root detection",
            0,
        ),
        # Strategy findings
        (
            "strategy",
            "12 systematic improvement strategies tested: all neutral or regressive",
            "Phase 0 upstream, decoy reduction, block ordering, FP suppression, pre-emptive Restore, etc.",
            "Confirms v9.1 is at heuristic Pareto frontier",
            0,
        ),
        (
            "strategy",
            "77% Sleep rate is optimal: no productive action available most of the time",
            "Action budget: 57682 Sleep / 74850 total = 77.1%",
            "Spurious actions waste steps that could address real threats",
            0,
        ),
        (
            "strategy",
            "Cost-benefit: missed Impact on OZ during mission (-100+) far exceeds unnecessary Restore (-51)",
            "OZ RIA=-10/step; Restore cost=-1 + 5*-10 LWF = -51 worst case",
            "Always err on side of Restoring in mission zones",
            0,
        ),
        # Oracle findings
        (
            "oracle",
            "Oracle v1 with perfect information scored -1558, WORSE than heuristic (-1039)",
            "Perfect info != perfect play; oracle did not optimize action ordering",
            "Information value depends on decision quality, not just information quantity",
            1,
        ),
    ]

    conn.executemany(
        """INSERT INTO findings
           (category, finding, evidence, impact, actionable)
           VALUES (?, ?, ?, ?, ?)""",
        findings,
    )
    conn.commit()


def populate_performance_comparison(conn: sqlite3.Connection, data: dict) -> None:
    """Populate performance comparison across all known agent versions."""
    agents = [
        ("SleepAgent", "baseline", data["sleep_baseline"]["mean"], data["sleep_baseline"]["std"], 50, 42),
        ("EnterpriseHeuristicAgent", "v7", -5025.0, 1570.0, 30, 42),
        ("EnterpriseHeuristicAgent", "v9", -1100.0, 400.0, 30, 42),
        ("EnterpriseHeuristicAgent", "v9.1", data["baseline"]["mean"], data["baseline"]["std"], 100, 42),
        ("EnterpriseHeuristicAgent", "v9.1 (30ep)", -868.7, 292.0, 30, 42),
        ("OracleAgent", "v1", -1558.0, None, 100, 42),
        ("Restore-only (no Remove)", "ablation", -931.7, 221.7, 30, 42),
    ]

    conn.executemany(
        """INSERT INTO performance_comparison
           (agent_name, version, mean_reward, std_reward, episodes, seed)
           VALUES (?, ?, ?, ?, ?, ?)""",
        agents,
    )
    conn.commit()


def populate_seed_experiments(conn: sqlite3.Connection, data: dict) -> None:
    """Add seed sensitivity experiments."""
    for seed_str, values in data["seed_sensitivity"].items():
        conn.execute(
            """INSERT INTO experiments
               (name, date, description, episodes, seed, mean_reward, std_reward, ci95)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"seed_sensitivity_{seed_str}",
                "2026-04-08",
                f"Seed sensitivity test with seed={seed_str}",
                30,
                int(seed_str),
                values["mean"],
                values["std"],
                None,
            ),
        )
    conn.commit()


def verify_database(conn: sqlite3.Connection) -> None:
    """Print summary statistics to verify the database was populated correctly."""
    tables = [
        "experiments",
        "ablations",
        "attack_chains",
        "decision_rules",
        "findings",
        "performance_comparison",
    ]

    print("\n=== Database Verification ===")
    for table in tables:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} rows")

    # Sample queries
    print("\n--- Performance Comparison ---")
    cursor = conn.execute(
        "SELECT agent_name, version, mean_reward FROM performance_comparison ORDER BY mean_reward"
    )
    for row in cursor:
        print(f"  {row[0]} {row[1]}: {row[2]}")

    print("\n--- Ablation Impact (sorted by delta) ---")
    cursor = conn.execute(
        "SELECT feature_disabled, mean_reward, delta_vs_baseline FROM ablations ORDER BY delta_vs_baseline"
    )
    for row in cursor:
        print(f"  No {row[0]}: {row[1]:.1f} (delta: {row[2]:+.1f})")

    print("\n--- Findings by Category ---")
    cursor = conn.execute(
        "SELECT category, COUNT(*) FROM findings GROUP BY category ORDER BY COUNT(*) DESC"
    )
    for row in cursor:
        print(f"  {row[0]}: {row[1]} findings")


def main() -> None:
    """Initialize and populate the findings database."""
    # Ensure data directory exists
    db_dir = os.path.dirname(os.path.abspath(DB_PATH))
    os.makedirs(db_dir, exist_ok=True)

    # Remove existing database to start fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing database: {DB_PATH}")

    # Load source data
    if not os.path.exists(DATA_JSON):
        print(f"ERROR: Data file not found: {DATA_JSON}")
        sys.exit(1)

    data = load_json_data()
    print(f"Loaded data from {DATA_JSON}")

    # Create and populate database
    conn = sqlite3.connect(DB_PATH)
    try:
        print("Creating schema...")
        create_schema(conn)

        print("Populating experiments...")
        exp_ids = populate_experiments(conn, data)

        print("Populating seed sensitivity experiments...")
        populate_seed_experiments(conn, data)

        print("Populating ablations...")
        populate_ablations(conn, data, exp_ids)

        print("Populating attack chains...")
        populate_attack_chains(conn)

        print("Populating decision rules...")
        populate_decision_rules(conn)

        print("Populating findings...")
        populate_findings(conn)

        print("Populating performance comparison...")
        populate_performance_comparison(conn, data)

        # Verify
        verify_database(conn)

        print(f"\nDatabase created successfully: {os.path.abspath(DB_PATH)}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
