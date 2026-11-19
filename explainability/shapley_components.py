#!/usr/bin/env python3
"""Component-decomposed Shapley attribution.

The team reward is a sum of components (LWF, ASF, RIA, EXP, PRIV, DEG, ANALYSE,
and an action-cost remainder). The Shapley value is linear in the value
function, so phi_i decomposes exactly:  phi_i = sum_c phi_i^c. This runs the 32
coalitions once, capturing per-component coalition values, and reports phi_i
split by component -- the journal centrepiece.

Components are captured from BlueRewardMachine's own per-step log
(CYBORG_REWARD_LOG_PATH, written unconditionally): reward_list is the per-subnet
per-component breakdown. We set a unique log file per episode, sum it, and let
an 'other' term absorb any gap to the scalar return so components reconcile to
v(S) exactly.

Usage:
  python explainability/shapley_components.py --backend heuristic --seeds 20 --profile fsm_default
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H                                                   # noqa: E402

from CybORG import CybORG                                             # noqa: E402
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent            # noqa: E402
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator     # noqa: E402

N = H.N_AGENTS
ZONES = {0: "RZ-A", 1: "OZ-A", 2: "RZ-B", 3: "OZ-B", 4: "corporate"}
COMPONENTS = ["LWF", "ASF", "RIA", "EXP", "PRIV", "DEG", "ANALYSE"]


def _parse_components(log_path: Path) -> dict:
    """Sum the per-subnet per-component reward over all steps in the log."""
    acc = {c: 0.0 for c in COMPONENTS}
    if not log_path.exists():
        return acc
    for line in log_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        for _subnet, comps in (entry.get("reward_list") or {}).items():
            for c in COMPONENTS:
                acc[c] += float(comps.get(c, 0) or 0)
    return acc


def run_episode(active_idx, submission, seed, steps, red_cls, logdir):
    """Scalar v(S) plus per-component totals for one coalition episode."""
    active = set(active_idx)
    log_path = Path(logdir) / f"rew_{'-'.join(map(str, sorted(active)))}_{seed}.jsonl"
    if log_path.exists():
        log_path.unlink()
    os.environ["CYBORG_REWARD_LOG_PATH"] = str(log_path)

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent,
        red_agent_class=red_cls, steps=steps)
    env = submission.Submission.wrap(CybORG(sg, "sim", seed=seed))
    agents = submission.Submission.AGENTS
    obs, _ = env.reset()
    sleep_idx = {a: env.action_labels(a).index("Sleep") for a in H.AGENT_NAMES}

    total = 0.0
    for _ in range(steps):
        live = [a for a in env.agents if a in obs]
        actions = {}
        for a in live:
            i = H.AGENT_NAMES.index(a)
            if i in active:
                actions[a] = agents[a].get_action(obs[a], env.action_space(a))
            else:
                actions[a] = sleep_idx[a]
                agents[a]._last_message = None
        obs, rew, term, trunc, _ = env.step(actions)
        # shared team reward: take one agent's value, not the sum (see harness.py)
        total += float(next(iter(rew.values()))) if rew else 0.0
        if (term and all(term.values())) or (trunc and all(trunc.values())):
            break

    comp = _parse_components(log_path)
    comp["cost"] = total - sum(comp.values())   # = action cost (Restore -1 each); reconciles exactly
    log_path.unlink(missing_ok=True)
    return total, comp


def shapley(vmean: dict) -> np.ndarray:
    phi = np.zeros(N)
    for i in range(N):
        for r in range(N):
            for S in combinations([j for j in range(N) if j != i], r):
                S = frozenset(S)
                w = factorial(len(S)) * factorial(N - len(S) - 1) / factorial(N)
                phi[i] += w * (vmean[S | {i}] - vmean[S])
    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["heuristic"], default="heuristic")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--profile", default="fsm_default")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    submission = H._load_submission()
    red_cls = H.red_registry()[args.profile]
    seeds = [args.base_seed + k for k in range(args.seeds)]
    masks = [frozenset(c) for r in range(N + 1) for c in combinations(range(N), r)]
    all_comp = COMPONENTS + ["cost", "total"]

    logdir = tempfile.mkdtemp(prefix="shapcomp_")
    v = {m: {c: [] for c in all_comp} for m in masks}
    t0 = time.time()
    for mi, m in enumerate(masks):
        for s in seeds:
            tot, comp = run_episode(m, submission, s, args.steps, red_cls, logdir)
            for c in COMPONENTS + ["cost"]:
                v[m][c].append(comp[c])
            v[m]["total"].append(tot)
        active = "".join(str(i) if i in m else "." for i in range(N))
        print(f"[{mi + 1}/32] active={active} "
              f"v={statistics.mean(v[m]['total']):+9.1f} ({time.time() - t0:.0f}s)",
              flush=True)

    vmean = {c: {m: statistics.mean(v[m][c]) for m in masks} for c in all_comp}
    phi = {c: shapley(vmean[c]) for c in all_comp}

    v_raw = {
        "".join(str(i) if i in m else "." for i in range(N)):
            {c: v[m][c] for c in all_comp}
        for m in masks
    }
    out = {
        "backend": args.backend, "profile": args.profile,
        "seeds": args.seeds, "steps": args.steps, "seed_list": seeds,
        "components": all_comp,
        "phi": {c: phi[c].tolist() for c in all_comp},
        "v_raw": v_raw,
        "zones": ZONES,
        "note": "fixed-env (cc4-optimized); component-decomposed Shapley",
    }
    outpath = args.out or f"explainability/results/shapley_components_{args.backend}_{args.profile}.json"
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outpath, "w"), indent=1)

    # report
    full, empty = frozenset(range(N)), frozenset()
    print("\n" + "=" * 78)
    hdr = f"  {'agent':11s}" + "".join(f"{c:>9s}" for c in ["total"] + COMPONENTS + ["cost"])
    print(hdr)
    for i in sorted(range(N), key=lambda k: -phi["total"][k]):
        row = f"  {ZONES[i]:11s}"
        for c in ["total"] + COMPONENTS + ["cost"]:
            row += f"{phi[c][i]:+9.0f}"
        print(row)
    # linearity check: sum of component phi == total phi
    lin = max(abs(phi["total"][i] - sum(phi[c][i] for c in COMPONENTS + ["cost"]))
              for i in range(N))
    print(f"\nlinearity check max|phi_total - sum_c phi_c| = {lin:.2e}")
    print(f"saved -> {outpath}")


if __name__ == "__main__":
    main()
