#!/usr/bin/env python3
"""Exact Shapley-value attribution over all agent coalitions.

With 5 agents there are 2^5 = 32 coalitions, few enough to evaluate exhaustively
rather than by Monte-Carlo sampling. For each coalition S the agents in S run
the real policy and the rest are replaced by Sleep (present but passive); v(S)
is the mean team return over the evaluation seeds. The Shapley value is then

    phi_i = sum_{S subseteq N\\{i}} |S|!(n-|S|-1)!/n! [ v(S+i) - v(S) ]

Saves the per-seed coalition values ('v_raw') so uncertainty can be
bootstrapped downstream (shapley_bootstrap.py, interaction_multiplicity.py)
without re-running episodes.

Usage:
  python explainability/shapley.py --backend heuristic --seeds 20 --profile fsm_default
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import sys
from itertools import combinations
from math import factorial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H                                                   # noqa: E402

N = H.N_AGENTS
ZONES = {0: "RZ-A", 1: "OZ-A", 2: "RZ-B", 3: "OZ-B", 4: "corporate"}


def shapley(vmean: dict) -> list:
    """vmean: frozenset(active idx) -> mean value. Returns [phi_0..phi_{n-1}]."""
    phi = [0.0] * N
    for i in range(N):
        others = [j for j in range(N) if j != i]
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                S = frozenset(S)
                w = factorial(len(S)) * factorial(N - len(S) - 1) / factorial(N)
                phi[i] += w * (vmean[S | {i}] - vmean[S])
    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["heuristic", "marl"], default="heuristic")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--profile", default="fsm_default")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_episode, team = H.make_backend(args.backend)
    red_cls = H.red_registry()[args.profile]
    seeds = [args.base_seed + k for k in range(args.seeds)]
    masks = [frozenset(c) for r in range(N + 1) for c in combinations(range(N), r)]  # 32

    v = {m: [] for m in masks}
    t0 = time.time()
    for mi, m in enumerate(masks):
        for s in seeds:
            v[m].append(run_episode(m, team, s, args.steps, red_cls, None))
        active = "".join(str(i) if i in m else "." for i in range(N))
        print(f"[{mi + 1}/32] active={active} "
              f"v={statistics.mean(v[m]):+9.1f} ({time.time() - t0:.0f}s)", flush=True)

    vmean = {m: statistics.mean(v[m]) for m in masks}
    phi = shapley(vmean)
    full, empty = frozenset(range(N)), frozenset()

    v_raw = {
        "".join(str(i) if i in m else "." for i in range(N)): {"total": v[m]}
        for m in masks
    }
    out = {
        "backend": args.backend, "profile": args.profile,
        "seeds": args.seeds, "steps": args.steps, "base_seed": args.base_seed,
        "seed_list": seeds,
        "v_full": vmean[full], "v_empty": vmean[empty],
        "phi": {"total": phi},
        "v_raw": v_raw,
        "zones": ZONES,
        "note": "fixed-env (cc4-optimized); not comparable to the paper's numbers",
    }
    outpath = args.out or f"explainability/shapley_{args.backend}_{args.profile}.json"
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outpath, "w"), indent=1)

    print("\n" + "=" * 60)
    print(f"v(empty)={vmean[empty]:+.1f}  v(full)={vmean[full]:+.1f}  "
          f"improvement={vmean[full] - vmean[empty]:+.1f}")
    for i in sorted(range(N), key=lambda k: -phi[k]):
        print(f"  {ZONES[i]:10s} phi={phi[i]:+9.1f}")
    eff = abs(sum(phi) - (vmean[full] - vmean[empty]))
    print(f"efficiency check |sum phi - (v(N)-v(0))| = {eff:.2e}")
    print(f"saved -> {outpath}")


if __name__ == "__main__":
    main()
