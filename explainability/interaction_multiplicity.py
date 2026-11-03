#!/usr/bin/env python3
"""Multiple-comparison correction for the pairwise interaction indices.

Ten pairs are tested per policy, so calling a pair significant whenever its 95%
CI excludes zero gives a family-wise error rate well above 5%. This adds:
  * a two-sided bootstrap p-value per pair, p = 2*min(P(I*<=0), P(I*>=0)),
    floored at 1/B;
  * Holm step-down correction across the 10 pairs (uniformly more powerful than
    Bonferroni at the same family-wise level).

Resampling unit is the seed, matching shapley_bootstrap.py. No episodes run.

Usage: python explainability/interaction_multiplicity.py <shapley_*.json> [B] [seed]
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from math import factorial

import numpy as np

N = 5
ALPHA = 0.05
ZONES = {0: "RZ-A", 1: "OZ-A", 2: "RZ-B", 3: "OZ-B", 4: "corporate"}


def key_to_mask(k: str) -> frozenset:
    return frozenset(int(c) for c in k if c != ".")


def interaction_from_v(v: dict, i: int, j: int) -> float:
    others = [k for k in range(N) if k not in (i, j)]
    tot = 0.0
    for r in range(len(others) + 1):
        for S in combinations(others, r):
            S = frozenset(S)
            w = factorial(len(S)) * factorial(N - len(S) - 2) / factorial(N - 1)
            tot += w * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
    return tot


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda k: pvals[k])
    adj, running = [0.0] * m, 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    path = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)

    d = json.load(open(path))
    raw = {key_to_mask(k): np.asarray(v["total"], float) for k, v in d["v_raw"].items()}
    n_seeds = len(next(iter(raw.values())))
    pairs = list(combinations(range(N), 2))

    vhat = {m: a.mean() for m, a in raw.items()}
    point = np.array([interaction_from_v(vhat, i, j) for i, j in pairs])

    bs = np.zeros((B, len(pairs)))
    for b in range(B):
        idx = rng.integers(0, n_seeds, n_seeds)
        vb = {m: a[idx].mean() for m, a in raw.items()}
        for pi, (i, j) in enumerate(pairs):
            bs[b, pi] = interaction_from_v(vb, i, j)

    p_raw = []
    for pi in range(len(pairs)):
        col = bs[:, pi]
        p = 2.0 * min((col <= 0).mean(), (col >= 0).mean())
        p_raw.append(max(min(p, 1.0), 1.0 / B))
    p_adj = holm(p_raw)

    print(f"\n{path}\n{n_seeds} seeds, B={B}, {len(pairs)} pairs, alpha={ALPHA}\n")
    print(f"  {'pair':16s}{'I(i,j)':>9s}{'95% CI':>22s}{'p_raw':>9s}{'p_holm':>9s}  verdict")
    n_sig = 0
    for pi in sorted(range(len(pairs)), key=lambda k: -abs(point[k])):
        i, j = pairs[pi]
        lo, hi = np.percentile(bs[:, pi], [2.5, 97.5])
        sig = p_adj[pi] < ALPHA
        n_sig += sig
        verdict = "significant" if sig else (
            "n.s. after Holm" if (lo > 0 or hi < 0) else "n.s.")
        print(f"  {ZONES[i]+'x'+ZONES[j]:16s}{point[pi]:+9.1f}  [{lo:+8.1f},{hi:+8.1f}]"
              f"{p_raw[pi]:9.4f}{p_adj[pi]:9.4f}  {verdict}")
    print(f"\n  {n_sig}/{len(pairs)} significant after Holm correction")


if __name__ == "__main__":
    main()
