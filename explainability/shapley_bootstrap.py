#!/usr/bin/env python3
"""Bootstrap confidence intervals for Shapley values and interaction indices.

The 2^5 coalition values are each a mean over evaluation seeds, so phi_i and
I(i,j) carry sampling error. We resample the seeds with replacement B times;
for each resample we recompute every coalition mean and then phi_i / I(i,j),
applying the SAME resampled seed indices to all coalitions (common random
numbers -- scenario variation cancels within each marginal contribution rather
than inflating the interval). Reports percentile 95% CIs.

Reads the per-seed 'v_raw' from a shapley_*.json (shapley.py). No episodes run.

Usage: python explainability/shapley_bootstrap.py explainability/shapley_heuristic_fsm_default.json [B]
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from math import factorial

import numpy as np

N = 5
ZONES = {0: "RZ-A", 1: "OZ-A", 2: "RZ-B", 3: "OZ-B", 4: "corporate"}


def key_to_mask(k: str) -> frozenset:
    return frozenset(int(c) for c in k if c != ".")


def shapley_from_v(v: dict) -> np.ndarray:
    phi = np.zeros(N)
    for i in range(N):
        others = [j for j in range(N) if j != i]
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                S = frozenset(S)
                w = factorial(len(S)) * factorial(N - len(S) - 1) / factorial(N)
                phi[i] += w * (v[S | {i}] - v[S])
    return phi


def interaction_from_v(v: dict, i: int, j: int) -> float:
    others = [k for k in range(N) if k not in (i, j)]
    tot = 0.0
    for r in range(len(others) + 1):
        for S in combinations(others, r):
            S = frozenset(S)
            w = factorial(len(S)) * factorial(N - len(S) - 2) / factorial(N - 1)
            tot += w * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
    return tot


def main():
    path = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    rng = np.random.default_rng(0)

    d = json.load(open(path))
    raw = {key_to_mask(k): np.asarray(val["total"], float) for k, val in d["v_raw"].items()}
    n_seeds = len(next(iter(raw.values())))
    pairs = list(combinations(range(N), 2))

    vhat = {m: a.mean() for m, a in raw.items()}
    phi_hat = shapley_from_v(vhat)
    I_hat = np.array([interaction_from_v(vhat, i, j) for i, j in pairs])

    phi_bs = np.zeros((B, N))
    I_bs = np.zeros((B, len(pairs)))
    for b in range(B):
        idx = rng.integers(0, n_seeds, n_seeds)
        vb = {m: a[idx].mean() for m, a in raw.items()}
        phi_bs[b] = shapley_from_v(vb)
        for pi, (i, j) in enumerate(pairs):
            I_bs[b, pi] = interaction_from_v(vb, i, j)

    print(f"{path}\n{n_seeds} seeds, B={B} resamples\n")
    print("Shapley values (95% CI):")
    for i in np.argsort(-phi_hat):
        lo, hi = np.percentile(phi_bs[:, i], [2.5, 97.5])
        flag = "" if lo < 0 < hi else "  *excludes 0"
        print(f"  {ZONES[i]:10s} {phi_hat[i]:+9.1f}  [{lo:+8.1f}, {hi:+8.1f}]{flag}")

    print("\nPairwise interaction indices (95% CI):")
    for pi in np.argsort(-np.abs(I_hat)):
        i, j = pairs[pi]
        lo, hi = np.percentile(I_bs[:, pi], [2.5, 97.5])
        flag = "" if lo < 0 < hi else "  *excludes 0"
        print(f"  {ZONES[i]+' x '+ZONES[j]:18s} {I_hat[pi]:+9.1f}  "
              f"[{lo:+8.1f}, {hi:+8.1f}]{flag}")


if __name__ == "__main__":
    main()
