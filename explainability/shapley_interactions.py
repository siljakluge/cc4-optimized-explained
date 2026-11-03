#!/usr/bin/env python3
"""Pairwise Shapley interaction indices (Grabisch & Roubens, 1999).

The Shapley values always sum to v(N)-v(0) by efficiency, regardless of whether
agents interact, so that sum is a sanity check -- not evidence about the game.
The interaction index is the actual measurement of non-additivity: for a pair
(i,j),

    I(i,j) = sum_{S subseteq N\\{i,j}} w(|S|) *
                 [ v(S+i+j) - v(S+i) - v(S+j) + v(S) ]
    w(|S|) = |S|!(n-|S|-2)! / (n-1)!

I(i,j) > 0 synergy (worth more together); < 0 redundancy/interference; ~0
additive. Reads the per-seed coalition means from a shapley_*.json produced by
shapley.py. Point estimates only; see interaction_multiplicity.py for CIs and
Holm-corrected significance.

Usage: python explainability/shapley_interactions.py explainability/shapley_heuristic_fsm_default.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from math import factorial

import numpy as np

N = 5
ZONES = {0: "RZ-A", 1: "OZ-A", 2: "RZ-B", 3: "OZ-B", 4: "corporate"}


def mask_of(key: str) -> frozenset:
    return frozenset(int(c) for c in key if c != ".")


def interaction(v: dict, i: int, j: int) -> float:
    others = [k for k in range(N) if k not in (i, j)]
    tot = 0.0
    for r in range(len(others) + 1):
        for S in combinations(others, r):
            S = frozenset(S)
            w = factorial(len(S)) * factorial(N - len(S) - 2) / factorial(N - 1)
            tot += w * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
    return tot


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "explainability/shapley_heuristic_fsm_default.json"
    d = json.load(open(path))
    vmean = {mask_of(k): float(np.mean(val["total"])) for k, val in d["v_raw"].items()}

    print(f"Pairwise Shapley interaction indices, from {path}\n")
    print(f"  {'pair':20s}{'I(i,j)':>12s}   interpretation")
    rows = [(i, j, interaction(vmean, i, j)) for i, j in combinations(range(N), 2)]
    for i, j, val in sorted(rows, key=lambda r: -abs(r[2])):
        interp = ("~additive" if abs(val) < 5 else
                  "synergistic (more together)" if val > 0 else
                  "redundant/interfering (less together)")
        print(f"  {ZONES[i]+' x '+ZONES[j]:20s}{val:+12.1f}   {interp}")

    full, empty = frozenset(range(N)), frozenset()
    print(f"\n  sum |I| = {sum(abs(v) for _, _, v in rows):.1f}   "
          f"(team improvement v(N)-v(0) = {vmean[full] - vmean[empty]:.1f})")


if __name__ == "__main__":
    main()
