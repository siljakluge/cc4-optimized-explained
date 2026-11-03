#!/usr/bin/env python3
"""Plot the team-level explainability figures for the fixed environment.

Reads the JSON produced by shapley.py and comms_ablation.py and writes:
  figures/shapley.(pdf|png)       phi_i with bootstrap 95% CIs
  figures/interaction.(pdf|png)   I(i,j) forest plot, Holm-significant filled
  figures/ablation.(pdf|png)      per-seed ON vs ablated, both arms

Usage:
  python explainability/plot_figures.py \
      --shapley explainability/results/shapley_heuristic_fsm_default.json \
      --comms   explainability/results/comms_heuristic_fsm_default.json
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "stix", "savefig.dpi": 300, "savefig.bbox": "tight",
})
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                   # noqa: E402

N = 5
ZONES = {0: "RZ-A", 1: "OZ-A", 2: "RZ-B", 3: "OZ-B", 4: "Corp"}
PAIRS = list(combinations(range(N), 2))
BLUE, ORANGE, INK = "#0072B2", "#E69F00", "#222222"
B = 5000
OUT = Path("explainability/figures")


def mask_of(k): return frozenset(int(c) for c in k if c != ".")


def shapley_from_v(v):
    phi = np.zeros(N)
    for i in range(N):
        for r in range(N):
            for S in combinations([j for j in range(N) if j != i], r):
                S = frozenset(S)
                w = factorial(len(S)) * factorial(N - len(S) - 1) / factorial(N)
                phi[i] += w * (v[S | {i}] - v[S])
    return phi


def interaction_from_v(v, i, j):
    tot = 0.0
    for r in range(N - 1):
        for S in combinations([k for k in range(N) if k not in (i, j)], r):
            S = frozenset(S)
            w = factorial(len(S)) * factorial(N - len(S) - 2) / factorial(N - 1)
            tot += w * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
    return tot


def holm(p):
    order = sorted(range(len(p)), key=lambda k: p[k])
    adj, run = [0.0] * len(p), 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[idx]))
        adj[idx] = run
    return adj


def bootstrap(raw, rng):
    K = len(next(iter(raw.values())))
    phi_bs = np.zeros((B, N)); I_bs = np.zeros((B, len(PAIRS)))
    for b in range(B):
        idx = rng.integers(0, K, K)
        vb = {m: a[idx].mean() for m, a in raw.items()}
        phi_bs[b] = shapley_from_v(vb)
        for pi, (i, j) in enumerate(PAIRS):
            I_bs[b, pi] = interaction_from_v(vb, i, j)
    return phi_bs, I_bs


def fig_shapley(raw, phi, phi_bs, stem):
    order = np.argsort(-phi)
    lo = phi - np.percentile(phi_bs, 2.5, axis=0)
    hi = np.percentile(phi_bs, 97.5, axis=0) - phi
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(N)
    ax.bar(x, phi[order], yerr=[lo[order], hi[order]], capsize=2.5,
           color=BLUE, zorder=3, error_kw={"elinewidth": 0.8, "capthick": 0.8})
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([ZONES[i] for i in order])
    ax.set_xlabel("Agent zone"); ax.set_ylabel(r"$\phi_i$  (Shapley value)")
    ax.grid(axis="y", lw=0.4, alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, stem)


def fig_interaction(I, I_bs, stem):
    lo = np.percentile(I_bs, 2.5, axis=0); hi = np.percentile(I_bs, 97.5, axis=0)
    p_raw = np.array([max(2 * min((I_bs[:, k] <= 0).mean(), (I_bs[:, k] >= 0).mean()),
                          1 / B) for k in range(len(PAIRS))])
    p_adj = holm(p_raw)
    order = np.argsort(-np.abs(I))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    y = np.arange(len(PAIRS))[::-1]
    for row, k in enumerate(order):
        sig = p_adj[k] < 0.05
        ax.errorbar(I[k], y[row], xerr=[[I[k] - lo[k]], [hi[k] - I[k]]], fmt="s",
                    color=BLUE, ms=4 if sig else 3, elinewidth=1.0 if sig else 0.7,
                    capsize=2, capthick=0.7, markerfacecolor=BLUE if sig else "white")
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{ZONES[PAIRS[k][0]]}$\\times${ZONES[PAIRS[k][1]]}" for k in order])
    ax.set_xlabel(r"$I(i,j)$  (interaction index)")
    ax.grid(axis="x", lw=0.4, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, stem)
    return p_adj, order


def fig_ablation(comms, stem):
    on = np.array(comms["per_seed"]["on"], float)
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6), sharey=True)
    for ax, key, title, c in ((axes[0], "content", "content zeroed", BLUE),
                              (axes[1], "empty", "empty message", ORANGE)):
        off = np.array(comms["per_seed"][key], float)
        lim_lo = min(on.min(), off.min()); lim_hi = max(on.max(), off.max())
        pad = 0.05 * (lim_hi - lim_lo)
        ax.plot([lim_lo - pad, lim_hi + pad], [lim_lo - pad, lim_hi + pad],
                color=INK, lw=0.7)
        ax.scatter(off, on, s=14, facecolor="none", edgecolor=c, linewidths=0.9)
        ax.set_xlabel(f"reward, {title}")
        n_above = int((on > off).sum())
        ax.text(0.05, 0.95, f"{n_above}/{len(on)} above", transform=ax.transAxes,
                fontsize=7, va="top", color="#555")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
    axes[0].set_ylabel("reward, comms on")
    fig.subplots_adjust(wspace=0.12)
    _save(fig, stem)


def _save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf"); fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)
    print(f"wrote {OUT / stem}.(pdf|png)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapley", required=True)
    ap.add_argument("--comms", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    d = json.load(open(args.shapley))
    raw = {mask_of(k): np.asarray(v["total"], float) for k, v in d["v_raw"].items()}
    vhat = {m: a.mean() for m, a in raw.items()}
    phi = shapley_from_v(vhat)
    I = np.array([interaction_from_v(vhat, i, j) for i, j in PAIRS])
    phi_bs, I_bs = bootstrap(raw, rng)

    full, empty = frozenset(range(N)), frozenset()
    print(f"v(empty)={vhat[empty]:+.1f}  v(full)={vhat[full]:+.1f}  "
          f"improvement={vhat[full] - vhat[empty]:+.1f}")
    fig_shapley(raw, phi, phi_bs, "shapley")
    p_adj, order = fig_interaction(I, I_bs, "interaction")
    sig = [f"{ZONES[PAIRS[k][0]]}x{ZONES[PAIRS[k][1]]}" for k in range(len(PAIRS)) if p_adj[k] < 0.05]
    print("Holm-significant interactions:", sig or "none")
    if args.comms:
        fig_ablation(json.load(open(args.comms)), "ablation")


if __name__ == "__main__":
    main()
