#!/usr/bin/env python3
"""Component-decomposed Shapley figure: phi_i split into reward channels.

Shows how each agent's Shapley value breaks into LWF / ASF / RIA / cost (the
components active in the default reward mode; EXP/PRIV/DEG/ANALYSE are zero and
omitted). Reads shapley_components_*.json.

Usage:
  python explainability/plot_components.py explainability/results/shapley_components_heuristic_fsm_default.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "mathtext.fontset": "stix",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                   # noqa: E402

N = 5
ZONES = {0: "RZ-A", 1: "OZ-A", 2: "RZ-B", 3: "OZ-B", 4: "Corp"}
# active components in default reward mode, with a colour each
ACTIVE = ["LWF", "ASF", "RIA", "cost"]
COLORS = {"LWF": "#0072B2", "ASF": "#E69F00", "RIA": "#CC79A7", "cost": "#999999"}
OUT = Path("explainability/figures")


def main():
    path = sys.argv[1]
    d = json.load(open(path))
    phi = {c: np.array(d["phi"][c]) for c in d["phi"]}
    total = phi["total"]
    order = np.argsort(-total)

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = np.arange(N)
    pos = np.zeros(N); neg = np.zeros(N)
    for c in ACTIVE:
        vals = phi[c][order]
        base = np.where(vals >= 0, pos, neg)
        ax.bar(x, vals, 0.6, bottom=base, color=COLORS[c], label=c, zorder=3)
        pos = pos + np.where(vals >= 0, vals, 0)
        neg = neg + np.where(vals < 0, vals, 0)
    # total marker
    ax.plot(x, total[order], "D", color="black", ms=4, zorder=5, label=r"total $\phi_i$")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([ZONES[i] for i in order])
    ax.set_xlabel("Agent zone"); ax.set_ylabel(r"$\phi_i$ by reward component")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    ax.grid(axis="y", lw=0.4, alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "shapley_components.pdf"); fig.savefig(OUT / "shapley_components.png")
    plt.close(fig)
    print("wrote", OUT / "shapley_components.(pdf|png)")

    # text table
    print(f"\n  {'agent':8s}" + "".join(f"{c:>8s}" for c in ["total"] + ACTIVE))
    for i in order:
        print(f"  {ZONES[i]:8s}" + "".join(f"{phi[c][i]:+8.0f}" for c in ["total"] + ACTIVE))


if __name__ == "__main__":
    main()
