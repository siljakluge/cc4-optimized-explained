#!/usr/bin/env python3
"""Paired statistics for the communication ablation.

Reads one or more comms_*.json (comms_ablation.py) and for each arm reports,
per profile: the paired difference Delta = v(ON)-v(OFF), a paired t-test, an
exact sign-flip permutation test (all 2^n patterns when n<=20), and, across
profiles, a Holm-corrected family. Also a pooled test and (if >1 profile) a
check of whether the effect varies by adversary.

Usage:
  python explainability/comms_stats.py explainability/comms_heuristic_*.json
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
from scipy import stats


def exact_sign_flip_p(d: np.ndarray) -> float:
    n = len(d)
    obs = abs(d.mean())
    if n <= 20:
        count = 0
        for mask in range(1 << n):
            signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)])
            if abs((d * signs).mean()) >= obs - 1e-9:
                count += 1
        return count / (1 << n)
    rng = np.random.default_rng(0)
    signs = rng.choice([-1, 1], size=(100_000, n))
    return float((np.abs((d[None, :] * signs).mean(axis=1)) >= obs - 1e-9).mean())


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def load(paths):
    data = {}
    for p in paths:
        d = json.load(open(p))
        data[d["profile"]] = d
    return data


def analyse_arm(arm_key: str, data: dict):
    print(f"\n{'=' * 70}\nARM: {arm_key}\n{'=' * 70}")
    print(f"  {'profile':16s}{'Delta':>9s}{'t(df)':>9s}{'p_t':>9s}{'p_perm':>9s}")
    diffs, raw_p = {}, {}
    for prof, d in data.items():
        x = np.asarray(d[arm_key], float)
        diffs[prof] = x
        t, p_t = stats.ttest_1samp(x, 0.0)
        p_perm = exact_sign_flip_p(x)
        raw_p[prof] = p_t
        print(f"  {prof:16s}{x.mean():+9.1f}{t:+9.2f}{p_t:9.3f}{p_perm:9.3f}")

    p_holm = holm(raw_p)
    print("\n  Holm-corrected per-profile (t-based):")
    for prof, p in sorted(p_holm.items(), key=lambda kv: kv[1]):
        print(f"    {prof:16s} p_holm={p:.3f}  "
              f"({'significant' if p < 0.05 else 'not significant'})")

    pooled = np.concatenate(list(diffs.values()))
    t, p = stats.ttest_1samp(pooled, 0.0)
    print(f"\n  Pooled ({pooled.size} profile x seed): "
          f"Delta={pooled.mean():+.1f}  t({pooled.size - 1})={t:+.2f}  p={p:.4f}")

    if len(diffs) > 1:
        try:
            import pandas as pd
            from statsmodels.stats.anova import AnovaRM
            mat = np.array(list(diffs.values())).T       # (seeds, profiles)
            ns, npf = mat.shape
            long = pd.DataFrame({
                "seed": np.repeat(np.arange(ns), npf),
                "profile": np.tile(list(diffs.keys()), ns),
                "delta": mat.flatten(),
            })
            res = AnovaRM(long, depvar="delta", subject="seed", within=["profile"]).fit()
            f = res.anova_table["F Value"].iloc[0]
            pv = res.anova_table["Pr > F"].iloc[0]
            verdict = ("effect DIFFERS by profile" if pv < 0.05
                       else "no evidence effect differs by profile")
            print(f"  Comms x Profile (rm-ANOVA): F={f:.2f}  p={pv:.4f}  -> {verdict}")
        except ImportError:
            print("  (statsmodels unavailable; skipping rm-ANOVA)")


def main():
    args = sys.argv[1:]
    paths = []
    for a in args:
        paths.extend(glob.glob(a))
    if not paths:
        sys.exit("usage: comms_stats.py explainability/comms_*.json")
    data = load(paths)
    print(f"loaded {len(data)} profile(s): {', '.join(data)}")
    analyse_arm("delta_content", data)
    analyse_arm("delta_empty", data)


if __name__ == "__main__":
    main()
