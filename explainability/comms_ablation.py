#!/usr/bin/env python3
"""Communication-channel ablation, paired per seed.

Runs the full five-agent team three ways on the SAME seeds:
  ON      : messages unchanged
  CONTENT : content bits zeroed, final validity bit kept (in-distribution)
  EMPTY   : whole message zeroed (CybORG EMPTY_MESSAGE; out-of-distribution)
and reports the paired per-seed difference Delta = v(ON) - v(OFF).

Unlike Silja's H_Agent (which ignored messages), EnterpriseHeuristicAgentV11a
DOES read peer messages (v9 protocol), so this ablation is meaningful for the
heuristic. Ablation is applied by zeroing each agent's OUTGOING message before
the env forwards it -- if everyone sends ablated, everyone receives ablated.

Saves per-seed arrays to JSON for comms_stats.py. No new training.

Usage:
  python explainability/comms_ablation.py --backend heuristic --seeds 20 --profile fsm_default
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H                                                   # noqa: E402

N = H.N_AGENTS
FULL = list(range(N))


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

    arms = {"on": None, "content": "content", "empty": "empty"}
    per_seed = {name: [] for name in arms}
    t0 = time.time()
    for s in seeds:
        for name, mode in arms.items():
            per_seed[name].append(run_episode(FULL, team, s, args.steps, red_cls, mode))
        print(f"[seed {s}] " + "  ".join(
            f"{k}={per_seed[k][-1]:+.0f}" for k in arms) +
            f"  ({time.time() - t0:.0f}s)", flush=True)

    def delta(off):
        return [on - of for on, of in zip(per_seed["on"], per_seed[off])]

    out = {
        "backend": args.backend, "profile": args.profile,
        "seeds": args.seeds, "steps": args.steps, "seed_list": seeds,
        "per_seed": per_seed,
        "delta_content": delta("content"),
        "delta_empty": delta("empty"),
        "note": "fixed-env (cc4-optimized); V11a reads messages so ablation is meaningful",
    }
    outpath = args.out or f"explainability/comms_{args.backend}_{args.profile}.json"
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outpath, "w"), indent=1)

    for off in ("content", "empty"):
        d = out[f"delta_{off}"]
        helped = sum(1 for x in d if x > 0)
        print(f"\n{off.upper():8s}  mean Delta = {statistics.mean(d):+.1f}  "
              f"helped {helped}/{len(d)} seeds")
    print(f"\nsaved -> {outpath}")


if __name__ == "__main__":
    main()
