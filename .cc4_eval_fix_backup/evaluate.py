from dataclasses import dataclass
from typing import Dict, List
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from statistics import mean, stdev
from datetime import datetime
import json
import sys
from pathlib import Path
import shutil
import argparse
import os, time
import re
from time import perf_counter

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import FiniteStateRedAgent

"""
evaluate.py — Robust Evaluation Script for CAGE Challenge 4 (Graph-based PPO MARL)

This script evaluates trained blue agents in the CybORG Scenario4 environment
under different red attacker profiles. It supports both single-profile
evaluation and full robustness sweeps across multiple attacker strategies.

Core Functionality
------------------
• Loads a trained Submission (multi-agent blue policy).
• Builds CybORG environments with configurable red attacker classes.
• Runs multiple evaluation episodes (parallelized via joblib).
• Computes mean and standard deviation of total rewards.
• Optionally logs full action and observation traces.
• Saves structured outputs (summary.txt, summary.json, scores.txt).

Evaluation Modes
----------------
1) Single Mode (--mode single)
   - Evaluate against:
       • one fixed attacker profile (--strategy single)
       • or a weighted mixture of profiles (--strategy mixture)
   - Results are written to a specified output folder.

2) Sweep Mode (--mode sweep)
   - Automatically runs:
       • N episodes per available attacker profile
       • N episodes using a mixed distribution
   - Creates structured output under:
       evals/<model_name>/
           <profile_name>/
           mixed/
   - Designed for robustness benchmarking and research comparison.

Red Attacker Profiles
---------------------
Profiles are registered via PROFILE_REGISTRY and may include:
    fsm_default
    stealth_pivot
    lateral_spread
    impact_rush
    deception_aware
    discovery
    verbose

Each evaluation episode builds a fresh CybORG environment,
since the red agent class is fixed at scenario construction time.

Parallelization
---------------
Evaluation uses joblib with process-based parallelism.
Each episode runs in its own environment instance to ensure
independent rollouts and reproducibility.

Reproducibility
---------------
• Per-episode seeds are derived from the base seed.
• Deterministic evaluation possible via --seed.
• Sweep mode offsets seeds per profile for controlled variation.

Output Files
------------
For each run:
    summary.txt     – human-readable results
    summary.json    – structured machine-readable output
    scores.txt      – compact reward summary
    full.txt        – optional detailed logs
    actions.txt     – optional action traces

Typical Usage
-------------
Single profile:
    python evaluate.py --mode single --strategy single --single_profile fsm_default

Mixed evaluation:
    python evaluate.py --mode single --strategy mixture \
        --profile_weights "fsm_default=0.35,stealth_pivot=0.25,..."

Full sweep:
    python evaluate.py --mode sweep --model-name RL_contractor_active_50k

Intended Use
------------
This script is designed for:
• Robustness benchmarking
• Research experiments
• Comparative model analysis
• Generating reproducible evaluation tables for papers

Author Context
--------------
Developed for Graph-based PPO MARL agents & heuristic agents in the
CAGE Challenge 4 enterprise cyber defense setting.


Runs:  python evaluate.py --mode sweep --model-name Baseline_50k_2 --eps-per-profile 100 --distribute 4 --submission-path . --seed 1337 --overwrite
"""
REWARD_MODES = ["default", "contractor_off", "red_only"]

def parse_reward_modes(s: str) -> List[str]:
    if not s.strip():
        return []
    modes = [x.strip() for x in s.split(",") if x.strip()]
    bad = [m for m in modes if m not in REWARD_MODES]
    if bad:
        raise ValueError(f"Unknown reward mode(s): {bad}. Available: {REWARD_MODES}")
    return modes

def parse_name_set(s: str) -> set[str]:
    if not s.strip():
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}
EPISODE_LENGTH = 500
# ----------------------------
# Red profiles (same pattern as train.py)
# ----------------------------
@dataclass(frozen=True)
class AttackProfile:
    name: str
    red_cls: type

def _safe_import_profiles():
    profiles: Dict[str, type] = {
        "fsm_default": FiniteStateRedAgent,
    }
    try:
        from CybORG.Agents.RedAgents import (
            DiscoveryFSRed,
            VerboseFSRed,
            StealthPivotFSRed,
            ImpactRushFSRed,
            DeceptionAwareFSRed,
            LateralSpreadFSRed,
        )
        profiles.update({
            "discovery": DiscoveryFSRed,
            "verbose": VerboseFSRed,
            "stealth_pivot": StealthPivotFSRed,
            "impact_rush": ImpactRushFSRed,
            "deception_aware": DeceptionAwareFSRed,
            "lateral_spread": LateralSpreadFSRed,
        })
    except Exception:
        print("[eval] Optional RedAgents variants not found; using baseline FiniteStateRedAgent only.")
        pass

    return profiles

PROFILE_REGISTRY: Dict[str, type] = _safe_import_profiles()

def parse_profile_weights(s: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if not s.strip():
        return weights
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad profile_weights token '{p}', expected name=weight.")
        name, w = p.split("=", 1)
        name = name.strip()
        w = float(w.strip())
        weights[name] = w

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("profile_weights sum must be > 0")
    for k in list(weights.keys()):
        weights[k] = weights[k] / total
    return weights

def sample_profile(rng: random.Random, weights: Dict[str, float], fallback: str = "fsm_default") -> str:
    candidates = [(name, w) for name, w in weights.items() if name in PROFILE_REGISTRY and w > 0]
    if not candidates:
        return fallback if fallback in PROFILE_REGISTRY else "fsm_default"
    names = [c[0] for c in candidates]
    ws = [c[1] for c in candidates]
    return rng.choices(names, weights=ws, k=1)[0]

def rmkdir(path: str):
    """Recursive mkdir"""
    partial_path = ""
    for p in path.split("/"):
        partial_path += p + "/"

        if os.path.exists(partial_path):
            if os.path.isdir(partial_path):
                continue
            if os.path.isfile(partial_path):
                raise RuntimeError(f"Cannot create {partial_path} (exists as file).")

        os.mkdir(partial_path)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def clean_dir(p: str | Path):
    p = Path(p)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)  # keep safe chars
    return s or "model"

def load_submission(source: str):
    """Load submission from a directory or zip file"""
    sys.path.insert(0, source)

    if source.endswith(".zip"):
        try:
            # Load submission from zip.
            from submission.submission import Submission
        except ImportError as e:
            raise ImportError(
                """
                Error loading submission from zip.
                Please ensure the zip contains the path submission/submission.py
                """
            ).with_traceback(e.__traceback__)
    else:
        # Load submission normally
        from submission import Submission

    # Remove submission from path.
    sys.path.remove(source)
    return Submission

def make_env_for_episode(seed: int, episode_len: int, red_agent_class: type, submission):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=red_agent_class,
        steps=episode_len,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    wrapped_cyborg = submission.wrap(cyborg)
    return cyborg, wrapped_cyborg

def evaluate_one_episode(
    submission,
    base_seed: int | None,
    episode_idx: int,
    max_eps: int,
    profile_name: str,
    write_to_file: bool,
    show_progress: bool,
):
    # per-episode env seed (deterministisch)
    # falls base_seed None: trotzdem stabil via episode_idx
    env_seed = (base_seed if base_seed is not None else 0) + 100_000 + episode_idx

    red_cls = PROFILE_REGISTRY.get(profile_name, PROFILE_REGISTRY.get("fsm_default"))
    cyborg, wrapped_cyborg = make_env_for_episode(env_seed, EPISODE_LENGTH, red_cls, submission)

    observations, _ = wrapped_cyborg.reset()

    r = []
    a = []
    o = []

    iterator = range(EPISODE_LENGTH)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Episode {episode_idx + 1}/{max_eps} {profile_name}")
    for _ in iterator:
        actions = {
            agent_name: agent.get_action(
                observations[agent_name], wrapped_cyborg.action_space(agent_name)
            )
            for agent_name, agent in submission.AGENTS.items()
            if agent_name in wrapped_cyborg.agents
        }

        observations, rew, term, trunc, info = wrapped_cyborg.step(actions)

        done = {
            agent: term.get(agent, False) or trunc.get(agent, False)
            for agent in wrapped_cyborg.agents
        }
        if all(done.values()):
            break

        r.append(mean(rew.values()))

        if write_to_file:
            a.append({agent_name: cyborg.get_last_action(agent_name) for agent_name in wrapped_cyborg.agents})
            o.append({agent_name: observations[agent_name] for agent_name in observations.keys()})

    total_reward = sum(r)
    return total_reward, a, o, profile_name

def _filter_available_weights(w: Dict[str, float]) -> Dict[str, float]:
    ww = {k: v for k, v in w.items() if (k in PROFILE_REGISTRY and v > 0)}
    if not ww:
        return {"fsm_default": 1.0}
    s = sum(ww.values())
    return {k: v / s for k, v in ww.items()}

def run_evaluation_parallel_profiles(
    submission,
    log_path: str,
    max_eps: int = 100,
    write_to_file: bool = False,
    seed: int | None = None,
    workers: int = 32,
    strategy: str = "mixture",               # single|mixture
    single_profile: str = "fsm_default",
    profile_weights: Dict[str, float] | None = None,
    log_per_profile: bool = True,
):
    cyborg_version = CYBORG_VERSION
    scenario = "Scenario4"
    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"

    print(version_header)
    print(author_header)
    print(f"Using agents {submission.AGENTS}")

    print("Available profiles:", sorted(PROFILE_REGISTRY.keys()))
    if strategy == "single":
        if single_profile not in PROFILE_REGISTRY:
            raise ValueError(f"single_profile='{single_profile}' not found. Available: {sorted(PROFILE_REGISTRY.keys())}")
        print("Strategy: single", "profile:", single_profile)
    else:
        w = _filter_available_weights(profile_weights or {})
        if not profile_weights:
            # default eval mixture (feel free to mirror train.py defaults)
            w = _filter_available_weights({
                "fsm_default": 0.35,
                "stealth_pivot": 0.25,
                "lateral_spread": 0.20,
                "impact_rush": 0.15,
                "deception_aware": 0.05,
            })
        print("Strategy: mixture", "weights:", w)

    if write_to_file:
        if not log_path.endswith("/"):
            log_path += "/"
        print(f"Results will be saved to {log_path}")

    # choose profile per episode deterministically
    episode_profiles = []
    for i in range(max_eps):
        if strategy == "single":
            episode_profiles.append(single_profile)
        else:
            rng = random.Random((seed if seed is not None else 0) + i)
            episode_profiles.append(sample_profile(rng, w))

    start = datetime.now()

    outs = Parallel(prefer="processes", n_jobs=workers)(
        delayed(evaluate_one_episode)(
            submission=submission,
            base_seed=seed,
            episode_idx=i,
            max_eps=max_eps,
            profile_name=episode_profiles[i],
            write_to_file=write_to_file,
            show_progress=False,
        )
        for i in range(max_eps)
    )

    total_reward, actions_log, obs_log, used_profiles = zip(*outs)

    end = datetime.now()
    difference = end - start

    reward_mean = mean(total_reward)
    reward_stdev = stdev(total_reward) if len(total_reward) > 1 else 0.0
    reward_string = f"Average reward is: {reward_mean} with a standard deviation of {reward_stdev}"
    print(reward_string)
    print(f"Evaluation took {difference}")

    # per-profile stats
    per_profile_stats = {}
    if log_per_profile:
        tmp = {}
        for r, p in zip(total_reward, used_profiles):
            tmp.setdefault(p, []).append(r)
        for p, rs in tmp.items():
            per_profile_stats[p] = {
                "n": len(rs),
                "mean": mean(rs),
                "stdev": (stdev(rs) if len(rs) > 1 else 0.0),
            }
        # print a compact table-like summary
        print("Per-profile:")
        for p in sorted(per_profile_stats.keys()):
            s = per_profile_stats[p]
            print(f"  {p:16s} n={s['n']:3d} mean={s['mean']:9.3f} stdev={s['stdev']:9.3f}")

    if not write_to_file:
        return

    # ---- writing results (keep your old files, but add profiles) ----
    os.makedirs(log_path, exist_ok=True)

    with open(log_path + "summary.txt", "w") as data:
        data.write(version_header + "\n")
        data.write(author_header + "\n")
        data.write(reward_string + "\n")
        data.write(f"Using agents {submission.AGENTS}\n")
        data.write(f"Strategy: {strategy}\n")
        if strategy == "single":
            data.write(f"Single profile: {single_profile}\n")
        else:
            data.write(f"Profile weights: {w}\n")
        if log_per_profile:
            data.write(f"Per-profile stats: {per_profile_stats}\n")

    with open(log_path + "full.txt", "w") as data:
        data.write(version_header + "\n")
        data.write(author_header + "\n")
        data.write(reward_string + "\n")
        for act, obs, sum_rew, prof in zip(actions_log, obs_log, total_reward, used_profiles):
            data.write(f"profile: {prof}\n")
            data.write(f"actions: {act},\n observations: {obs},\n total reward: {sum_rew}\n\n")

    with open(log_path + "actions.txt", "w") as data:
        data.write(version_header + "\n")
        data.write(author_header + "\n")
        data.write(reward_string + "\n")
        for act, prof in zip(actions_log, used_profiles):
            data.write(f"profile: {prof}\n")
            data.write(f"actions: {act}\n\n")

    with open(log_path + "summary.json", "w") as output:
        payload = {
            "submission": {
                "author": submission.NAME,
                "team": submission.TEAM,
                "technique": submission.TECHNIQUE,
            },
            "parameters": {
                "seed": seed,
                "episode_length": EPISODE_LENGTH,
                "max_episodes": max_eps,
                "strategy": strategy,
                "single_profile": single_profile if strategy == "single" else None,
                "profile_weights": (w if strategy != "single" else None),
            },
            "time": {
                "start": str(start),
                "end": str(end),
                "elapsed": str(difference),
            },
            "reward": {
                "mean": reward_mean,
                "stdev": reward_stdev,
            },
            "profiles": {
                "used": list(used_profiles),
                "per_profile": per_profile_stats,
            },
            "agents": {agent: str(submission.AGENTS[agent]) for agent in submission.AGENTS},
        }
        json.dump(payload, output, indent=2)

    with open(log_path + "scores.txt", "w") as scores:
        scores.write(f"reward_mean: {reward_mean}\n")
        scores.write(f"reward_stdev: {reward_stdev}\n")
        if log_per_profile:
            for p in sorted(per_profile_stats.keys()):
                s = per_profile_stats[p]
                scores.write(f"{p}: n={s['n']} mean={s['mean']} stdev={s['stdev']}\n")

def run_profile_sweep(
    submission,
    model_name: str,
    base_out_dir: str = "evals",
    eps_per_profile: int = 100,
    workers: int = 32,
    seed: int | None = 1337,
    mixed_weights_str: str = "fsm_default=0.35,stealth_pivot=0.25,lateral_spread=0.20,impact_rush=0.15,deception_aware=0.05",
    write_to_file: bool = True,
    overwrite: bool = False,
):
    """
    Runs:
      - for each available profile: strategy=single, max_eps=eps_per_profile
      - then mixed: strategy=mixture, max_eps=eps_per_profile

    Writes into:
      evals/<model_name>/<profile_name>/...
      evals/<model_name>/mixed/...
    """
    # base directory for this model
    model_dir = Path(base_out_dir) / model_name
    if overwrite:
        clean_dir(model_dir)
    else:
        ensure_dir(model_dir)

    # ---- SINGLE PROFILE runs ----
    profiles = [p for p in sorted(PROFILE_REGISTRY.keys()) if p != "verbose"]
    print(f"[sweep] Model={model_name} profiles={profiles}")

    total_runs = len(profiles) + 1
    pbar = tqdm(total=total_runs, desc="Sweep", unit="run", dynamic_ncols=True)

    try:
        for idx, prof in enumerate(profiles):
            out_dir = model_dir / prof
            ensure_dir(out_dir)

            prof_seed = (seed if seed is not None else 0) + 10_000 * (idx + 1)

            pbar.set_postfix_str(f"profile={prof}, eps={eps_per_profile}, workers={workers}")
            t0 = perf_counter()

            run_evaluation_parallel_profiles(
                submission=submission,
                log_path=str(out_dir),
                max_eps=eps_per_profile,
                write_to_file=write_to_file,
                seed=prof_seed,
                workers=workers,
                strategy="single",
                single_profile=prof,
                profile_weights=None,
                log_per_profile=True,
            )

            dt = perf_counter() - t0
            pbar.write(f"[sweep] done profile='{prof}' in {dt / 60:.1f} min -> {out_dir}")
            pbar.update(1)

        # ---- MIXED run ----
        mixed_dir = model_dir / "mixed"
        ensure_dir(mixed_dir)

        mixed_seed = (seed if seed is not None else 0) + 999_999
        mixed_weights = parse_profile_weights(mixed_weights_str) if mixed_weights_str else {}

        pbar.set_postfix_str(f"profile=mixed, eps={eps_per_profile}, workers={workers}")
        t0 = perf_counter()

        run_evaluation_parallel_profiles(
            submission=submission,
            log_path=str(mixed_dir),
            max_eps=eps_per_profile,
            write_to_file=write_to_file,
            seed=mixed_seed,
            workers=workers,
            strategy="mixture",
            single_profile="fsm_default",
            profile_weights=mixed_weights,
            log_per_profile=True,
        )

        dt = perf_counter() - t0
        pbar.write(f"[sweep] done profile='mixed' in {dt / 60:.1f} min -> {mixed_dir}")
        pbar.update(1)

    finally:
        pbar.close()

    print(f"\n[sweep] Done. Results under: {model_dir}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("CybORG Evaluation Script")

    parser.add_argument("--append-timestamp", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--distribute", type=int, default=1)
    parser.add_argument("--max-eps", type=int, default=100)

    parser.add_argument("--mode", choices=["single", "sweep"], default="single")

    # single-mode controls (also usable for single mixed run)
    parser.add_argument("--strategy", choices=["single", "mixture"], default="mixture")
    parser.add_argument("--single_profile", default="fsm_default")
    parser.add_argument("--profile_weights", default="")
    parser.add_argument("--log_per_profile", action="store_true")

    # sweep-mode controls
    parser.add_argument("--model-name", dest="model_name", type=str, default="my_model")
    parser.add_argument("--out-root", dest="out_root", type=str, default="evals")
    parser.add_argument("--eps-per-profile", dest="eps_per_profile", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mixed-weights",
        dest="mixed_weights",
        type=str,
        default="fsm_default=0.35,stealth_pivot=0.25,lateral_spread=0.20,impact_rush=0.15,deception_aware=0.05",
    )

    # IMPORTANT: make these CLI args, don't hardcode
    parser.add_argument("--output-path", type=str, default="evaluation_out")
    parser.add_argument("--submission-path", type=str, default=".")

    REWARD_MODES = ["default", "contractor_off", "red_only"]

    parser.add_argument(
        "--phase_reward_mode",
        default="default",
        choices=REWARD_MODES,
        help="Single reward mode to use if --reward_mode_strategy=single."
    )
    parser.add_argument(
        "--reward_mode_strategy",
        choices=["single", "sweep"],
        default="single",
        help="Run one reward mode or sweep across multiple reward modes."
    )
    parser.add_argument(
        "--reward_modes",
        default="",
        help='Comma-separated reward modes for sweep, e.g. "default,red_only". Empty => all.'
    )
    parser.add_argument(
        "--reward_blue",
        action="store_true",
        help="Enable blue analyse reward for single-mode runs."
    )
    parser.add_argument(
        "--reward_blue_modes",
        default="",
        help='Optional comma-separated reward modes for blue analyse reward in sweep mode, e.g. "red_only".'
    )

    args = parser.parse_args()

    # Optional timestamp behavior:
    # - single mode: append to output-path
    # - sweep mode: append to model folder name (so you don't overwrite runs)
    if args.append_timestamp:
        ts = time.strftime("%Y%m%d_%H%M%S")
        if args.mode == "single":
            args.output_path = os.path.join(args.output_path, ts)
        else:
            args.model_name = f"{args.model_name}_{ts}"

    # Load submission once
    submission = load_submission(os.path.abspath(args.submission_path))
    if isinstance(submission, type):
        submission = submission()

    # Auto-fill model_name if user didn't set it (or left default)
    if getattr(args, "model_name", None) in (None, "", "my_model"):
        args.model_name = slugify(getattr(submission, "NAME", "model"))
    else:
        args.model_name = slugify(args.model_name)
    if args.mode == "single" and (args.output_path == "evaluation_out"):
        args.output_path = os.path.join("evals", args.model_name, "single")

    if args.mode == "single":
        out_dir = os.path.abspath(args.output_path)
        if not out_dir.endswith("/"):
            out_dir += "/"
        rmkdir(out_dir)

        weights = parse_profile_weights(args.profile_weights) if args.profile_weights else {}

        run_evaluation_parallel_profiles(
            submission=submission,
            log_path=out_dir,
            max_eps=args.max_eps,
            write_to_file=True,
            seed=args.seed,
            workers=args.distribute,
            strategy=args.strategy,
            single_profile=args.single_profile,
            profile_weights=weights,
            log_per_profile=args.log_per_profile,
        )

    else:  # sweep
        run_profile_sweep(
            submission=submission,
            model_name=args.model_name,
            base_out_dir=args.out_root,
            eps_per_profile=args.eps_per_profile,
            workers=args.distribute,
            seed=args.seed,
            mixed_weights_str=args.mixed_weights,
            write_to_file=True,
            overwrite=args.overwrite,
        )

