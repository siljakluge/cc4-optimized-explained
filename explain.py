from __future__ import annotations

import os, sys, json, time, random, shutil, re, importlib.util
from typing import Dict, List, Any, Optional, Tuple
from statistics import mean, stdev
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from plotting.plot_actions import log_actions_jsonl
from SHAP.policy_shap_heuristic import train_surrogate_and_shap
from SHAP.shap_gnn import run_shap


# ----------------------------
# Utilities
# ----------------------------

def rmkdir(path: str):
    partial_path = ""
    for p in path.split("/"):
        if not p:
            continue
        partial_path = os.path.join(partial_path, p)
        if os.path.exists(partial_path):
            if os.path.isdir(partial_path):
                continue
            raise RuntimeError(f"Cannot create {partial_path} (exists as file).")
        os.mkdir(partial_path)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def clean_dir(p: str | Path):
    p = Path(p)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def load_submission(source: str):
    source_path = Path(source).resolve()
    candidates = []

    if source_path.is_file() and source_path.name == "submission.py":
        candidates.append(source_path)
    else:
        candidates.extend([
            source_path / "submission.py",
            source_path / "marl" / "submission.py",
        ])

    for submission_file in candidates:
        if submission_file.exists():
            module_dir = str(submission_file.parent)
            sys.path.insert(0, module_dir)
            try:
                spec = importlib.util.spec_from_file_location("cc4_submission", submission_file)
                module = importlib.util.module_from_spec(spec)
                assert spec is not None and spec.loader is not None
                spec.loader.exec_module(module)
                return module.Submission
            finally:
                try:
                    sys.path.remove(module_dir)
                except ValueError:
                    pass

    raise FileNotFoundError(
        f"No submission.py found at {source_path}, {source_path / 'submission.py'}, "
        f"or {source_path / 'marl' / 'submission.py'}"
    )

# ----------------------------
# Red profiles (same as evaluate.py)
# ----------------------------

@dataclass(frozen=True)
class AttackProfile:
    name: str
    red_cls: type

def _safe_import_profiles():
    profiles: Dict[str, type] = {
        "fsm_default": FiniteStateRedAgent,
    }

    import_paths = [
        (
            "CybORG.Agents.RedAgents",
            [
                "DiscoveryFSRed",
                "VerboseFSRed",
                "StealthPivotFSRed",
                "ImpactRushFSRed",
                "DeceptionAwareFSRed",
                "LateralSpreadFSRed",
            ],
        ),
        (
            "CybORG.Agents.SimpleAgents.FSMRedVariants",
            [
                "DiscoveryFSRed",
                "VerboseFSRed",
                "StealthPivotFSRed",
                "ImpactRushFSRed",
                "DeceptionAwareFSRed",
                "LateralSpreadFSRed",
            ],
        ),
    ]

    for module_name, class_names in import_paths:
        try:
            module = __import__(module_name, fromlist=class_names)
        except Exception:
            continue
        for class_name in class_names:
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            profile = class_name.removesuffix("FSRed")
            snake = "".join(
                ("_" + ch.lower() if ch.isupper() and idx else ch.lower())
                for idx, ch in enumerate(profile)
            )
            profiles[snake] = cls

    if len(profiles) == 1:
        print("[explain] Optional Red profile variants not found; using baseline FiniteStateRedAgent only.")
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

def _filter_available_weights(w: Dict[str, float]) -> Dict[str, float]:
    ww = {k: v for k, v in w.items() if (k in PROFILE_REGISTRY and v > 0)}
    if not ww:
        return {"fsm_default": 1.0}
    s = sum(ww.values())
    return {k: v / s for k, v in ww.items()}

def sample_profile(rng: random.Random, weights: Dict[str, float], fallback: str = "fsm_default") -> str:
    candidates = [(name, w) for name, w in weights.items() if name in PROFILE_REGISTRY and w > 0]
    if not candidates:
        return fallback if fallback in PROFILE_REGISTRY else "fsm_default"
    names = [c[0] for c in candidates]
    ws = [c[1] for c in candidates]
    return rng.choices(names, weights=ws, k=1)[0]

# ----------------------------
# SHAP dataset helper (unchanged)
# ----------------------------

def build_shap_dataset_all_features(infos, fill_value=0):
    rows = []
    all_keys = set()

    for step_i, step_info in enumerate(infos):
        for agent, d in step_info.items():
            sf = d.get("shap_features")
            y = d.get("chosen_action_type")
            if not sf or y is None:
                continue
            all_keys.update(sf.keys())
            rows.append({"step": step_i, "agent": agent, "y": str(y), **sf})

    df = pd.DataFrame(rows)

    for k in sorted(all_keys):
        if k not in df.columns:
            df[k] = fill_value

    feature_cols = sorted(all_keys)
    if feature_cols:
        df[feature_cols] = df[feature_cols].fillna(fill_value)

    return df

# ----------------------------
# Env builder
# ----------------------------

def make_env(seed: int | None, episode_len: int, red_agent_class: type, submission):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=red_agent_class,
        steps=episode_len,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    try:
        wrapped_cyborg = submission.wrap(cyborg)
    except TypeError as exc:
        if "positional argument" not in str(exc):
            raise
        wrapped_cyborg = submission.__class__.wrap(cyborg)
    return cyborg, wrapped_cyborg

# ----------------------------
# Core runner for ONE EPISODE (given a fixed red profile)
# ----------------------------

def run_one_episode_and_log(
    submission,
    episode_idx: int,
    env_seed: int | None,
    episode_length: int,
    red_profile_name: str,
    red_cls: type,
    log_path_reward: Path,
    log_path_actions: Path,
    is_heuristic: bool,
    shap: bool,
    prof_dir: Path | None,
    epi: int | None,
) -> Tuple[float, List[Any], List[Dict[str, Any]]]:
    """
    Runs 1 episode in an env with fixed red agent class and appends to logs.

    Returns:
      total_reward_scalar, policy_rows (heuristic SHAP), all_info_steps (for GNN SHAP)
    """

    _, wrapped_cyborg = make_env(env_seed, episode_length, red_cls, submission)
    observations, _ = wrapped_cyborg.reset()

    reward_sum = 0.0
    policy_rows: List[Any] = []
    all_info_steps: List[Dict[str, Any]] = []

    for t in range(episode_length):
        actions = {
            agent_name: agent.get_action(
                observations[agent_name], wrapped_cyborg.action_space(agent_name)
            )
            for agent_name, agent in submission.AGENTS.items()
            if agent_name in wrapped_cyborg.agents
        }

        # ---- step env ----
        step_result = wrapped_cyborg.step(actions)
        if len(step_result) == 5:
            observations, rewards_scalar, term, trunc, info = step_result
            translated_actions = None
        elif len(step_result) >= 7:
            observations, rewards_scalar, term, trunc, info, translated_actions, _raw_observation = step_result[:7]
        else:
            raise RuntimeError(f"Unexpected wrapped_cyborg.step return length: {len(step_result)}")

        # Prefer translated Action objects from newer wrappers. Fall back to the
        # older manual translation path, then finally to the raw action ids.
        if translated_actions is not None:
            actions_actual = translated_actions
        elif not is_heuristic and submission.NAME != "Sleep" and hasattr(wrapped_cyborg, "action_translator"):
            actions_actual = {
                agent_name: wrapped_cyborg.action_translator(agent_name, actions[agent_name])
                for agent_name in actions
            }
        else:
            actions_actual = actions

        # ---- log actions (per profile) ----
        log_actions_jsonl(log_path_actions, episode=episode_idx, step=t, actions=actions_actual, mode="type")

        all_info_steps.append(info)
        if prof_dir is not None:
            scalar_log = prof_dir / "scalar_rewards.jsonl"  # or output_dir/<profile>/scalar_rewards.jsonl
            with scalar_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "profile": red_profile_name,
                    "episode": epi,
                    "step": t,
                    "rewards": rewards_scalar.get("blue_agent_0", 0),
                }, default=str) + "\n")


        # ---- scalar reward (sum over blue agents) ----
        reward_sum += rewards_scalar["blue_agent_0"]
        """
        # ---- append reward log entry (per profile) ----
        # IMPORTANT: keep this format compatible with your parse_log(...)
        # If your parse_log expects extra fields, extend here accordingly.
        log_entry = {
            "profile": red_profile_name,
            "episode": episode_idx,
            "step": t,
        }
        with open(log_path_reward, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")"""

        # ---- heuristic SHAP ----
        if is_heuristic and shap:
            for agent_name, agent in submission.AGENTS.items():
                if not hasattr(agent, "info") or len(agent.info) == 0:
                    continue
                step_info = agent.info[-1]
                if "Predicates" in step_info and "ActionClass" in step_info:
                    policy_rows.append((step_info["Predicates"], step_info["ActionClass"]))

        # termination?
        done = {
            agent: term.get(agent, False) or trunc.get(agent, False)
            for agent in wrapped_cyborg.agents
        }
        if done and all(done.values()):
            break

    return reward_sum, policy_rows, all_info_steps

# ----------------------------
# Main Explainability driver
# ----------------------------

def run_explainability_profiles(
    submission,
    output_dir: str,
    max_eps: int = 10,
    seed: Optional[int] = None,
    episode_length: int = 500,
    shap: bool = False,
    is_heuristic: bool = False,

    # profile selection mode
    mode: str = "single",                 # single | sweep
    strategy: str = "single",             # single | mixture (only relevant for mode=single)
    single_profile: str = "fsm_default",
    profile_weights: Optional[Dict[str, float]] = None,

    # plotting controls
    include_mixed_in_sweep: bool = False,
):
    cyborg_version = CYBORG_VERSION
    scenario = "Scenario4"

    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"
    print(version_header)
    print(author_header)
    print(f"Using agents {submission.AGENTS}")
    print("Available profiles:", sorted(PROFILE_REGISTRY.keys()))

    out_root = Path(output_dir)
    ensure_dir(out_root)

    # ----------------------------
    # Decide which profiles to run
    # ----------------------------
    profiles_to_run: List[str] = []
    weights = _filter_available_weights(profile_weights or {})

    if mode == "sweep":
        profiles_to_run = [p for p in sorted(PROFILE_REGISTRY.keys()) if p != "verbose"]
        if include_mixed_in_sweep:
            profiles_to_run.append("mixed")
    else:
        # single mode:
        if strategy == "single":
            if single_profile not in PROFILE_REGISTRY:
                raise ValueError(f"single_profile='{single_profile}' not found. Available: {sorted(PROFILE_REGISTRY.keys())}")
            profiles_to_run = [single_profile]
        else:
            # mixture in single-mode: we still want logs PER red profile,
            # so we create/append to separate per-profile folders while sampling profiles per episode.
            if not profile_weights:
                weights = _filter_available_weights({
                    "fsm_default": 0.35,
                    "stealth_pivot": 0.25,
                    "lateral_spread": 0.20,
                    "impact_rush": 0.15,
                    "deception_aware": 0.05,
                })
            profiles_to_run = sorted([p for p in weights.keys() if p in PROFILE_REGISTRY])
            if not profiles_to_run:
                profiles_to_run = ["fsm_default"]

    # Prepare per-profile directories & clean logs
    per_profile_dirs: Dict[str, Path] = {}
    for prof in profiles_to_run:
        prof_dir = out_root / prof
        ensure_dir(prof_dir)
        per_profile_dirs[prof] = prof_dir

        # clear logs (fresh run)
        (prof_dir / "reward_log.jsonl").write_text("")
        (prof_dir / "actions.jsonl").write_text("")
    # pro Profil-Ordner:
    # ----------------------------
    # Episode loop
    # ----------------------------
    # In sweep mode: run max_eps episodes per profile separately.
    # In single/mixture mode: run max_eps total episodes, but route each episode into the sampled profile log.
    start = datetime.now()

    # store scalar totals per profile
    totals_by_profile: Dict[str, List[float]] = {p: [] for p in profiles_to_run if p != "mixed"}
    # also store SHAP data per profile (so SHAP is per profile)
    shap_policy_rows: Dict[str, List[Any]] = {p: [] for p in profiles_to_run}
    shap_infos: Dict[str, List[Dict[str, Any]]] = {p: [] for p in profiles_to_run}

    def prof_seed(base: int | None, offset: int) -> int:
        return (base if base is not None else 0) + offset

    if mode == "sweep":
        pbar = tqdm(total=len([p for p in profiles_to_run]), desc="Explainability sweep", unit="run", dynamic_ncols=True)
        try:
            for idx, prof in enumerate(profiles_to_run):

                if prof == "mixed":
                    # mixed run is still logged per profile by sampling profiles per episode
                    run_label = "mixed"
                    weights_m = weights
                    ensure_dir(per_profile_dirs["mixed"])
                    # note: we will also append into the concrete profile folders, not only mixed/
                    # so mixed/ can hold a summary.json if you want, but logs remain per-profile.
                    pbar.set_postfix_str(f"profile=mixed eps={max_eps}")
                    for epi in tqdm(range(max_eps), desc="Episodes(mixed)", leave=False):
                        rng = random.Random(prof_seed(seed, 10_000_000 + epi))
                        chosen = sample_profile(rng, weights_m)
                        red_cls = PROFILE_REGISTRY.get(chosen, PROFILE_REGISTRY["fsm_default"])

                        env_seed = prof_seed(seed, 100_000 + epi + 999_999)
                        prof_dir = per_profile_dirs[chosen]
                        reward_log = Path(prof_dir) / "reward_log.jsonl"

                        os.environ["CYBORG_REWARD_LOG_PATH"] = str(reward_log)
                        os.environ["CYBORG_ATTACK_PROFILE"] = prof
                        os.environ["CYBORG_RUN_TAG"] = out_root.name  # oder submission.NAME + timestamp
                        r, pol_rows, infos = run_one_episode_and_log(
                            submission=submission,
                            episode_idx=epi,
                            env_seed=env_seed,
                            episode_length=episode_length,
                            red_profile_name=chosen,
                            red_cls=red_cls,
                            log_path_reward=prof_dir / "reward_log.jsonl",
                            log_path_actions=prof_dir / "actions.jsonl",
                            is_heuristic=is_heuristic,
                            shap=shap,
                            prof_dir=prof_dir,
                            epi=epi,
                        )
                        totals_by_profile.setdefault(chosen, []).append(r)
                        if shap:
                            shap_policy_rows[chosen].extend(pol_rows)
                            shap_infos[chosen].extend(infos)

                    # write mixed summary
                    save_json(
                        {
                            "mode": "sweep",
                            "run": "mixed",
                            "weights": weights_m,
                            "parameters": {"seed": seed, "episode_length": episode_length, "max_episodes": max_eps},
                            "time": {"start": str(start), "end": str(datetime.now())},
                        },
                        str(per_profile_dirs["mixed"] / "summary_mixed.json"),
                    )

                else:
                    pbar.set_postfix_str(f"profile={prof} eps={max_eps}")
                    red_cls = PROFILE_REGISTRY[prof]
                    for epi in tqdm(range(max_eps), desc=f"Episodes({prof})", leave=False):
                        env_seed = prof_seed(seed, 100_000 + epi + 10_000 * (idx + 1))
                        prof_dir = per_profile_dirs[prof]
                        reward_log = Path(prof_dir) / "reward_log.jsonl"

                        os.environ["CYBORG_REWARD_LOG_PATH"] = str(reward_log)
                        os.environ["CYBORG_ATTACK_PROFILE"] = prof
                        os.environ["CYBORG_RUN_TAG"] = out_root.name  # oder submission.NAME + timestamp
                        r, pol_rows, infos = run_one_episode_and_log(
                            submission=submission,
                            episode_idx=epi,
                            env_seed=env_seed,
                            episode_length=episode_length,
                            red_profile_name=prof,
                            red_cls=red_cls,
                            log_path_reward=prof_dir / "reward_log.jsonl",
                            log_path_actions=prof_dir / "actions.jsonl",
                            is_heuristic=is_heuristic,
                            shap=shap,
                            prof_dir=prof_dir,
                            epi=epi,
                        )
                        totals_by_profile[prof].append(r)
                        if shap:
                            shap_policy_rows[prof].extend(pol_rows)
                            shap_infos[prof].extend(infos)

                pbar.update(1)
        finally:
            pbar.close()

    else:
        # single mode
        if strategy == "single":
            chosen_profiles = [single_profile for _ in range(max_eps)]
        else:
            # mixture: sample profile per episode, but log to that profile’s logs
            chosen_profiles = []
            for epi in range(max_eps):
                rng = random.Random(prof_seed(seed, 42_000 + epi))
                chosen_profiles.append(sample_profile(rng, weights))

        for epi in tqdm(range(max_eps), desc="Episodes(single)"):
            prof = chosen_profiles[epi]
            red_cls = PROFILE_REGISTRY.get(prof, PROFILE_REGISTRY["fsm_default"])
            env_seed = prof_seed(seed, 100_000 + epi)

            prof_dir = per_profile_dirs[prof]
            reward_log = Path(prof_dir) / "reward_log.jsonl"

            os.environ["CYBORG_REWARD_LOG_PATH"] = str(reward_log)
            os.environ["CYBORG_ATTACK_PROFILE"] = prof
            os.environ["CYBORG_RUN_TAG"] = out_root.name  # oder submission.NAME + timestamp
            r, pol_rows, infos = run_one_episode_and_log(
                submission=submission,
                episode_idx=epi,
                env_seed=env_seed,
                episode_length=episode_length,
                red_profile_name=prof,
                red_cls=red_cls,
                log_path_reward=prof_dir / "reward_log.jsonl",
                log_path_actions=prof_dir / "actions.jsonl",
                is_heuristic=is_heuristic,
                shap=shap,
                prof_dir=prof_dir,
                epi=epi,
            )
            totals_by_profile.setdefault(prof, []).append(r)
            if shap:
                shap_policy_rows[prof].extend(pol_rows)
                shap_infos[prof].extend(infos)

    end = datetime.now()

    # ----------------------------
    # Save per-profile scalar summaries + run SHAP + Reward Decomp plots
    # ----------------------------
    for prof, prof_dir in per_profile_dirs.items():
        if prof == "mixed":
            continue

        totals = totals_by_profile.get(prof, [])
        reward_mean = mean(totals) if totals else 0.0
        reward_stdev = stdev(totals) if len(totals) > 1 else 0.0
        print(f"[{prof}] Average total (scalar): {reward_mean:.2f} ± {reward_stdev:.2f}")

        save_json(
            {
                "submission": {
                    "author": submission.NAME,
                    "team": submission.TEAM,
                    "technique": submission.TECHNIQUE,
                },
                "parameters": {
                    "seed": seed,
                    "episode_length": episode_length,
                    "max_episodes": max_eps,
                    "mode": mode,
                    "strategy": strategy if mode == "single" else None,
                    "profile": prof,
                    "profile_weights": weights if (mode == "single" and strategy == "mixture") else None,
                },
                "time": {"start": str(start), "end": str(end), "elapsed": str(end - start)},
                "reward_scalar": {
                    "mean": reward_mean,
                    "stdev": reward_stdev,
                    "per_episode": totals,
                },
            },
            str(prof_dir / "summary_scalar.json"),
        )

        # ---- SHAP (per profile) ----
        if shap and is_heuristic:
            out_dir_shap = prof_dir / "SHAPAnalysis"
            ensure_dir(out_dir_shap)
            summary = train_surrogate_and_shap(
                shap_policy_rows.get(prof, []),
                out_dir=str(out_dir_shap),
            )
            print(f"[{prof}] SHAP(heuristic) summary:", summary)

        elif shap and (not is_heuristic):
            out_dir_shap = prof_dir / "SHAPAnalysis"
            ensure_dir(out_dir_shap)
            df = build_shap_dataset_all_features(shap_infos.get(prof, []))
            if not df.empty and "prev_action_success" in df.columns:
                df["prev_action_success"] = pd.to_numeric(df["prev_action_success"], errors="coerce").fillna(-1)

            if not df.empty:
                run_shap(df, max_classes=8, out_dir=str(out_dir_shap))
                print(f"[{prof}] SHAP(GNN) done.")
            else:
                print(f"[{prof}] SHAP skipped: empty dataset (no shap_features/chosen_action_type).")



    print(f"\n[explain] Done. Results under: {out_root}")

# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CybORG Explainability Script (per red profile logs)")

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-eps", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=500)

    parser.add_argument("--shap", action="store_true")

    parser.add_argument("--output", type=str, default=os.path.abspath("Results"))
    parser.add_argument("--submission-path", type=str, default=os.path.abspath(""))

    parser.add_argument("--is-heuristic", action="store_true")
    parser.add_argument("--non-heuristic", dest="is_heuristic", action="store_false")
    parser.set_defaults(is_heuristic=False)
    parser.add_argument("--phase_reward_mode", default="default",
                    choices=["default", "contractor_off", "red_only"])
    parser.add_argument("--reward_blue", action="store_true")

    # profile control
    parser.add_argument("--mode", choices=["single", "sweep"], default="sweep")
    parser.add_argument("--strategy", choices=["single", "mixture"], default="single")
    parser.add_argument("--single_profile", default="fsm_default")
    parser.add_argument("--profile_weights", default="")  # name=w,name=w,...
    parser.add_argument("--include-mixed-in-sweep", action="store_true")

    args = parser.parse_args()
    os.environ["CYBORG_PHASE_REWARD_MODE"] = args.phase_reward_mode
    os.environ["CYBORG_REWARD_BLUE"] = "1" if args.reward_blue else "0"

    submission = load_submission(args.submission_path)
    if isinstance(submission, type):
        submission = submission()

    os.makedirs(args.output, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    agent_type = submission.NAME
    run_dir = os.path.join(args.output, f"{agent_type}_{ts}")
    rmkdir(run_dir)

    weights = parse_profile_weights(args.profile_weights) if args.profile_weights else None

    run_explainability_profiles(
        submission=submission,
        output_dir=run_dir,
        max_eps=args.max_eps,
        seed=args.seed,
        episode_length=args.episode_length,
        shap=args.shap,
        is_heuristic=args.is_heuristic,
        mode=args.mode,
        strategy=args.strategy,
        single_profile=args.single_profile,
        profile_weights=weights,
        include_mixed_in_sweep=args.include_mixed_in_sweep,
    )

    """python explain.py \
  --mode sweep \
  --max-eps 10 \
  --episode-length 100 \
  --non-heuristic \
  --seed 1337 \
  --phase_reward_mode red_only \
  --submission-path ."""
