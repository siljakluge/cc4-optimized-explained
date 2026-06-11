from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
"""The OG Script"""

SUBNET_LABEL_MAP = {
    "admin_network_subnet": "Admin",
    "contractor_network_subnet": "Contractor",
    "office_network_subnet": "Office",
    "operational_zone_a_subnet": "Operational A",
    "operational_zone_b_subnet": "Operational B",
    "public_access_zone_subnet": "Public",
    "restricted_zone_a_subnet": "Restricted A",
    "restricted_zone_b_subnet": "Restricted B",
    "internet_subnet": "Internet",
    "action_cost": "Action Cost",
}

ATTACKER_ORDER = [
    "deception_aware",
    "discovery",
    "fsm_default",
    "impact_rush",
    "lateral_spread",
    "stealth_pivot",
]
BASE_DIR = Path(__file__).resolve().parent.parent  # geht von plotting/ eine Ebene hoch
RESULTS_DIR = BASE_DIR / "Result"
MODEL_DIRS_DEFAULT = {
    "SimpleGNN": RESULTS_DIR / "SimpleGNN",
    "Heuristic": RESULTS_DIR / "Heuristic",
    "RedVariants": RESULTS_DIR / "RedVariants",
    "Sleep": RESULTS_DIR / "Sleep",
}

OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

SUBNET_ORDER = [
    "operational_zone_a_subnet",
    "operational_zone_b_subnet",
    "restricted_zone_a_subnet",
    "restricted_zone_b_subnet",
    "contractor_network_subnet",
    "admin_network_subnet",
    "office_network_subnet",
    "public_access_zone_subnet",
    "internet_subnet",
    "action_cost",
]


def _style_rcparams():
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 600,
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.borderpad": 0.3,
        }
    )


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out_png


def _apply_common_axes_style(ax: plt.Axes, y_label: str):
    ax.set_xlabel("Subnet")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.axhline(0.0, linewidth=1.2, color="black", zorder=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", labelsize=11)


def reconstruct_episode_from_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct episode ids when broken logging always wrote episode=0.
    Assumes rows are in logging order and each new episode starts when step decreases.
    """
    if df.empty:
        return df

    df = df.copy().reset_index(drop=True)
    episodes = []
    cur_ep = 0
    prev_step = None

    for step in df["step"].tolist():
        if prev_step is not None and step < prev_step:
            cur_ep += 1
        episodes.append(cur_ep)
        prev_step = step

    df["episode_recon"] = episodes
    return df


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_reward_log(log_path: Path) -> pd.DataFrame:
    """
    Output columns:
      profile, step, phase, subnet, value, reward_total
    """
    rows = []
    entries = read_jsonl(log_path)

    for obj in entries:
        if not {"phase", "reward_list", "total", "step"}.issubset(obj.keys()):
            continue

        profile = obj.get("profile")
        step = int(obj["step"])
        phase = obj["phase"]
        reward_total = float(obj.get("total", 0.0))
        reward_list = obj.get("reward_list", {})

        # even if reward_list is empty, keep total row implicit via join later
        if reward_list:
            for subnet, compdict in reward_list.items():
                subnet_val = 0.0
                for _comp, val in compdict.items():
                    subnet_val += float(val)
                rows.append(
                    {
                        "profile": profile,
                        "step": step,
                        "phase": phase,
                        "subnet": subnet,
                        "value": subnet_val,
                        "reward_total": reward_total,
                    }
                )
        else:
            rows.append(
                {
                    "profile": profile,
                    "step": step,
                    "phase": phase,
                    "subnet": None,
                    "value": 0.0,
                    "reward_total": reward_total,
                }
            )

    df = pd.DataFrame(rows, columns=["profile", "step", "phase", "subnet", "value", "reward_total"])
    if df.empty:
        return df

    df = reconstruct_episode_from_steps(df)
    return df


def parse_scalar_log(log_path: Path) -> pd.DataFrame:
    """
    Output columns:
      profile, step, total_scalar
    Supports either:
      {"rewards": -5}
    or
      {"total_scalar": -5}
    """
    rows = []
    entries = read_jsonl(log_path)

    for obj in entries:
        if "step" not in obj:
            continue

        total_scalar = None
        if "total_scalar" in obj:
            total_scalar = float(obj["total_scalar"])
        elif "rewards" in obj:
            total_scalar = float(obj["rewards"])

        if total_scalar is None:
            continue

        rows.append(
            {
                "profile": obj.get("profile"),
                "step": int(obj["step"]),
                "total_scalar": total_scalar,
            }
        )

    df = pd.DataFrame(rows, columns=["profile", "step", "total_scalar"])
    if df.empty:
        return df

    df = reconstruct_episode_from_steps(df)
    return df


def build_attacker_subnet_table(
    reward_log_path: Path,
    scalar_log_path: Path,
    avg: bool = True,
) -> pd.DataFrame:
    """
    Returns:
      attacker x subnet table with subnets + action_cost
    """
    df_r = parse_reward_log(reward_log_path)
    df_s = parse_scalar_log(scalar_log_path)

    if df_r.empty and df_s.empty:
        return pd.DataFrame()

    # ---- subnet reward aggregation ----
    subnet_part = pd.DataFrame(columns=["profile", "subnet", "value"])

    if not df_r.empty:
        subnet_rows = df_r[df_r["subnet"].notna()].copy()

        if avg:
            subnet_part = (
                subnet_rows.groupby(["profile", "episode_recon", "subnet"], dropna=True)["value"]
                .sum()
                .groupby(["profile", "subnet"], dropna=True)
                .mean()
                .reset_index()
            )
        else:
            subnet_part = (
                subnet_rows.groupby(["profile", "subnet"], dropna=True)["value"]
                .sum()
                .reset_index()
            )

    # ---- reward_total per step (deduplicate step-level rows) ----
    reward_total_step = pd.DataFrame(columns=["profile", "episode_recon", "step", "reward_total"])

    if not df_r.empty:
        reward_total_step = (
            df_r.groupby(["profile", "episode_recon", "step"], dropna=True)["reward_total"]
            .first()
            .reset_index()
        )

    # ---- scalar_total per step ----
    scalar_total_step = pd.DataFrame(columns=["profile", "episode_recon", "step", "total_scalar"])

    if not df_s.empty:
        scalar_total_step = (
            df_s.groupby(["profile", "episode_recon", "step"], dropna=True)["total_scalar"]
            .first()
            .reset_index()
        )

    # ---- action cost ----
    action_cost_part = pd.DataFrame(columns=["profile", "subnet", "value"])

    if not reward_total_step.empty and not scalar_total_step.empty:
        merged = pd.merge(
            scalar_total_step,
            reward_total_step,
            on=["profile", "episode_recon", "step"],
            how="inner",
        )
        merged["action_cost"] = merged["total_scalar"] - merged["reward_total"]

        if avg:
            action_cost_part = (
                merged.groupby(["profile", "episode_recon"], dropna=True)["action_cost"]
                .sum()
                .groupby("profile", dropna=True)
                .mean()
                .reset_index()
            )
        else:
            action_cost_part = (
                merged.groupby("profile", dropna=True)["action_cost"]
                .sum()
                .reset_index()
            )

        action_cost_part["subnet"] = "action_cost"
        action_cost_part = action_cost_part.rename(columns={"action_cost": "value"})

    # ---- combine ----
    combined = pd.concat([subnet_part, action_cost_part], ignore_index=True)

    if combined.empty:
        return combined

    combined["subnet"] = pd.Categorical(combined["subnet"], categories=SUBNET_ORDER, ordered=True)
    combined = combined.sort_values(["profile", "subnet"]).reset_index(drop=True)
    return combined


def plot_model_attacker_lines(
    model_name: str,
    model_dir: Path,
    out_dir: Path,
    attackers: Optional[list[str]] = None,
    avg: bool = True,
    show_legend: bool = False,
):
    if attackers is None:
        attackers = ATTACKER_ORDER

    rows = []

    for attacker in attackers:
        reward_log = model_dir / attacker / "reward_log.jsonl"
        scalar_log = model_dir / attacker / "scalar_rewards.jsonl"

        if not reward_log.exists() or not scalar_log.exists():
            print(f"[warn] Missing logs for {model_name}/{attacker}")
            continue

        table = build_attacker_subnet_table(reward_log, scalar_log, avg=avg)
        if table.empty:
            continue

        table = table.copy()
        table["attacker"] = attacker
        rows.append(table)

    if not rows:
        print(f"[warn] No data found for model {model_name}")
        return None

    df = pd.concat(rows, ignore_index=True)

    with plt.style.context(["science", "ieee", "bright", "no-latex"]):
        _style_rcparams()

        fig, ax = plt.subplots(figsize=(6.8, 3.2))

        x_labels = [s for s in SUBNET_ORDER if s in set(df["subnet"].astype(str))]
        x = np.arange(len(x_labels))

        color_map = {
            attacker: OKABE_ITO[i % len(OKABE_ITO)]
            for i, attacker in enumerate(attackers)
        }

        for attacker in attackers:
            sub = df[df["attacker"] == attacker].copy()
            if sub.empty:
                continue

            y_vals = []
            for subnet in x_labels:
                val = sub.loc[sub["subnet"].astype(str) == subnet, "value"]
                y_vals.append(float(val.iloc[0]) if len(val) else 0.0)

            ax.plot(
                x,
                y_vals,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                label=attacker,
                color=color_map[attacker],
            )

        pretty_labels = [SUBNET_LABEL_MAP.get(s, s) for s in x_labels]
        ax.set_xticks(x)
        ax.set_xticklabels(pretty_labels, rotation=25, ha="right", fontsize=9)

        _apply_common_axes_style(ax, y_label="Avg Reward" if avg else "Cumulative Reward")
        ax.set_title(model_name, fontsize=12, fontweight="bold")

        if "sleep" in model_name.lower():
            ax.set_yticks([0, -1000, -2000, -3000])
        else:
            ax.set_yticks([0, -50, -100, -150])

        if show_legend:
            ax.legend(
                loc="lower right",
                frameon=True,
                fontsize=10,
                handlelength=1.5,
                borderpad=0.4,
            )

        stem = f"{model_name}_reward_decomposition_lines_{'avg' if avg else 'sum'}"
        return _save_fig(fig, out_dir, stem)

def plot_all_models_2x2(
    model_dirs: dict[str, Path],
    out_dir: Path,
    attackers: Optional[list[str]] = None,
    avg: bool = True,
):
    if attackers is None:
        attackers = ATTACKER_ORDER

    model_tables = {}

    # ---- load data for all models ----
    for model_name, model_dir in model_dirs.items():
        rows = []

        for attacker in attackers:
            reward_log = model_dir / attacker / "reward_log.jsonl"
            scalar_log = model_dir / attacker / "scalar_rewards.jsonl"

            if not reward_log.exists() or not scalar_log.exists():
                print(f"[warn] Missing logs for {model_name}/{attacker}")
                continue

            table = build_attacker_subnet_table(reward_log, scalar_log, avg=avg)
            if table.empty:
                continue

            table = table.copy()
            table["attacker"] = attacker
            rows.append(table)

        if rows:
            model_tables[model_name] = pd.concat(rows, ignore_index=True)

    if not model_tables:
        print("[warn] No model data found.")
        return None

    with plt.style.context(["science", "ieee", "bright", "no-latex"]):
        _style_rcparams()

        fig, axes = plt.subplots(
            2, 2,
            figsize=(7.16, 3.6),  # breiter Eindruck, weniger Höhe
            sharex=True,
            sharey=False
        )
        fig.subplots_adjust(
            left=0.07,
            right=0.995,
            bottom=0.16,
            top=0.86,
            wspace=0.18,
            hspace=0.25
        )
        axes = axes.flatten()

        legend_handles = None
        legend_labels = None

        color_map = {
            attacker: OKABE_ITO[i % len(OKABE_ITO)]
            for i, attacker in enumerate(attackers)
        }

        x_labels = SUBNET_ORDER
        x = np.arange(len(x_labels))

        global_min = 0.0
        for df in model_tables.values():
            if not df.empty:
                global_min = min(global_min, float(df["value"].min()))

        for ax, (model_name, df) in zip(axes, model_tables.items()):
            for attacker in attackers:
                sub = df[df["attacker"] == attacker].copy()
                if sub.empty:
                    continue

                y_vals = []
                for subnet in x_labels:
                    val = sub.loc[sub["subnet"].astype(str) == subnet, "value"]
                    y_vals.append(float(val.iloc[0]) if len(val) else 0.0)

                ax.plot(
                    x,
                    y_vals,
                    marker="o",
                    linewidth=1.3,
                    markersize=3.0,
                    label=attacker,
                    color=color_map[attacker],
                )

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            ax.set_title(model_name, fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.25, zorder=0)
            ax.axhline(0.0, linewidth=1.2, color="black", zorder=3)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            pretty_labels = [SUBNET_LABEL_MAP.get(s, s) for s in x_labels]
            ax.set_xticks(x)
            ax.set_xticklabels(pretty_labels, rotation=25, ha="right", fontsize=9)

            if "sleep" in model_name.lower():
                ax.set_yticks([0, -1000, -2000, -3000])
            else:
                ax.set_yticks([0, -100, -200, -300])

        # axis labels only where needed
        axes[0].set_ylabel("Avg Reward" if avg else "Cumulative Reward")
        axes[2].set_ylabel("Avg Reward" if avg else "Cumulative Reward")
        axes[2].set_xlabel("Subnet", fontsize=10)
        axes[3].set_xlabel("Subnet", fontsize=10)

        for ax in axes:
            ax.tick_params(axis="x", pad=1)
        for ax in axes:
            ax.xaxis.labelpad = 2
            ax.yaxis.labelpad = 2

        # one shared legend above all plots
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=len(legend_labels),   # alles in eine Zeile
            frameon=True,
            fontsize=11,
            handlelength=1.5,
            columnspacing=0.9,
            borderpad=0.3,
        )

        fig.subplots_adjust(
            left=0.07,
            right=0.995,
            bottom=0.12,
            top=0.855,
            wspace=0.20,
            hspace=0.18
        )

        stem = f"reward_decomposition_all_models_2x2_{'avg' if avg else 'sum'}"
        return _save_fig(fig, out_dir, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("Plots/RewardDecomp"))
    ap.add_argument("--sum", action="store_true")
    args = ap.parse_args()

    avg = not args.sum
    out_dir = Path(args.out)

    saved = []
    for model_name, model_dir in MODEL_DIRS_DEFAULT.items():
        p = plot_model_attacker_lines(
            model_name=model_name,
            model_dir=model_dir,
            out_dir=out_dir,
            attackers=ATTACKER_ORDER,
            avg=avg,
        )
        if p is not None:
            saved.append(p)

    print("Saved figures:")
    for p in saved:
        print(f" - {p}")

    model_dirs = MODEL_DIRS_DEFAULT

    out = plot_all_models_2x2(
        model_dirs=model_dirs,
        out_dir=out_dir,
        attackers=ATTACKER_ORDER,
        avg=avg,
    )

    print("Saved figures:")
    if out is not None:
        print(f" - {out}")


if __name__ == "__main__":
    main()