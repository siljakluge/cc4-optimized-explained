from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    HAS_SCIENCEPLOTS = True
except Exception:
    HAS_SCIENCEPLOTS = False

BASE_DIR = Path(__file__).resolve().parent.parent  # geht von plotting/ eine Ebene hoch
RESULTS_DIR = BASE_DIR / "Result"

# -----------------------
# Action label + colors
# -----------------------

ACTION_COLORS = {
    "Sleep": "#7f7f7f",
    "Analyse": "#1f77b4",
    "Monitor": "#17becf",
    "Restore": "#2ca02c",
    "Remove": "#d62728",
    "DeployDecoy": "#9467bd",
    "AllowTrafficZone": "#bcbd22",
    "BlockTrafficZone": "#ff7f0e",
    "None": "#c7c7c7",
    "OTHER": "#8c564b",
}

ACTION_ORDER = [
    "Sleep", "Analyse", "Monitor", "Restore", "Remove", "DeployDecoy",
    "AllowTrafficZone", "BlockTrafficZone", "None", "OTHER"
]

DEFAULT_ATTACKERS = [
    "deception_aware",
    "discovery",
    "fsm_default",
    "impact_rush",
    "lateral_spread",
    "stealth_pivot",
    "verbose",
]

DEFAULT_BLUE_AGENTS = [
    "blue_agent_0",
    "blue_agent_1",
    "blue_agent_2",
    "blue_agent_3",
    "blue_agent_4",
]


def apply_style():
    if HAS_SCIENCEPLOTS:
        plt.style.use(["science", "ieee", "no-latex"])
    else:
        plt.style.use("default")


def stable_color(action: str):
    """Stable color for known actions + deterministic fallback for unknown."""
    if action in ACTION_COLORS:
        return ACTION_COLORS[action]
    cmap = plt.get_cmap("tab20")
    h = int(hashlib.md5(action.encode("utf-8")).hexdigest(), 16)
    return cmap(h % cmap.N)


def action_to_label(act, mode: str = "type") -> str:
    """
    act is typically like [AllowTrafficZone] or [DeployDecoy host_x]
    mode:
      - "type": only the action name (first token)
      - "full": full string representation (keeps parameters/targets)
    """
    if isinstance(act, (list, tuple)) and len(act) == 1:
        act = act[0]
    elif isinstance(act, (list, tuple)) and len(act) == 0:
        return "None"

    s = str(act).strip()
    if mode == "full":
        return s
    return s.split()[0] if s else "Unknown"


def actions_to_label(act: Any, mode: str = "type") -> str:
    if act is None:
        return "None"
    if isinstance(act, (list, tuple)):
        if len(act) == 0:
            return "None"
        if len(act) == 1:
            act = act[0]
        else:
            act = " | ".join(str(a) for a in act)

    s = str(act).strip()
    if not s:
        return "None"
    if mode == "full":
        return s
    return s.split()[0]


# -----------------------
# Logging in eval loop
# -----------------------

def log_actions_jsonl(
    log_path: Path,
    episode: int,
    step: int,
    actions: Dict[str, Any],
    mode: str = "type",
):
    """
    Append one JSONL line:
      {"episode": int, "step": int, "actions": {"blue_agent_0": "AllowTrafficZone", ...}}
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    actions_str = {agent: action_to_label(act, mode=mode) for agent, act in actions.items()}
    entry = {"episode": int(episode), "step": int(step), "actions": actions_str}

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# -----------------------
# Read logs
# -----------------------

def read_actions_log_jsonl(log_path: Path) -> List[Dict[str, Any]]:
    """Reads the JSONL action log into a list of entries."""
    log_path = Path(log_path)
    entries = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def load_all_action_logs(
    base_dir: Path,
    attackers: List[str],
) -> pd.DataFrame:
    """
    Load all logs from:
      base_dir/<attacker>/actions.jsonl
    Returns long dataframe with columns:
      attacker, episode, step, agent, action
    """
    rows = []

    for attacker in attackers:
        log_path = base_dir / attacker / "actions.jsonl"
        if not log_path.exists():
            print(f"[warn] Missing log: {log_path}")
            continue

        entries = read_actions_log_jsonl(log_path)
        for e in entries:
            epi = e.get("episode")
            step = e.get("step")
            actions = e.get("actions", {})
            for agent, action_label in actions.items():
                rows.append({
                    "attacker": attacker,
                    "episode": epi,
                    "step": step,
                    "agent": str(agent),
                    "action": str(action_label),
                })

    if not rows:
        return pd.DataFrame(columns=["attacker", "episode", "step", "agent", "action"])

    return pd.DataFrame(rows)


# -----------------------
# Plot all attackers x agents
# -----------------------

def plot_action_frequencies_all_attackers_agents(
    base_dir: Path,
    out_dir: Path,
    attackers: Optional[List[str]] = None,
    blue_agents: Optional[List[str]] = None,
    normalize: bool = True,
    top_k: Optional[int] = 25,
):
    """
    Create one stacked bar plot with bars for attacker x blue_agent.

    Desired structure:
      6x5 (or Nx5) bars = attacker profiles x blue agents
    Each bar is stacked by action frequency.

    Also saves:
      - CSV table
      - heatmap
    """
    apply_style()

    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if attackers is None:
        attackers = DEFAULT_ATTACKERS
    if blue_agents is None:
        blue_agents = DEFAULT_BLUE_AGENTS

    df = load_all_action_logs(base_dir=base_dir, attackers=attackers)

    if df.empty:
        print("No actions found in any attacker logs.")
        return None

    # keep only requested blue agents
    df = df[df["agent"].isin(blue_agents)].copy()

    if df.empty:
        print("No matching blue agents found.")
        return None

    # compress rare actions globally across all logs
    if top_k is not None:
        vc = df["action"].value_counts()
        keep = set(vc.head(top_k).index)
        df["action"] = df["action"].where(df["action"].isin(keep), other="OTHER")

    # aggregate: one bar per (attacker, agent)
    pivot = pd.crosstab(
        index=[df["attacker"], df["agent"]],
        columns=df["action"]
    )

    # enforce complete order
    full_index = pd.MultiIndex.from_product(
        [attackers, blue_agents],
        names=["attacker", "agent"]
    )
    pivot = pivot.reindex(full_index, fill_value=0)

    # stable action order
    cols = list(pivot.columns)
    ordered = [c for c in ACTION_ORDER if c in cols]
    rest = sorted([c for c in cols if c not in ordered])
    pivot = pivot[ordered + rest]

    if normalize:
        pivot = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100.0

    # save table
    out_csv = out_dir / f"action_freq_attacker_agent_{'pct' if normalize else 'count'}.csv"
    pivot.to_csv(out_csv)

    # -----------------------
    # stacked barplot
    # -----------------------
    n_bars = len(pivot.index)
    x = np.arange(n_bars)
    bottom = np.zeros(n_bars, dtype=float)

    fig, ax = plt.subplots(figsize=(16,6))

    for action in pivot.columns:
        vals = pivot[action].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=action,
            color=stable_color(str(action)),
            width=0.88,
            edgecolor="none",
        )
        bottom = bottom + vals

    ylabel = "Percent of actions (%)" if normalize else "Action count"
    ax.set_ylabel(ylabel, fontsize=14)

    # only show agent numbers (0–4)
    xticklabels = [ag.split("_")[-1] for _, ag in pivot.index]

    ax.set_xticks(x)
    ax.set_xticklabels(
        xticklabels,
        rotation=0,
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

    # visual separators between attackers
    n_agents = len(blue_agents)
    for i in range(1, len(attackers)):
        ax.axvline(i * n_agents - 0.5, color="black", linewidth=0.8, alpha=0.4)

    # attacker labels above groups
    group_centers = [i * n_agents + (n_agents - 1) / 2 for i in range(len(attackers))]
    ymax = ax.get_ylim()[1]
    for center, attacker in zip(group_centers, attackers):
        ax.text(
            center,
            ymax * 1.05,
            attacker,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold"
        )

    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlabel("Blue Agent", fontsize=14)

    fig.tight_layout()

    out_bar = out_dir / f"action_freq_attacker_agent_stacked_{'pct' if normalize else 'count'}.png"
    fig.savefig(out_bar, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -----------------------
    # heatmap
    # -----------------------
    fig, ax = plt.subplots(figsize=(max(12, 0.5 * len(pivot.columns) + 8), max(8, 0.25 * n_bars + 4)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto")

    ax.set_title("Action Frequency Heatmap across Attacker Profiles and Blue Agents")
    ax.set_xlabel("Action")
    ax.set_ylabel("Attacker / Agent")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")

    ylabels = [f"{att} | {ag}" for att, ag in pivot.index]
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(ylabels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(ylabel)

    fig.tight_layout()
    out_heat = out_dir / f"action_freq_attacker_agent_heatmap_{'pct' if normalize else 'count'}.png"
    fig.savefig(out_heat, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "stacked_bar": out_bar,
        "heatmap": out_heat,
        "table_csv": out_csv,
    }

def plot_action_frequencies_multiple_models(
    model_dirs: Dict[str, Path],
    out_dir: Path,
    attackers: Optional[List[str]] = None,
    blue_agents: Optional[List[str]] = None,
    normalize: bool = True,
    top_k: Optional[int] = 25,
):
    """
    Plot multiple models stacked vertically.
    One subplot per model, one shared legend for all.
    """
    apply_style()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if attackers is None:
        attackers = [
            "deception_aware",
            "discovery",
            "fsm_default",
            "impact_rush",
            "lateral_spread",
            "stealth_pivot",
        ]
    if blue_agents is None:
        blue_agents = DEFAULT_BLUE_AGENTS

    # ---- load all data first ----
    pivots = {}
    all_actions_global = set()

    for model_name, base_dir in model_dirs.items():
        df = load_all_action_logs(Path(base_dir), attackers=attackers)

        if df.empty:
            print(f"[warn] No actions found for model {model_name} in {base_dir}")
            continue

        df = df[df["agent"].isin(blue_agents)].copy()
        if df.empty:
            print(f"[warn] No matching blue agents for model {model_name}")
            continue

        if top_k is not None:
            vc = df["action"].value_counts()
            keep = set(vc.head(top_k).index)
            df["action"] = df["action"].where(df["action"].isin(keep), other="OTHER")

        pivot = pd.crosstab(
            index=[df["attacker"], df["agent"]],
            columns=df["action"]
        )

        full_index = pd.MultiIndex.from_product(
            [attackers, blue_agents],
            names=["attacker", "agent"]
        )
        pivot = pivot.reindex(full_index, fill_value=0)

        all_actions_global.update(pivot.columns)
        pivots[model_name] = pivot

    if not pivots:
        print("No data found for any model.")
        return None

    # ---- align all pivots to same action columns ----
    ordered = [c for c in ACTION_ORDER if c in all_actions_global]
    rest = sorted([c for c in all_actions_global if c not in ordered])
    all_actions = ordered + rest

    for model_name in list(pivots.keys()):
        pivot = pivots[model_name].reindex(columns=all_actions, fill_value=0)
        if normalize:
            pivot = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100.0
        pivots[model_name] = pivot

    # ---- plotting ----
    n_models = len(pivots)
    n_agents = len(blue_agents)
    n_bars = len(attackers) * n_agents

    fig_w = max(18, 0.62 * n_bars)
    fig_h = 2.35 * n_models + 0.7
    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(fig_w, fig_h),
        sharex=True,
        squeeze=False
    )

    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for row_idx, (model_name, pivot) in enumerate(pivots.items()):
        ax = axes[row_idx]
        x = np.arange(n_bars)
        bottom = np.zeros(n_bars, dtype=float)

        for action in pivot.columns:
            vals = pivot[action].to_numpy(dtype=float)
            bars = ax.bar(
                x,
                vals,
                bottom=bottom,
                label=action,
                color=stable_color(str(action)),
                width=0.82,
                edgecolor="none",
            )
            bottom += vals

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ylabel = "%" if normalize else "Count"
        ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
        ax.set_title(model_name, fontsize=15, fontweight="bold", loc="left")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        ax.set_ylim(0, 100)

        ax.tick_params(
            axis="y",
            labelsize=13
        )

        # separators between attackers
        for i in range(1, len(attackers)):
            ax.axvline(i * n_agents - 0.5, color="black", linewidth=2.0, alpha=0.35)

        # attacker labels above each group
        group_centers = [i * n_agents + (n_agents - 1) / 2 for i in range(len(attackers))]
        ymax = ax.get_ylim()[1]
        for center, attacker in zip(group_centers, attackers):
            ax.text(
                center,
                ymax * 1.02,
                attacker,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold"
            )

    # x tick labels only on bottom subplot
    xticklabels = [ag.split("_")[-1] for _att in attackers for ag in blue_agents]
    axes[-1].set_xticks(np.arange(n_bars))
    axes[-1].set_xticklabels(
        xticklabels,
        fontsize=14,
        fontweight="bold",
        rotation=0,
        ha="center"
    )

    axes[-1].margins(x=0.01)  # tighter fit
    axes[-1].set_xlabel("Blue Agent", fontsize=14, fontweight="bold")

    # remove x tick labels on upper plots
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    # one shared legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        fontsize=15,
        frameon=True,
        ncol=len(legend_labels),
        handlelength=1.6,
        columnspacing=1.0,
        borderpad=0.4,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    suffix = "pct" if normalize else "count"
    out_path = out_dir / f"action_freq_multi_model_{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # save tables too
    for model_name, pivot in pivots.items():
        pivot.to_csv(out_dir / f"{model_name}_action_freq_{suffix}.csv")

    return {"figure": out_path}
# -----------------------
# Main entry point
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="Plots/ActionPlots")
    ap.add_argument("--normalize", type=bool, default=True)
    ap.add_argument("--top_k", type=int, default=25)
    ap.add_argument(
        "--attackers",
        type=str,
        default="deception_aware,discovery,fsm_default,impact_rush,lateral_spread,stealth_pivot",
        help="Comma-separated attacker list"
    )
    args = ap.parse_args()

    top_k = None if args.top_k == -1 else args.top_k
    attackers = [a.strip() for a in args.attackers.split(",") if a.strip()]


    model_dirs = {
        "Heuristic": RESULTS_DIR / "Heuristic",
        "SimpleGNN": RESULTS_DIR / "SimpleGNN",
    }

    res = plot_action_frequencies_multiple_models(
        model_dirs=model_dirs,
        out_dir=Path(args.out_dir),
        attackers=attackers,
        blue_agents=DEFAULT_BLUE_AGENTS,
        normalize=args.normalize,
        top_k=top_k,
    )

    if res is None:
        print("No actions found.")
    else:
        print("Saved:", res)


if __name__ == "__main__":
    main()
