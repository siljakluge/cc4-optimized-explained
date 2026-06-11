#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scienceplots  # noqa: F401

# ----------------------------
# Inputs
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # geht von plotting/ eine Ebene hoch
CSV_DIR = BASE_DIR / "exported_csv"

RUNS = [
    ("SimpleGNN", CSV_DIR / "contractoractive.csv"),
]

OUTDIR = Path("Plots/trainingCurves")
OUTDIR.mkdir(parents=False, exist_ok=True)

# ----------------------------
# Smoothing
# ----------------------------
def ema(series: pd.Series, alpha: float) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()

EMA_ALPHA = 0.05  # lower => smoother

# ----------------------------
# Load & validate
# ----------------------------
dfs = []
for label, path in RUNS:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    df = pd.read_csv(path)
    required = ["step_e", "avg_reward", "avg_loss"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}. Has: {list(df.columns)}")

    df = df.sort_values("step_e").copy()
    df["Scenario"] = label
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
scenario_order = [r[0] for r in RUNS]

# ----------------------------
# SciencePlots styling
# ----------------------------
# Good defaults for papers. If you want LaTeX-rendered text, add "tex" to the list.
STYLE = ["science", "ieee", "grid"]  # try also: ["science", "nature", "grid"]

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": True,   # ieee style often uses framed legends; set False if you prefer
    "legend.framealpha": 0.9,
    "legend.borderpad": 0.3,
    "lines.linewidth": 1.8,
})

# ----------------------------
# Plot helper
# ----------------------------
def plot_metric(
    y_col: str,
    ylabel: str,
    title: str,
    outfile: Path,
    yscale: str = "linear",
    linthresh: float = 1e-2,
):
    with plt.style.context(STYLE):
        plt.rcParams["text.usetex"] = False
        fig, ax = plt.subplots(figsize=(6.0, 3.6))

        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
        color_map = {}
        if prop_cycle:
            for i, scen in enumerate(scenario_order):
                color_map[scen] = prop_cycle[i % len(prop_cycle)]

        for scenario in scenario_order:
            df = data[data["Scenario"] == scenario]
            x = df["step_e"].to_numpy()
            y = df[y_col].to_numpy()
            y_s = ema(pd.Series(y), EMA_ALPHA).to_numpy()

            c = color_map.get(scenario, None)

            ax.plot(x, y, alpha=0.25, linewidth=1.0, color=c)
            ax.plot(x, y_s, label=scenario, linewidth=2.2, color=c)

        ax.set_xlabel("Training Episodes")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if yscale == "symlog":
            ax.set_yscale("symlog", linthresh=linthresh)

        ax.legend(loc="best")
        fig.savefig(outfile)
        plt.close(fig)

# ----------------------------
# Reward (linear)
# ----------------------------
plot_metric(
    y_col="avg_reward",
    ylabel="Average Episode Reward",
    title="Training Performance",
    outfile=OUTDIR / "reward_over_training.pdf",
    yscale="linear",
)

# ----------------------------
# Loss (linear)
# ----------------------------
plot_metric(
    y_col="avg_loss",
    ylabel="Mean PPO Training Loss",
    title="Optimization Stability",
    outfile=OUTDIR / "loss_over_training_linear.pdf",
    yscale="linear",
)

# ----------------------------
# Loss (symlog)
# ----------------------------
plot_metric(
    y_col="avg_loss",
    ylabel="Mean PPO Training Loss",
    title="Optimization Stability",
    outfile=OUTDIR / "loss_over_training_symlog.pdf",
    yscale="symlog",
    linthresh=1e-2,
)

print(f"Wrote figures to: {OUTDIR.resolve()}")

# ----------------------------
# Style (IEEE / two-column)
# ----------------------------
STYLE = ["science", "ieee"]

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

FIGSIZE_ONECOL_SIDE = (3.4, 1.6)  # width x height (inches)

# ----------------------------
# EMA smoothing
# ----------------------------
def ema(series: pd.Series, alpha: float) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()

EMA_ALPHA = 0.05

# ----------------------------
# Combined side-by-side figure
# ----------------------------
with plt.style.context(STYLE):
    plt.rcParams["text.usetex"] = False
    fig, (ax_r, ax_l) = plt.subplots(
        1, 2,
        figsize=FIGSIZE_ONECOL_SIDE,
        sharex=True,
        gridspec_kw={"wspace": 0.28}
    )

    # ---------- Reward ----------
    for scenario in scenario_order:
        df = data[data["Scenario"] == scenario]
        x = df["step_e"]
        y = df["avg_reward"]
        y_s = ema(y, EMA_ALPHA)

        ax_r.plot(x, y, alpha=0.25)
        ax_r.plot(x, y_s, label=scenario)

    ax_r.set_xlabel("Episodes")
    ax_r.set_ylabel("Avg. Reward")

    # --- FORCE scientific notation on y only (robust) ---
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))  # always use scientific notation
    fmt.set_useOffset(False)  # optional: avoids "+1e3" style offsets
    ax_r.yaxis.set_major_formatter(fmt)
    ax_r.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

    # Make sure it re-renders ticks with the formatter
    ax_r.relim()
    ax_r.autoscale_view()
    ax_r.figure.canvas.draw_idle()

    ax_r.legend(loc="best", frameon=False)

    # ---------- Loss (linear) ----------
    for scenario in scenario_order:
        df = data[data["Scenario"] == scenario]
        x = df["step_e"]
        y = df["avg_loss"]
        y_s = ema(y, EMA_ALPHA)

        ax_l.plot(x, y, alpha=0.25)
        ax_l.plot(x, y_s)

    ax_l.set_xlabel("Episodes")
    ax_l.set_ylabel("Mean PPO Loss")
#    ax_l.set_title("Loss")

    fig.tight_layout(pad=0.35)
    fig.savefig(OUTDIR / "training_curves.pdf")
    plt.close(fig)

