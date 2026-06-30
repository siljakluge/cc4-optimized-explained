import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
from pathlib import Path


def infer_feature_group(feature_name: str) -> str:
    name = str(feature_name).lower()
    if "prev_action" in name or name.startswith("action") or "_action" in name:
        return "previous_action"
    if "success" in name or "fail" in name:
        return "action_outcome"
    if "comprom" in name or "session" in name or "privilege" in name or "root" in name or "user" in name:
        return "host_compromise"
    if "service" in name or "process" in name or "proc" in name or "conn" in name:
        return "service_process"
    if "subnet" in name or "zone" in name or "network" in name or "reachable" in name:
        return "network_reachability"
    if "decoy" in name or "decept" in name:
        return "deception"
    if "step" in name or "time" in name or "phase" in name:
        return "mission_phase"
    if "host" in name:
        return "host_state"
    return "other"


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._@" else "_" for c in str(name))


def _save_bar(df: pd.DataFrame, value_col: str, label_col: str, title: str, out_path: Path, top_n: int = 25):
    if df.empty:
        return
    plot_df = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    height = max(4.0, 0.28 * len(plot_df) + 1.5)
    plt.figure(figsize=(9, height))
    plt.barh(plot_df[label_col].astype(str), plot_df[value_col].astype(float))
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _write_shap_tables(
    out_dir: str,
    classes,
    feature_names: list[str],
    shap_values,
    x_values,
    max_display: int,
) -> dict:
    out_path = Path(out_dir)
    rows = []
    group_rows = []

    for ci, cname in enumerate(classes):
        values = shap_values[:, :, ci].values
        mean_abs = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)
        class_name = str(cname)

        class_rows = []
        for feature, abs_val, signed_val in zip(feature_names, mean_abs, mean_signed):
            group = infer_feature_group(feature)
            row = {
                "action_class": class_name,
                "feature": feature,
                "feature_group": group,
                "mean_abs_shap": float(abs_val),
                "mean_signed_shap": float(signed_val),
            }
            rows.append(row)
            class_rows.append(row)

        class_df = pd.DataFrame(class_rows).sort_values("mean_abs_shap", ascending=False)
        class_dir = out_path / f"class_{ci}_{_sanitize(class_name)}"
        class_dir.mkdir(parents=True, exist_ok=True)
        class_df.to_csv(class_dir / "mean_abs_shap_by_feature.csv", index=False)
        _save_bar(
            class_df,
            "mean_abs_shap",
            "feature",
            f"Top SHAP features for action: {class_name}",
            class_dir / "top_features_bar.png",
            top_n=max_display,
        )

        group_df = (
            class_df.groupby("feature_group", as_index=False)
            .agg(mean_abs_shap=("mean_abs_shap", "sum"), mean_signed_shap=("mean_signed_shap", "sum"))
            .sort_values("mean_abs_shap", ascending=False)
        )
        group_df.insert(0, "action_class", class_name)
        group_rows.extend(group_df.to_dict("records"))
        group_df.to_csv(class_dir / "mean_abs_shap_by_feature_group.csv", index=False)
        _save_bar(
            group_df,
            "mean_abs_shap",
            "feature_group",
            f"Top SHAP feature groups for action: {class_name}",
            class_dir / "top_feature_groups_bar.png",
            top_n=max_display,
        )

    feature_df = pd.DataFrame(rows)
    group_df = pd.DataFrame(group_rows)

    feature_csv = out_path / "mean_abs_shap_by_action_feature.csv"
    group_csv = out_path / "mean_abs_shap_by_action_feature_group.csv"
    global_feature_csv = out_path / "mean_abs_shap_by_feature.csv"
    global_group_csv = out_path / "mean_abs_shap_by_feature_group.csv"

    feature_df.to_csv(feature_csv, index=False)
    group_df.to_csv(group_csv, index=False)

    global_feature_df = (
        feature_df.groupby(["feature", "feature_group"], as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"), mean_signed_shap=("mean_signed_shap", "mean"))
        .sort_values("mean_abs_shap", ascending=False)
    )
    global_group_df = (
        group_df.groupby("feature_group", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"), mean_signed_shap=("mean_signed_shap", "mean"))
        .sort_values("mean_abs_shap", ascending=False)
    )

    global_feature_df.to_csv(global_feature_csv, index=False)
    global_group_df.to_csv(global_group_csv, index=False)
    _save_bar(
        global_feature_df,
        "mean_abs_shap",
        "feature",
        "Top SHAP features across explained actions",
        out_path / "top_features_global_bar.png",
        top_n=max_display,
    )
    _save_bar(
        global_group_df,
        "mean_abs_shap",
        "feature_group",
        "Top SHAP feature groups across explained actions",
        out_path / "top_feature_groups_global_bar.png",
        top_n=max_display,
    )

    return {
        "feature_csv": str(feature_csv),
        "group_csv": str(group_csv),
        "global_feature_csv": str(global_feature_csv),
        "global_group_csv": str(global_group_csv),
        "n_explained_samples": int(len(x_values)),
        "n_features": int(len(feature_names)),
        "classes": [str(c) for c in classes],
    }


def write_profile_shap_comparison(run_dir: str | Path, profiles: list[str], out_dir: str | Path | None = None) -> list[Path]:
    run_path = Path(run_dir)
    out_path = Path(out_dir) if out_dir is not None else run_path / "SHAPProfileComparison"
    out_path.mkdir(parents=True, exist_ok=True)

    feature_frames = []
    group_frames = []
    for profile in profiles:
        shap_dir = run_path / profile / "SHAPAnalysis"
        feature_csv = shap_dir / "mean_abs_shap_by_feature.csv"
        group_csv = shap_dir / "mean_abs_shap_by_feature_group.csv"
        if feature_csv.exists():
            df = pd.read_csv(feature_csv)
            df.insert(0, "profile", profile)
            feature_frames.append(df)
        if group_csv.exists():
            df = pd.read_csv(group_csv)
            df.insert(0, "profile", profile)
            group_frames.append(df)

    generated: list[Path] = []
    if feature_frames:
        all_features = pd.concat(feature_frames, ignore_index=True)
        feature_out = out_path / "mean_abs_shap_by_profile_feature.csv"
        all_features.to_csv(feature_out, index=False)
        generated.append(feature_out)

        baseline = "fsm_default" if "fsm_default" in set(all_features["profile"]) else sorted(all_features["profile"].unique())[0]
        base = all_features[all_features["profile"] == baseline][["feature", "mean_abs_shap"]].rename(
            columns={"mean_abs_shap": "baseline_mean_abs_shap"}
        )
        deltas = all_features.merge(base, on="feature", how="left")
        deltas["delta_vs_" + baseline] = deltas["mean_abs_shap"] - deltas["baseline_mean_abs_shap"].fillna(0.0)
        delta_out = out_path / f"profile_delta_shap_vs_{baseline}.csv"
        deltas.sort_values(["profile", "delta_vs_" + baseline], ascending=[True, False]).to_csv(delta_out, index=False)
        generated.append(delta_out)

    if group_frames:
        all_groups = pd.concat(group_frames, ignore_index=True)
        group_out = out_path / "mean_abs_shap_by_profile_feature_group.csv"
        all_groups.to_csv(group_out, index=False)
        generated.append(group_out)

        pivot = all_groups.pivot_table(
            index="profile",
            columns="feature_group",
            values="mean_abs_shap",
            aggfunc="mean",
            fill_value=0.0,
        )
        heatmap_csv = out_path / "feature_group_heatmap.csv"
        pivot.to_csv(heatmap_csv)
        generated.append(heatmap_csv)

        plt.figure(figsize=(max(7, 0.8 * len(pivot.columns)), max(3.5, 0.45 * len(pivot.index) + 1.5)))
        im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="mean |SHAP|")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title("SHAP feature-group importance by red profile")
        plt.tight_layout()
        heatmap_png = out_path / "feature_group_heatmap.png"
        plt.savefig(heatmap_png, dpi=150, bbox_inches="tight")
        plt.close()
        generated.append(heatmap_png)

    return generated

def run_shap(
    df: pd.DataFrame,
    max_classes: int | None = None,
    out_dir: str = None,
    background_samples: int = 200,
    explain_samples: int = 500,
    random_state: int = 0,
):
    if out_dir is None:
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)
    # ---- Choose columns ----
    y = df["y"].astype(str)

    # Optionally restrict to top-K action classes to keep SHAP readable
    if max_classes is not None:
        top = y.value_counts().head(max_classes).index
        mask = y.isin(top)
        df = df.loc[mask].copy()
        y = df["y"].astype(str)

    # Drop non-features
    drop_cols = {"y", "step", "agent"}
    X = df[[c for c in df.columns if c not in drop_cols]].copy()

    # Identify categorical/numeric
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    clf = Pipeline([("pre", pre), ("model", model)])

    class_counts = y.value_counts()
    stratify = y if not class_counts.empty and class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=stratify
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred, digits=3))

    # Get feature names after one-hot encoding
    ohe = clf.named_steps["pre"].named_transformers_["cat"]
    cat_feature_names = []
    if len(cat_cols) > 0:
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = cat_feature_names + num_cols

    import scipy.sparse as sp

    def to_dense(X):
        return X.toarray() if sp.issparse(X) else np.asarray(X)

    # after training your pipeline `clf`...
    X_test_t = to_dense(clf.named_steps["pre"].transform(X_test))
    rf = clf.named_steps["model"]
    if len(X_test_t) == 0:
        raise ValueError("Cannot run SHAP on an empty transformed test set.")

    bg_n = min(max(1, background_samples), len(X_test_t))
    ex_n = min(max(1, explain_samples), len(X_test_t))
    background = shap.sample(X_test_t, bg_n, random_state=random_state)
    X_explain = shap.sample(X_test_t, ex_n, random_state=random_state + 1)
    explainer = shap.Explainer(rf, background)
    sv = explainer(X_explain)

    # save per-class summary plots
    max_disp = min(25, len(feature_names))
    summary = _write_shap_tables(out_dir, rf.classes_, feature_names, sv, X_explain, max_disp)
    summary.update({
        "n_background_samples": int(bg_n),
        "n_explain_samples": int(ex_n),
        "random_state": int(random_state),
    })
    for ci, cname in enumerate(rf.classes_):
        plt.figure()
        shap.summary_plot(
            sv[:, :, ci].values,  # (n_samples, n_features)
            X_explain,
            feature_names=feature_names,
            show=False,
            max_display=max_disp,
        )
        plt.savefig(os.path.join(out_dir, f"shap_summary_class_{_sanitize(cname)}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    return clf, explainer, sv, feature_names, summary
