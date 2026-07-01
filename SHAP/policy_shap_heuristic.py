# policy_shap_heuristic.py
import os, json, warnings
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt
import pandas as pd

from SHAP.shap_gnn import infer_feature_group, _save_bar

def _matrix(rows: List[Dict[str, float]]):
    df = pd.DataFrame(rows)
    if df.empty:
        return np.zeros((0, 0), dtype=np.float32), []

    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().all():
            df[col] = numeric.astype(np.float32)

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    df = df.fillna(0.0).reindex(sorted(df.columns), axis=1)
    return df.to_numpy(dtype=np.float32), list(df.columns)

def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._@" else "_" for c in str(name))

def _write_comparable_shap_tables(
    out_dir: str,
    classes,
    feature_names: list[str],
    shap_values,
    max_display: int,
) -> dict:
    out_path = os.path.abspath(out_dir)
    rows = []
    group_rows = []

    for ci, cname in enumerate(classes):
        values = shap_values[..., int(ci)].values
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
        class_dir = os.path.join(out_path, f"class_{ci}_{_sanitize(class_name)}")
        os.makedirs(class_dir, exist_ok=True)
        class_df.to_csv(os.path.join(class_dir, "mean_abs_shap_by_feature.csv"), index=False)
        _save_bar(
            class_df,
            "mean_abs_shap",
            "feature",
            f"Top SHAP features for action: {class_name}",
            os.path.join(class_dir, "top_features_bar.png"),
            top_n=max_display,
        )

        group_df = (
            class_df.groupby("feature_group", as_index=False)
            .agg(mean_abs_shap=("mean_abs_shap", "sum"), mean_signed_shap=("mean_signed_shap", "sum"))
            .sort_values("mean_abs_shap", ascending=False)
        )
        group_df.insert(0, "action_class", class_name)
        group_rows.extend(group_df.to_dict("records"))
        group_df.to_csv(os.path.join(class_dir, "mean_abs_shap_by_feature_group.csv"), index=False)
        _save_bar(
            group_df,
            "mean_abs_shap",
            "feature_group",
            f"Top SHAP feature groups for action: {class_name}",
            os.path.join(class_dir, "top_feature_groups_bar.png"),
            top_n=max_display,
        )

    feature_df = pd.DataFrame(rows)
    group_df = pd.DataFrame(group_rows)

    feature_csv = os.path.join(out_path, "mean_abs_shap_by_action_feature.csv")
    group_csv = os.path.join(out_path, "mean_abs_shap_by_action_feature_group.csv")
    global_feature_csv = os.path.join(out_path, "mean_abs_shap_by_feature.csv")
    global_group_csv = os.path.join(out_path, "mean_abs_shap_by_feature_group.csv")

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
        os.path.join(out_path, "top_features_global_bar.png"),
        top_n=max_display,
    )
    _save_bar(
        global_group_df,
        "mean_abs_shap",
        "feature_group",
        "Top SHAP feature groups across explained actions",
        os.path.join(out_path, "top_feature_groups_global_bar.png"),
        top_n=max_display,
    )

    return {
        "feature_csv": feature_csv,
        "group_csv": group_csv,
        "global_feature_csv": global_feature_csv,
        "global_group_csv": global_group_csv,
        "n_features": int(len(feature_names)),
        "classes": [str(c) for c in classes],
    }

def train_surrogate_and_shap(
    policy_rows: List[Tuple[Dict[str, float], str]],
    out_dir: str,
    background_samples: int = 200,
    explain_samples: int = 500,
    random_state: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    feats = [r[0] for r in policy_rows]
    labels = [r[1] for r in policy_rows]
    if len(feats) < 200:
        warnings.warn("Few samples (<200); SHAP may be noisy.")

    X, feature_names = _matrix(feats)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    class_counts = pd.Series(y).value_counts()
    stratify = y if not class_counts.empty and class_counts.min() >= 2 else None

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06)
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    bacc = balanced_accuracy_score(yte, ypred)
    rep = classification_report(
        yte,
        ypred,
        labels=np.arange(len(le.classes_)),
        target_names=list(le.classes_),
        output_dict=True,
        zero_division=0,
    )

    surrogate_report = {
        "n_samples": int(len(y)),
        "classes": [str(c) for c in le.classes_],
        "test_accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "report": rep,
    }
    with open(os.path.join(out_dir, "surrogate_report.json"), "w") as f:
        json.dump(surrogate_report, f, indent=2)

    # --- SHAP with model-agnostic Explainer on predict_proba ---
    rng = np.random.default_rng(random_state)
    if len(Xtr) > 0:
        bg_idx = rng.choice(len(Xtr), size=min(background_samples, len(Xtr)), replace=False)
        bg_X = Xtr[bg_idx]
    else:
        bg_X = X[: min(background_samples, len(X))]

    # callable that returns class probabilities
    explainer = shap.Explainer(clf.predict_proba, bg_X)

    # choose samples to explain
    ex_idx = rng.choice(len(X), size=min(explain_samples, len(X)), replace=False)
    X_explain = X[ex_idx]
    y_explain = y[ex_idx]

    shap_values = explainer(X_explain)  # PermutationExplainer, multiclass
    # shap_values shape semantics: (n_samples, n_features, n_classes)

    max_disp = min(25, len(feature_names))
    table_summary = _write_comparable_shap_tables(
        out_dir,
        le.classes_,
        feature_names,
        shap_values,
        max_display=max_disp,
    )

    # Per-class outputs
    for ci, cname in enumerate(le.classes_):
        class_dir = os.path.join(out_dir, f"class_{ci}_{_sanitize(cname)}")
        os.makedirs(class_dir, exist_ok=True)

        # Slice the multiclass Explanation down to this class
        sv_ci = shap_values[..., int(ci)]  # (n_samples, n_features)

        # Rewrap with explicit metadata for reliable plotting
        sv_plot = shap.Explanation(
            values=sv_ci.values,  # (n_samples, n_features)
            base_values=None,
            data=X_explain,  # (n_samples, n_features)
            feature_names=feature_names
        )

        # ---- Global bar (mean |SHAP|) for this class ----
        plt.figure()
        shap.plots.bar(sv_plot, max_display=max_disp, show=False)
        plt.title(f"Global SHAP – top 25 features (class: {cname})")
        plt.tight_layout()
        plt.savefig(os.path.join(class_dir, "shap_global_bar.png"), dpi=150)
        plt.close()

        # ---- Beeswarm for this class ----
        plt.figure()
        shap.plots.beeswarm(sv_plot, max_display=max_disp, show=False)
        plt.title(f"SHAP Beeswarm – top 25 (class: {cname})")
        plt.tight_layout()
        plt.savefig(os.path.join(class_dir, "shap_beeswarm.png"), dpi=150)
        plt.close()


    summary = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "classes": [str(c) for c in le.classes_],
        "n_samples": int(len(y)),
        "n_background_samples": int(len(bg_X)),
        "n_explain_samples": int(len(X_explain)),
        "random_state": int(random_state),
    }
    summary.update(table_summary)
    with open(os.path.join(out_dir, "shap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
