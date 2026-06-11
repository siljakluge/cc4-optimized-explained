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
import csv

def _matrix(rows: List[Dict[str, float]]):
    keys = sorted({k for r in rows for k in r.keys()})
    X = np.zeros((len(rows), len(keys)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            X[i, j] = float(r.get(k, 0.0))
    return X, keys

def train_surrogate_and_shap(policy_rows: List[Tuple[Dict[str, float], str]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    feats = [r[0] for r in policy_rows]
    labels = [r[1] for r in policy_rows]
    if len(feats) < 200:
        warnings.warn("Few samples (<200); SHAP may be noisy.")

    X, feature_names = _matrix(feats)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)

    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06)
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    bacc = balanced_accuracy_score(yte, ypred)
    rep = classification_report(yte, ypred, target_names=list(le.classes_), output_dict=True)

    with open(os.path.join(out_dir, "surrogate_report.json"), "w") as f:
        json.dump({
            "n_samples": int(len(y)),
            "classes": list(le.classes_),
            "test_accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "report": rep
        }, f, indent=2)

    # --- SHAP with model-agnostic Explainer on predict_proba ---
    rng = np.random.default_rng(42)
    if len(Xtr) > 0:
        bg_idx = rng.choice(len(Xtr), size=min(200, len(Xtr)), replace=False)
        bg_X = Xtr[bg_idx]
    else:
        bg_X = X[: min(200, len(X))]

    # callable that returns class probabilities
    explainer = shap.Explainer(clf.predict_proba, bg_X)

    # choose samples to explain
    ex_idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
    X_explain = X[ex_idx]
    y_explain = y[ex_idx]

    shap_values = explainer(X_explain)  # PermutationExplainer, multiclass
    # shap_values shape semantics: (n_samples, n_features, n_classes)

    def sanitize(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-._@" else "_" for c in name)

    # Per-class outputs
    for ci, cname in enumerate(le.classes_):
        class_dir = os.path.join(out_dir, f"class_{ci}_{sanitize(cname)}")
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
        shap.plots.bar(sv_plot, max_display=25, show=False)
        plt.title(f"Global SHAP – top 25 features (class: {cname})")
        plt.tight_layout()
        plt.savefig(os.path.join(class_dir, "shap_global_bar.png"), dpi=150)
        plt.close()

        # ---- Beeswarm for this class ----
        plt.figure()
        shap.plots.beeswarm(sv_plot, max_display=25, show=False)
        plt.title(f"SHAP Beeswarm – top 25 (class: {cname})")
        plt.tight_layout()
        plt.savefig(os.path.join(class_dir, "shap_beeswarm.png"), dpi=150)
        plt.close()


    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "classes": list(le.classes_)
    }
