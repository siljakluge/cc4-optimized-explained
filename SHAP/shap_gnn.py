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

def run_shap(df: pd.DataFrame, max_classes: int | None = None, out_dir: str = None):
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
        random_state=0,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    clf = Pipeline([("pre", pre), ("model", model)])

    class_counts = y.value_counts()
    stratify = y if not class_counts.empty and class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=stratify
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

    import shap
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import scipy.sparse as sp

    def to_dense(X):
        return X.toarray() if sp.issparse(X) else np.asarray(X)

    # after training your pipeline `clf`...
    X_test_t = to_dense(clf.named_steps["pre"].transform(X_test))
    rf = clf.named_steps["model"]
    sample = shap.sample(X_test_t, 100)
    explainer = shap.Explainer(rf, sample)  # background = X_test_t (or a sample)
    sv = explainer(X_test_t)  # shap.Explanation

    # save per-class summary plots
    max_disp = len(feature_names)
    for ci, cname in enumerate(rf.classes_):
        plt.figure()
        shap.summary_plot(
            sv[:, :, ci].values,  # (n_samples, n_features)
            X_test_t,
            feature_names=feature_names,
            show=False,
            max_display=max_disp,
        )
        plt.savefig(os.path.join(out_dir, f"shap_summary_class_{cname}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    return clf, explainer, sv, feature_names
