# Leave-One-Attack-Out (LOAO) evaluator using families_order from configs/dataset.yaml
# Usage: python -m scripts.run_loao
import os
import datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils.data_prep import load_dataset, load_config, encode_labels
from src.utils.metrics import write_metrics_csv

RESULTS_CSV = "results/metrics.csv"


def make_Xy(df, label_map):
    feats = ["Length", "Source Port", "Destination Port", "FunctionCodeNum"]
    X = df[feats].astype(float).values
    y = encode_labels(df, label_map).values
    return X, y


def loao_avg_recall(df, family_name, model):
    # train on all families except the held-out; test only on held-out family
    train_df = df[df["Attack Family"] != family_name]
    test_df = df[df["Attack Family"] == family_name]

    # Only evaluate recall on ATTACKs in the held-out family
    # If there are no attack labels in that slice, skip (return np.nan)
    if not ((test_df["Label"] == "Attack").any()):
        return np.nan

    Xtr, ytr = make_Xy(train_df, label_map)
    Xte, yte = make_Xy(test_df, label_map)

    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)

    # Recall = TP / (TP + FN) on attack class
    tp = ((yte == 1) & (ypred == 1)).sum()
    fn = ((yte == 1) & (ypred == 0)).sum()
    rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    return rec


if __name__ == "__main__":
    cfg = load_config("configs/dataset.yaml")
    families = cfg["families_order"]
    df = load_dataset("configs/dataset.yaml")
    label_map = cfg["label_encoding"]

    footer = {
        "dataset": Path(cfg["dataset_path"]).name,
        "commit": os.getenv("GIT_COMMIT", "local"),
        "seed": 42,
        "date": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    # Models (same as baselines)
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, n_jobs=-1, random_state=42
    )

    for name, model in [("LogReg", logreg), ("RandomForest", rf)]:
        recalls = []
        for fam in families:
            r = loao_avg_recall(df, fam, model)
            if not np.isnan(r):
                recalls.append(r)
        avg_rec = float(np.mean(recalls)) if len(recalls) else None
        write_metrics_csv(
            RESULTS_CSV,
            {
                "model": f"{name}-LOAO",
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "pr_auc": None,
                "avg_loao_recall": avg_rec,
                "notes": f"LOAO families={families}",
            },
            footer_meta=footer,
        )

    print(f"[OK] LOAO complete. Wrote metrics to {RESULTS_CSV}")
