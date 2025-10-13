# ML baselines runner (LogReg, RF, optional IF) with EIP-standard outputs.
# Usage: python -m scripts.run_baselines
import os
import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils.data_prep import load_dataset, load_config, encode_labels
from src.utils.metrics import compute_metrics, write_metrics_csv
from sklearn.ensemble import IsolationForest

# import numpy as np -- to be added later


RESULTS_CSV = "results/metrics.csv"


def make_Xy(df, label_map):
    # minimal, stable numeric features per our schema
    feats = ["Length", "Source Port", "Destination Port", "FunctionCodeNum"]
    X = df[feats].astype(float).values
    y = encode_labels(df, label_map).values
    return X, y


def main():
    cfg = load_config("configs/dataset.yaml")
    df = load_dataset("configs/dataset.yaml")
    label_map = cfg["label_encoding"]

    X, y = make_Xy(df, label_map)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    footer = {
        "dataset": Path(cfg["dataset_path"]).name,
        "commit": os.getenv("GIT_COMMIT", "local"),
        "seed": 42,
        "date": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    # 1) Logistic Regression
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, n_jobs=None)),
        ]
    )
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    try:
        y_score = logreg.predict_proba(X_test)[:, 1]
    except Exception:
        y_score = None
    m = compute_metrics(y_test, y_pred, y_score)
    write_metrics_csv(
        RESULTS_CSV,
        {
            "model": "LogReg",
            **m,
            "avg_loao_recall": None,
            "notes": "Baseline; features=[Length,SPort,DPort,FCNum]",
        },
        footer_meta=footer,
    )

    # 2) Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    try:
        y_score = rf.predict_proba(X_test)[:, 1]
    except Exception:
        y_score = None
    m = compute_metrics(y_test, y_pred, y_score)
    write_metrics_csv(
        RESULTS_CSV,
        {
            "model": "RandomForest",
            **m,
            "avg_loao_recall": None,
            "notes": "Baseline; features=[Length,SPort,DPort,FCNum]",
        },
        footer_meta=footer,
    )

    # 3) Isolation Forest (unsupervised)
    # It needs decision_function (higher=more normal). Score->attack=low score.
    iforest = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    iforest.fit(X_train)
    # attack if anomaly (predict=-1)
    y_pred = (iforest.predict(X_test) == -1).astype(int)
    y_score = None
    m = compute_metrics(y_test, y_pred, y_score)
    write_metrics_csv(
        RESULTS_CSV,
        {
            "model": "IsolationForest",
            **m,
            "avg_loao_recall": None,
            "notes": "Unsupervised; threshold=default",
        },
        footer_meta=footer,
    )

    print(f"[OK] Baselines complete. Wrote metrics to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
