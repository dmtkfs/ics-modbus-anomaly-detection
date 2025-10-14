# ML baselines runner (LogReg, RF, optional IF) with EIP-standard outputs.
# Usage: python -m scripts.run_baselines

import os
import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.utils.class_weight import compute_sample_weight

from src.utils.data_prep import (
    load_config,
    verify_checksum,
    sha256_file,
    check_schema,
)
from src.utils.ml_data_prep import load_ml_config, make_supervised_xy
from src.utils.metrics import (
    compute_metrics,
    write_metrics_csv,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)

RESULTS_CSV = "results/metrics.csv"
RESULTS_DIR = "results"


def load_dataset_light(ds_yaml_path: str, ml_cfg: dict) -> pd.DataFrame:
    """
    EIP-compliant light loader:
    - Verifies checksum
    - Validates schema from header only (nrows=0)
    - Reads only ML-needed columns with explicit dtypes
    """
    ds_cfg = load_config(ds_yaml_path)
    csv_path = ds_cfg["dataset_path"]

    # 1) Checksum enforcement
    expected = ds_cfg.get("sha256")
    if expected:
        if not verify_checksum(csv_path, expected):
            actual = sha256_file(csv_path)
            raise ValueError(
                f"Dataset checksum mismatch.\nExpected: {expected}\nActual:   {actual}"
            )

    # 2) Schema check from header only (very cheap)
    header_df = pd.read_csv(csv_path, nrows=0)
    check_schema(header_df)

    # 3) Read only needed columns with explicit dtypes (prevents heavy inference)
    needed_cols = list(ml_cfg["features"]) + [
        ml_cfg["label_column"],
        ml_cfg["attack_family_column"],
    ]

    # dtype choices: fit into smaller ints where safe; keeping labels as strings (mapped later)
    dtype_map = {
        "Length": "int32",
        "Source Port": "int32",
        "Destination Port": "int32",
        "FunctionCodeNum": "int16",
        ml_cfg["label_column"]: "string",
        ml_cfg["attack_family_column"]: "string",
    }

    df = pd.read_csv(
        csv_path,
        usecols=needed_cols,
        dtype=dtype_map,
        engine="c",
        low_memory=True,
        memory_map=True,
    )
    return df


def lr_note(max_iter, solver, weighted=True, scaler="StandardScaler"):
    return (
        f"Baseline; features from ml.yaml; "
        f"LR(solver={solver}, max_iter={max_iter}, weighted={weighted}, scaler={scaler})"
    )


def rf_note_from_params(p):
    return (
        "Baseline; features from ml.yaml; weighted; RF("
        f"n_estimators={p['n_estimators']}, "
        f"max_samples={p['max_samples']}, "
        f"max_depth={p['max_depth']}, "
        f"min_samples_leaf={p['min_samples_leaf']}, "
        f"n_jobs={p['n_jobs']}, bootstrap={p['bootstrap']})"
    )


def main():
    # ---- Load configs
    ds_yaml = "configs/dataset.yaml"
    ml_cfg = load_ml_config("configs/ml.yaml")
    seed = int(ml_cfg.get("random_seed", 42))
    test_size = float(ml_cfg.get("test_size", 0.2))

    # ---- Light, memory-safe dataset load
    df = load_dataset_light(ds_yaml, ml_cfg)

    # ---- X, y using ml.yaml (features + label_map)
    X_df, y_ser = make_supervised_xy(df, ml_cfg)
    X = X_df.values
    y = y_ser.values

    # ---- Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    ds_cfg = load_config(ds_yaml)
    footer = {
        "dataset": Path(ds_cfg["dataset_path"]).name,
        "commit": os.getenv("GIT_COMMIT", "local"),
        "seed": seed,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }

    # ----- sample weights for imbalance
    sample_w = compute_sample_weight(class_weight="balanced", y=y_train)

    # =========================
    # 1) Logistic Regression
    # =========================
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                    n_jobs=-1,
                    solver="saga",
                    class_weight=None,  # use sample_weight
                ),
            ),
        ]
    )
    logreg.fit(X_train, y_train, clf__sample_weight=sample_w)
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
            "notes": lr_note(max_iter=1000, solver="saga", weighted=True),
        },
        footer_meta=footer,
    )
    plot_confusion_matrix(
        y_test, y_pred, "CM — Logistic Regression", f"{RESULTS_DIR}/cm_logreg.png"
    )
    if y_score is not None:
        plot_pr_curve(
            y_test, y_score, "PR — Logistic Regression", f"{RESULTS_DIR}/pr_logreg.png"
        )
        plot_roc_curve(
            y_test,
            y_score,
            "ROC — Logistic Regression",
            f"{RESULTS_DIR}/roc_logreg.png",
        )

    # =========================
    # 2) Random Forest
    # =========================
    rf_params = {
        "n_estimators": 50,
        "bootstrap": True,
        "max_samples": 1_000_000,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "n_jobs": 2,  # testing memory
        "random_state": seed,
        "class_weight": None,
    }
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train, sample_weight=sample_w)
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
            "notes": rf_note_from_params(rf_params),
        },
        footer_meta=footer,
    )
    plot_confusion_matrix(
        y_test, y_pred, "CM — Random Forest", f"{RESULTS_DIR}/cm_random_forest.png"
    )
    if y_score is not None:
        plot_pr_curve(
            y_test, y_score, "PR — Random Forest", f"{RESULTS_DIR}/pr_random_forest.png"
        )
        plot_roc_curve(
            y_test,
            y_score,
            "ROC — Random Forest",
            f"{RESULTS_DIR}/roc_random_forest.png",
        )

    # =========================
    # 3) Isolation Forest (unsupervised)
    # =========================
    if_params = {
        "n_estimators": 200,
        "contamination": "auto",
        "random_state": seed,
        "n_jobs": 2,  # parallelize (adjust if memory gets tight)
        "bootstrap": False,  # default; include for clarity
        "max_samples": "auto",  # default; include for clarity
    }

    iforest = IsolationForest(**if_params)
    iforest.fit(X_train)

    # predict: -1 => anomaly => Attack(1)
    y_pred = (iforest.predict(X_test) == -1).astype(int)
    y_score = None  # no calibrated probabilities by default

    # human-readable note and filename tag
    if_note = (
        "Unsupervised; IF("
        f"n_estimators={if_params['n_estimators']}, "
        f"contamination={if_params['contamination']}, "
        f"max_samples={if_params['max_samples']}, "
        f"bootstrap={if_params['bootstrap']}, "
        f"n_jobs={if_params['n_jobs']})"
    )
    if_tag = (
        f"ne{if_params['n_estimators']}_"
        f"ms{str(if_params['max_samples']).replace('%','p')}_"
        f"nj{if_params['n_jobs']}"
    )

    m = compute_metrics(y_test, y_pred, y_score)
    write_metrics_csv(
        RESULTS_CSV,
        {
            "model": "IsolationForest",
            **m,
            "avg_loao_recall": None,
            "notes": if_note,
        },
        footer_meta=footer,
    )
    # no PR/ROC curves without scores
    plot_confusion_matrix(
        y_test,
        y_pred,
        "CM — Isolation Forest",
        f"{RESULTS_DIR}/cm_isolation_forest.png",
    )

    print(
        f"[OK] Baselines complete. Wrote metrics to {RESULTS_CSV} and plots to {RESULTS_DIR}/"
    )


if __name__ == "__main__":
    main()
