# Phase II calibration with balanced training for LogReg & RF.
# Outputs -> results/ml/feat-ml-phase2/calib_*

import argparse
import csv
import json
import pathlib
from datetime import datetime, UTC

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.ml.calibration as calib_mod
from src.utils.ml_data_prep import load_ml_config, make_supervised_xy
from src.utils.data_prep import load_config, check_schema, verify_checksum, sha256_file
from src.ml.balanced import (
    make_balanced,
    ForestGrowConfig,
    grow_forest,
    stable_class_weight,
)


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _light_load(ds_yaml_path: str, ml_cfg: dict) -> pd.DataFrame:
    ds_cfg = load_config(ds_yaml_path)
    csv_path = ds_cfg["dataset_path"]
    exp = ds_cfg.get("sha256")
    if exp and not verify_checksum(csv_path, exp):
        actual = sha256_file(csv_path)
        raise ValueError(f"Checksum mismatch.\nExpected: {exp}\nActual:   {actual}")
    check_schema(pd.read_csv(csv_path, nrows=0))
    needed = list(ml_cfg["features"]) + [
        ml_cfg["label_column"],
        ml_cfg["attack_family_column"],
    ]
    dtype_map = {
        "Length": "int32",
        "Source Port": "int32",
        "Destination Port": "int32",
        "FunctionCodeNum": "int16",
        ml_cfg["label_column"]: "string",
        ml_cfg["attack_family_column"]: "string",
    }
    return pd.read_csv(
        csv_path,
        usecols=needed,
        dtype=dtype_map,
        engine="c",
        low_memory=True,
        memory_map=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-prefix",
        default=f"results/ml/feat-ml-phase2/calib_{_ts()}",
        help="Prefix for outputs (CSV, models, plots).",
    )
    ap.add_argument(
        "--fig-prefix",
        help="Figure prefix; defaults to figures/... mirroring out-prefix (recommended).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.2)

    # RF grow strategy
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-trees-per-pass", type=int, default=25)
    ap.add_argument("--rf-max-samples", type=int, default=1_000_000)
    ap.add_argument("--rf-n-jobs", type=int, default=1)

    # Threshold policy
    ap.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="If set, choose the smallest threshold with recall >= this value (on validation).",
    )

    args = ap.parse_args()

    # Mirror out-prefix to a figure prefix by default
    if not args.fig_prefix:
        # results/ml/feat-ml-phase2/calib_runX  ->  figures/ml/feat-ml-phase2/calib_runX
        rp = pathlib.Path(args.out_prefix)
        args.fig_prefix = str(
            pathlib.Path("figures") / rp.parent.relative_to("results") / rp.name
        )

    ml_cfg = load_ml_config("configs/ml.yaml")
    df = _light_load("configs/dataset.yaml", ml_cfg)

    X_df, y_ser = make_supervised_xy(df, ml_cfg)
    X = X_df.values.astype(np.float32, copy=False)
    y = y_ser.values.astype(np.int8, copy=False)

    # ===== Balanced train / natural validation =====
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=args.val_split, random_state=args.seed, stratify=y
    )

    # Stable RF weights computed on full (pre-balance) train labels
    cw = stable_class_weight(y_tr)
    print(f"[RF] stable class_weight (pre-balance): {cw}")

    # Balance only the training split for actual fitting
    X_tr, y_tr = make_balanced(X_tr, y_tr, seed=args.seed)

    # ---- Logistic Regression via existing calibrate() (only LR)
    orig_build = calib_mod.build_models

    def lr_only(seed=42):
        return {
            "LogReg": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            solver="saga",
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            )
        }

    calib_mod.build_models = lr_only
    try:
        pathlib.Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.fig_prefix).parent.mkdir(parents=True, exist_ok=True)
        lr_rows = calib_mod.calibrate(
            X_tr, y_tr, X_va, y_va, args.out_prefix, seed=args.seed
        )
    finally:
        calib_mod.build_models = orig_build

    # 2) Grow the RF incrementally (balanced passes + fixed weights)
    rf_cfg = ForestGrowConfig(
        total_trees=args.rf_trees,
        trees_per_pass=args.rf_trees_per_pass,
        max_depth=20,
        min_samples_leaf=2,
        max_samples_per_tree=args.rf_max_samples,
        class_weight=cw,  # fixed dict -> no warm_start warning
        n_jobs=args.rf_n_jobs,
        random_state=args.seed,
        verbose=True,
    )

    def rebalance_each_pass(X_full, y_full, pass_id):
        # different seed per pass to get a fresh balanced slice
        return make_balanced(X_full, y_full, seed=args.seed + pass_id)

    rf_model = grow_forest(X_tr, y_tr, cfg=rf_cfg, sampler=rebalance_each_pass)
    print(f"[RF] Training complete ({rf_cfg.total_trees} trees with rebalanced passes)")

    # ==== Evaluate RF on validation with robust threshold sweep ====
    rf_score = rf_model.predict_proba(X_va)[:, 1]
    ts = np.linspace(0.0, 1.0, 401, dtype=np.float64)  # finer sweep
    P = np.empty_like(ts)
    R = np.empty_like(ts)
    F1 = np.empty_like(ts)

    best_t, best_f1, best_idx = 0.5, -1.0, 0
    n_filled = 0

    for i, t in enumerate(ts):
        yhat = (rf_score >= t).astype(np.int8, copy=False)
        P[i] = precision_score(y_va, yhat, zero_division=0)
        R[i] = recall_score(y_va, yhat, zero_division=0)
        F1[i] = f1_score(y_va, yhat, zero_division=0)
        n_filled = i + 1

        if args.target_recall is not None:
            # choose smallest threshold with recall >= target
            if R[i] >= args.target_recall:
                best_t, best_f1, best_idx = t, F1[i], i
                break
        else:
            if F1[i] > best_f1:
                best_f1, best_t, best_idx = F1[i], t, i

    # Trim to filled portion (handles early break)
    ts_used = ts[:n_filled]
    P = P[:n_filled]
    R = R[:n_filled]
    F1 = F1[:n_filled]

    print(
        f"[RF] threshold sweep points: {n_filled}, "
        f"best_t={best_t:.4f}, best_f1={best_f1:.4f}, "
        f"policy={'target_recall' if args.target_recall is not None else 'best_f1'}"
    )

    pr_p, pr_r, _ = precision_recall_curve(y_va, rf_score)
    fpr, tpr, _ = roc_curve(y_va, rf_score)
    pr_auc = auc(pr_r, pr_p)
    roc_auc = auc(fpr, tpr)

    # plots (scoped by run)
    pathlib.Path(args.fig_prefix).parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(pr_r, pr_p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR — RandomForest (balanced)")
    plt.tight_layout()
    plt.savefig(f"{args.fig_prefix}_RandomForest_pr.png")
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC — RandomForest (balanced)")
    plt.tight_layout()
    plt.savefig(f"{args.fig_prefix}_RandomForest_roc.png")
    plt.close()

    plt.figure()
    plt.plot(ts_used, F1)
    plt.axvline(best_t, linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title("Threshold sweep — RandomForest (balanced)")
    plt.tight_layout()
    plt.savefig(f"{args.fig_prefix}_RandomForest_thresh_f1.png")
    plt.close()

    # save model + meta
    joblib.dump(rf_model, f"{args.out_prefix}_RandomForest.pkl")
    with open(f"{args.out_prefix}_RandomForest.json", "w") as f:
        json.dump(
            {
                "model": "RandomForest",
                "chosen_threshold": float(best_t),
                "seed": args.seed,
                "date_utc": datetime.now(UTC).isoformat(),
                "policy": (
                    "target_recall" if args.target_recall is not None else "best_f1"
                ),
                "target_recall": args.target_recall,
                "index": int(best_idx),
                "sweep_points": int(n_filled),
            },
            f,
            indent=2,
        )

    rf_row = {
        "model": "RandomForest",
        "precision": float(P[best_idx]),
        "recall": float(R[best_idx]),
        "f1": float(F1[best_idx]),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(best_t),
        "date_utc": datetime.now(UTC).isoformat(),
    }

    rows = lr_rows + [rf_row]

    # write combined CSV
    csv_path = f"{args.out_prefix}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "threshold",
                "date_utc",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[calibration-balanced] wrote {csv_path}")


if __name__ == "__main__":
    main()
