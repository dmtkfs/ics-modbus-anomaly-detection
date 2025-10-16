# Phase III calibration with balanced training for LogReg & RF
# Outputs -> results/ml/feat-ml-phase3/calib_*

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
from sklearn.metrics import precision_recall_curve, roc_curve, auc
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
    print("[load] Loading dataset (light mode)...")
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
    df = pd.read_csv(
        csv_path,
        usecols=needed,
        dtype=dtype_map,
        engine="c",
        low_memory=True,
        memory_map=True,
    )
    print(f"[load] Done. Rows: {len(df):,}, Columns: {len(df.columns)}")
    return df


def _confusion(y_true: np.ndarray, yhat: np.ndarray):
    tp = int(((y_true == 1) & (yhat == 1)).sum())
    fp = int(((y_true == 0) & (yhat == 1)).sum())
    fn = int(((y_true == 1) & (yhat == 0)).sum())
    tn = int(((y_true == 0) & (yhat == 0)).sum())
    return tp, fp, fn, tn


def eval_rf_at_policy(
    rf_model,
    X_va,
    y_va,
    policy: str,
    target_recall: float | None,
    min_precision: float | None,
    max_fpr: float | None,
):
    """Evaluate RF on validation set under a threshold policy + optional constraints."""
    print(
        f"[eval] RF policy={policy} target_recall={target_recall} "
        f"min_precision={min_precision} max_fpr={max_fpr}"
    )
    rf_score = rf_model.predict_proba(X_va)[:, 1]
    ts = np.linspace(0.0, 1.0, 401, dtype=np.float64)
    P = np.empty_like(ts)
    R = np.empty_like(ts)
    F1 = np.empty_like(ts)
    FPR = np.empty_like(ts)

    best_t, best_f1, best_idx = 0.5, -1.0, 0
    n_filled = 0
    first_feasible_idx = None

    for i, t in enumerate(ts):
        yhat = (rf_score >= t).astype(np.int8, copy=False)
        tp, fp, fn, tn = _confusion(y_va, yhat)
        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)

        P[i], R[i], F1[i], FPR[i] = prec, rec, f1, fpr
        n_filled = i + 1

        # feasibility under constraints
        feasible = True
        if policy == "target_recall" and target_recall is not None:
            feasible &= rec >= target_recall
        if min_precision is not None:
            feasible &= prec >= min_precision
        if max_fpr is not None:
            feasible &= fpr <= max_fpr

        if feasible and first_feasible_idx is None:
            first_feasible_idx = i

        # best-F1 tracking (subject to constraints if any provided)
        if (min_precision is not None) or (max_fpr is not None):
            # constrained best-F1
            if feasible and f1 > best_f1:
                best_f1, best_t, best_idx = f1, t, i
        else:
            # unconstrained best-F1 (legacy)
            if f1 > best_f1:
                best_f1, best_t, best_idx = f1, t, i

    _ = best_t  # silence linter warning (used below)

    # choose threshold
    chosen_idx = None
    if policy == "target_recall" and target_recall is not None:
        if first_feasible_idx is not None:
            chosen_idx = first_feasible_idx
        else:
            print("[eval] WARN: no threshold met constraints; falling back to best-F1.")
            chosen_idx = best_idx
    else:
        chosen_idx = best_idx

    chosen_t = ts[chosen_idx]
    print(
        f"[eval] → chosen_t={chosen_t:.4f}, "
        f"P={P[chosen_idx]:.3f}, R={R[chosen_idx]:.3f}, F1={F1[chosen_idx]:.3f}, "
        f"FPR={FPR[chosen_idx]:.4f}"
    )

    # Curves once
    pr_p, pr_r, _ = precision_recall_curve(y_va, rf_score)
    fpr_curve, tpr_curve, _ = roc_curve(y_va, rf_score)
    pr_auc = auc(pr_r, pr_p)
    roc_auc = auc(fpr_curve, tpr_curve)

    return {
        "precision": float(P[chosen_idx]),
        "recall": float(R[chosen_idx]),
        "f1": float(F1[chosen_idx]),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(chosen_t),
        "policy": policy,
        "target_recall": (None if target_recall is None else float(target_recall)),
        "ts": ts[:n_filled],
        "f1_curve": F1[:n_filled],
        "pr_curve": (pr_r, pr_p),
        "roc_curve": (fpr_curve, tpr_curve),
        "best_idx": int(best_idx),
        "sweep_points": int(n_filled),
    }


def main():
    ap = argparse.ArgumentParser()
    # Phase III controls
    ap.add_argument("--target-recall-list", type=float, nargs="*", default=None)
    ap.add_argument("--include-best-f1", action="store_true")
    ap.add_argument("--benign-mult", type=int, default=1)
    ap.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="Minimum precision constraint for threshold selection.",
    )
    ap.add_argument(
        "--max-fpr",
        type=float,
        default=None,
        help="Maximum false-positive rate constraint for threshold selection.",
    )

    # Outputs
    ap.add_argument(
        "--out-prefix",
        default=f"results/ml/feat-ml-phase3/calib_{_ts()}",
        help="Output prefix (CSV/models/plots)",
    )
    ap.add_argument("--fig-prefix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.2)

    # RF grow strategy
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-trees-per-pass", type=int, default=25)
    ap.add_argument("--rf-max-samples", type=int, default=1_000_000)
    ap.add_argument("--rf-n-jobs", type=int, default=1)

    # Legacy single-target
    ap.add_argument("--target-recall", type=float, default=None)
    args = ap.parse_args()

    if not args.fig_prefix:
        rp = pathlib.Path(args.out_prefix)
        args.fig_prefix = str(
            pathlib.Path("figures") / rp.parent.relative_to("results") / rp.name
        )

    print("=== Phase III Calibration Run ===")
    print(f"Output prefix: {args.out_prefix}")

    ml_cfg = load_ml_config("configs/ml.yaml")
    df = _light_load("configs/dataset.yaml", ml_cfg)
    X_df, y_ser = make_supervised_xy(df, ml_cfg)
    X = X_df.values.astype(np.float32, copy=False)
    y = y_ser.values.astype(np.int8, copy=False)

    print("[split] Splitting train/validation...")
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=args.val_split, random_state=args.seed, stratify=y
    )

    if args.benign_mult > 1:
        print(
            f"[enrich] Duplicating benign samples {args.benign_mult}× before balancing..."
        )
        ben_mask = y_tr == 0
        Xb = X_tr[ben_mask]
        yb = y_tr[ben_mask]
        Xb_enr = np.concatenate([Xb] * args.benign_mult, axis=0)
        yb_enr = np.concatenate([yb] * args.benign_mult, axis=0)
        Xa = X_tr[~ben_mask]
        ya = y_tr[~ben_mask]
        X_tr = np.concatenate([Xa, Xb_enr], axis=0)
        y_tr = np.concatenate([ya, yb_enr], axis=0)

    print("[balance] Computing stable class weights...")
    cw = stable_class_weight(y_tr)
    print(f"[balance] Weights: {cw}")
    X_tr, y_tr = make_balanced(X_tr, y_tr, seed=args.seed)
    print(f"[balance] After balancing: {len(y_tr):,} samples")

    print("[LR] Starting Logistic Regression calibration...")
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
        # NEW: pass constraints through to calibration
        lr_rows = calib_mod.calibrate(
            X_tr,
            y_tr,
            X_va,
            y_va,
            args.out_prefix,
            seed=args.seed,
            min_precision=args.min_precision,
            max_fpr=args.max_fpr,
            target_recall_list=args.target_recall_list
            or ([args.target_recall] if args.target_recall is not None else None),
            include_best_f1=(
                args.include_best_f1
                or not (args.target_recall_list or args.target_recall)
            ),
        )
    finally:
        calib_mod.build_models = orig_build
    print("[LR] Logistic Regression calibration complete.")

    # Clamp rf_max_samples
    eff_max_samples = int(min(args.rf_max_samples, len(y_tr)))
    if eff_max_samples < args.rf_max_samples:
        print(
            f"[RF] Note: lowering max_samples_per_tree from {args.rf_max_samples:,} "
            f"to {eff_max_samples:,} (n_train={len(y_tr):,})."
        )

    print("[RF] Growing RandomForest...")
    rf_cfg = ForestGrowConfig(
        total_trees=args.rf_trees,
        trees_per_pass=args.rf_trees_per_pass,
        max_depth=20,
        min_samples_leaf=2,
        max_samples_per_tree=eff_max_samples,
        class_weight=cw,
        n_jobs=args.rf_n_jobs,
        random_state=args.seed,
        verbose=True,
    )

    def rebalance_each_pass(X_full, y_full, pass_id):
        return make_balanced(X_full, y_full, seed=args.seed + pass_id)

    rf_model = grow_forest(X_tr, y_tr, cfg=rf_cfg, sampler=rebalance_each_pass)
    print(f"[RF] Training complete ({rf_cfg.total_trees} trees).")

    # Policies
    policies = []
    if args.target_recall_list:
        for tr in args.target_recall_list:
            policies.append(("target_recall", float(tr)))
    elif args.target_recall is not None:
        policies.append(("target_recall", float(args.target_recall)))
    if args.include_best_f1 or not policies:
        policies.append(("best_f1", None))
    print(f"[RF] Threshold policies to evaluate: {policies}")

    rf_rows = []
    plotted = False
    last_res = None
    for pol, tr in policies:
        res = eval_rf_at_policy(
            rf_model,
            X_va,
            y_va,
            pol,
            tr,
            min_precision=args.min_precision,
            max_fpr=args.max_fpr,
        )
        last_res = res
        rf_rows.append(
            {
                "model": f"RandomForest[{pol}{'' if tr is None else f'={tr:.2f}'}]",
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
                "roc_auc": res["roc_auc"],
                "pr_auc": res["pr_auc"],
                "threshold": res["threshold"],
                "date_utc": datetime.now(UTC).isoformat(),
            }
        )
        if not plotted:
            print("[plot] Generating PR/ROC/F1 plots...")
            pr_r, pr_p = res["pr_curve"]
            fpr, tpr = res["roc_curve"]
            ts_used = res["ts"]
            F1_curve = res["f1_curve"]

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
            plt.plot(ts_used, F1_curve)
            plt.axvline(res["threshold"], linestyle="--")
            plt.xlabel("Threshold")
            plt.ylabel("F1")
            plt.title("Threshold sweep — RandomForest (balanced)")
            plt.tight_layout()
            plt.savefig(f"{args.fig_prefix}_RandomForest_thresh_f1.png")
            plt.close()
            plotted = True

    print("[save] Saving model and metrics...")
    joblib.dump(rf_model, f"{args.out_prefix}_RandomForest.pkl")
    with open(f"{args.out_prefix}_RandomForest.json", "w") as f:
        json.dump(
            {
                "model": "RandomForest",
                "chosen_threshold": float(rf_rows[-1]["threshold"]),
                "seed": args.seed,
                "date_utc": datetime.now(UTC).isoformat(),
                "policy": policies[-1][0],
                "target_recall": policies[-1][1],
                "trees": rf_cfg.total_trees,
                "trees_per_pass": rf_cfg.trees_per_pass,
                "max_samples_per_tree": rf_cfg.max_samples_per_tree,
                "index": (None if last_res is None else last_res.get("best_idx")),
                "sweep_points": (
                    None if last_res is None else last_res.get("sweep_points")
                ),
            },
            f,
            indent=2,
        )

    rows = lr_rows + rf_rows
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

    print(f"[done] Wrote {csv_path}\n=== Phase III calibration complete ===")


if __name__ == "__main__":
    main()
