# Phase III LOAO with balanced training (LogReg, RF) and benign-only IF.
# Outputs -> results/ml/feat-ml-phase3/loao_v2/loao_*

import argparse
import json
import pathlib
from datetime import datetime, UTC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.utils.ml_data_prep import load_ml_config, make_supervised_xy
from src.utils.data_prep import load_config, check_schema, verify_checksum, sha256_file
from src.ml.balanced import (
    make_balanced,
    ForestGrowConfig,
    grow_forest,
    IsoGrowConfig,
    grow_iforest,
    stable_class_weight,
)


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _light_load_with_family(ds_yaml_path: str, ml_cfg: dict) -> pd.DataFrame:
    cfg = load_config(ds_yaml_path)
    csv_path = cfg["dataset_path"]
    exp = cfg.get("sha256")
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


def _tpfn_bar(tp: int, fn: int, title: str, path: str):
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.bar(["TP", "FN"], [tp, fn])
    ax.set_title(title)
    for i, v in enumerate([tp, fn]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _score_hist(scores: np.ndarray, thresh: float, title: str, path: str):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(scores, bins=60)
    ax.axvline(thresh, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Score / predicted probability")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _compute_counts_and_metrics(y_true: np.ndarray, yhat: np.ndarray):
    tp = int(((y_true == 1) & (yhat == 1)).sum())
    fp = int(((y_true == 0) & (yhat == 1)).sum())
    fn = int(((y_true == 1) & (yhat == 0)).sum())
    tn = int(((y_true == 0) & (yhat == 0)).sum())
    prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
    return tp, fp, fn, tn, prec, rec, fpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-prefix",
        default=f"results/ml/feat-ml-phase3/loao_v2/loao_{_ts()}",
        help="Prefix for outputs (CSV).",
    )
    ap.add_argument(
        "--fig-prefix",
        help="Figure prefix; defaults to figures/... mirroring out-prefix (recommended).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--family-col", default="Attack Family")
    ap.add_argument("--benign-label", default="Benign", help="Name of benign family.")

    # RF growing strategy
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-trees-per-pass", type=int, default=25)
    ap.add_argument("--rf-max-samples", type=int, default=1_000_000)
    ap.add_argument("--rf-n-jobs", type=int, default=1)

    # IF growing strategy
    ap.add_argument("--if-trees", type=int, default=200)
    ap.add_argument("--if-trees-per-pass", type=int, default=50)
    ap.add_argument(
        "--if-benign-cap",
        type=int,
        default=1_000_000,
        help="Max benign rows per pass for IF sampler (None to disable).",
    )

    # Threshold controls
    ap.add_argument("--lr-thresh", type=float, default=0.5)
    ap.add_argument("--rf-thresh", type=float, default=0.5)
    ap.add_argument(
        "--use-calibrated-thresholds",
        action="store_true",
        help="Load thresholds from calibration JSONs (prefix via --calib-prefix).",
    )
    ap.add_argument(
        "--calib-prefix",
        type=str,
        default=None,
        help="Prefix used by run_calibration_balanced (without _RandomForest.json suffix).",
    )
    ap.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="If set, retune threshold on TRAIN split to achieve recall >= target "
        "(also applies min-precision/max-fpr constraints if provided).",
    )
    ap.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="Minimum precision constraint when retuning threshold.",
    )
    ap.add_argument(
        "--max-fpr",
        type=float,
        default=None,
        help="Maximum false-positive rate constraint when retuning threshold.",
    )

    # Plots
    ap.add_argument("--write-tpfn", action="store_true")
    ap.add_argument("--write-hist", action="store_true")

    args = ap.parse_args()

    # Mirror out-prefix to a figure prefix by default
    if not args.fig_prefix:
        rp = pathlib.Path(args.out_prefix)
        args.fig_prefix = str(
            pathlib.Path("figures") / rp.parent.relative_to("results") / rp.name
        )

    ml_cfg = load_ml_config("configs/ml.yaml")
    df = _light_load_with_family("configs/dataset.yaml", ml_cfg)
    X_df, y_ser = make_supervised_xy(df, ml_cfg)

    X_all = X_df.values.astype(np.float32, copy=False)
    y_all = y_ser.values.astype(np.int8, copy=False)
    fam = df[args.family_col].values

    # Optionally pull calibrated thresholds (as starting points)
    if args.use_calibrated_thresholds:
        if not args.calib_prefix:
            raise SystemExit("--use-calibrated-thresholds requires --calib-prefix")
        try:
            with open(f"{args.calib_prefix}_RandomForest.json", "r") as f:
                rf_meta = json.load(f)
            args.rf_thresh = rf_meta.get("chosen_threshold", args.rf_thresh)
        except FileNotFoundError:
            print(
                "[warn] RF calibration JSON not found; using --rf-thresh as provided."
            )
        try:
            with open(f"{args.calib_prefix}_LogReg.json", "r") as f:
                lr_meta = json.load(f)
            args.lr_thresh = lr_meta.get("chosen_threshold", args.lr_thresh)
        except FileNotFoundError:
            print(
                "[warn] LR calibration JSON not found; using --lr-thresh as provided."
            )

    def _retune_threshold(scores: np.ndarray, y: np.ndarray, start_t=0.0):
        """Find smallest t meeting recall≥target_recall AND optional constraints."""
        if (
            args.target_recall is None
            and args.min_precision is None
            and args.max_fpr is None
        ):
            return None
        ts = np.linspace(start_t, 1.0, 401)
        chosen = None
        best_f1, best_t = -1.0, ts[0]
        for t in ts:
            yhat = (scores >= t).astype(np.int8)
            tp = int(((y == 1) & (yhat == 1)).sum())
            fp = int(((y == 0) & (yhat == 1)).sum())
            fn = int(((y == 1) & (yhat == 0)).sum())
            tn = int(((y == 0) & (yhat == 0)).sum())
            prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
            rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
            fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)

            feasible = True
            if args.target_recall is not None:
                feasible &= rec >= args.target_recall
            if args.min_precision is not None:
                feasible &= prec >= args.min_precision
            if args.max_fpr is not None:
                feasible &= fpr <= args.max_fpr
            if feasible and chosen is None:
                chosen = float(t)

            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            if feasible and f1 > best_f1:
                best_f1, best_t = f1, float(t)

        return chosen if chosen is not None else best_t

    families = sorted(np.unique(fam))
    rows = []

    # Prep dirs
    pathlib.Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.fig_prefix).parent.mkdir(parents=True, exist_ok=True)

    benign_label = args.benign_label

    for held in families:
        if held == benign_label:
            print(f"[LOAO] held-out family: {held}")
            print("  -> skipping: this is the benign family.")
            continue

        print(f"[LOAO] held-out family: {held}")

        # Test = held-out attacks + ALL benign
        test_idx = (fam == held) | (fam == benign_label)
        # Train = ALL rows except held-out attack family (includes benign)
        train_idx = fam != held

        X_tr = X_all[train_idx]
        y_tr = y_all[train_idx]
        X_te = X_all[test_idx]
        y_te = y_all[test_idx]

        if not ((fam == held) & (y_all == 1)).any():
            print("  -> no Attack rows in held-out family; skipping.")
            continue

        # ===== Logistic Regression (balanced training) =====
        X_lr, y_lr = make_balanced(X_tr, y_tr, seed=args.seed)
        lr = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        solver="saga",
                        class_weight="balanced",
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        print(f"  [LR] fit on {len(y_lr):,} rows (balanced)")
        lr.fit(X_lr, y_lr)
        lr_scores_tr = lr.predict_proba(X_lr)[:, 1]
        lr_thresh = args.lr_thresh
        tuned_lr = _retune_threshold(lr_scores_tr, y_lr, start_t=0.0)
        if tuned_lr is not None:
            lr_thresh = tuned_lr
        lr_scores_te = lr.predict_proba(X_te)[:, 1]
        yhat_lr = (lr_scores_te >= lr_thresh).astype(np.int8)

        tp, fp, fn, tn, prec, rec, fpr = _compute_counts_and_metrics(y_te, yhat_lr)
        rows.append(
            {
                "family": held,
                "model": "LogReg(balanced)",
                "precision": prec,
                "recall": rec,
                "fpr": fpr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "threshold": float(lr_thresh),
            }
        )
        run_tag = args.fig_prefix  # already run-scoped
        if args.write_tpfn:
            _tpfn_bar(
                tp,
                fn,
                f"TP/FN — LR — {held}",
                f"{run_tag}_tpfn_lr_{str(held).replace(' ','_')}.png",
            )
        if args.write_hist:
            _score_hist(
                lr_scores_te,
                lr_thresh,
                f"Score hist — LR — {held}",
                f"{run_tag}_hist_lr_{str(held).replace(' ','_')}.png",
            )

        # ===== Random Forest (balanced training, grown in passes) =====
        cw = stable_class_weight(y_tr)
        X_rf, y_rf = make_balanced(X_tr, y_tr, seed=args.seed)
        eff_max_samples = int(min(args.rf_max_samples, len(y_rf)))
        if eff_max_samples < args.rf_max_samples:
            print(
                f"  [RF] lower max_samples_per_tree {args.rf_max_samples:,} -> {eff_max_samples:,} "
                f"(n_train_bal={len(y_rf):,})"
            )
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

        rf = grow_forest(X_rf, y_rf, rf_cfg, sampler=rebalance_each_pass)

        rf_scores_tr = rf.predict_proba(X_rf)[:, 1]
        rf_thresh = args.rf_thresh
        tuned_rf = _retune_threshold(rf_scores_tr, y_rf, start_t=0.0)
        if tuned_rf is not None:
            rf_thresh = tuned_rf

        rf_scores_te = rf.predict_proba(X_te)[:, 1]
        yhat_rf = (rf_scores_te >= rf_thresh).astype(np.int8)

        tp, fp, fn, tn, prec, rec, fpr = _compute_counts_and_metrics(y_te, yhat_rf)
        rows.append(
            {
                "family": held,
                "model": "RandomForest(balanced,grown)",
                "precision": prec,
                "recall": rec,
                "fpr": fpr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "threshold": float(rf_thresh),
            }
        )
        if args.write_tpfn:
            _tpfn_bar(
                tp,
                fn,
                f"TP/FN — RF — {held}",
                f"{run_tag}_tpfn_rf_{str(held).replace(' ','_')}.png",
            )
        if args.write_hist:
            _score_hist(
                rf_scores_te,
                rf_thresh,
                f"Score hist — RF — {held}",
                f"{run_tag}_hist_rf_{str(held).replace(' ','_')}.png",
            )

        # ===== Isolation Forest (benign-only training, grown in passes) =====
        X_benign = X_tr[y_tr == 0]  # benign-only
        if len(X_benign) > 0:
            if_cfg = IsoGrowConfig(
                total_estimators=args.if_trees,
                trees_per_pass=args.if_trees_per_pass,
                max_samples="auto",
                contamination="auto",
                n_jobs=1,
                random_state=args.seed,
                verbose=True,
            )

            def benign_sampler(X_benign_full: np.ndarray, pass_id: int) -> np.ndarray:
                cap = args.if_benign_cap
                if cap is None or len(X_benign_full) <= cap:
                    return X_benign_full
                rng = np.random.default_rng(args.seed + pass_id)
                idx = rng.choice(len(X_benign_full), size=cap, replace=False)
                return X_benign_full[idx]

            iforest = grow_iforest(X_benign, if_cfg, sampler=benign_sampler)

            # IF: predict = -1 => anomaly => attack (no threshold value)
            yhat_if = (iforest.predict(X_te) == -1).astype(np.int8)
            tp, fp, fn, tn, prec, rec, fpr = _compute_counts_and_metrics(y_te, yhat_if)
            rows.append(
                {
                    "family": held,
                    "model": "IsolationForest(benign-only,grown)",
                    "precision": prec,
                    "recall": rec,
                    "fpr": fpr,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "threshold": None,
                }
            )
            if args.write_tpfn:
                _tpfn_bar(
                    tp,
                    fn,
                    f"TP/FN — IF — {held}",
                    f"{run_tag}_tpfn_if_{str(held).replace(' ','_')}.png",
                )

    # Write CSV (now includes precision/recall/FPR and counts)
    out_csv = f"{args.out_prefix}_metrics.csv"
    pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[loao-balanced] wrote {out_csv}")

    # --- Bar chart (recall only, for a quick glance) ---
    df_rows = pd.DataFrame(rows)
    try:
        pv = df_rows.pivot(
            index="family", columns="model", values="recall"
        ).sort_index()
        figpath = f"{args.fig_prefix}_loao_bars.png"
        ax = pv.plot(kind="bar", figsize=(10, 4))
        ax.set_ylabel("Recall")
        ax.set_xlabel("Attack Family")
        ax.set_title("LOAO Recall by Model (Balanced/Grown)")
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(figpath, bbox_inches="tight")
        plt.close(fig)
        print(f"[loao-balanced] bar chart: {figpath}")
    except Exception as e:
        print(f"[loao-balanced] bar chart skipped: {e}")


if __name__ == "__main__":
    main()
