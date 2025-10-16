# Phase II LOAO with balanced training (LogReg, RF) and benign-only IF.
# Outputs -> results/ml/feat-ml-phase2/loao_*

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-prefix",
        default=f"results/ml/feat-ml-phase2/loao_{_ts()}",
        help="Prefix for outputs (CSV).",
    )
    ap.add_argument(
        "--fig-prefix",
        help="Figure prefix; defaults to figures/... mirroring out-prefix (recommended).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--family-col", default="Attack Family")

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
        help="If set, sweep thresholds on the TRAIN side to achieve recall >= target.",
    )

    # Plots
    ap.add_argument(
        "--write-tpfn",
        action="store_true",
        help="Write TP/FN bar charts per family instead of CMs.",
    )
    ap.add_argument(
        "--write-hist",
        action="store_true",
        help="Write score histograms with threshold line for LR/RF per family.",
    )

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

    # Optionally pull calibrated thresholds
    if args.use_calibrated_througholds if False else False:
        pass  # dead branch to keep linters quiet if user toggles flags later

    if args.use_calibrated_thresholds:
        if not args.calib_prefix:
            raise SystemExit("--use-calibrated-thresholds requires --calib-prefix")
        try:
            with open(f"{args.calib_prefix}_RandomForest.json", "r") as f:
                rf_meta = json.load(f)
            args.rf_thres = rf_meta.get("chosen_threshold", args.rf_thres)  # typo guard
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

    families = sorted(np.unique(fam))
    rows = []

    # Prep dirs
    pathlib.Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.fig_prefix).parent.mkdir(parents=True, exist_ok=True)

    for held in families:
        print(f"[LOAO] held-out family: {held}")
        train_idx = fam != held
        test_idx = fam == held

        X_tr = X_all[train_idx]
        y_tr = y_all[train_idx]
        X_te = X_all[test_idx]
        y_te = y_all[test_idx]

        if not (y_te == 1).any():
            print("  -> no Attack rows in held-out split; skipping.")
            continue

        # Optional threshold retuning to a target recall (using TRAIN side only)
        def fit_thresh_from_train(scores: np.ndarray, y: np.ndarray, start_t=0.0):
            if args.target_recall is None:
                return None
            ts = np.linspace(start_t, 1.0, 401)
            for t in ts:
                rec = (((scores >= t).astype(np.int8) & (y == 1)).sum()) / max(
                    1, (y == 1).sum()
                )
                if rec >= args.target_recall:
                    return float(t)
            return float(ts[-1])

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
        if args.target_recall is not None:
            lr_thresh = fit_thresh_from_train(lr_scores_tr, y_lr) or lr_thresh
        lr_scores_te = lr.predict_proba(X_te)[:, 1]
        yhat_lr = (lr_scores_te >= lr_thresh).astype(np.int8)

        tp = int(((y_te == 1) & (yhat_lr == 1)).sum())
        fn = int(((y_te == 1) & (yhat_lr == 0)).sum())
        rec_lr = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        rows.append({"family": held, "model": "LogReg(balanced)", "recall": rec_lr})
        run_tag = args.fig_prefix  # already run-scoped
        if args.write_tpfn:
            _tpfn_bar(
                tp,
                fn,
                f"Attack-only TP/FN — LR — {held}",
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
        # stable weights from full (pre-balance) y_tr
        cw = stable_class_weight(y_tr)
        X_rf, y_rf = make_balanced(X_tr, y_tr, seed=args.seed)
        rf_cfg = ForestGrowConfig(
            total_trees=args.rf_trees,
            trees_per_pass=args.rf_trees_per_pass,
            max_depth=20,
            min_samples_leaf=2,
            max_samples_per_tree=args.rf_max_samples,
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
        if args.target_recall is not None:
            rf_thresh = fit_thresh_from_train(rf_scores_tr, y_rf) or rf_thresh

        rf_scores_te = rf.predict_proba(X_te)[:, 1]
        yhat_rf = (rf_scores_te >= rf_thresh).astype(np.int8)

        tp = int(((y_te == 1) & (yhat_rf == 1)).sum())
        fn = int(((y_te == 1) & (yhat_rf == 0)).sum())
        rec_rf = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        rows.append(
            {"family": held, "model": "RandomForest(balanced,grown)", "recall": rec_rf}
        )
        if args.write_tpfn:
            _tpfn_bar(
                tp,
                fn,
                f"Attack-only TP/FN — RF — {held}",
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

            # IF: -1 => anomaly => attack; no calibrated threshold here
            yhat_if = (iforest.predict(X_te) == -1).astype(np.int8)
            tp = int(((y_te == 1) & (yhat_if == 1)).sum())
            fn = int(((y_te == 1) & (yhat_if == 0)).sum())
            rec_if = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
            rows.append(
                {
                    "family": held,
                    "model": "IsolationForest(benign-only,grown)",
                    "recall": rec_if,
                }
            )
            if args.write_tpfn:
                _tpfn_bar(
                    tp,
                    fn,
                    f"Attack-only TP/FN — IF — {held}",
                    f"{run_tag}_tpfn_if_{str(held).replace(' ','_')}.png",
                )

    out_csv = f"{args.out_prefix}_metrics.csv"
    pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[loao-balanced] wrote {out_csv}")

    # --- Bar chart from LOAO metrics (scoped to run via fig-prefix) ---
    df_rows = pd.DataFrame(rows)
    pv = df_rows.pivot(index="family", columns="model", values="recall").sort_index()

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


if __name__ == "__main__":
    main()
