# Generic calibration helpers for supervised models (constraint-aware)

from __future__ import annotations

import json
import pathlib
from datetime import datetime, UTC
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(seed: int = 42) -> Dict[str, Pipeline]:
    """Default model builder (overridden by callers when needed)."""
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


def _confusion(y_true: np.ndarray, yhat: np.ndarray):
    tp = int(((y_true == 1) & (yhat == 1)).sum())
    fp = int(((y_true == 0) & (yhat == 1)).sum())
    fn = int(((y_true == 1) & (yhat == 0)).sum())
    tn = int(((y_true == 0) & (yhat == 0)).sum())
    return tp, fp, fn, tn


def _sweep_with_constraints(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: Optional[float],
    min_precision: Optional[float],
    max_fpr: Optional[float],
) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Return chosen threshold + metrics dict + curves dict using constraints.
    Policy: if target_recall set -> choose smallest t meeting all constraints;
            else choose best-F1 subject to constraints (if provided).
    """
    ts = np.linspace(0.0, 1.0, 401)
    best_f1, best_t = -1.0, ts[0]
    chosen_t = None

    # arrays for plotting
    pr_p, pr_r, _ = precision_recall_curve(y_true, scores)
    fpr_curve, tpr_curve, _ = roc_curve(y_true, scores)
    pr_auc = auc(pr_r, pr_p)
    roc_auc = auc(fpr_curve, tpr_curve)

    for t in ts:
        yhat = (scores >= t).astype(np.int8)
        tp, fp, fn, tn = _confusion(y_true, yhat)
        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
        feasible = True
        if target_recall is not None:
            feasible &= rec >= target_recall
        if min_precision is not None:
            feasible &= prec >= min_precision
        if max_fpr is not None:
            feasible &= fpr <= max_fpr
        if feasible and chosen_t is None:
            chosen_t = float(t)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        if feasible and f1 > best_f1:
            best_f1, best_t = f1, float(t)

    if chosen_t is None:
        chosen_t = best_t  # fallback to constrained best-F1 (or unconstrained if no constraints)

    # final metrics at chosen_t
    yhat = (scores >= chosen_t).astype(np.int8)
    tp, fp, fn, tn = _confusion(y_true, yhat)
    prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    return (
        chosen_t,
        {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        },
        {"pr": (pr_r, pr_p), "roc": (fpr_curve, tpr_curve)},
    )


def calibrate(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    out_prefix: str,
    seed: int = 42,
    # NEW:
    min_precision: Optional[float] = None,
    max_fpr: Optional[float] = None,
    target_recall_list: Optional[List[float]] = None,
    include_best_f1: bool = True,
) -> List[Dict[str, float]]:
    """
    Fit models from build_models(seed), sweep thresholds with optional constraints,
    write per-model artifacts, and return a list of rows for the caller's summary CSV.
    For each model:
      - If target_recall_list: one row per target.
      - If include_best_f1: add a best-F1 row (subject to constraints if provided).
    """
    models = build_models(seed)
    pathlib.Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        scores = pipe.predict_proba(X_va)[:, 1]

        # Which policies to evaluate
        policies: List[Tuple[str, Optional[float]]] = []
        if target_recall_list:
            for tr in target_recall_list:
                policies.append(("target_recall", float(tr)))
        if include_best_f1 or not policies:
            policies.append(("best_f1", None))

        plotted = False
        last_threshold = None
        for pol, tr in policies:
            chosen_t, m, curves = _sweep_with_constraints(
                y_va,
                scores,
                target_recall=(tr if pol == "target_recall" else None),
                min_precision=min_precision,
                max_fpr=max_fpr,
            )
            label = f"{name}[{pol}{'' if tr is None else f'={tr:.2f}'}]"
            rows.append(
                {
                    "model": label,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "threshold": float(chosen_t),
                    "date_utc": datetime.now(UTC).isoformat(),
                }
            )
            last_threshold = chosen_t

            if not plotted:
                # PR
                pr_r, pr_p = curves["pr"]
                plt.figure()
                plt.plot(pr_r, pr_p)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"PR — {name}")
                plt.tight_layout()
                plt.savefig(f"{out_prefix}_{name}_pr.png")
                plt.close()
                # ROC
                fpr, tpr = curves["roc"]
                plt.figure()
                plt.plot(fpr, tpr)
                plt.plot([0, 1], [0, 1], "--")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title(f"ROC — {name}")
                plt.tight_layout()
                plt.savefig(f"{out_prefix}_{name}_roc.png")
                plt.close()
                # F1 sweep (for visibility): compute once
                ts = np.linspace(0.0, 1.0, 401)
                f1s = []
                for t in ts:
                    yhat = (scores >= t).astype(np.int8)
                    tp = int(((y_va == 1) & (yhat == 1)).sum())
                    fp = int(((y_va == 0) & (yhat == 1)).sum())
                    fn = int(((y_va == 1) & (yhat == 0)).sum())
                    prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
                    rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
                    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
                    f1s.append(f1)
                plt.figure()
                plt.plot(ts, f1s)
                if last_threshold is not None:
                    plt.axvline(last_threshold, linestyle="--")
                plt.xlabel("Threshold")
                plt.ylabel("F1")
                plt.title(f"Threshold sweep — {name}")
                plt.tight_layout()
                plt.savefig(f"{out_prefix}_{name}_thresh_f1.png")
                plt.close()
                plotted = True

        # Persist model and meta for the *last* evaluated policy
        joblib.dump(pipe, f"{out_prefix}_{name}.pkl")
        with open(f"{out_prefix}_{name}.json", "w") as f:
            json.dump(
                {
                    "model": name,
                    "chosen_threshold": (
                        float(last_threshold) if last_threshold is not None else None
                    ),
                    "seed": seed,
                    "date_utc": datetime.now(UTC).isoformat(),
                    "policies": [p[0] for p in policies],
                    "targets": [p[1] for p in policies],
                    "min_precision": min_precision,
                    "max_fpr": max_fpr,
                },
                f,
                indent=2,
            )

    return rows
