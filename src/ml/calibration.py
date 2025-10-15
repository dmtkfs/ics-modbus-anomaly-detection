from __future__ import annotations
import json
import datetime
import pathlib
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _ts():
    return datetime.datetime.utcnow().isoformat()


def _ensuredir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def build_models(seed=42):
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
        ),
        "RandomForest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=50,
                        max_depth=20,
                        min_samples_leaf=2,
                        bootstrap=True,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=2,
                    ),
                )
            ]
        ),
    }


def _sweep_thresholds(y_true, y_score, steps=81):
    ts = np.linspace(0, 1, steps)
    best_t, best_f1 = 0.5, -1.0
    P, R, F1 = [], [], []
    for t in ts:
        yhat = (y_score >= t).astype(int)
        p = precision_score(y_true, yhat, zero_division=0)
        r = recall_score(y_true, yhat, zero_division=0)
        f = f1_score(y_true, yhat, zero_division=0)
        P.append(p)
        R.append(r)
        F1.append(f)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t), {"t": ts, "p": P, "r": R, "f1": F1}


def calibrate(Xtr, ytr, Xva, yva, out_prefix: str, seed=42):
    _ensuredir(pathlib.Path(out_prefix).parent.as_posix())
    rows = []
    for name, pipe in build_models(seed).items():
        pipe.fit(Xtr, ytr)
        score = (
            pipe.predict_proba(Xva)[:, 1]
            if hasattr(pipe[-1], "predict_proba")
            else pipe.decision_function(Xva)
        )
        best_t, curves = _sweep_thresholds(yva, score)

        pr_p, pr_r, _ = precision_recall_curve(yva, score)
        fpr, tpr, _ = roc_curve(yva, score)
        pr_auc = auc(pr_r, pr_p)
        roc_auc = auc(fpr, tpr)

        # plots
        plt.figure()
        plt.plot(pr_r, pr_p)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR — {name}")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{name}_pr.png")
        plt.close()
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC — {name}")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{name}_roc.png")
        plt.close()
        plt.figure()
        plt.plot(curves["t"], curves["f1"])
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.title(f"Threshold sweep — {name}")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{name}_thresh_f1.png")
        plt.close()

        # persist
        joblib.dump(pipe, f"{out_prefix}_{name}.pkl")
        with open(f"{out_prefix}_{name}.json", "w") as f:
            json.dump(
                {
                    "model": name,
                    "chosen_threshold": best_t,
                    "seed": seed,
                    "date_utc": _ts(),
                },
                f,
                indent=2,
            )

        yhat = (score >= best_t).astype(int)
        rows.append(
            {
                "model": name,
                "precision": float(precision_score(yva, yhat, zero_division=0)),
                "recall": float(recall_score(yva, yhat, zero_division=0)),
                "f1": float(f1_score(yva, yhat, zero_division=0)),
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "threshold": best_t,
                "date_utc": _ts(),
            }
        )
    return rows
