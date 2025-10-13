from __future__ import annotations
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    confusion_matrix,
)


def _fmt(x):
    return None if x is None else float(f"{x:.3f}")


def compute_metrics(y_true, y_pred, y_score=None):
    m = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": None,
        "pr_auc": None,
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_score)
            m["pr_auc"] = average_precision_score(y_true, y_score)
        except Exception:
            pass
    for k in m:
        m[k] = _fmt(m[k])
    return m


def write_metrics_csv(csv_path, row_dict, footer_meta=None):
    header = [
        "model",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "avg_loao_recall",
        "notes",
    ]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        for k in header:
            row_dict.setdefault(k, None)
        for k in ["precision", "recall", "f1", "roc_auc", "pr_auc", "avg_loao_recall"]:
            if row_dict.get(k) is not None:
                row_dict[k] = float(f"{row_dict[k]:.3f}")
        w.writerow(row_dict)
        if footer_meta:
            f.write(
                "# " + " | ".join([f"{k}: {v}" for k, v in footer_meta.items()]) + "\n"
            )


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Attack", "Benign"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Attack", "Benign"])
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = 100.0 * count / total if total else 0.0
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true, y_score, title, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_score, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr)
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
