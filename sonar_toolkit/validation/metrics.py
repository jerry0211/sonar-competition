"""Metric suite for detection tasks.

The headline metric for a detector is almost never raw accuracy. A submarine
detector is judged on how many real targets it catches (recall) while keeping
false alarms low. So `recall_at_fpr` is the one to watch on the day.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def recall_at_fpr(y_true, y_score, max_fpr: float = 0.1) -> float:
    """Highest recall (true-positive rate) achievable without exceeding `max_fpr`.

    This mirrors how operational detectors are tuned: pick the most sensitive
    threshold you can afford given a false-alarm budget.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    allowed = fpr <= max_fpr
    if not allowed.any():
        return 0.0
    return float(tpr[allowed].max())


def evaluate(y_true, y_score, threshold: float = 0.5, max_fpr: float = 0.1) -> dict:
    """Return a dict of the metrics worth tracking for a binary detector."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        f"recall@fpr<={max_fpr}": recall_at_fpr(y_true, y_score, max_fpr),
    }
    # AUC is undefined if only one class is present in a fold.
    if len(np.unique(y_true)) > 1:
        out["auc"] = roc_auc_score(y_true, y_score)
    else:
        out["auc"] = float("nan")
    return out


def aggregate(fold_metrics: list[dict]) -> dict:
    """Mean +/- std across folds, so you report a range, not a lucky number."""
    keys = fold_metrics[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        summary[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return summary
