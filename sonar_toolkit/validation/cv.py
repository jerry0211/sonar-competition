"""Cross-validation harness.

The single most important rule for audio/sonar: if you have multiple segments
from the same recording, they MUST stay together in a fold. Otherwise the model
memorises the recording, segments leak across train/test, and your validation
score is fiction that collapses on the real test set.

Pass `groups` (e.g. recording IDs) and this uses StratifiedGroupKFold. Omit it
only when every row is genuinely independent (e.g. the UCI tabular dataset).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from .metrics import evaluate, aggregate


def cross_validate(
    X,
    y,
    fit_predict: Callable,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    max_fpr: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run K-fold CV.

    `fit_predict(X_tr, y_tr, X_va) -> y_score` trains on the fold's train split
    and returns a probability/score for the validation split. Keeping the model
    behind this callable is what makes the harness model-agnostic: swap a GBM for
    a CNN and nothing else changes.
    """
    y = np.asarray(y)
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)

    fold_metrics = []
    for i, (tr, va) in enumerate(split_iter):
        X_tr = X[tr] if hasattr(X, "__getitem__") else [X[j] for j in tr]
        X_va = X[va] if hasattr(X, "__getitem__") else [X[j] for j in va]
        y_score = fit_predict(X_tr, y[tr], X_va)
        m = evaluate(y[va], y_score, max_fpr=max_fpr)
        fold_metrics.append(m)
        if verbose:
            print(f"  fold {i}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))

    return aggregate(fold_metrics)
