"""Gradient-boosted trees: the fastest strong baseline on tabular sonar features.

Uses sklearn's HistGradientBoosting so there are zero extra dependencies. Swap in
LightGBM/XGBoost later if you want, behind the same fit_predict signature.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def make_gbm(**kwargs):
    params = dict(max_iter=300, learning_rate=0.06, max_depth=None, l2_regularization=1.0)
    params.update(kwargs)
    return HistGradientBoostingClassifier(**params)


def gbm_fit_predict(X_tr, y_tr, X_va, **kwargs):
    """fit_predict callable for the CV harness. Returns P(positive) on X_va."""
    model = make_gbm(**kwargs)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1]
