"""Single source of truth for CV-splitting policy.

Both the sklearn harness (`cross_validate`) and the torch harness
(`cross_validate_torch`) call `make_splits`, so the rule "group by recording AND
stratify by label" lives in exactly one place and can't drift between them. This
is the fix for the 02_audio_cnn runner having quietly re-implemented its own
(label-blind) GroupKFold.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold


def make_splits(y, groups: Optional[np.ndarray] = None, n_splits: int = 5, seed: int = 42):
    """Return a list of (train_idx, val_idx) folds.

    groups given -> StratifiedGroupKFold: keeps each recording intact across folds
                    (no segment leakage) AND balances classes (no single-class folds,
                    which would make recall@FPR meaningless).
    groups None  -> StratifiedKFold: only for genuinely independent rows, e.g. UCI tabular.
    """
    y = np.asarray(y)
    dummy = np.zeros(len(y))  # splitters only need shape; features aren't required here
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(dummy, y, np.asarray(groups)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(dummy, y))
