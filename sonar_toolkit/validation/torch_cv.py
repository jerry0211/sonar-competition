"""Group-aware, stratified cross-validation for PyTorch models.

The deep-learning counterpart to `cross_validate`. It exists so torch experiments
stop hand-rolling their own fold loop: it calls the SAME `make_splits` and the SAME
metric suite (`evaluate`/`aggregate`) as the tabular harness, so the CV policy and
the scoring live in one place for both worlds.

torch is imported lazily inside the function, so `import sonar_toolkit.validation`
stays torch-free for the Phase 1 tabular path.

Contract:
    make_fold_datasets(fold, train_idx, val_idx) -> (train_ds, val_ds)
    make_model()                                  -> a fresh nn.Module per fold
The experiment supplies those two thin factories; everything else is shared.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from .metrics import evaluate, aggregate, recall_at_fpr
from .splits import make_splits


def _balanced_weights(y_tr) -> np.ndarray:
    """Inverse-frequency class weights for rare-target imbalance."""
    y_tr = np.asarray(y_tr)
    pos = float(y_tr.mean())
    return np.array([1.0 / (1.0 - pos + 1e-6), 1.0 / (pos + 1e-6)], dtype="float32")


def cross_validate_torch(
    labels,
    groups,
    make_fold_datasets: Callable,   # (fold, train_idx, val_idx) -> (train_ds, val_ds)
    make_model: Callable,           # () -> nn.Module
    n_splits: int = 5,
    seed: int = 42,
    max_fpr: float = 0.1,
    train_cfg=None,                 # sonar_toolkit.training.TrainConfig
    device=None,
    class_weight: bool = True,
    ckpt_dir: str = "checkpoints",
    verbose: bool = True,
) -> dict:
    # Lazy import keeps the Phase 1 tabular path free of any torch dependency.
    from sonar_toolkit.training import fit, TrainConfig, get_device, predict_scores

    labels = np.asarray(labels)
    device = device or get_device()
    base_cfg = train_cfg or TrainConfig()
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    for i, (tr, va) in enumerate(make_splits(labels, groups, n_splits, seed)):
        train_ds, val_ds = make_fold_datasets(i, tr, va)
        cfg = replace(base_cfg, ckpt_path=str(Path(ckpt_dir) / f"fold{i}.pt"))
        weights = _balanced_weights(labels[tr]) if class_weight else None

        # Early-stop / checkpoint on recall@FPR (passed positionally so it works
        # regardless of whether the metric's kwarg is named max_fpr or fpr).
        out = fit(make_model(), train_ds, val_ds,
                  lambda yt, ys: recall_at_fpr(yt, ys, max_fpr),
                  cfg=cfg, device=device, class_weights=weights, verbose=verbose)

        # Score the best checkpoint through the shared metric suite, exactly like cross_validate.
        scores, ys = predict_scores(out["model"], val_ds, device=device, pos_index=cfg.pos_index)
        m = evaluate(ys, scores, max_fpr=max_fpr)
        fold_metrics.append(m)
        if verbose:
            print(f"[fold {i}] " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))

    return aggregate(fold_metrics)
