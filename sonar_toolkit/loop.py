"""The GPU training loop — the piece the repo was missing.

Deliberately framework-light and decoupled: `fit` takes a model, two datasets,
and a *scoring function*, and knows nothing about which model or which feature.
That keeps it in the durable toolkit while experiments stay thin.

What it does for you (i.e. what makes a GPU worth having):
  - mixed precision (AMP) — the biggest free speedup on a T4; auto-off on CPU
  - AdamW + cosine/plateau LR schedule
  - class weighting for imbalance (rare-target detection)
  - early stopping on the *detection metric*, not loss
  - best-checkpoint saving — survives a Kaggle session disconnect

`predict_scores` returns calibrated P(positive) so results plug straight into
your existing metric suite and into cheap ensembling later.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .device import get_device

# (y_true, y_score) -> float, higher is better. e.g. your recall_at_fpr.
ScoreFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 6              # early-stop patience, in epochs
    num_workers: int = 2
    use_amp: bool = True           # only actually engages on CUDA
    grad_clip: float | None = 1.0
    pos_index: int = 1             # which logit column means "detection"
    scheduler: str = "cosine"      # 'cosine' | 'plateau' | 'none'
    min_delta: float = 1e-4
    ckpt_path: str = "checkpoints/model.pt"


def _make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )


@torch.no_grad()
def predict_scores(model, ds_or_loader, device=None, pos_index=1,
                   batch_size=64, num_workers=2):
    """Return (scores, y_true) where scores = P(positive)."""
    device = device or get_device()
    model.eval().to(device)
    loader = (ds_or_loader if isinstance(ds_or_loader, DataLoader)
              else _make_loader(ds_or_loader, batch_size, False, num_workers))
    scores, ys = [], []
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1]
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, pos_index]
        scores.append(p.detach().cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(scores), np.concatenate(ys)


def fit(model, train_ds, val_ds, score_fn: ScoreFn, cfg: TrainConfig | None = None,
        device=None, class_weights=None, verbose: bool = True) -> dict:
    cfg = cfg or TrainConfig()
    device = device or get_device()
    model = model.to(device)

    train_loader = _make_loader(train_ds, cfg.batch_size, True, cfg.num_workers)
    val_loader = _make_loader(val_ds, cfg.batch_size, False, cfg.num_workers)

    if class_weights is not None:
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    elif cfg.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    else:
        sched = None

    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, best_state, no_improve = -np.inf, None, 0
    history: list[dict] = []
    Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_ds)

        val_scores, val_y = predict_scores(model, val_loader, device, cfg.pos_index)
        val_metric = float(score_fn(val_y, val_scores))
        if sched is not None:
            sched.step(val_metric) if cfg.scheduler == "plateau" else sched.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_metric": val_metric})
        if verbose:
            print(f"  epoch {epoch:3d}  loss {train_loss:.4f}  val {val_metric:.4f}")

        if val_metric > best_score + cfg.min_delta:
            best_score = val_metric
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({"model_state": best_state, "val_metric": best_score, "epoch": epoch},
                       cfg.ckpt_path)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                if verbose:
                    print(f"  early stop @ epoch {epoch} (best {best_score:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "best_score": best_score,
            "history": history, "ckpt_path": cfg.ckpt_path}
