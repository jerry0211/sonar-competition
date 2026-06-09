"""Device selection and reproducibility helpers.

One import the whole repo uses so device handling lives in exactly one place.
On Kaggle this resolves to the T4/P100; on your laptop it falls back to CPU,
so the same experiment code runs in both places unchanged.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if prefer_gpu and mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def describe_device(device: torch.device | None = None) -> str:
    device = device or get_device()
    if device.type == "cuda":
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        return f"CUDA: {name} ({mem:.1f} GB)"
    return device.type
