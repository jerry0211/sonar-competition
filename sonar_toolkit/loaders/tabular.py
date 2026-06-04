"""Data loaders. Each returns numpy-friendly objects the toolkit understands.

The loader layer is the seam that lets you re-point the whole pipeline at a new
data format on competition day without touching models or validation.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_tabular(csv_path, label_col=-1, positive_label=None):
    """Load a feature CSV. Returns (X: float array, y: int array).

    `positive_label` marks which class counts as a detection (1). If None and
    labels are non-numeric, the alphabetically-last class is treated as positive.
    """
    df = pd.read_csv(csv_path, header=None) if _headerless(csv_path) else pd.read_csv(csv_path)
    label = df.columns[label_col]
    X = df.drop(columns=[label]).to_numpy(dtype=float)
    raw = df[label]
    if not pd.api.types.is_numeric_dtype(raw):
        pos = positive_label if positive_label is not None else sorted(raw.unique())[-1]
        y = (raw == pos).astype(int).to_numpy()
    else:
        y = raw.astype(int).to_numpy()
    return X, y


def _headerless(csv_path) -> bool:
    first = Path(csv_path).read_text().splitlines()[0].split(",")
    # If the first row is mostly numbers, assume there's no header.
    numeric = sum(_is_float(c) for c in first)
    return numeric >= len(first) - 1


def _is_float(s) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_audio(path, sr=22050, mono=True):
    """Load a single wav. Returns (waveform, sr). Requires librosa."""
    try:
        import librosa
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install librosa to load audio") from e
    y, sr = librosa.load(path, sr=sr, mono=mono)
    return y, sr


def segment(y, sr, win_s=2.0, hop_s=1.0):
    """Slice a long recording into fixed windows for frame-level inference."""
    win, hop = int(win_s * sr), int(hop_s * sr)
    return [y[i:i + win] for i in range(0, max(1, len(y) - win + 1), hop)]
