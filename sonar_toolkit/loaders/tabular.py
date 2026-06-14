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


# ShipsEar class labels embedded in filenames: <id>_<CLASS>.wav
_SHIPSEAR_CLASSES = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def load_shipsear(data_dir, sr=22050, win_s=2.0, hop_s=1.0):
    """Scan a ShipsEar directory and return windowed waveforms + labels + groups.

    Returns:
        waveforms : list[np.ndarray]  — each is a fixed-length window
        labels    : np.ndarray[int]   — class index 0-4 (A–E)
        groups    : np.ndarray[int]   — recording id (for group-aware CV)
        class_names: dict             — {index: letter}
    """
    data_dir = Path(data_dir)
    wav_files = sorted(data_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")

    waveforms, labels, groups = [], [], []
    for rec_id, path in enumerate(wav_files):
        # Filename convention: <anything>_<CLASS>.wav  e.g. "01_A.wav"
        stem = path.stem.upper()
        letter = stem.split("_")[-1]
        if letter not in _SHIPSEAR_CLASSES:
            continue
        label = _SHIPSEAR_CLASSES[letter]

        y, _ = load_audio(path, sr=sr)
        for win in segment(y, sr, win_s=win_s, hop_s=hop_s):
            waveforms.append(win)
            labels.append(label)
            groups.append(rec_id)

    class_names = {v: k for k, v in _SHIPSEAR_CLASSES.items()}
    return waveforms, np.asarray(labels, dtype=int), np.asarray(groups, dtype=int), class_names


# DeepShip class labels are subfolder names.
_DEEPSHIP_CLASSES = {"Cargo": 0, "Passengership": 1, "Tanker": 2, "Tug": 3}


def load_deepship(data_dir, sr=22050, win_s=2.0, hop_s=1.0):
    """Scan a DeepShip directory tree and return windowed waveforms + labels + groups.

    Expected layout:
        data_dir/
            Cargo/          *.wav
            Passengership/  *.wav
            Tanker/         *.wav
            Tug/            *.wav

    Returns:
        waveforms  : list[np.ndarray]  — fixed-length windows
        labels     : np.ndarray[int]   — 0=Cargo 1=Passengership 2=Tanker 3=Tug
        groups     : np.ndarray[int]   — recording id (for group-aware CV)
        class_names: dict              — {index: class_name}
    """
    data_dir = Path(data_dir)
    waveforms, labels, groups = [], [], []
    rec_id = 0
    for class_name, label in sorted(_DEEPSHIP_CLASSES.items(), key=lambda x: x[1]):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.glob("*.wav")):
            y, _ = load_audio(path, sr=sr)
            for win in segment(y, sr, win_s=win_s, hop_s=hop_s):
                waveforms.append(win)
                labels.append(label)
                groups.append(rec_id)
            rec_id += 1

    if not waveforms:
        raise FileNotFoundError(
            f"No DeepShip WAV files found under {data_dir}. "
            "Expected subfolders: Cargo, Passengership, Tanker, Tug"
        )
    class_names = {v: k for k, v in _DEEPSHIP_CLASSES.items()}
    return waveforms, np.asarray(labels, dtype=int), np.asarray(groups, dtype=int), class_names
