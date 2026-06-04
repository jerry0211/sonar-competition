"""Turn a variable-length waveform into a fixed-length feature vector.

Combines broadband shape (Welch PSD bands + spectral shape stats), narrowband
tonal cues (LOFAR line strength), and shaft-rate cues (DEMON peak prominence).
Pure scipy/numpy, so it runs without librosa or torch — ideal for a fast GBM
baseline on either real audio or the synthetic simulator.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .narrowband import lofargram, demon

N_BANDS = 16


def summary_features(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.std() < 1e-9:                     # guard near-silence
        y = y + 1e-6 * np.random.standard_normal(len(y))

    feats = []

    # --- broadband: log energy in N log-spaced bands from Welch PSD ---
    f, pxx = welch(y, fs=sr, nperseg=min(1024, len(y)))
    edges = np.linspace(0, len(f), N_BANDS + 1).astype(int)
    band_e = [pxx[a:b].mean() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    feats.extend(np.log10(np.array(band_e) + 1e-12))

    # --- spectral shape stats ---
    p = pxx / (pxx.sum() + 1e-12)
    centroid = float((f * p).sum())
    spread = float(np.sqrt(((f - centroid) ** 2 * p).sum()))
    flatness = float(np.exp(np.mean(np.log(pxx + 1e-12))) / (pxx.mean() + 1e-12))
    cum = np.cumsum(p)
    rolloff = float(f[np.searchsorted(cum, 0.85)]) if cum[-1] > 0 else 0.0
    feats.extend([centroid, spread, flatness, rolloff])

    # --- narrowband tonals via LOFAR: strongest persistent line ---
    _, _, lof = lofargram(y, sr, fmax=min(1000.0, sr / 2 - 1))
    line = lof.max(axis=1) - np.median(lof, axis=1)   # per-freq line strength
    feats.extend([float(line.max()), float(line.mean()),
                  float((line > line.mean() + 2 * line.std()).sum())])  # tonal count

    # --- shaft-rate cue via DEMON: peak prominence + frequency ---
    nyq = sr / 2
    df, dmag = demon(y, sr, band=(0.25 * nyq, 0.45 * nyq), fmax=80.0)
    if len(dmag) > 2:
        peak = int(np.argmax(dmag))
        prominence = float(dmag[peak] / (dmag.mean() + 1e-12))
        peak_freq = float(df[peak])
    else:
        prominence, peak_freq = 0.0, 0.0
    feats.extend([prominence, peak_freq])

    # --- time domain ---
    rms = float(np.sqrt(np.mean(y ** 2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(y))) > 0))
    feats.extend([rms, zcr])

    return np.asarray(feats, dtype=float)


def featurize_batch(waveforms, sr) -> np.ndarray:
    return np.vstack([summary_features(w, sr) for w in waveforms])
