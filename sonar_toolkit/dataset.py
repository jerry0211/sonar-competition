"""A torch Dataset that turns waveform windows into model inputs.

Decoupled by design: you inject a `feature_fn` (your mel / STFT / LOFAR
extractor) rather than importing the features module here, so this stays in the
durable toolkit and the experiment does the wiring.

The on-disk cache is the other half of "getting the best out of a GPU": once the
model runs fast on the card, recomputing spectrograms every epoch on CPU becomes
the bottleneck. Compute each feature once, memoise to .npy, and the GPU stays fed
across epochs *and* across CV folds.

`groups` is carried through so this composes with your group-aware CV — split by
recording, never by window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

FeatureFn = Callable[[np.ndarray], np.ndarray]  # waveform -> 2D [freq, time]


class WindowDataset(Dataset):
    """Fixed-length windows -> (tensor, label).

    waveforms : list of 1D float arrays, OR list of file paths (loaded lazily).
    feature_fn: waveform -> 2D feature. If None, returns the raw waveform as [1, L].
    cache_dir : if set, features are memoised. NOTE: keyed by index, so clear the
                cache dir whenever the underlying recordings change.
    transform : applied to the tensor at access time (train-time augmentation only).
    """

    def __init__(self, waveforms: Sequence, labels, groups=None,
                 feature_fn: FeatureFn | None = None, cache_dir=None,
                 transform: Callable | None = None, sr: int = 22050):
        assert len(waveforms) == len(labels)
        self.waveforms = list(waveforms)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.groups = (np.asarray(groups) if groups is not None
                       else np.arange(len(labels)))
        self.feature_fn = feature_fn
        self.transform = transform
        self.sr = sr
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.waveforms)

    def _load_waveform(self, item) -> np.ndarray:
        if isinstance(item, (str, Path)):
            import librosa
            y, _ = librosa.load(str(item), sr=self.sr, mono=True)
            return y.astype(np.float32)
        return np.asarray(item, dtype=np.float32)

    def _feature_for(self, idx: int) -> np.ndarray:
        fp = self.cache_dir / f"{idx:07d}.npy" if self.cache_dir else None
        if fp is not None and fp.exists():
            return np.load(fp)
        wav = self._load_waveform(self.waveforms[idx])
        feat = self.feature_fn(wav) if self.feature_fn is not None else wav[None, :]
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim == 2:           # [freq, time] -> [1, freq, time]
            feat = feat[None, :, :]
        if fp is not None:
            np.save(fp, feat)
        return feat

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.ascontiguousarray(self._feature_for(idx))).float()
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[idx])
