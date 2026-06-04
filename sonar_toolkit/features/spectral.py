"""General spectral features. Thin wrappers over librosa so experiments stay short.

Install librosa to use these (it's optional; the UCI tabular path doesn't need it).
"""
from __future__ import annotations

import numpy as np


def _require_librosa():
    try:
        import librosa  # noqa
        return librosa
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install librosa to use spectral features") from e


def stft_db(y, sr, n_fft=1024, hop=256):
    librosa = _require_librosa()
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    return librosa.amplitude_to_db(S, ref=np.max)


def mel_spectrogram(y, sr, n_mels=128, n_fft=1024, hop=256):
    librosa = _require_librosa()
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
    return librosa.power_to_db(S, ref=np.max)


def mfcc(y, sr, n_mfcc=40):
    librosa = _require_librosa()
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
