"""Narrowband passive-sonar features. These are the domain-specific tools that
beat generic spectrograms when the target is a quiet machine in loud water.

LOFAR (Low-Frequency Analysis and Recording): a normalised low-frequency
spectrogram that surfaces stable narrowband *tonals* from rotating machinery.

DEMON (Detection of Envelope Modulation On Noise): demodulates the broadband
propeller-cavitation noise to recover the *shaft rate* and blade count. Two
vessels can share a spectrum but have different shaft signatures, so DEMON is
often what separates target classes.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt, spectrogram, resample_poly


def lofargram(y, sr, n_fft=2048, hop=512, fmax=1000.0):
    """Normalised low-frequency spectrogram (dB), clipped to [0, fmax] Hz.

    Per-frequency mean removal ("2-pass split-window" in spirit) flattens the
    broadband background so narrowband lines stand out.
    """
    f, t, Sxx = spectrogram(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, scaling="spectrum")
    keep = f <= fmax
    f, Sxx = f[keep], Sxx[keep]
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    Sxx_db -= Sxx_db.mean(axis=1, keepdims=True)  # background normalisation
    return f, t, Sxx_db


def demon(y, sr, band=(5000.0, 15000.0), decim_rate=1000, fmax=120.0):
    """DEMON spectrum: envelope of a high-freq band, then low-freq FFT.

    Returns (freqs_hz, magnitude). Peaks correspond to shaft rate and harmonics
    (blade rate = shaft rate x number of blades).
    """
    nyq = sr / 2
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.99)
    sos = butter(4, [lo, hi], btype="band", output="sos")
    filtered = sosfiltfilt(sos, y)

    envelope = np.abs(hilbert(filtered))            # AM demodulation
    envelope = envelope - envelope.mean()

    up, down = 1, max(1, int(sr // decim_rate))     # decimate to resolve low freqs
    env_ds = resample_poly(envelope, up, down)
    fs_ds = sr / down

    spectrum = np.abs(np.fft.rfft(env_ds * np.hanning(len(env_ds))))
    freqs = np.fft.rfftfreq(len(env_ds), 1 / fs_ds)
    keep = freqs <= fmax
    return freqs[keep], spectrum[keep]
