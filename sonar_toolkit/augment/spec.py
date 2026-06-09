"""Spectrogram augmentations — fills the `augment/` placeholder.

These are item-level transforms (passed as `transform=` to WindowDataset) plus a
batch-level mixup utility. On small/noisy sonar data, augmentation is usually
worth more than a bigger model, so this earns its place in the toolkit.
"""
from __future__ import annotations

import torch


class SpecAugment:
    """Random frequency/time masking on a spectrogram tensor [C, F, T]."""

    def __init__(self, freq_mask=12, time_mask=20, n_freq=2, n_time=2, p=0.5):
        self.freq_mask, self.time_mask = freq_mask, time_mask
        self.n_freq, self.n_time, self.p = n_freq, n_time, p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or torch.rand(1).item() > self.p:
            return x
        _, F, T = x.shape
        for _ in range(self.n_freq):
            f = int(torch.randint(0, self.freq_mask + 1, (1,)).item())
            if 0 < f < F:
                f0 = int(torch.randint(0, F - f, (1,)).item())
                x[:, f0:f0 + f, :] = 0
        for _ in range(self.n_time):
            t = int(torch.randint(0, self.time_mask + 1, (1,)).item())
            if 0 < t < T:
                t0 = int(torch.randint(0, T - t, (1,)).item())
                x[:, :, t0:t0 + t] = 0
        return x


class AddNoise:
    """Add Gaussian noise at a random SNR (dB). Works on raw or spectrogram input."""

    def __init__(self, snr_db_range=(5, 20), p=0.5):
        self.lo, self.hi = snr_db_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        snr = self.lo + (self.hi - self.lo) * torch.rand(1).item()
        sig_p = x.pow(2).mean()
        noise_p = sig_p / (10 ** (snr / 10))
        return x + torch.randn_like(x) * noise_p.sqrt()


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Batch-level mixup. Returns mixed_x, (y_a, y_b, lam).

    To use, mix inside the training step and combine the two losses:
        xm, (ya, yb, lam) = mixup_batch(x, y)
        loss = lam * crit(model(xm), ya) + (1 - lam) * crit(model(xm), yb)
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], (y, y[idx], lam)
