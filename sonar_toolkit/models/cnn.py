"""PyTorch CNN templates: a 1D net for raw waveform and a 2D net for
spectrograms. Deliberately small and readable so they train fast in a hackathon
and are easy to modify. Both expose the same `(in_shape) -> logits` contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """For raw waveform input of shape (batch, 1, samples)."""

    def __init__(self, n_classes: int = 2, base: int = 16):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv1d(ci, co, 9, padding=4), nn.BatchNorm1d(co),
                nn.ReLU(), nn.MaxPool1d(4),
            )
        self.features = nn.Sequential(
            block(1, base), block(base, base * 2),
            block(base * 2, base * 4), block(base * 4, base * 8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(base * 8, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class CNN2D(nn.Module):
    """For spectrogram input of shape (batch, 1, freq, time)."""

    def __init__(self, n_classes: int = 2, base: int = 16):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(), nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(1, base), block(base, base * 2),
            block(base * 2, base * 4), block(base * 4, base * 8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(base * 8, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# NOTE: For small/noisy datasets, prefer transfer learning (PANNs CNN14 / YAMNet
# embeddings + a light classifier) over training these from scratch. Add that
# wrapper here when you reach Phase 2.
