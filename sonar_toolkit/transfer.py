"""Transfer-learning classifier — the flagged "do this on small/noisy data" path.

A torchvision ResNet treats the mel-spectrogram as an image. The single-channel
input is repeated to 3 channels so the pretrained stem stays intact. On the kind
of small, noisy sonar data you're rehearsing for, fine-tuning a pretrained net
reliably beats training the small CNNs from scratch — and it's exactly the
workload a GPU makes practical.

Kaggle note: pretrained weights download on first use, so toggle **Internet: ON**
in the notebook once. With Internet off, add the weights as a Kaggle Dataset,
construct with pretrained=False, and load_state_dict from /kaggle/input.

For audio-native transfer (PANNs CNN14 / YAMNet) swap the backbone construction
here; the Trainer and metric stay identical.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransferClassifier(nn.Module):
    def __init__(self, n_classes=2, backbone="resnet18",
                 pretrained=True, freeze_backbone=False):
        super().__init__()
        import torchvision

        ctor = getattr(torchvision.models, backbone)
        self.net = ctor(weights="DEFAULT" if pretrained else None)
        in_feats = self.net.fc.in_features
        self.net.fc = nn.Linear(in_feats, n_classes)
        if freeze_backbone:
            for name, p in self.net.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:               # grayscale spectrogram -> 3 channels
            x = x.repeat(1, 3, 1, 1)
        return self.net(x)
