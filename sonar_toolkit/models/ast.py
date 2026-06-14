"""Audio Spectrogram Transformer (AST) classifier.

Expects mel-spectrogram input as tensors of shape (batch, 1, n_mels, time).
Uses a Vision Transformer (DeiT) backbone with patch embedding adapted for
spectrograms. Same fit/predict contract as CNN2D and TransferClassifier.

Reference: Gong et al., "AST: Audio Spectrogram Transformer", Interspeech 2021.

Requires: timm  (pip install timm)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _require_timm():
    try:
        import timm
        return timm
    except ImportError as e:
        raise ImportError("pip install timm to use ASTClassifier") from e


class PatchEmbed(nn.Module):
    """Project spectrogram patches into the transformer embedding dimension.

    Splits a (1, n_mels, time) spectrogram into non-overlapping patches of
    size (patch_h, patch_w) and linearly projects each to `embed_dim`.
    """

    def __init__(self, n_mels: int = 128, patch_h: int = 16, patch_w: int = 16,
                 embed_dim: int = 768):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(patch_h, patch_w),
                              stride=(patch_h, patch_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = self.proj(x)          # (B, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)          # (B, embed_dim, n_patches)
        return x.transpose(1, 2)  # (B, n_patches, embed_dim)


class ASTClassifier(nn.Module):
    """Audio Spectrogram Transformer for binary/multi-class classification.

    Parameters
    ----------
    n_classes : int
        Number of output classes (default 2 for detection).
    n_mels : int
        Height of the input mel spectrogram.
    target_length : int
        Expected time-axis length of the spectrogram (in frames). Inputs are
        padded or truncated to this length for consistent patch counts.
    pretrained : bool
        If True, initialise the transformer encoder from a pretrained DeiT
        checkpoint (ImageNet). The patch embedding and head are always trained
        from scratch.
    backbone : str
        timm model name for the ViT/DeiT backbone.
    embed_dim : int
        Embedding dimension (must match the backbone).
    patch_h, patch_w : int
        Patch size along frequency and time axes.
    drop_rate : float
        Dropout before the classification head.
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_mels: int = 128,
        target_length: int = 200,
        pretrained: bool = True,
        backbone: str = "deit_base_distilled_patch16_224",
        embed_dim: int = 768,
        patch_h: int = 16,
        patch_w: int = 16,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        timm = _require_timm()

        self.n_mels = n_mels
        self.target_length = target_length
        self.patch_h = patch_h
        self.patch_w = patch_w

        # Number of patches along each axis.
        self.n_patches_h = n_mels // patch_h
        self.n_patches_w = target_length // patch_w
        n_patches = self.n_patches_h * self.n_patches_w

        # Custom patch embedding for single-channel spectrograms.
        self.patch_embed = PatchEmbed(
            n_mels=n_mels, patch_h=patch_h, patch_w=patch_w, embed_dim=embed_dim,
        )

        # CLS token + learned positional embedding.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder from a pretrained DeiT.
        deit = timm.create_model(backbone, pretrained=pretrained)
        # Take only the transformer blocks and the layer norm.
        self.blocks = deit.blocks
        self.norm = deit.norm

        # Classification head.
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, n_classes),
        )

    # ------------------------------------------------------------------
    def _pad_or_trim(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure the time axis is exactly `target_length` frames."""
        T = x.shape[-1]
        if T < self.target_length:
            x = nn.functional.pad(x, (0, self.target_length - T))
        elif T > self.target_length:
            x = x[..., : self.target_length]
        return x

    def _interpolate_pos_embed(self, n_patches: int) -> torch.Tensor:
        """Resize positional embedding if input patch count differs from init."""
        if n_patches + 1 == self.pos_embed.shape[1]:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        patch_pos = patch_pos.transpose(1, 2)  # (1, embed_dim, old_n_patches)
        patch_pos = nn.functional.interpolate(
            patch_pos, size=n_patches, mode="linear", align_corners=False,
        )
        patch_pos = patch_pos.transpose(1, 2)  # (1, n_patches, embed_dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, 1, n_mels, T)
            Mel-spectrogram input (single channel).

        Returns
        -------
        logits : Tensor of shape (B, n_classes)
        """
        # Normalise time axis.
        x = self._pad_or_trim(x)

        # Patch embedding -> (B, n_patches, embed_dim).
        tokens = self.patch_embed(x)
        B, N, _ = tokens.shape

        # Prepend CLS token.
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+N, embed_dim)

        # Add positional embedding (interpolate if patch count changed).
        tokens = tokens + self._interpolate_pos_embed(N)

        # Transformer encoder.
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # Classify from CLS token.
        return self.head(tokens[:, 0])
