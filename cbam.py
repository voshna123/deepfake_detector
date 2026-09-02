"""
Convolutional Block Attention Module (CBAM).

Reference:
  Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
  https://arxiv.org/abs/1807.06521

CBAM applies two sequential attention sub-modules:
  1. Channel attention  – "what" to focus on
  2. Spatial attention  – "where" to focus on
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel attention module.

    Uses both average-pooling and max-pooling paths and combines them
    through a shared MLP before applying a sigmoid gate.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid = max(1, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_pool = x.mean(dim=(2, 3))          # (B, C)
        max_pool = x.amax(dim=(2, 3))          # (B, C)
        scale = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * scale.unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    """
    Spatial attention module.

    Concatenates channel-wise average and max features and applies
    a single 2-D convolution followed by a sigmoid gate.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        scale = torch.sigmoid(self.conv(concat))
        return x * scale


class CBAM(nn.Module):
    """
    Full CBAM block (channel attention followed by spatial attention).

    Args:
        in_channels      : Number of input feature-map channels.
        reduction_ratio  : Reduction ratio for the channel-attention MLP.
        kernel_size      : Kernel size for the spatial-attention convolution (3 or 7).
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
