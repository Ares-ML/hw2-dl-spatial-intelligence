"""Attention modules for Task 1 classification experiments."""

from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """CBAM channel attention with shared 1x1 MLP."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return self.activation(attention)


class SpatialAttention(nn.Module):
    """CBAM spatial attention over average and max channel maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in {3, 7}:
            raise ValueError("kernel_size must be 3 or 7.")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        attention = torch.cat([avg_map, max_map], dim=1)
        return self.activation(self.conv(attention))


class CBAM(nn.Module):
    """Convolutional Block Attention Module preserving input shape."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
