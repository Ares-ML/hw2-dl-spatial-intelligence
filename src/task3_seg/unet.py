"""Handwritten U-Net modules for Task 3 semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    """Two 3x3 conv-batchnorm-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscale with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upscale decoder features, concatenate the encoder skip, then double conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, bilinear: bool = False) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels, mid_channels=out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    @staticmethod
    def _match_spatial(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        target_h, target_w = reference.shape[-2:]
        height, width = x.shape[-2:]

        if height > target_h:
            top = (height - target_h) // 2
            x = x[..., top : top + target_h, :]
            height = target_h
        if width > target_w:
            left = (width - target_w) // 2
            x = x[..., :, left : left + target_w]
            width = target_w

        diff_h = target_h - height
        diff_w = target_w - width
        if diff_h or diff_w:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._match_spatial(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 classifier head that returns raw logits."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Task 3 U-Net with four encoder-decoder scales."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 32,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        factor = 2 if bilinear else 1
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        self.up1 = Up(base_channels * 16 // factor, base_channels * 8, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8 // factor, base_channels * 4, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4 // factor, base_channels * 2, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2 // factor, base_channels, base_channels, bilinear)
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
