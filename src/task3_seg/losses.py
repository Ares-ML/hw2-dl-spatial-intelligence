"""Loss functions for Task 3 semantic segmentation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss with ignore-index masking."""

    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("logits must have shape [B, C, H, W].")
        if target.ndim != 3:
            raise ValueError("target must have shape [B, H, W].")
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"logits channel count must equal num_classes={self.num_classes}.")

        valid = target != self.ignore_index
        target_clamped = target.clone()
        target_clamped[~valid] = 0

        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target_clamped.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        denominator = torch.sum(probs + one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    """Cross-entropy plus weighted Dice loss."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index, smooth=smooth)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def build_segmentation_loss(
    name: str,
    *,
    num_classes: int,
    ignore_index: int = 255,
    dice_weight: float = 1.0,
    smooth: float = 1.0,
    **_: Any,
) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    if key == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index, smooth=smooth)
    if key in {"ce_dice", "ce+dice"}:
        return CEDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            dice_weight=dice_weight,
            smooth=smooth,
        )
    raise ValueError(f"Unsupported segmentation loss: {name}")
