"""Model builders for Task 1 Flower102 classification."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.task1_cls.attention import CBAM


def build_resnet18_classifier(
    *,
    num_classes: int,
    init: str = "pretrained",
    attention: str | None = None,
    cbam_reduction: int = 16,
    cbam_spatial_kernel_size: int = 7,
) -> nn.Module:
    """Build ResNet-18 with optional CBAM before global average pooling."""

    weights = ResNet18_Weights.DEFAULT if init == "pretrained" else None
    model = resnet18(weights=weights)
    feature_channels = model.fc.in_features
    if (attention or "").lower() == "cbam":
        model.avgpool = nn.Sequential(
            CBAM(feature_channels, reduction=cbam_reduction, spatial_kernel_size=cbam_spatial_kernel_size),
            model.avgpool,
        )
    model.fc = nn.Linear(feature_channels, num_classes)
    return model
