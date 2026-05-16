"""Model builders for Task 1 Flower102 classification."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

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


def build_resnet34_classifier(
    *,
    num_classes: int,
    init: str = "pretrained",
    attention: str | None = None,
    cbam_reduction: int = 16,
    cbam_spatial_kernel_size: int = 7,
) -> nn.Module:
    """Build ResNet-34 with optional CBAM before global average pooling.

    Mirrors ``build_resnet18_classifier`` so the CBAM hookup is structurally
    identical — the only difference is the deeper backbone (3-4-6-3 vs 2-2-2-2
    BasicBlocks). Both share ImageNet normalization so the training transform
    can be reused unchanged.
    """

    weights = ResNet34_Weights.DEFAULT if init == "pretrained" else None
    model = resnet34(weights=weights)
    feature_channels = model.fc.in_features
    if (attention or "").lower() == "cbam":
        model.avgpool = nn.Sequential(
            CBAM(feature_channels, reduction=cbam_reduction, spatial_kernel_size=cbam_spatial_kernel_size),
            model.avgpool,
        )
    model.fc = nn.Linear(feature_channels, num_classes)
    return model


def build_vit_tiny_classifier(
    *,
    num_classes: int,
    init: str = "pretrained",
    **_unused,
) -> nn.Module:
    """ViT-Tiny (``timm.vit_tiny_patch16_224``) with a replaced classifier head.

    Bonus comparison against the convolutional ResNet backbones. CBAM is not
    applicable on ViT (no spatial-conv stack), so any ``attention``/``cbam_*``
    kwargs from the shared factory call are silently ignored.
    """

    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "timm is required for vit_tiny; install with `pip install \"timm>=0.9\"`."
        ) from exc

    return timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=(init == "pretrained"),
        num_classes=num_classes,
    )


def build_classifier(model_name: str, **kwargs) -> nn.Module:
    """Dispatch to the right builder based on the config ``model`` field."""

    name = (model_name or "resnet18").lower()
    if name == "resnet18":
        return build_resnet18_classifier(**kwargs)
    if name == "resnet34":
        return build_resnet34_classifier(**kwargs)
    if name == "vit_tiny":
        return build_vit_tiny_classifier(**kwargs)
    raise ValueError(
        f"Unknown classifier model: {name!r} (supported: resnet18, resnet34, vit_tiny)."
    )
