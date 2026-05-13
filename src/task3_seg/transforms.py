"""PIL/PyTorch transforms for Task 3 semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps


@dataclass
class SegmentationTransform:
    image_size: int
    horizontal_flip_prob: float = 0.0

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")

        if self.horizontal_flip_prob > 0.0 and float(torch.rand(()).item()) < self.horizontal_flip_prob:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        size = (self.image_size, self.image_size)
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = mask.resize(size, Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.int64)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask_array).long()
        return image_tensor, mask_tensor


def build_segmentation_transform(
    image_size: int,
    *,
    train: bool,
    horizontal_flip_prob: float = 0.5,
) -> SegmentationTransform:
    return SegmentationTransform(
        image_size=image_size,
        horizontal_flip_prob=horizontal_flip_prob if train else 0.0,
    )
