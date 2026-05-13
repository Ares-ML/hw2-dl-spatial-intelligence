"""Stanford Background dataset helpers for Task 3 segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.task3_seg.transforms import build_segmentation_transform


CLASS_NAMES = ["sky", "tree", "road", "grass", "water", "building", "mountain", "foreground"]
PALETTE = [
    (128, 128, 128),
    (129, 127, 38),
    (120, 69, 125),
    (53, 125, 34),
    (0, 11, 123),
    (118, 20, 12),
    (122, 81, 25),
    (241, 134, 51),
]
IGNORE_INDEX = 255

SegmentationTransformFn = Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]]


class StanfordSegmentationDataset(Dataset):
    """Load prepared Stanford Background RGB images and mapped mask PNGs."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: SegmentationTransformFn | None = None,
        image_size: int = 256,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / "iccv09Data" / "images"
        self.mask_dir = self.root / "masks"
        self.split_path = self.root / "splits" / f"{split}.txt"
        self.transform = transform or build_segmentation_transform(image_size, train=False)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Missing Stanford image directory: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Missing Stanford mask directory: {self.mask_dir}")
        if not self.split_path.is_file():
            raise FileNotFoundError(f"Missing Stanford split file: {self.split_path}")

        self.ids = [line.strip() for line in self.split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.ids:
            raise RuntimeError(f"No sample ids found in {self.split_path}")

    def __len__(self) -> int:
        return len(self.ids)

    def paths_for(self, sample_id: str) -> tuple[Path, Path]:
        return self.image_dir / f"{sample_id}.jpg", self.mask_dir / f"{sample_id}.png"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.ids[index]
        image_path, mask_path = self.paths_for(sample_id)
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing image or mask for sample id {sample_id}")

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            return self.transform(image, mask)
