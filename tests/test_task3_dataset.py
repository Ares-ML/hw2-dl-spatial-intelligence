from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - local machines may not have torch.
    torch = None

if torch is not None:
    from src.task3_seg.datasets import IGNORE_INDEX, StanfordSegmentationDataset
    from src.task3_seg.transforms import SegmentationTransform, build_segmentation_transform


def _write_fixture(root: Path, sample_id: str = "sample_001") -> None:
    image_dir = root / "iccv09Data" / "images"
    mask_dir = root / "masks"
    split_dir = root / "splits"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    image = Image.new("RGB", (2, 2))
    image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)])
    image.save(image_dir / f"{sample_id}.jpg")

    mask = Image.new("L", (2, 2))
    mask.putdata([0, 1, 2, IGNORE_INDEX])
    mask.save(mask_dir / f"{sample_id}.png")

    (split_dir / "train.txt").write_text(f"{sample_id}\n", encoding="utf-8")


@unittest.skipIf(torch is None, "PyTorch is not installed")
class Task3DatasetTest(unittest.TestCase):
    def test_dataset_loads_split_and_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_fixture(root)
            transform = build_segmentation_transform(4, train=False)
            dataset = StanfordSegmentationDataset(root, "train", transform=transform)

            image, mask = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(tuple(image.shape), (3, 4, 4))
        self.assertEqual(tuple(mask.shape), (4, 4))
        self.assertEqual(mask.dtype, torch.long)
        self.assertIn(IGNORE_INDEX, set(int(value) for value in torch.unique(mask).tolist()))

    def test_train_transform_flips_image_and_mask_together(self) -> None:
        image = Image.new("RGB", (2, 2))
        image.putdata([(10, 0, 0), (20, 0, 0), (30, 0, 0), (40, 0, 0)])
        mask = Image.new("L", (2, 2))
        mask.putdata([0, 1, 2, IGNORE_INDEX])
        transform = SegmentationTransform(image_size=2, horizontal_flip_prob=1.0)

        image_tensor, mask_tensor = transform(image, mask)

        self.assertEqual(mask_tensor.tolist(), [[1, 0], [IGNORE_INDEX, 2]])
        red_channel = image_tensor[0].mul(255).round().long().tolist()
        self.assertEqual(red_channel, [[20, 10], [40, 30]])


if __name__ == "__main__":
    unittest.main()
