"""Smoke test a handwritten U-Net forward pass on Stanford Background."""

from __future__ import annotations

import argparse
import math
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "stanford_background"
DEFAULT_LOG_FILE = REPO_ROOT / "logs" / "smoke" / "t3_unet_smoke.log"
CLASS_COUNT = 8
IGNORE_INDEX = 255
ALLOWED_MASK_VALUES = set(range(CLASS_COUNT)) | {IGNORE_INDEX}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Stanford Background root.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Split file to read.")
    parser.add_argument("--batch-size", type=int, default=2, help="Single smoke-test batch size.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize for image and mask.")
    parser.add_argument("--num-classes", type=int, default=CLASS_COUNT, help="Semantic class count.")
    parser.add_argument("--ignore-index", type=int, default=IGNORE_INDEX, help="Mask ignore index.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"], help="Smoke-test device.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE, help="Smoke-test log path.")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "On the course GPU server, run through scripts/run_smoke_server.sh "
            "or source scripts/cuda_driver_shim.sh first."
        )
    return torch.device(requested)


class StanfordSegmentationDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: int, ignore_index: int) -> None:
        self.root = root
        self.image_dir = root / "iccv09Data" / "images"
        self.mask_dir = root / "masks"
        split_path = root / "splits" / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        self.ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.ids:
            raise RuntimeError(f"No sample ids found in {split_path}")
        self.image_size = image_size
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.ids[index]
        image_path = self.image_dir / f"{sample_id}.jpg"
        mask_path = self.mask_dir / f"{sample_id}.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing image or mask for sample id {sample_id}")

        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
        with Image.open(mask_path) as mask:
            mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            mask_array = np.asarray(mask, dtype=np.int64)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array)
        return image_tensor, mask_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.enc1 = DoubleConv(3, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(64, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)
        self.shape_flow: list[tuple[str, list[int]]] = []

    def record(self, name: str, tensor: torch.Tensor) -> None:
        self.shape_flow.append((name, list(tensor.shape)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.shape_flow = []
        self.record("input", x)

        enc1 = self.enc1(x)
        self.record("enc1", enc1)
        pool1 = self.pool(enc1)
        self.record("pool1", pool1)

        enc2 = self.enc2(pool1)
        self.record("enc2", enc2)
        pool2 = self.pool(enc2)
        self.record("pool2", pool2)

        enc3 = self.enc3(pool2)
        self.record("enc3", enc3)
        pool3 = self.pool(enc3)
        self.record("pool3", pool3)

        bottleneck = self.bottleneck(pool3)
        self.record("bottleneck", bottleneck)

        up3 = self.up3(bottleneck)
        self.record("up3", up3)
        cat3 = torch.cat([up3, enc3], dim=1)
        self.record("cat3", cat3)
        dec3 = self.dec3(cat3)
        self.record("dec3", dec3)

        up2 = self.up2(dec3)
        self.record("up2", up2)
        cat2 = torch.cat([up2, enc2], dim=1)
        self.record("cat2", cat2)
        dec2 = self.dec2(cat2)
        self.record("dec2", dec2)

        up1 = self.up1(dec2)
        self.record("up1", up1)
        cat1 = torch.cat([up1, enc1], dim=1)
        self.record("cat1", cat1)
        dec1 = self.dec1(cat1)
        self.record("dec1", dec1)

        logits = self.classifier(dec1)
        self.record("logits", logits)
        return logits


def validate_mask_values(mask: torch.Tensor, allowed_values: set[int]) -> list[int]:
    values = sorted(int(value) for value in torch.unique(mask).cpu().tolist())
    unexpected = [value for value in values if value not in allowed_values]
    if unexpected:
        raise RuntimeError(f"Unexpected mask values: {unexpected}; observed={values}")
    return values


def main() -> int:
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    with args.log_file.open("w", encoding="utf-8") as log:
        log.write("Task 3 U-Net Stanford Background smoke test\n")
        log.write(f"repo_root={REPO_ROOT}\n")
        log.write(f"data_root={args.data_root}\n")
        log.write(f"split={args.split}\n")
        log.write(f"batch_size={args.batch_size}\n")
        log.write(f"image_size={args.image_size}\n")
        log.write(f"num_classes={args.num_classes}\n")
        log.write(f"ignore_index={args.ignore_index}\n")
        log.flush()

        try:
            with redirect_stdout(log), redirect_stderr(log):
                torch.manual_seed(42)
                device = choose_device(args.device)
                dataset = StanfordSegmentationDataset(args.data_root, args.split, args.image_size, args.ignore_index)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                images, masks = next(iter(loader))
                observed_mask_values = validate_mask_values(masks, set(range(args.num_classes)) | {args.ignore_index})

                model = SmallUNet(args.num_classes).to(device)
                criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

                images = images.to(device)
                masks = masks.to(device)
                with torch.no_grad():
                    logits = model(images)
                    loss = criterion(logits, masks)

                expected_logits_shape = [args.batch_size, args.num_classes, args.image_size, args.image_size]
                if list(logits.shape) != expected_logits_shape:
                    raise RuntimeError(f"Unexpected logits shape: {list(logits.shape)} != {expected_logits_shape}")
                if not math.isfinite(float(loss.item())):
                    raise RuntimeError(f"Loss is not finite: {loss.item()}")

                log.write(f"device={device}\n")
                log.write(f"dataset_size={len(dataset)}\n")
                log.write(f"batch_images_shape={list(images.shape)}\n")
                log.write(f"batch_masks_shape={list(masks.shape)}\n")
                log.write(f"mask_values={observed_mask_values}\n")
                log.write(f"logits_shape={list(logits.shape)}\n")
                log.write(f"loss={loss.item():.8f}\n")
                log.write("shape_flow:\n")
                for name, shape in model.shape_flow:
                    log.write(f"- {name}: {shape}\n")
                log.write("PASS unet_forward_shape_and_loss\n")
        except Exception as exc:
            log.write(f"FAIL unet smoke failed: {exc}\n")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
