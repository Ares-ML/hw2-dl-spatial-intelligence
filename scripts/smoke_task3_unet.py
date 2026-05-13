"""Smoke test the shared Task 3 U-Net forward pass on Stanford Background."""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task3_seg.datasets import CLASS_NAMES, IGNORE_INDEX, StanfordSegmentationDataset
from src.task3_seg.transforms import build_segmentation_transform
from src.task3_seg.unet import UNet


DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "stanford_background"
DEFAULT_LOG_FILE = REPO_ROOT / "logs" / "smoke" / "t3_unet_smoke.log"
CLASS_COUNT = len(CLASS_NAMES)
ALLOWED_MASK_VALUES = set(range(CLASS_COUNT)) | {IGNORE_INDEX}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Stanford Background root.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Split file to read.")
    parser.add_argument("--batch-size", type=int, default=2, help="Single smoke-test batch size.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize for image and mask.")
    parser.add_argument("--num-classes", type=int, default=CLASS_COUNT, help="Semantic class count.")
    parser.add_argument("--base-channels", type=int, default=32, help="U-Net base channel width.")
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


def validate_mask_values(mask: torch.Tensor, allowed_values: set[int]) -> list[int]:
    values = sorted(int(value) for value in torch.unique(mask).cpu().tolist())
    unexpected = [value for value in values if value not in allowed_values]
    if unexpected:
        raise RuntimeError(f"Unexpected mask values: {unexpected}; observed={values}")
    return values


def attach_shape_hooks(model: UNet, shape_flow: list[tuple[str, list[int]]]) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            shape_flow.append((name, list(output.shape)))

        return hook

    for name in ("inc", "down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4", "outc"):
        handles.append(getattr(model, name).register_forward_hook(make_hook(name)))
    return handles


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
        log.write(f"base_channels={args.base_channels}\n")
        log.write(f"ignore_index={args.ignore_index}\n")
        log.flush()

        try:
            with redirect_stdout(log), redirect_stderr(log):
                torch.manual_seed(42)
                device = choose_device(args.device)
                transform = build_segmentation_transform(args.image_size, train=False)
                dataset = StanfordSegmentationDataset(args.data_root, args.split, transform=transform)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                images, masks = next(iter(loader))
                observed_mask_values = validate_mask_values(masks, set(range(args.num_classes)) | {args.ignore_index})

                model = UNet(num_classes=args.num_classes, base_channels=args.base_channels).to(device)
                criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
                shape_flow: list[tuple[str, list[int]]] = [("input", list(images.shape))]
                hooks = attach_shape_hooks(model, shape_flow)

                images = images.to(device)
                masks = masks.to(device)
                try:
                    with torch.no_grad():
                        logits = model(images)
                        loss = criterion(logits, masks)
                finally:
                    for hook in hooks:
                        hook.remove()

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
                for name, shape in shape_flow:
                    log.write(f"- {name}: {shape}\n")
                log.write("PASS unet_forward_shape_and_loss\n")
        except Exception as exc:
            log.write(f"FAIL unet smoke failed: {exc}\n")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
