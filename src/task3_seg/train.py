"""Train Task 3 Stanford Background U-Net with SwanLab logging."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.common.logger import setup_logging
from src.common.metrics import mean_iou, pixel_accuracy
from src.common.seed import make_generator, seed_worker, set_seed
from src.common.swanlab_logger import finish, format_run_name, init_run, log_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "task3_unet_ce.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--swanlab-mode", default="disabled", choices=["cloud", "offline", "local", "disabled"])
    parser.add_argument("--require-swanlab", action="store_true")
    parser.add_argument("--links-file", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "logs" / "swanlab" / "task3_train.log")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; source scripts/cuda_driver_shim.sh on the server.")
    return torch.device(requested)


def append_link(path: Path | None, task: str, run_name: str, project_url: str, experiment_url: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"| {task} | {run_name} | {project_url} | {experiment_url} |\n")


class StanfordSegmentationDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: int) -> None:
        self.root = root
        self.image_dir = root / "iccv09Data" / "images"
        self.mask_dir = root / "masks"
        split_path = root / "splits" / f"{split}.txt"
        self.ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.ids[index]
        with Image.open(self.image_dir / f"{sample_id}.jpg") as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            image_arr = np.asarray(image, dtype=np.float32) / 255.0
        with Image.open(self.mask_dir / f"{sample_id}.png") as mask:
            mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            mask_arr = np.asarray(mask, dtype=np.int64)
        return torch.from_numpy(image_arr).permute(2, 0, 1), torch.from_numpy(mask_arr)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self.dec3(torch.cat([self.up3(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        return self.out(dec1)


def iter_limit(loader: DataLoader, limit: int | None):
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield batch


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    epochs = args.epochs or int(cfg.get("epochs", 1))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    lr = args.lr if args.lr is not None else float(cfg.get("lr", 1e-3))
    set_seed(seed)
    device = choose_device(args.device)
    logger = setup_logging("hw2.task3", args.log_file, level=logging.INFO, file_mode="w")

    root = Path(cfg.get("data_root", "data/stanford_background"))
    image_size = int(cfg.get("image_size", 256))
    batch_size = int(cfg.get("batch_size", 4))
    train_set = StanfordSegmentationDataset(root, "train", image_size)
    val_set = StanfordSegmentationDataset(root, "val", image_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=int(cfg.get("num_workers", 4)), generator=make_generator(seed), worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=int(cfg.get("num_workers", 4)))

    num_classes = int(cfg.get("num_classes", 8))
    ignore_index = int(cfg.get("ignore_index", 255))
    model = UNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    run_values = {**cfg, "epochs": epochs, "seed": seed, "lr": lr}
    run_name = format_run_name(run_values)
    run = init_run(task="task3", run_name=run_name, config=run_values, mode=args.swanlab_mode, tags=["day1", "smoke", "segmentation"])
    if args.require_swanlab and not run.enabled:
        raise RuntimeError(f"SwanLab cloud run was required but initialization failed: {run.error or 'no error captured'}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for images, masks in iter_limit(train_loader, args.max_batches):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        model.eval()
        val_losses: list[float] = []
        val_preds: list[torch.Tensor] = []
        val_targets: list[torch.Tensor] = []
        with torch.no_grad():
            for images, masks in iter_limit(val_loader, args.max_val_batches):
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                val_losses.append(float(criterion(logits, masks).item()))
                val_preds.append(logits.argmax(dim=1).cpu())
                val_targets.append(masks.cpu())
        pred = torch.cat(val_preds)
        target = torch.cat(val_targets)
        iou = mean_iou(pred, target, num_classes=num_classes, ignore_index=ignore_index)
        metrics = {
            "train_loss": total_loss / max(batches, 1),
            "val_loss": sum(val_losses) / max(len(val_losses), 1),
            "val_miou": float(iou["miou"]),
            "val_pixel_acc": pixel_accuracy(pred, target, num_classes=num_classes, ignore_index=ignore_index),
        }
        logger.info("epoch=%s metrics=%s", epoch, metrics)
        log_metrics(metrics, step=epoch, task="task3")

    run = finish()
    append_link(args.links_file, "task3", run_name, run.project_url, run.experiment_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
