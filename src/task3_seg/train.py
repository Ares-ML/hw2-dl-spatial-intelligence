"""Train Task 3 Stanford Background U-Net with SwanLab logging."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from src.common.checkpoint import CheckpointManager
from src.common.logger import setup_logging
from src.common.metrics import mean_iou, pixel_accuracy
from src.common.seed import make_generator, seed_worker, set_seed
from src.common.swanlab_logger import finish, format_run_name, init_run, log_metrics
from src.task3_seg.datasets import IGNORE_INDEX, StanfordSegmentationDataset
from src.task3_seg.losses import build_segmentation_loss
from src.task3_seg.transforms import build_segmentation_transform
from src.task3_seg.unet import UNet


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "task3_unet_ce.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--swanlab-mode", default="disabled", choices=["cloud", "offline", "local", "disabled"])
    parser.add_argument("--require-swanlab", action="store_true")
    parser.add_argument("--links-file", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "logs" / "swanlab" / "task3_train.log")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
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


def swanlab_tags(cfg: dict[str, Any], default: list[str]) -> list[str]:
    swan_cfg = cfg.get("swanlab")
    tags = swan_cfg.get("tags") if isinstance(swan_cfg, dict) else None
    tags = tags or cfg.get("tags")
    if tags is None:
        return default
    if isinstance(tags, str):
        return [tags]
    return [str(tag) for tag in tags]


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
    horizontal_flip_prob = float(cfg.get("horizontal_flip_prob", cfg.get("hflip_prob", 0.5)))
    train_transform = build_segmentation_transform(image_size, train=True, horizontal_flip_prob=horizontal_flip_prob)
    val_transform = build_segmentation_transform(image_size, train=False)
    train_set = StanfordSegmentationDataset(root, "train", transform=train_transform)
    val_set = StanfordSegmentationDataset(root, "val", transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=int(cfg.get("num_workers", 4)), generator=make_generator(seed), worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=int(cfg.get("num_workers", 4)))

    num_classes = int(cfg.get("num_classes", 8))
    ignore_index = int(cfg.get("ignore_index", IGNORE_INDEX))
    base_channels = int(cfg.get("base_channels", 32))
    bilinear = bool(cfg.get("bilinear", False))
    model = UNet(num_classes=num_classes, base_channels=base_channels, bilinear=bilinear).to(device)
    criterion = build_segmentation_loss(
        str(cfg.get("loss", "ce")),
        num_classes=num_classes,
        ignore_index=ignore_index,
        dice_weight=float(cfg.get("dice_weight", 1.0)),
        smooth=float(cfg.get("dice_smooth", 1.0)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    run_values = {**cfg, "epochs": epochs, "seed": seed, "lr": lr}
    run_name = format_run_name(run_values)
    checkpoint_dir = args.checkpoint_dir or (REPO_ROOT / "checkpoints" / "task3" / f"unet_{cfg.get('loss', 'ce')}")
    checkpoint_manager = CheckpointManager(checkpoint_dir, monitor="val_miou", mode="max")
    logger.info("checkpoint_dir=%s", checkpoint_dir)
    run = init_run(
        task="task3",
        run_name=run_name,
        config=run_values,
        mode=args.swanlab_mode,
        tags=swanlab_tags(cfg, ["segmentation"]),
    )
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
        checkpoint_result = checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=run_values,
            extra={"run_name": run_name},
        )
        logger.info(
            "epoch=%s checkpoint last=%s best=%s is_best=%s",
            epoch,
            checkpoint_result["last"],
            checkpoint_result["best"],
            checkpoint_result["is_best"],
        )

    run = finish()
    append_link(args.links_file, "task3", run_name, run.project_url, run.experiment_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
