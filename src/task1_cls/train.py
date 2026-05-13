"""Train Task 1 Flower102 classifiers with SwanLab logging."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from torchvision.models import ResNet18_Weights
from torchvision.transforms import v2

from src.common.checkpoint import CheckpointManager
from src.common.logger import setup_logging
from src.common.metrics import topk_accuracy
from src.common.seed import make_generator, seed_worker, set_seed
from src.common.swanlab_logger import finish, format_run_name, init_run, log_metrics
from src.task1_cls.models import build_resnet18_classifier


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "task1_resnet18_pretrained.yaml"


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
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "logs" / "swanlab" / "task1_train.log")
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


def iter_limit(loader: DataLoader, limit: int | None):
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield index, batch


def append_link(path: Path | None, task: str, run_name: str, project_url: str, experiment_url: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"| {task} | {run_name} | {project_url} | {experiment_url} |\n")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    epochs = args.epochs or int(cfg.get("epochs", 1))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    lr = args.lr if args.lr is not None else float(cfg.get("lr", 1e-4))
    batch_size = int(cfg.get("batch_size", 32))
    set_seed(seed)
    device = choose_device(args.device)
    logger = setup_logging("hw2.task1", args.log_file, level=logging.INFO, file_mode="w")

    norm = ResNet18_Weights.DEFAULT.transforms()
    transform = v2.Compose(
        [
            v2.Resize(int(cfg.get("image_size", 224)) + 32),
            v2.CenterCrop(int(cfg.get("image_size", 224))),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=norm.mean, std=norm.std),
        ]
    )
    train_set = Flowers102(root=str(cfg.get("data_root", "data/flower102")), split="train", transform=transform, download=False)
    val_set = Flowers102(root=str(cfg.get("data_root", "data/flower102")), split="val", transform=transform, download=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=int(cfg.get("num_workers", 4)), generator=make_generator(seed), worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=int(cfg.get("num_workers", 4)))

    model = build_resnet18_classifier(
        num_classes=int(cfg.get("num_classes", 102)),
        init=str(cfg.get("init", "pretrained")),
        attention=cfg.get("attention"),
        cbam_reduction=int(cfg.get("cbam_reduction", 16)),
        cbam_spatial_kernel_size=int(cfg.get("cbam_spatial_kernel_size", 7)),
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    run_values = {**cfg, "epochs": epochs, "seed": seed, "lr": lr}
    run_name = format_run_name(run_values)
    checkpoint_dir = args.checkpoint_dir or (
        REPO_ROOT / "checkpoints" / "task1" / f"{cfg.get('model', 'resnet18')}_{cfg.get('init', 'pretrained')}"
    )
    checkpoint_manager = CheckpointManager(checkpoint_dir, monitor="val_acc", mode="max")
    logger.info("checkpoint_dir=%s", checkpoint_dir)
    run = init_run(task="task1", run_name=run_name, config=run_values, mode=args.swanlab_mode, tags=["day1", "smoke", "classification"])
    if args.require_swanlab and not run.enabled:
        raise RuntimeError(f"SwanLab cloud run was required but initialization failed: {run.error or 'no error captured'}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        seen_batches = 0
        for _, (images, labels) in iter_limit(train_loader, args.max_batches):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            acc = topk_accuracy(logits.detach(), labels.detach(), topk=(1,))[0]
            total_loss += float(loss.item())
            total_acc += acc
            seen_batches += 1
            global_step += 1
        train_loss = total_loss / max(seen_batches, 1)
        train_acc = total_acc / max(seen_batches, 1)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for _, (images, labels) in iter_limit(val_loader, args.max_val_batches):
                logits = model(images.to(device))
                labels = labels.to(device)
                val_loss += float(criterion(logits, labels).item())
                val_acc += topk_accuracy(logits, labels, topk=(1,))[0]
                val_batches += 1
        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss / max(val_batches, 1),
            "val_acc": val_acc / max(val_batches, 1),
        }
        logger.info("epoch=%s metrics=%s", epoch, metrics)
        log_metrics(metrics, step=epoch, task="task1")
        checkpoint_result = checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=run_values,
            extra={"global_step": global_step, "run_name": run_name},
        )
        logger.info(
            "epoch=%s checkpoint last=%s best=%s is_best=%s",
            epoch,
            checkpoint_result["last"],
            checkpoint_result["best"],
            checkpoint_result["is_best"],
        )

    run = finish()
    append_link(args.links_file, "task1", run_name, run.project_url, run.experiment_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
