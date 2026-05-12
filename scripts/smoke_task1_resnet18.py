"""Smoke test Flower102 classification with a pretrained ResNet-18.

The test intentionally optimizes only the final classifier head on one
deterministic batch, then verifies that the loss on that same batch decreases.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "flower102"
DEFAULT_LOG_FILE = REPO_ROOT / "logs" / "smoke" / "t1_resnet18_smoke.log"
NUM_CLASSES = 102


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="TorchVision Flower102 root.")
    parser.add_argument("--device", default="cuda", help="Device to run on, usually cuda on the GPU server.")
    parser.add_argument("--batch-size", type=int, default=16, help="Single smoke-test batch size.")
    parser.add_argument("--lr", type=float, default=1e-1, help="Classifier-head learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic DataLoader seed.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE, help="Smoke-test log path.")
    return parser.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("t1_resnet18_smoke")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "On the course GPU server, run this through scripts/run_smoke_server.sh "
            "or source scripts/cuda_driver_shim.sh and call prepare_cuda_driver_shim first."
        )
    return torch.device(requested)


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.log_file)

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    logger.info("Task 1 ResNet-18 Flower102 smoke test")
    logger.info("repo_root=%s", REPO_ROOT)
    logger.info("data_root=%s", args.data_root.resolve())
    logger.info("device=%s", device)
    logger.info("torch=%s torchvision=%s", torch.__version__, sys.modules["torchvision"].__version__)

    weights = ResNet18_Weights.DEFAULT
    transform = v2.Compose(
        [
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )
    dataset = Flowers102(root=str(args.data_root), split="train", transform=transform, download=False)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.0)

    with torch.no_grad():
        loss_before = criterion(model(images), labels).item()

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        loss_after = criterion(model(images), labels).item()

    loss_delta = loss_before - loss_after
    logger.info("batch_shape=%s", tuple(images.shape))
    logger.info("label_min=%s label_max=%s", int(labels.min().item()), int(labels.max().item()))
    logger.info("loss_before=%.8f", loss_before)
    logger.info("loss_after=%.8f", loss_after)
    logger.info("loss_delta=%.8f", loss_delta)

    if loss_after < loss_before:
        logger.info("PASS loss_after < loss_before")
        return 0

    logger.error("FAIL loss did not decrease")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
