"""Compare Task 3 U-Net checkpoints trained with three different losses.

Generates Table 3 (loss summary + class-wise IoU) and Fig 5 (2x2 loss curves)
referenced in the report. Re-evaluates every ``best.pt`` on the validation
set so that the published numbers are the *global* mIoU rather than an
average of per-batch mIoUs.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


VARIANT_LOG: dict[str, str] = {
    "ce": "task3_unet_ce_e100.log",
    "dice": "task3_unet_dice.log",
    "ce_dice": "task3_unet_ce_dice.log",
}
VARIANT_LABEL: dict[str, str] = {"ce": "CE", "dice": "Dice", "ce_dice": "CE+Dice"}
VARIANT_CHECKPOINT_DIR: dict[str, str] = {
    "ce": "unet_ce",
    "dice": "unet_dice",
    "ce_dice": "unet_ce_dice",
}
EPOCH_LINE_RE = re.compile(r"epoch=(\d+)\s+metrics=(\{[^}]*\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="ce,dice,ce_dice")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "stanford_background")
    parser.add_argument("--checkpoints-root", type=Path, default=REPO_ROOT / "checkpoints" / "task3")
    parser.add_argument("--logs-root", type=Path, default=REPO_ROOT / "logs" / "swanlab")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "report")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast if CUDA is unavailable instead of falling back to CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--bilinear", action="store_true")
    return parser.parse_args()


def parse_epoch_log(log_path: Path) -> list[dict]:
    """Extract per-epoch metrics from a training log file."""

    series: list[dict] = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        match = EPOCH_LINE_RE.search(line)
        if not match:
            continue
        epoch = int(match.group(1))
        try:
            metrics = ast.literal_eval(match.group(2))
        except (ValueError, SyntaxError):
            continue
        if not isinstance(metrics, dict):
            continue
        row = {"epoch": epoch}
        row.update({str(k): float(v) for k, v in metrics.items()})
        series.append(row)
    series.sort(key=lambda r: r["epoch"])
    return series


def resolve_device(device_str: str, *, require_cuda: bool = False) -> str:
    """Same shape as ``count_video.resolve_device`` — keeps CPU fallback safe."""

    device_str = (device_str or "cpu").strip()
    lowered = device_str.lower()
    if lowered in {"cpu", "cuda", "mps"}:
        return lowered
    if lowered.startswith("cuda:"):
        return lowered
    try:
        import torch
    except ImportError:
        if require_cuda:
            raise RuntimeError("PyTorch is not installed, cannot use CUDA.")
        return "cpu"
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("CUDA unavailable but --require-cuda was set.")
        print("[eval_compare] CUDA unavailable; falling back to CPU.")
        return "cpu"
    try:
        first_index = int(device_str.split(",")[0])
    except (ValueError, IndexError):
        first_index = 0
    if torch.cuda.device_count() <= first_index:
        if require_cuda:
            raise RuntimeError(
                f"Invalid CUDA device '{device_str}' (count={torch.cuda.device_count()})."
            )
        print(f"[eval_compare] Invalid CUDA device {device_str}; falling back to CPU.")
        return "cpu"
    return f"cuda:{first_index}"


def build_val_loader(args, transforms_mod, dataset_cls):
    import torch
    from torch.utils.data import DataLoader

    val_transform = transforms_mod.build_segmentation_transform(args.image_size, train=False)
    dataset = dataset_cls(
        root=args.data_root,
        split="val",
        transform=val_transform,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def evaluate_checkpoint(
    weights: Path,
    val_loader,
    device: str,
    *,
    num_classes: int,
    ignore_index: int,
    base_channels: int,
    bilinear: bool,
) -> dict:
    """Evaluate a single best.pt; returns mIoU/class_iou/pixel_acc/best_epoch/best_score."""

    import torch

    from src.common.checkpoint import load_checkpoint
    from src.common.metrics import intersection_and_union, pixel_accuracy
    from src.task3_seg.unet import UNet

    model = UNet(num_classes=num_classes, base_channels=base_channels, bilinear=bilinear)
    checkpoint = load_checkpoint(weights, model=model, map_location="cpu", strict=True)
    model.eval().to(device)

    total_inter = torch.zeros(num_classes, dtype=torch.long)
    total_union = torch.zeros(num_classes, dtype=torch.long)
    correct = 0
    total_valid = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            inter, union = intersection_and_union(
                preds, masks, num_classes=num_classes, ignore_index=ignore_index
            )
            total_inter += inter.cpu()
            total_union += union.cpu()
            valid_mask = masks != ignore_index
            correct += int((preds == masks)[valid_mask].sum().item())
            total_valid += int(valid_mask.sum().item())

    class_iou: list[float | None] = []
    valid_values: list[float] = []
    for inter, denom in zip(total_inter.tolist(), total_union.tolist()):
        if denom == 0:
            class_iou.append(None)
            continue
        value = float(inter) / float(denom)
        class_iou.append(value)
        valid_values.append(value)
    miou = sum(valid_values) / len(valid_values) if valid_values else 0.0
    pixel_acc = (correct / total_valid) if total_valid > 0 else 0.0

    return {
        "miou": float(miou),
        "class_iou": class_iou,
        "valid_classes": len(valid_values),
        "pixel_acc": float(pixel_acc),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "best_score": float(checkpoint.get("best_score") or 0.0),
        "checkpoint_metrics": {
            k: float(v) for k, v in (checkpoint.get("metrics") or {}).items() if isinstance(v, (int, float))
        },
    }


def write_loss_table(
    results: "OrderedDict[str, dict]",
    series: "OrderedDict[str, list[dict]]",
    path: Path,
) -> None:
    """Write report/tables/task3_loss_comparison.md."""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| Loss | Best Epoch | Best Val mIoU (training) | Eval mIoU (val) "
        "| Eval Pixel Acc | Final Train Loss | Final Val Loss |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = []
    for variant, eval_result in results.items():
        last_row = series[variant][-1] if series[variant] else {}
        rows.append(
            "| {label} | {best_epoch} | {best_score:.4f} | {eval_miou:.4f} | "
            "{pixel_acc:.4f} | {final_train_loss:.4f} | {final_val_loss:.4f} |".format(
                label=VARIANT_LABEL[variant],
                best_epoch=eval_result.get("best_epoch", -1),
                best_score=eval_result.get("best_score", 0.0),
                eval_miou=eval_result.get("miou", 0.0),
                pixel_acc=eval_result.get("pixel_acc", 0.0),
                final_train_loss=float(last_row.get("train_loss", 0.0)),
                final_val_loss=float(last_row.get("val_loss", 0.0)),
            )
        )
    body = (
        "# Task 3 — Three-Loss Comparison Summary\n\n"
        "Validation re-evaluation done with `python -m src.task3_seg.eval_compare`, "
        "after loading `checkpoints/task3/unet_{ce,dice,ce_dice}/best.pt`. "
        "Eval mIoU is computed by accumulating per-class intersection/union over the "
        "full validation split (143 images), then averaging IoU across classes — "
        "this is the global (not per-batch-mean) mIoU.\n\n"
        + header
        + "\n".join(rows)
        + "\n"
    )
    path.write_text(body, encoding="utf-8")


def write_class_iou_table(
    results: "OrderedDict[str, dict]",
    class_names: list[str],
    path: Path,
) -> None:
    """Write report/tables/task3_class_iou.md."""

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [VARIANT_LABEL[v] for v in results.keys()]
    header = "| Class | " + " | ".join(columns) + " |\n"
    align = "| --- " + "| ---: " * len(columns) + "|\n"
    rows = []
    for idx, name in enumerate(class_names):
        cells = []
        for variant in results.keys():
            value = results[variant]["class_iou"][idx]
            cells.append(f"{value:.4f}" if value is not None else "—")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    rows.append("| **mIoU** | " + " | ".join(
        f"**{results[v]['miou']:.4f}**" for v in results.keys()
    ) + " |")
    body = (
        "# Task 3 — Class-wise IoU\n\n"
        "Per-class IoU on the validation split (143 images, 8 classes). "
        "Best.pt for each loss variant loaded; argmax predictions are accumulated "
        "into a confusion matrix and IoU = TP / (TP + FP + FN) per class.\n\n"
        + header
        + align
        + "\n".join(rows)
        + "\n"
    )
    path.write_text(body, encoding="utf-8")


def plot_curves(series: "OrderedDict[str, list[dict]]", path: Path) -> None:
    """Save a 2x2 grid: train_loss / val_loss / val_miou / val_pixel_acc."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("train_loss", "Train Loss", "loss"),
        ("val_loss", "Val Loss", "loss"),
        ("val_miou", "Val mIoU", "mIoU"),
        ("val_pixel_acc", "Val Pixel Acc", "pixel accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    palette = {"ce": "#1f77b4", "dice": "#ff7f0e", "ce_dice": "#2ca02c"}
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        for variant, rows in series.items():
            if not rows or key not in rows[0]:
                continue
            xs = [row["epoch"] for row in rows]
            ys = [row[key] for row in rows]
            ax.plot(
                xs, ys,
                label=VARIANT_LABEL[variant],
                color=palette.get(variant),
                linewidth=1.4,
            )
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Task 3 — U-Net training curves under three losses", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def gather_environment(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {"device_requested": device}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    except ImportError:
        info["torch"] = "missing"
    return info


def main() -> int:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANT_LOG:
            raise SystemExit(f"Unknown variant {v!r}; expected one of {sorted(VARIANT_LOG)}.")

    from src.task3_seg import datasets as datasets_mod
    from src.task3_seg import transforms as transforms_mod

    device = resolve_device(args.device, require_cuda=args.require_cuda)

    print(f"[eval_compare] variants={variants} device={device} data_root={args.data_root}")

    series: "OrderedDict[str, list[dict]]" = OrderedDict()
    for variant in variants:
        log_path = args.logs_root / VARIANT_LOG[variant]
        rows = parse_epoch_log(log_path)
        series[variant] = rows
        print(f"[eval_compare] {variant}: {len(rows)} epoch rows from {log_path}")
        if not rows:
            raise SystemExit(f"No epoch rows parsed from {log_path}.")

    dataset, loader = build_val_loader(args, transforms_mod, datasets_mod.StanfordSegmentationDataset)
    print(f"[eval_compare] val dataset: {len(dataset)} samples (image_size={args.image_size})")

    results: "OrderedDict[str, dict]" = OrderedDict()
    wallclock_start = time.perf_counter()
    for variant in variants:
        weights = (args.checkpoints_root / VARIANT_CHECKPOINT_DIR[variant] / "best.pt").resolve()
        if not weights.is_file():
            raise SystemExit(f"Missing checkpoint: {weights}")
        t0 = time.perf_counter()
        result = evaluate_checkpoint(
            weights, loader, device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            base_channels=args.base_channels,
            bilinear=args.bilinear,
        )
        try:
            result["weights_path"] = str(weights.relative_to(REPO_ROOT).as_posix())
        except ValueError:
            result["weights_path"] = str(weights.as_posix())
        result["eval_seconds"] = time.perf_counter() - t0
        results[variant] = result
        print(
            f"[eval_compare] {variant}: eval miou={result['miou']:.4f} "
            f"pixel_acc={result['pixel_acc']:.4f} "
            f"best_epoch={result['best_epoch']} ({result['eval_seconds']:.1f}s)"
        )

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures" / "task3"
    write_loss_table(results, series, tables_dir / "task3_loss_comparison.md")
    write_class_iou_table(results, datasets_mod.CLASS_NAMES, tables_dir / "task3_class_iou.md")
    plot_curves(series, figures_dir / "fig5_loss_curves.png")

    payload: dict[str, Any] = {
        "variants": variants,
        "labels": {v: VARIANT_LABEL[v] for v in variants},
        "results": {
            v: {
                "miou": results[v]["miou"],
                "class_iou": results[v]["class_iou"],
                "valid_classes": results[v]["valid_classes"],
                "pixel_acc": results[v]["pixel_acc"],
                "best_epoch": results[v]["best_epoch"],
                "best_score": results[v]["best_score"],
                "checkpoint_metrics": results[v]["checkpoint_metrics"],
                "weights_path": results[v]["weights_path"],
                "eval_seconds": results[v]["eval_seconds"],
            }
            for v in variants
        },
        "class_names": datasets_mod.CLASS_NAMES,
        "image_size": args.image_size,
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
        "val_samples": len(dataset),
        "wallclock_sec": time.perf_counter() - wallclock_start,
        "environment": gather_environment(args.device),
    }
    (tables_dir / "task3_eval_compare.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[eval_compare] wrote {tables_dir / 'task3_loss_comparison.md'}")
    print(f"[eval_compare] wrote {tables_dir / 'task3_class_iou.md'}")
    print(f"[eval_compare] wrote {figures_dir / 'fig5_loss_curves.png'}")
    print(f"[eval_compare] wrote {tables_dir / 'task3_eval_compare.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
