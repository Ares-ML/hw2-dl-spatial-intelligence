"""Pick representative validation samples and render an image / GT / pred grid.

Produces ``report/figures/task3/mask_comparison.png`` — a 6×5 grid showing,
for each of 6 samples (chosen from CE-model per-image mIoU buckets: best /
median / median / hard / hard / worst), columns ``Image | GT | pred-CE |
pred-Dice | pred-CE+Dice``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


VARIANT_LABEL: dict[str, str] = {"ce": "CE", "dice": "Dice", "ce_dice": "CE+Dice"}
VARIANT_CHECKPOINT_DIR: dict[str, str] = {
    "ce": "unet_ce",
    "dice": "unet_dice",
    "ce_dice": "unet_ce_dice",
}
DEFAULT_BUCKETS = {"best": 1, "median": 2, "hard": 2, "worst": 1}
BUCKET_ORDER = ("best", "median", "hard", "worst")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="ce,dice,ce_dice")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "stanford_background")
    parser.add_argument("--checkpoints-root", type=Path, default=REPO_ROOT / "checkpoints" / "task3")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "report" / "figures" / "task3" / "mask_comparison.png",
    )
    parser.add_argument(
        "--samples-json",
        type=Path,
        default=REPO_ROOT / "report" / "figures" / "task3" / "mask_comparison_samples.json",
    )
    parser.add_argument(
        "--bucket-counts",
        default="best=1,median=2,hard=2,worst=1",
        help="Comma-separated counts per bucket.",
    )
    parser.add_argument("--scoring-variant", default="ce")
    parser.add_argument("--device", default="0")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--bilinear", action="store_true")
    return parser.parse_args()


def parse_bucket_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"--bucket-counts items must be 'name=int'; got {piece!r}.")
        name, num = piece.split("=", 1)
        name = name.strip().lower()
        if name not in BUCKET_ORDER:
            raise ValueError(f"Unknown bucket {name!r}; expected one of {BUCKET_ORDER}.")
        counts[name] = int(num)
    for name in BUCKET_ORDER:
        counts.setdefault(name, 0)
    if sum(counts.values()) == 0:
        raise ValueError("At least one bucket must request samples.")
    return counts


def pick_buckets(
    scored: list[tuple[str, float]],
    counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Split a list sorted ascending by score into best/median/hard/worst buckets.

    ``scored`` is expected to be ascending (worst first, best last). Returns a
    selection in presentation order: best → median → hard → worst, with each
    chosen sample carrying its ``(sample_id, score, bucket)`` triple. Dedupes
    by sample_id while honouring bucket sizes — useful for small datasets.
    """

    if not scored:
        return []
    n = len(scored)
    if any(c < 0 for c in counts.values()):
        raise ValueError("bucket counts must be non-negative.")

    desired = {
        "best": min(counts.get("best", 0), n),
        "worst": min(counts.get("worst", 0), n),
        "median": min(counts.get("median", 0), n),
        "hard": min(counts.get("hard", 0), n),
    }

    used: set[str] = set()
    chosen: list[dict[str, Any]] = []

    def take(index: int, bucket: str) -> None:
        if 0 <= index < n:
            sample_id, score = scored[index]
            if sample_id not in used:
                used.add(sample_id)
                chosen.append({"sample_id": sample_id, "score": float(score), "bucket": bucket})

    best_pick = []
    for offset in range(desired["best"]):
        best_pick.append(n - 1 - offset)
    median_anchor = n // 2
    median_pick = []
    half_lo = desired["median"] // 2
    half_hi = desired["median"] - half_lo
    for offset in range(half_lo):
        median_pick.append(median_anchor - offset - 1)
    for offset in range(half_hi):
        median_pick.append(median_anchor + offset)
    hard_anchor = max(0, n // 4)
    hard_pick = []
    for offset in range(desired["hard"]):
        hard_pick.append(hard_anchor + offset)
    worst_pick = list(range(desired["worst"]))

    for idx in best_pick:
        take(idx, "best")
    for idx in median_pick:
        take(idx, "median")
    for idx in hard_pick:
        take(idx, "hard")
    for idx in worst_pick:
        take(idx, "worst")

    bucket_rank = {name: rank for rank, name in enumerate(BUCKET_ORDER)}
    chosen.sort(key=lambda item: (bucket_rank[item["bucket"]], -item["score"]))
    return chosen


def resolve_device(device_str: str, *, require_cuda: bool = False) -> str:
    device_str = (device_str or "cpu").strip()
    if device_str.lower() in {"cpu", "cuda", "mps"}:
        return device_str.lower()
    try:
        import torch
    except ImportError:
        if require_cuda:
            raise RuntimeError("PyTorch is not installed, cannot use CUDA.")
        return "cpu"
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("CUDA unavailable but --require-cuda was set.")
        print("[visualize_masks] CUDA unavailable; falling back to CPU.")
        return "cpu"
    try:
        first_index = int(device_str.split(",")[0])
    except (ValueError, IndexError):
        first_index = 0
    if torch.cuda.device_count() <= first_index:
        if require_cuda:
            raise RuntimeError(f"Invalid CUDA device '{device_str}'.")
        return "cpu"
    return device_str


def load_unet(weights: Path, *, num_classes: int, base_channels: int, bilinear: bool, device: str):
    import torch  # noqa: F401

    from src.common.checkpoint import load_checkpoint
    from src.task3_seg.unet import UNet

    model = UNet(num_classes=num_classes, base_channels=base_channels, bilinear=bilinear)
    load_checkpoint(weights, model=model, map_location="cpu", strict=True)
    model.eval().to(device)
    return model


def per_image_miou(
    model, dataset, device: str,
    *, num_classes: int, ignore_index: int,
) -> list[tuple[str, float]]:
    """Compute per-image mIoU on the dataset, return ``(sample_id, miou)`` list."""

    import torch

    from src.common.metrics import mean_iou

    results: list[tuple[str, float]] = []
    with torch.no_grad():
        for index, (image, mask) in enumerate(dataset):
            sample_id = dataset.ids[index]
            image_batch = image.unsqueeze(0).to(device)
            mask_batch = mask.unsqueeze(0).to(device)
            logits = model(image_batch)
            preds = logits.argmax(dim=1)
            stats = mean_iou(preds, mask_batch, num_classes=num_classes, ignore_index=ignore_index)
            results.append((sample_id, float(stats["miou"])))
    results.sort(key=lambda item: item[1])
    return results


def render_grid(
    samples: list[dict[str, Any]],
    variants: list[str],
    palette,
    output_path: Path,
) -> None:
    """Save the 6×5 PNG using matplotlib."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.task3_seg.colorize import colorize_mask

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(samples)
    cols = 2 + len(variants)  # image, GT, then one per variant
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), dpi=120)
    if rows == 1:
        axes = axes.reshape(1, -1)

    column_titles = ["Image", "GT"] + [f"pred-{VARIANT_LABEL[v]}" for v in variants]

    for r, sample in enumerate(samples):
        for c in range(cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        # Column 0 — RGB image
        axes[r, 0].imshow(sample["image_np"])
        # Column 1 — GT colorized
        axes[r, 1].imshow(colorize_mask(sample["gt_np"], palette))
        # Columns 2..N — per-variant predictions with mIoU caption
        for ci, variant in enumerate(variants):
            ax = axes[r, 2 + ci]
            ax.imshow(colorize_mask(sample["preds"][variant], palette))
            miou_value = sample["per_variant_miou"].get(variant)
            if miou_value is not None:
                ax.set_title(f"mIoU={miou_value:.3f}", fontsize=9)

        bucket = sample["bucket"]
        scoring_miou = sample.get("score")
        row_label = f"{bucket}\n{sample['sample_id']}"
        if scoring_miou is not None:
            row_label += f"\nCE mIoU={scoring_miou:.3f}"
        axes[r, 0].set_ylabel(row_label, fontsize=9, rotation=0, labelpad=42, va="center")

    for c, title in enumerate(column_titles):
        axes[0, c].set_title(title if c < 2 else (axes[0, c].get_title() or title), fontsize=10)
        # Column 0/1 titles always set; column 2+ already had per-cell mIoU titles
        if c < 2:
            axes[0, c].set_title(title, fontsize=10)

    fig.suptitle("Task 3 — Mask comparison across three losses", fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def tensor_to_uint8_image(image_tensor) -> "Any":
    import numpy as np

    array = image_tensor.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    return array


def main() -> int:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.scoring_variant not in variants:
        raise SystemExit(
            f"--scoring-variant {args.scoring_variant!r} must be one of --variants {variants}."
        )
    counts = parse_bucket_counts(args.bucket_counts)

    device = resolve_device(args.device, require_cuda=args.require_cuda)

    from src.task3_seg.datasets import PALETTE, StanfordSegmentationDataset
    from src.task3_seg.transforms import build_segmentation_transform

    val_transform = build_segmentation_transform(args.image_size, train=False)
    dataset = StanfordSegmentationDataset(
        root=args.data_root,
        split="val",
        transform=val_transform,
        image_size=args.image_size,
    )
    print(f"[visualize_masks] val dataset: {len(dataset)} samples device={device}")

    scoring_weights = args.checkpoints_root / VARIANT_CHECKPOINT_DIR[args.scoring_variant] / "best.pt"
    scoring_model = load_unet(
        scoring_weights,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        bilinear=args.bilinear,
        device=device,
    )

    t0 = time.perf_counter()
    scored = per_image_miou(
        scoring_model, dataset, device,
        num_classes=args.num_classes, ignore_index=args.ignore_index,
    )
    print(f"[visualize_masks] {args.scoring_variant} per-image mIoU: "
          f"min={scored[0][1]:.3f} median={scored[len(scored) // 2][1]:.3f} "
          f"max={scored[-1][1]:.3f} ({time.perf_counter() - t0:.1f}s)")

    selection = pick_buckets(scored, counts)
    sample_ids = [s["sample_id"] for s in selection]
    print(f"[visualize_masks] selected {len(selection)} samples: " +
          ", ".join(f"{s['bucket']}={s['sample_id']}({s['score']:.3f})" for s in selection))

    index_of: dict[str, int] = {sid: i for i, sid in enumerate(dataset.ids)}
    samples: list[dict[str, Any]] = []
    import torch
    import numpy as np

    for entry in selection:
        idx = index_of[entry["sample_id"]]
        image_tensor, mask_tensor = dataset[idx]
        samples.append({
            "sample_id": entry["sample_id"],
            "bucket": entry["bucket"],
            "score": entry["score"],
            "image_tensor": image_tensor,
            "mask_tensor": mask_tensor,
            "image_np": tensor_to_uint8_image(image_tensor),
            "gt_np": mask_tensor.detach().cpu().numpy().astype(np.int64),
            "preds": {},
            "per_variant_miou": {},
        })

    from src.common.metrics import mean_iou

    for variant in variants:
        weights = args.checkpoints_root / VARIANT_CHECKPOINT_DIR[variant] / "best.pt"
        if not weights.is_file():
            raise SystemExit(f"Missing checkpoint: {weights}")
        if variant == args.scoring_variant:
            model = scoring_model
        else:
            model = load_unet(
                weights,
                num_classes=args.num_classes,
                base_channels=args.base_channels,
                bilinear=args.bilinear,
                device=device,
            )
        with torch.no_grad():
            for sample in samples:
                image_batch = sample["image_tensor"].unsqueeze(0).to(device)
                mask_batch = sample["mask_tensor"].unsqueeze(0).to(device)
                logits = model(image_batch)
                pred = logits.argmax(dim=1)
                stats = mean_iou(
                    pred, mask_batch,
                    num_classes=args.num_classes, ignore_index=args.ignore_index,
                )
                sample["preds"][variant] = pred.squeeze(0).detach().cpu().numpy().astype(np.int64)
                sample["per_variant_miou"][variant] = float(stats["miou"])

    render_grid(samples, variants, PALETTE, args.output_path)
    print(f"[visualize_masks] wrote {args.output_path}")

    args.samples_json.parent.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for sample in samples:
        summary.append({
            "sample_id": sample["sample_id"],
            "bucket": sample["bucket"],
            "scoring_variant": args.scoring_variant,
            "scoring_miou": sample["score"],
            "per_variant_miou": {v: sample["per_variant_miou"][v] for v in variants},
        })
    payload = {
        "samples": summary,
        "variants": variants,
        "scoring_variant": args.scoring_variant,
        "bucket_counts": counts,
        "image_size": args.image_size,
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
        "val_samples": len(dataset),
        "environment": {
            "device_requested": args.device,
            "device_resolved": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    args.samples_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[visualize_masks] wrote {args.samples_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
