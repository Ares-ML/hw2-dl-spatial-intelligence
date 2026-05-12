"""Prepare the Flower102 classification dataset and a class check image."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "flower102"
SPLITS = ("train", "val", "test")
NUM_CLASSES = 102


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Flower102 output root.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    parser.add_argument("--samples", type=int, default=12, help="Number of samples in class_check.png.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Validate an existing TorchVision Flower102 cache without downloading.",
    )
    return parser.parse_args()


def labels_for_dataset(dataset: Any) -> list[int]:
    labels = getattr(dataset, "_labels", None)
    if labels is not None and len(labels) == len(dataset):
        return [int(label) for label in labels]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def fit_image(image: Any, size: tuple[int, int]) -> Any:
    from PIL import Image

    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def draw_bar_chart(draw: Any, counts: Counter[int], box: tuple[int, int, int, int], font: Any) -> None:
    x0, y0, x1, y1 = box
    chart_w = x1 - x0
    chart_h = y1 - y0
    max_count = max(counts.values()) if counts else 1
    bar_gap = 2
    bar_w = max(2, (chart_w - bar_gap * (NUM_CLASSES - 1)) // NUM_CLASSES)

    draw.rectangle(box, outline=(180, 180, 180))
    for class_id in range(NUM_CLASSES):
        value = counts.get(class_id, 0)
        height = int((value / max_count) * (chart_h - 24))
        left = x0 + class_id * (bar_w + bar_gap)
        top = y1 - 18 - height
        color = (54, 119, 195) if value else (210, 210, 210)
        draw.rectangle((left, top, left + bar_w, y1 - 18), fill=color)
        if class_id % 10 == 0:
            draw.text((left, y1 - 15), str(class_id), fill=(60, 60, 60), font=font)

    draw.text((x0 + 8, y0 + 6), f"Flower102 class distribution, max={max_count}", fill=(20, 20, 20), font=font)


def make_check_image(
    datasets: dict[str, Any],
    split_labels: dict[str, list[int]],
    output_path: Path,
    seed: int,
    samples: int,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(seed)
    font = ImageFont.load_default()
    total_counts: Counter[int] = Counter()
    sample_pool: list[tuple[str, Any, int, int]] = []

    for split, labels in split_labels.items():
        total_counts.update(labels)
        for index, label in enumerate(labels):
            sample_pool.append((split, datasets[split], index, label))

    sample_pool = rng.sample(sample_pool, k=min(samples, len(sample_pool))) if sample_pool else []

    width = 1400
    thumb_w, thumb_h = 160, 130
    label_h = 34
    margin = 24
    gap = 12
    cols = max(1, min(6, width // (thumb_w + gap)))
    rows = (len(sample_pool) + cols - 1) // cols
    height = 320 + max(1, rows) * (thumb_h + label_h + gap) + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = "Flower102 dataset check"
    title_w, _ = text_size(draw, title, font)
    draw.text(((width - title_w) // 2, 16), title, fill=(20, 20, 20), font=font)

    summary = " | ".join(
        f"{split}: {len(split_labels[split])} images, {len(set(split_labels[split]))} classes"
        for split in SPLITS
    )
    draw.text((margin, 44), summary, fill=(60, 60, 60), font=font)
    draw_bar_chart(draw, total_counts, (margin, 76, width - margin, 280), font)

    top = 306
    for item_index, (split, dataset, index, label) in enumerate(sample_pool):
        row = item_index // cols
        col = item_index % cols
        left = margin + col * (thumb_w + gap)
        item_top = top + row * (thumb_h + label_h + gap)
        image, _ = dataset[index]
        canvas.paste(fit_image(image, (thumb_w, thumb_h)), (left, item_top))
        draw.rectangle((left, item_top, left + thumb_w, item_top + thumb_h), outline=(190, 190, 190))
        draw.text(
            (left + 4, item_top + thumb_h + 4),
            f"{split} | class_{label:03d}",
            fill=(30, 30, 30),
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    try:
        from torchvision.datasets import Flowers102
    except Exception as exc:  # pragma: no cover - environment smoke guard
        raise SystemExit(f"Failed to import torchvision.datasets.Flowers102: {exc}") from exc

    datasets: dict[str, Any] = {}
    split_labels: dict[str, list[int]] = {}
    for split in SPLITS:
        dataset = Flowers102(root=str(args.root), split=split, download=not args.skip_download)
        labels = labels_for_dataset(dataset)
        datasets[split] = dataset
        split_labels[split] = labels

    print("Flower102 summary")
    for split in SPLITS:
        counts = Counter(split_labels[split])
        print(f"- {split}: images={len(split_labels[split])}, classes={len(counts)}")
        if len(counts) != NUM_CLASSES:
            print(f"  [WARN] expected {NUM_CLASSES} classes, found {len(counts)}")

    check_path = args.root / "checks" / "class_check.png"
    make_check_image(datasets, split_labels, check_path, seed=args.seed, samples=args.samples)
    print(f"Wrote {check_path}")


if __name__ == "__main__":
    main()
