"""Prepare the Road Vehicle YOLO dataset and a class check image."""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "road_vehicle"
DEFAULT_SLUG = "ashfakyeafi/road-vehicle-images-dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = [
    "car",
    "bus",
    "motorbike",
    "three wheelers -CNG-",
    "rickshaw",
    "truck",
    "pickup",
    "minivan",
    "suv",
    "van",
    "bicycle",
    "auto rickshaw",
    "human hauler",
    "wheelbarrow",
    "ambulance",
    "minibus",
    "taxi",
    "army vehicle",
    "scooter",
    "policecar",
    "garbagevan",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Road Vehicle output root.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    parser.add_argument("--samples", type=int, default=12, help="Number of samples in class_check.png.")
    parser.add_argument("--skip-download", action="store_true", help="Validate existing data only.")
    parser.add_argument("--archive", type=Path, default=None, help="Manual .zip/.tar archive to extract.")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Kaggle dataset slug.")
    return parser.parse_args()


def ensure_inside(path: Path, root: Path) -> None:
    root = root.resolve()
    path = path.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Unsafe archive member path: {path}")


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive) as file:
            for member in file.namelist():
                ensure_inside(destination / member, destination)
            file.extractall(destination)
        return
    if suffixes.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive) as file:
            for member in file.getmembers():
                ensure_inside(destination / member.name, destination)
            file.extractall(destination)
        return
    raise ValueError(f"Unsupported archive type: {archive}")


def split_dirs(split_root: Path) -> tuple[Path, Path] | None:
    images = split_root / "images"
    labels = split_root / "labels"
    if images.is_dir() and labels.is_dir():
        return images, labels
    return None


def find_split(root: Path, names: set[str]) -> tuple[Path, Path]:
    candidates = [root] + [path for path in root.rglob("*") if path.is_dir()]
    for path in candidates:
        if path.name.lower() in names:
            found = split_dirs(path)
            if found is not None:
                return found
    raise FileNotFoundError(f"Could not find split {sorted(names)} under {root}")


def has_dataset(root: Path) -> bool:
    try:
        find_split(root, {"train"})
        find_split(root, {"valid", "val", "validation"})
    except FileNotFoundError:
        return False
    return True


def download_with_kaggle(root: Path, slug: str) -> None:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise SystemExit(
            "Kaggle CLI was not found. Install/configure kaggle, or download the dataset "
            "manually and rerun with --archive /path/to/road-vehicle-images-dataset.zip"
        )
    subprocess.run(
        [kaggle, "datasets", "download", "-d", slug, "-p", str(root), "--unzip"],
        check=True,
    )


def image_files(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_label_file(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            errors.append(f"{path}:{line_number}: expected at least 5 columns")
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, w, h = (float(value) for value in parts[1:5])
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric YOLO label")
            continue
        if not 0 <= class_id < len(CLASS_NAMES):
            errors.append(f"{path}:{line_number}: class id {class_id} outside 0..{len(CLASS_NAMES) - 1}")
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            errors.append(f"{path}:{line_number}: bbox values must be normalized with positive width/height")
            continue
        boxes.append((class_id, x, y, w, h))
    return boxes, errors


def validate_split(images_dir: Path, labels_dir: Path) -> dict[str, Any]:
    files = image_files(images_dir)
    counts: Counter[int] = Counter()
    missing: list[Path] = []
    empty: list[Path] = []
    errors: list[str] = []
    samples: list[tuple[Path, Path, list[tuple[int, float, float, float, float]]]] = []

    for image_path in files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing.append(image_path)
            continue
        boxes, label_errors = read_label_file(label_path)
        errors.extend(label_errors)
        if not boxes:
            empty.append(label_path)
        for box in boxes:
            counts[box[0]] += 1
        if boxes:
            samples.append((image_path, label_path, boxes))

    return {
        "images": files,
        "counts": counts,
        "missing": missing,
        "empty": empty,
        "errors": errors,
        "samples": samples,
    }


def relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_data_yaml(root: Path, train_images: Path, valid_images: Path) -> Path:
    path = root / "data.yaml"
    lines = [
        f"path: {root.resolve().as_posix()}",
        f"train: {relative_to_root(train_images, root)}",
        f"val: {relative_to_root(valid_images, root)}",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def color_for_class(class_id: int) -> tuple[int, int, int]:
    colors = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (255, 183, 3),
        (33, 158, 188),
    ]
    return colors[class_id % len(colors)]


def fit_image(image: Any, size: tuple[int, int]) -> Any:
    from PIL import Image

    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def make_check_image(
    output_path: Path,
    stats: dict[str, dict[str, Any]],
    seed: int,
    samples: int,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(seed)
    font = ImageFont.load_default()
    total_counts: Counter[int] = Counter()
    sample_pool: list[tuple[str, Path, list[tuple[int, float, float, float, float]]]] = []
    for split, split_stats in stats.items():
        total_counts.update(split_stats["counts"])
        for image_path, _, boxes in split_stats["samples"]:
            sample_pool.append((split, image_path, boxes))

    sample_pool = rng.sample(sample_pool, k=min(samples, len(sample_pool))) if sample_pool else []

    width = 1400
    margin = 24
    chart_top = 86
    chart_h = 240
    thumb_w, thumb_h = 260, 170
    label_h = 42
    gap = 14
    cols = max(1, min(4, (width - 2 * margin) // (thumb_w + gap)))
    rows = (len(sample_pool) + cols - 1) // cols
    height = 360 + max(1, rows) * (thumb_h + label_h + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "Road Vehicle dataset check", fill=(20, 20, 20), font=font)
    draw.text(
        (margin, 42),
        "YOLO class counts and sampled bounding boxes",
        fill=(70, 70, 70),
        font=font,
    )

    max_count = max(total_counts.values()) if total_counts else 1
    bar_area = (margin, chart_top, width - margin, chart_top + chart_h)
    draw.rectangle(bar_area, outline=(180, 180, 180))
    bar_w = max(12, (bar_area[2] - bar_area[0] - 8 * (len(CLASS_NAMES) - 1)) // len(CLASS_NAMES))
    for class_id, name in enumerate(CLASS_NAMES):
        value = total_counts.get(class_id, 0)
        left = bar_area[0] + class_id * (bar_w + 8)
        height_px = int((value / max_count) * (chart_h - 52))
        top = bar_area[3] - 28 - height_px
        draw.rectangle((left, top, left + bar_w, bar_area[3] - 28), fill=color_for_class(class_id))
        draw.text((left, bar_area[3] - 24), str(class_id), fill=(40, 40, 40), font=font)
        if class_id < 8:
            draw.text((left, top - 14), str(value), fill=(40, 40, 40), font=font)
    draw.text((bar_area[0] + 8, bar_area[1] + 6), f"max objects per class={max_count}", fill=(20, 20, 20), font=font)
    legend = " | ".join(f"{idx}:{name}" for idx, name in enumerate(CLASS_NAMES[:8])) + " | ..."
    draw.text((margin, chart_top + chart_h + 8), legend, fill=(65, 65, 65), font=font)

    grid_top = chart_top + chart_h + 42
    for item_index, (split, image_path, boxes) in enumerate(sample_pool):
        row = item_index // cols
        col = item_index % cols
        left = margin + col * (thumb_w + gap)
        top = grid_top + row * (thumb_h + label_h + gap)
        with Image.open(image_path) as image:
            annotated = image.convert("RGB")
        draw_image = ImageDraw.Draw(annotated)
        image_w, image_h = annotated.size
        for class_id, x, y, w, h in boxes[:20]:
            color = color_for_class(class_id)
            x0 = int((x - w / 2.0) * image_w)
            y0 = int((y - h / 2.0) * image_h)
            x1 = int((x + w / 2.0) * image_w)
            y1 = int((y + h / 2.0) * image_h)
            draw_image.rectangle((x0, y0, x1, y1), outline=color, width=max(2, image_w // 320))
            draw_image.text((max(0, x0), max(0, y0)), str(class_id), fill=color, font=font)
        fitted = fit_image(annotated, (thumb_w, thumb_h))
        canvas.paste(fitted, (left, top))
        draw.rectangle((left, top, left + thumb_w, top + thumb_h), outline=(190, 190, 190))
        draw.text((left + 4, top + thumb_h + 4), f"{split} | {image_path.name}", fill=(30, 30, 30), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    if args.skip_download and args.archive is not None:
        raise SystemExit("--skip-download and --archive are mutually exclusive.")

    if not args.skip_download:
        if args.archive is not None:
            print(f"Extracting {args.archive} to {args.root}")
            extract_archive(args.archive, args.root)
        elif has_dataset(args.root):
            print(f"Road Vehicle data already found under {args.root}; skipping Kaggle download.")
        else:
            print(f"Downloading Kaggle dataset {args.slug} to {args.root}")
            download_with_kaggle(args.root, args.slug)

    train_images, train_labels = find_split(args.root, {"train"})
    valid_images, valid_labels = find_split(args.root, {"valid", "val", "validation"})

    data_yaml = write_data_yaml(args.root, train_images, valid_images)
    stats = {
        "train": validate_split(train_images, train_labels),
        "valid": validate_split(valid_images, valid_labels),
    }

    all_errors = stats["train"]["errors"] + stats["valid"]["errors"]
    if all_errors:
        print("Invalid YOLO labels found:")
        for error in all_errors[:50]:
            print(f"- {error}")
        if len(all_errors) > 50:
            print(f"- ... {len(all_errors) - 50} more errors")
        raise SystemExit(1)

    print("Road Vehicle summary")
    for split, split_stats in stats.items():
        print(
            f"- {split}: images={len(split_stats['images'])}, "
            f"objects={sum(split_stats['counts'].values())}, "
            f"missing_labels={len(split_stats['missing'])}, empty_labels={len(split_stats['empty'])}"
        )
        if split_stats["missing"]:
            print(f"  [WARN] missing label files, first: {split_stats['missing'][0]}")
        if split_stats["empty"]:
            print(f"  [WARN] empty label files, first: {split_stats['empty'][0]}")

    check_path = args.root / "checks" / "class_check.png"
    make_check_image(check_path, stats, seed=args.seed, samples=args.samples)
    print(f"Wrote {data_yaml}")
    print(f"Wrote {check_path}")


if __name__ == "__main__":
    main()
