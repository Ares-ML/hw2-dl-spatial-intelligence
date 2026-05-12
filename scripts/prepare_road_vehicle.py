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
MIN_BOX_SIZE = 1e-6
CLASS_NAMES = [
    "ambulance",
    "army vehicle",
    "auto rickshaw",
    "bicycle",
    "bus",
    "car",
    "garbagevan",
    "human hauler",
    "minibus",
    "minivan",
    "motorbike",
    "pickup",
    "policecar",
    "rickshaw",
    "scooter",
    "suv",
    "taxi",
    "three wheelers -CNG-",
    "truck",
    "van",
    "wheelbarrow",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Road Vehicle output root.")
    parser.add_argument(
        "--yaml-root",
        default=None,
        help="Override the `path:` value written to data.yaml, e.g. the GPU server dataset root.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples in class_check.png.")
    parser.add_argument("--skip-download", action="store_true", help="Validate existing data only.")
    parser.add_argument("--archive", type=Path, default=None, help="Manual .zip/.tar archive to extract.")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Kaggle dataset slug.")
    parser.add_argument(
        "--strict-labels",
        action="store_true",
        help="Fail on invalid bbox rows instead of repairing or dropping them.",
    )
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


def normalize_class_names(raw_names: Any) -> list[str] | None:
    if isinstance(raw_names, list):
        names = [str(name) for name in raw_names]
    elif isinstance(raw_names, dict):
        try:
            ordered_keys = sorted(raw_names, key=lambda key: int(key))
        except (TypeError, ValueError):
            return None
        names = [str(raw_names[key]) for key in ordered_keys]
    else:
        return None
    return names if names else None


def load_class_names(root: Path) -> list[str]:
    try:
        import yaml
    except ImportError:
        return list(CLASS_NAMES)

    preferred = [root / "trafic_data" / "data_1.yaml", root / "data_1.yaml"]
    discovered = sorted(path for path in root.rglob("*.yaml") if path.name != "data.yaml")
    candidates: list[Path] = []
    for path in preferred + discovered:
        if path not in candidates:
            candidates.append(path)

    for path in candidates:
        if not path.is_file():
            continue
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(config, dict):
            continue
        names = normalize_class_names(config.get("names"))
        if names is not None:
            return names

    return list(CLASS_NAMES)


def read_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def format_box(box: tuple[int, float, float, float, float]) -> str:
    class_id, x, y, w, h = box
    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def repair_box(
    values: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[tuple[float, float, float, float] | None, str]:
    x, y, w, h = values
    original = values

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
        return values, "valid"

    source = "normalized"
    if max(abs(value) for value in values) > 4.0:
        image_w, image_h = image_size
        if image_w <= 0 or image_h <= 0:
            return None, "invalid image size"
        x, y, w, h = x / image_w, y / image_h, w / image_w, h / image_h
        source = "pixel_xywh"

    x0 = max(0.0, min(1.0, x - w / 2.0))
    y0 = max(0.0, min(1.0, y - h / 2.0))
    x1 = max(0.0, min(1.0, x + w / 2.0))
    y1 = max(0.0, min(1.0, y + h / 2.0))
    new_w = x1 - x0
    new_h = y1 - y0
    if new_w <= MIN_BOX_SIZE or new_h <= MIN_BOX_SIZE:
        return None, f"degenerate after clipping from {source}: {original}"

    repaired = (x0 + new_w / 2.0, y0 + new_h / 2.0, new_w, new_h)
    return repaired, f"repaired from {source}: {original} -> {repaired}"


def read_label_file(
    path: Path,
    image_size: tuple[int, int],
    repair_labels: bool,
    class_names: list[str],
) -> tuple[list[tuple[int, float, float, float, float]], list[str], list[str]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    errors: list[str] = []
    warnings: list[str] = []
    changed = False
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
        if not 0 <= class_id < len(class_names):
            errors.append(f"{path}:{line_number}: class id {class_id} outside 0..{len(class_names) - 1}")
            continue

        repaired, message = repair_box((x, y, w, h), image_size)
        if repaired is None:
            if repair_labels:
                warnings.append(f"{path}:{line_number}: dropped invalid bbox; {message}")
                changed = True
            else:
                errors.append(f"{path}:{line_number}: bbox values must be normalized with positive width/height")
            continue

        if message != "valid":
            if repair_labels:
                warnings.append(f"{path}:{line_number}: {message}")
                changed = True
            else:
                errors.append(f"{path}:{line_number}: bbox values must be normalized with positive width/height")
                continue

        x, y, w, h = repaired
        boxes.append((class_id, x, y, w, h))

    if changed and repair_labels and not errors:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        path.write_text("\n".join(format_box(box) for box in boxes) + ("\n" if boxes else ""), encoding="utf-8")

    return boxes, errors, warnings


def validate_split(images_dir: Path, labels_dir: Path, repair_labels: bool, class_names: list[str]) -> dict[str, Any]:
    files = image_files(images_dir)
    counts: Counter[int] = Counter()
    missing: list[Path] = []
    empty: list[Path] = []
    errors: list[str] = []
    warnings: list[str] = []
    samples: list[tuple[Path, Path, list[tuple[int, float, float, float, float]]]] = []

    for image_path in files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing.append(image_path)
            continue
        try:
            size = read_image_size(image_path)
        except Exception as exc:
            errors.append(f"{image_path}: failed to read image size: {exc}")
            continue
        boxes, label_errors, label_warnings = read_label_file(
            label_path,
            size,
            repair_labels=repair_labels,
            class_names=class_names,
        )
        errors.extend(label_errors)
        warnings.extend(label_warnings)
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
        "warnings": warnings,
        "samples": samples,
    }


def relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_data_yaml(
    root: Path,
    train_images: Path,
    valid_images: Path,
    class_names: list[str],
    yaml_root: str | None,
) -> Path:
    path = root / "data.yaml"
    output_root = yaml_root or root.resolve().as_posix()
    lines = [
        f"path: {output_root}",
        f"train: {relative_to_root(train_images, root)}",
        f"val: {relative_to_root(valid_images, root)}",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
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
    class_names: list[str],
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
    bar_w = max(12, (bar_area[2] - bar_area[0] - 8 * (len(class_names) - 1)) // len(class_names))
    for class_id, name in enumerate(class_names):
        value = total_counts.get(class_id, 0)
        left = bar_area[0] + class_id * (bar_w + 8)
        height_px = int((value / max_count) * (chart_h - 52))
        top = bar_area[3] - 28 - height_px
        draw.rectangle((left, top, left + bar_w, bar_area[3] - 28), fill=color_for_class(class_id))
        draw.text((left, bar_area[3] - 24), str(class_id), fill=(40, 40, 40), font=font)
        if class_id < 8:
            draw.text((left, top - 14), str(value), fill=(40, 40, 40), font=font)
    draw.text((bar_area[0] + 8, bar_area[1] + 6), f"max objects per class={max_count}", fill=(20, 20, 20), font=font)
    legend = " | ".join(f"{idx}:{name}" for idx, name in enumerate(class_names[:8])) + " | ..."
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
    class_names = load_class_names(args.root)

    data_yaml = write_data_yaml(args.root, train_images, valid_images, class_names, yaml_root=args.yaml_root)
    repair_labels = not args.strict_labels
    stats = {
        "train": validate_split(train_images, train_labels, repair_labels=repair_labels, class_names=class_names),
        "valid": validate_split(valid_images, valid_labels, repair_labels=repair_labels, class_names=class_names),
    }

    all_errors = stats["train"]["errors"] + stats["valid"]["errors"]
    if all_errors:
        print("Fatal YOLO label errors found:")
        for error in all_errors[:50]:
            print(f"- {error}")
        if len(all_errors) > 50:
            print(f"- ... {len(all_errors) - 50} more errors")
        raise SystemExit(1)

    all_warnings = stats["train"]["warnings"] + stats["valid"]["warnings"]
    if all_warnings:
        action = "Repaired/dropped invalid YOLO bbox rows" if repair_labels else "Invalid YOLO bbox rows"
        print(f"{action}: {len(all_warnings)}")
        for warning in all_warnings[:30]:
            print(f"- {warning}")
        if len(all_warnings) > 30:
            print(f"- ... {len(all_warnings) - 30} more warnings")

    print("Road Vehicle summary")
    for split, split_stats in stats.items():
        print(
            f"- {split}: images={len(split_stats['images'])}, "
            f"objects={sum(split_stats['counts'].values())}, "
            f"missing_labels={len(split_stats['missing'])}, "
            f"empty_labels={len(split_stats['empty'])}, "
            f"label_warnings={len(split_stats['warnings'])}"
        )
        if split_stats["missing"]:
            print(f"  [WARN] missing label files, first: {split_stats['missing'][0]}")
        if split_stats["empty"]:
            print(f"  [WARN] empty label files, first: {split_stats['empty'][0]}")

    check_path = args.root / "checks" / "class_check.png"
    make_check_image(check_path, stats, class_names, seed=args.seed, samples=args.samples)
    print(f"Wrote {data_yaml}")
    print(f"Wrote {check_path}")


if __name__ == "__main__":
    main()
