"""Prepare Stanford Background masks, splits, and a class check image."""

from __future__ import annotations

import argparse
import random
import tarfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "stanford_background"
DEFAULT_URL = "http://dags.stanford.edu/data/iccv09Data.tar.gz"
CLASS_NAMES = ["sky", "tree", "road", "grass", "water", "building", "mountain", "foreground"]
PALETTE = [
    (128, 128, 128),
    (129, 127, 38),
    (120, 69, 125),
    (53, 125, 34),
    (0, 11, 123),
    (118, 20, 12),
    (122, 81, 25),
    (241, 134, 51),
]
IGNORE_INDEX = 255
EXPECTED_RAW_VALUES = {-1, *range(len(CLASS_NAMES))}
EXPECTED_MASK_VALUES = {*range(len(CLASS_NAMES)), IGNORE_INDEX}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Stanford Background output root.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split and samples.")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples in mask_check.png.")
    parser.add_argument("--skip-download", action="store_true", help="Validate existing data only.")
    parser.add_argument("--archive", type=Path, default=None, help="Manual iccv09Data archive to extract.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Stanford Background archive URL.")
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


def download_archive(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Archive already exists: {destination}")
        return destination
    print(f"Downloading {url} to {destination}")
    urllib.request.urlretrieve(url, destination)
    return destination


def find_iccv09(root: Path) -> Path:
    direct = root / "iccv09Data"
    if (direct / "images").is_dir() and (direct / "labels").is_dir():
        return direct
    for path in root.rglob("iccv09Data"):
        if path.is_dir() and (path / "images").is_dir() and (path / "labels").is_dir():
            return path
    raise FileNotFoundError(f"Could not find iccv09Data/images and labels under {root}")


def infer_label_offset(region_paths: list[Path]) -> int:
    import numpy as np

    min_value: int | None = None
    max_value: int | None = None
    for regions_path in region_paths:
        raw = np.loadtxt(regions_path).astype(np.int64)
        current_min = int(raw.min())
        current_max = int(raw.max())
        min_value = current_min if min_value is None else min(min_value, current_min)
        max_value = current_max if max_value is None else max(max_value, current_max)
    if min_value is not None and max_value == len(CLASS_NAMES) and min_value >= 1:
        return 1
    return 0


def check_mask_mapping(
    raw_values: set[int],
    mask_values: set[int],
    raw_unknown_pixels: int,
    mask_ignore_pixels: int,
) -> None:
    if raw_values != EXPECTED_RAW_VALUES:
        raise RuntimeError(
            f"Unexpected Stanford region values: {sorted(raw_values)}; "
            f"expected {sorted(EXPECTED_RAW_VALUES)}"
        )
    if mask_values != EXPECTED_MASK_VALUES:
        raise RuntimeError(
            f"Unexpected mask values: {sorted(mask_values)}; "
            f"expected {sorted(EXPECTED_MASK_VALUES)}"
        )
    if raw_unknown_pixels != mask_ignore_pixels:
        raise RuntimeError(
            f"ignore_index mismatch: raw -1 pixels={raw_unknown_pixels}, "
            f"mask {IGNORE_INDEX} pixels={mask_ignore_pixels}"
        )


def convert_masks(iccv09_root: Path, masks_dir: Path) -> tuple[list[str], Counter[int], dict[str, Any]]:
    import numpy as np
    from PIL import Image

    labels_dir = iccv09_root / "labels"
    masks_dir.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    pixel_counts: Counter[int] = Counter()
    raw_values: set[int] = set()
    mask_values: set[int] = set()
    raw_unknown_pixels = 0
    mask_ignore_pixels = 0
    region_paths = sorted(labels_dir.glob("*.regions.txt"))
    if not region_paths:
        raise RuntimeError(f"No *.regions.txt files found in {labels_dir}")

    label_offset = infer_label_offset(region_paths)
    if label_offset:
        print("Detected 1-based Stanford labels; converting to 0-based class ids.")

    for regions_path in region_paths:
        image_id = regions_path.name.replace(".regions.txt", "")
        raw = np.loadtxt(regions_path).astype(np.int64) - label_offset
        raw_values.update(int(value) for value in np.unique(raw))
        raw_unknown_pixels += int((raw == -1).sum())
        valid = (raw >= 0) & (raw < len(CLASS_NAMES))
        mask = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)
        mask[valid] = raw[valid].astype(np.uint8)
        mask_values.update(int(value) for value in np.unique(mask))
        mask_ignore_pixels += int((mask == IGNORE_INDEX).sum())
        for class_id in range(len(CLASS_NAMES)):
            pixel_counts[class_id] += int((mask == class_id).sum())
        Image.fromarray(mask).save(masks_dir / f"{image_id}.png")
        ids.append(image_id)

    check_mask_mapping(raw_values, mask_values, raw_unknown_pixels, mask_ignore_pixels)
    mapping_check = {
        "raw_values": sorted(raw_values),
        "mask_values": sorted(mask_values),
        "raw_unknown_pixels": raw_unknown_pixels,
        "mask_ignore_pixels": mask_ignore_pixels,
    }
    return ids, pixel_counts, mapping_check


def write_splits(ids: list[str], split_dir: Path, seed: int) -> tuple[Path, Path]:
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * 0.8)
    train_ids = sorted(shuffled[:train_count])
    val_ids = sorted(shuffled[train_count:])

    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    train_path.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    return train_path, val_path


def colorize_mask(mask: Any) -> Any:
    import numpy as np

    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(PALETTE):
        color[mask == class_id] = rgb
    color[mask == IGNORE_INDEX] = (0, 0, 0)
    return color


def overlay_mask(image: Any, mask: Any) -> Any:
    import numpy as np

    image_arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = colorize_mask(mask).astype(np.float32)
    valid = mask != IGNORE_INDEX
    blended = image_arr.copy()
    blended[valid] = 0.58 * image_arr[valid] + 0.42 * color[valid]
    return blended.astype(np.uint8)


def fit_image(image: Any, size: tuple[int, int]) -> Any:
    from PIL import Image

    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def make_check_image(
    iccv09_root: Path,
    masks_dir: Path,
    output_path: Path,
    ids: list[str],
    pixel_counts: Counter[int],
    seed: int,
    samples: int,
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(seed)
    sample_ids = rng.sample(ids, k=min(samples, len(ids))) if ids else []
    font = ImageFont.load_default()

    width = 1400
    margin = 24
    chart_top = 86
    chart_h = 230
    cell_w, cell_h = 260, 170
    label_h = 34
    gap = 14
    cols = 4
    rows = (len(sample_ids) + cols - 1) // cols
    height = 356 + max(1, rows) * (cell_h + label_h + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "Stanford Background dataset check", fill=(20, 20, 20), font=font)
    draw.text((margin, 42), "Semantic class pixel counts and image/mask overlays", fill=(70, 70, 70), font=font)

    chart_box = (margin, chart_top, width - margin, chart_top + chart_h)
    draw.rectangle(chart_box, outline=(180, 180, 180))
    max_count = max(pixel_counts.values()) if pixel_counts else 1
    bar_w = 120
    for class_id, name in enumerate(CLASS_NAMES):
        value = pixel_counts.get(class_id, 0)
        left = chart_box[0] + 24 + class_id * (bar_w + 38)
        height_px = int((value / max_count) * (chart_h - 70))
        top = chart_box[3] - 38 - height_px
        draw.rectangle((left, top, left + bar_w, chart_box[3] - 38), fill=PALETTE[class_id])
        draw.text((left, chart_box[3] - 34), str(class_id), fill=(30, 30, 30), font=font)
        draw.text((left, top - 14), str(value), fill=(30, 30, 30), font=font)
        draw.text((left, chart_box[3] - 18), name[:16], fill=(60, 60, 60), font=font)

    grid_top = chart_top + chart_h + 38
    for item_index, image_id in enumerate(sample_ids):
        row = item_index // cols
        col = item_index % cols
        left = margin + col * (cell_w + gap)
        top = grid_top + row * (cell_h + label_h + gap)
        image_path = find_image_path(iccv09_root / "images", image_id)
        mask_path = masks_dir / f"{image_id}.png"
        with Image.open(image_path) as image, Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image, dtype=np.uint8)
            overlay = Image.fromarray(overlay_mask(image, mask))
            fitted = fit_image(overlay, (cell_w, cell_h))
        canvas.paste(fitted, (left, top))
        draw.rectangle((left, top, left + cell_w, top + cell_h), outline=(190, 190, 190))
        draw.text((left + 4, top + cell_h + 4), image_id, fill=(30, 30, 30), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def find_image_path(images_dir: Path, image_id: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = images_dir / f"{image_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find image for id {image_id} under {images_dir}")


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    if args.skip_download and args.archive is not None:
        raise SystemExit("--skip-download and --archive are mutually exclusive.")

    if not args.skip_download:
        if args.archive is not None:
            print(f"Extracting {args.archive} to {args.root}")
            extract_archive(args.archive, args.root)
        elif not (args.root / "iccv09Data").exists():
            archive_path = download_archive(args.url, args.root / "iccv09Data.tar.gz")
            extract_archive(archive_path, args.root)
        else:
            print(f"iccv09Data already found under {args.root}; skipping download.")

    iccv09_root = find_iccv09(args.root)
    masks_dir = args.root / "masks"
    split_dir = args.root / "splits"
    ids, pixel_counts, mapping_check = convert_masks(iccv09_root, masks_dir)
    train_path, val_path = write_splits(ids, split_dir, args.seed)
    check_path = args.root / "checks" / "mask_check.png"
    make_check_image(iccv09_root, masks_dir, check_path, ids, pixel_counts, args.seed, args.samples)

    print("Stanford Background summary")
    print(f"- images/masks: {len(ids)}")
    print(f"- classes: {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
    print(f"- raw regions values: {mapping_check['raw_values']}")
    print(f"- mask values: {mapping_check['mask_values']}")
    print(
        f"- ignore_index={IGNORE_INDEX}: "
        f"raw -1 pixels={mapping_check['raw_unknown_pixels']}, "
        f"mask pixels={mapping_check['mask_ignore_pixels']}"
    )
    print(f"Wrote {masks_dir}")
    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")
    print(f"Wrote {check_path}")


if __name__ == "__main__":
    main()
