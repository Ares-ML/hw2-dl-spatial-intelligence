"""Fetch YOLOv8 weights with mirror-aware fallback for server smoke tests / bonus runs.

Default model is ``yolov8n.pt`` to keep ``scripts/run_smoke_server.sh`` and the
Day 1 smoke flow unchanged; pass ``--model yolov8s.pt`` (or any other release
asset) to pre-fetch a different weight before running ``yolo detect train``.
Pre-fetching also sidesteps a known Ultralytics false-positive where
``check_disk_space`` reads ``shutil.disk_usage(...).free == 0`` under some
Docker overlay filesystems and aborts the download with ``MemoryError``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "yolov8n.pt"
MIN_BYTES = 1_000_000
OFFICIAL_URL_TEMPLATE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}"
HF_MIRROR_URL_TEMPLATE = "https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/{name}"


def model_urls(name: str) -> list[str]:
    """Return mirror-fallback URL list for a given weight filename."""

    official = OFFICIAL_URL_TEMPLATE.format(name=name)
    return [
        f"https://ghfast.top/{official}",
        f"https://gh-proxy.com/{official}",
        HF_MIRROR_URL_TEMPLATE.format(name=name),
        official,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Weight filename, e.g. yolov8n.pt, yolov8s.pt, yolov8m.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination .pt path. Defaults to REPO_ROOT/weights/<model>.",
    )
    parser.add_argument("--url", action="append", default=[], help="Override download URL, repeatable.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument("--min-bytes", type=int, default=MIN_BYTES, help="Minimum valid file size.")
    return parser.parse_args()


def split_env_urls(raw: str | None) -> list[str]:
    if not raw:
        return []
    normalized = raw.replace("\n", ",").replace(";", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def candidate_urls(cli_urls: list[str], model_name: str) -> list[str]:
    return cli_urls or split_env_urls(os.environ.get("YOLO_MODEL_URLS")) or model_urls(model_name)


def is_valid_weight(path: Path, min_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def download(url: str, output: Path, timeout: float, min_bytes: int) -> None:
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "HW2-smoke-fetch/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as tmp:
            shutil.copyfileobj(response, tmp)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    if tmp_path.stat().st_size < min_bytes:
        size = tmp_path.stat().st_size
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded file is too small: {size} bytes")
    tmp_path.replace(output)


def main() -> int:
    args = parse_args()
    output = args.output if args.output is not None else (REPO_ROOT / "weights" / args.model)
    if is_valid_weight(output, args.min_bytes):
        print(f"Using existing YOLO weight: {output} ({output.stat().st_size} bytes)")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for url in candidate_urls(args.url, args.model):
        print(f"Fetching YOLO weight from: {url}", flush=True)
        try:
            download(url, output, args.timeout, args.min_bytes)
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            failures.append(f"{url}: {exc}")
            print(f"Download failed: {exc}", file=sys.stderr, flush=True)
            continue
        print(f"Fetched YOLO weight: {output} ({output.stat().st_size} bytes)")
        return 0

    print("Failed to fetch YOLO weight from all candidates.", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
