"""Fetch YOLOv8n weights with mirror-aware fallback for server smoke tests."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "weights" / "yolov8n.pt"
MIN_BYTES = 1_000_000
OFFICIAL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
DEFAULT_URLS = [
    f"https://ghfast.top/{OFFICIAL_URL}",
    f"https://gh-proxy.com/{OFFICIAL_URL}",
    "https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/yolov8n.pt",
    OFFICIAL_URL,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination .pt path.")
    parser.add_argument("--url", action="append", default=[], help="Override download URL, repeatable.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument("--min-bytes", type=int, default=MIN_BYTES, help="Minimum valid file size.")
    return parser.parse_args()


def split_env_urls(raw: str | None) -> list[str]:
    if not raw:
        return []
    normalized = raw.replace("\n", ",").replace(";", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def candidate_urls(cli_urls: list[str]) -> list[str]:
    return cli_urls or split_env_urls(os.environ.get("YOLO_MODEL_URLS")) or DEFAULT_URLS


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
    output = args.output
    if is_valid_weight(output, args.min_bytes):
        print(f"Using existing YOLO weight: {output} ({output.stat().st_size} bytes)")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for url in candidate_urls(args.url):
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
