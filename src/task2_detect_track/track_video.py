"""Run YOLOv8 tracking on the Task 2 traffic video."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "task2" / "road_vehicle_yolov8n_e120" / "weights" / "best.pt"
DEFAULT_VIDEO_DIR = REPO_ROOT / "data" / "videos"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "track"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO weights path.")
    parser.add_argument("--source", default=None, help="Video path. Defaults to the newest .mp4 under data/videos.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config, e.g. bytetrack.yaml.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Ultralytics device string.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Tracking output project directory.")
    parser.add_argument("--name", default="campus_bytetrack", help="Tracking run name.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Video frame stride.")
    parser.add_argument("--no-exist-ok", dest="exist_ok", action="store_false", help="Create a fresh incremented run directory.")
    parser.set_defaults(exist_ok=True)
    parser.add_argument("--save-txt", action="store_true", help="Save tracking labels as text files.")
    parser.add_argument("--save-conf", action="store_true", help="Include confidences in saved text labels.")
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def newest_video(video_dir: Path) -> Path:
    candidates = sorted(video_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .mp4 files found under {video_dir}. Pass --source explicitly.")
    return candidates[0]


def resolve_source(source: str | None) -> str:
    if source:
        path = resolve_repo_path(source)
        return str(path if path.exists() else source)
    return str(newest_video(DEFAULT_VIDEO_DIR))


def main() -> int:
    args = parse_args()
    weights = resolve_repo_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Missing weights file: {weights}")
    source = resolve_source(args.source)
    source_path = Path(source)
    if not (source.startswith(("http://", "https://", "rtsp://")) or source_path.exists()):
        raise FileNotFoundError(f"Missing tracking source: {source}")

    from ultralytics import YOLO

    project = args.project if args.project.is_absolute() else REPO_ROOT / args.project
    model = YOLO(str(weights))
    model.track(
        source=source,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project),
        name=args.name,
        persist=True,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        vid_stride=args.vid_stride,
        exist_ok=args.exist_ok,
    )

    run_dir = project / args.name
    if not run_dir.is_dir():
        raise RuntimeError(f"Ultralytics tracking finished but output directory is missing: {run_dir}")
    media = sorted(path for path in run_dir.iterdir() if path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"})
    print(f"tracking_run_dir={run_dir}")
    if media:
        print(f"tracking_video={media[0]}")
    else:
        print("tracking_video=not-found; inspect the run directory for saved outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
