"""Run YOLOv8 + ByteTrack on a video and overlay line-crossing counts.

This is the Day 4 11:00 Task 2 deliverable script. It wraps
``ultralytics.YOLO.track(stream=True)``, feeds each frame's tracks into
``LineCounter``, draws the counting line + dots + a translucent stat panel on
top of ``Result.plot()``, and writes both the annotated video and a
``counts.json`` summary with per-class breakdowns.

It is intentionally separate from ``track_video.py``: that script keeps its
``YOLO.track(save=True)`` shape (already referenced by the occlusion analysis
and SwanLab logs); this one owns its own ``cv2.VideoWriter`` because the count
overlay has to be drawn frame-by-frame.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import os

from src.task2_detect_track.line_counter import LineCounter
from src.task2_detect_track.render_counts import (
    bbox_bottom_center,
    draw_count_dots,
    draw_count_overlay,
    draw_counting_line,
    parse_class_filter,
    parse_line_arg,
    pick_video_writer,
)
from src.task2_detect_track.track_video import (
    DEFAULT_PROJECT,
    DEFAULT_VIDEO_DIR,
    DEFAULT_WEIGHTS,
    REPO_ROOT,
    newest_video,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO weights path.")
    parser.add_argument("--source", default=None, help="Video path. Defaults to the newest .mp4 under data/videos.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--line", required=True, help='Counting line as "x1,y1,x2,y2".')
    parser.add_argument("--line-margin", type=float, default=4.0, help="Perpendicular pixel margin to suppress jitter.")
    parser.add_argument(
        "--exclude-classes",
        default="3,20",
        help="Comma-separated class ids to exclude from counting (default: bicycle, wheelbarrow).",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast if CUDA is unavailable instead of falling back to CPU.",
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="second_video_bytetrack_counts")
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--no-exist-ok", dest="exist_ok", action="store_false")
    parser.set_defaults(exist_ok=True)
    parser.add_argument("--sample-frame", type=int, default=0, help="Save raw + annotated PNG at this frame index. -1 disables.")
    parser.add_argument("--smoke-frames", type=int, default=-1, help="If > 0, stop after N frames (for fast iteration).")
    parser.add_argument("--counts-json", default=None, help="Override counts.json path.")
    return parser.parse_args()


def resolve_source(source: str | None) -> str:
    if source:
        candidate = resolve_repo_path(source)
        return str(candidate if candidate.exists() else source)
    return str(newest_video(DEFAULT_VIDEO_DIR))


def build_class_names(model, weights_path: Path) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def get_cuda_peak_mem_mb(device_str: str) -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        device_index = int(device_str.split(",")[0])
    except (ValueError, IndexError):
        device_index = 0
    try:
        peak = torch.cuda.max_memory_allocated(device_index)
    except Exception:
        return None
    return peak / (1024.0 * 1024.0)


def _collect_cuda_diagnostics() -> dict[str, str | int | bool | None]:
    try:
        import torch
    except ImportError:
        return {"torch": "missing"}
    diagnostics: dict[str, str | int | bool | None] = {
        "torch": getattr(torch, "__version__", "unknown"),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    try:
        diagnostics["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        diagnostics["nvidia_visible_devices"] = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    except Exception:
        pass
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            diagnostics["cuda_device_0"] = torch.cuda.get_device_name(0)
    except Exception:
        diagnostics["cuda_device_0"] = None
    return diagnostics


def resolve_device(device_str: str, *, require_cuda: bool = False) -> str:
    """Return a safe device string for Ultralytics/torch.

    Falls back to CPU when CUDA is unavailable or the requested index is invalid.
    If require_cuda is True, raises a RuntimeError when CUDA cannot be used.
    """
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
        diagnostics = _collect_cuda_diagnostics()
        if require_cuda:
            raise RuntimeError(f"CUDA unavailable. Diagnostics: {diagnostics}")
        print(f"[count_video] CUDA unavailable; falling back to CPU. Diagnostics: {diagnostics}")
        return "cpu"
    try:
        first_index = int(device_str.split(",")[0])
    except (ValueError, IndexError):
        first_index = 0
    if torch.cuda.device_count() <= first_index:
        diagnostics = _collect_cuda_diagnostics()
        message = (
            f"Invalid CUDA device '{device_str}' (count={torch.cuda.device_count()}). "
            f"Diagnostics: {diagnostics}"
        )
        if require_cuda:
            raise RuntimeError(message)
        print(f"[count_video] {message} Falling back to CPU.")
        return "cpu"
    return device_str


def main() -> int:
    args = parse_args()
    import cv2  # required from here on
    from ultralytics import YOLO

    weights = resolve_repo_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Missing weights file: {weights}")
    source = resolve_source(args.source)
    source_path = Path(source)
    if not (source.startswith(("http://", "https://", "rtsp://")) or source_path.exists()):
        raise FileNotFoundError(f"Missing tracking source: {source}")

    line = parse_line_arg(args.line)
    exclude_classes = parse_class_filter(args.exclude_classes)

    project = args.project if args.project.is_absolute() else REPO_ROOT / args.project
    run_dir = project / args.name
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Run directory already exists and --no-exist-ok was set: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = source_path.stem
    counted_video_path = run_dir / f"{stem}_counted.mp4"
    first_raw_path = run_dir / f"{stem}_first_raw.png"
    first_annotated_path = run_dir / f"{stem}_first_annotated.png"
    counts_json_path = Path(args.counts_json) if args.counts_json else run_dir / "counts.json"

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source for fps/size probe: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Source reports invalid size ({width}x{height}); cannot init VideoWriter.")

    model = YOLO(str(weights))
    class_names = build_class_names(model, weights)
    all_class_ids = set(class_names.keys()) if class_names else set(range(21))
    include_classes = all_class_ids - exclude_classes if exclude_classes else None
    excluded_class_names = sorted(class_names.get(cid, str(cid)) for cid in exclude_classes)

    counter = LineCounter(line=line, margin=args.line_margin, include_classes=include_classes)
    per_class_counts: dict[int, int] = defaultdict(int)
    track_classes: dict[int, int] = {}

    writer, codec_label, actual_video_path = pick_video_writer(
        counted_video_path, fps=fps, size=(width, height)
    )

    print(f"[count_video] source={source}")
    print(f"[count_video] weights={weights}")
    print(f"[count_video] line={line} margin={args.line_margin} exclude={sorted(exclude_classes)}")
    print(f"[count_video] writer={codec_label} -> {actual_video_path}")

    resolved_device = resolve_device(args.device, require_cuda=args.require_cuda)
    if resolved_device != args.device:
        print(f"[count_video] device requested={args.device} resolved={resolved_device}")

    results_iter = model.track(
        source=source,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=resolved_device,
        stream=True,
        persist=True,
        save=False,
        vid_stride=args.vid_stride,
        verbose=False,
    )

    wallclock_start = time.perf_counter()
    frame_count = 0
    frames_with_tracks = 0

    try:
        for frame_idx, result in enumerate(results_iter):
            frame_count += 1
            plotted = result.plot()
            if plotted is None:
                continue

            boxes = result.boxes
            counted_centers: list[tuple[float, float]] = []
            if boxes is not None and boxes.id is not None and len(boxes) > 0:
                frames_with_tracks += 1
                ids = boxes.id.detach().cpu().numpy().astype(int)
                clss = boxes.cls.detach().cpu().numpy().astype(int)
                xyxys = boxes.xyxy.detach().cpu().numpy()

                records: list[dict] = []
                for tid, cid, box in zip(ids, clss, xyxys):
                    center = bbox_bottom_center(box.tolist())
                    track_classes[int(tid)] = int(cid)
                    records.append({"track_id": int(tid), "center": center, "class_id": int(cid)})

                events = counter.update_many(records)
                for rec, event in zip(records, events):
                    if event.counted:
                        per_class_counts[int(rec["class_id"])] += 1
                        counted_centers.append(rec["center"])

            plotted = draw_counting_line(plotted, counter.line)
            if counted_centers:
                plotted = draw_count_dots(plotted, counted_centers)
            plotted = draw_count_overlay(
                plotted,
                total=counter.total_count,
                direction_counts=counter.direction_counts,
                per_class_counts=per_class_counts,
                class_names=class_names,
            )

            if args.sample_frame == frame_idx:
                cv2.imwrite(str(first_raw_path), result.orig_img)
                cv2.imwrite(str(first_annotated_path), plotted)

            writer.write(plotted)

            if args.smoke_frames > 0 and frame_count >= args.smoke_frames:
                print(f"[count_video] hit --smoke-frames={args.smoke_frames}, stopping early.")
                break
    finally:
        writer.release()

    wallclock_sec = time.perf_counter() - wallclock_start
    gpu_peak_mem_mb = get_cuda_peak_mem_mb(resolved_device)

    payload: dict = {
        "source": str(Path(source).as_posix()),
        "weights": str(weights.as_posix()),
        "tracker": args.tracker,
        "line": [float(v) for v in line],
        "line_margin": float(args.line_margin),
        "excluded_class_ids": sorted(exclude_classes),
        "excluded_class_names": excluded_class_names,
        "total_count": int(counter.total_count),
        "direction_counts": {k: int(v) for k, v in counter.direction_counts.items()},
        "per_class_counts": {
            class_names.get(cid, str(cid)): int(per_class_counts[cid])
            for cid in sorted(per_class_counts.keys())
        },
        "frame_count": int(frame_count),
        "frames_with_tracks": int(frames_with_tracks),
        "fps": float(fps),
        "duration_sec": float(frame_count / fps) if fps > 0 else 0.0,
        "writer_codec": codec_label,
        "video_path": str(actual_video_path.as_posix()),
        "first_raw_path": str(first_raw_path.as_posix()) if first_raw_path.exists() else None,
        "first_annotated_path": str(first_annotated_path.as_posix()) if first_annotated_path.exists() else None,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "wallclock_sec": float(wallclock_sec),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "imgsz": int(args.imgsz),
    "device": resolved_device,
    "device_requested": args.device,
        "smoke_frames": int(args.smoke_frames),
    }

    counts_json_path.parent.mkdir(parents=True, exist_ok=True)
    with counts_json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"[count_video] frames={frame_count} with_tracks={frames_with_tracks} "
          f"total={counter.total_count} pos2neg={counter.direction_counts['positive_to_negative']} "
          f"neg2pos={counter.direction_counts['negative_to_positive']} wallclock={wallclock_sec:.1f}s")
    print(f"[count_video] counts_json={counts_json_path}")
    print(f"[count_video] video={actual_video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
