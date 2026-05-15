"""Extract and summarize consecutive annotated frames for Task 2 occlusion analysis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from io import BytesIO
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = REPO_ROOT / "runs" / "track" / "campus_bytetrack" / "0fcfd11681a83b812c459e3bdd30a58b.avi"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "report" / "figures" / "task2" / "occlusion"


@dataclass(frozen=True)
class TrackBox:
    frame_index: int
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    conf: float | None = None
    track_id: int | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        return (
            self.x_center - half_w,
            self.y_center - half_h,
            self.x_center + half_w,
            self.y_center + half_h,
        )


@dataclass(frozen=True)
class FrameOverlapStats:
    max_iou: float
    mean_iou: float
    pair_count: int
    box_count: int


@dataclass(frozen=True)
class WindowSelection:
    start_frame: int
    max_iou: float
    mean_iou: float
    pair_count: int
    box_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Annotated tracking video path.")
    parser.add_argument("--start-frame", type=int, default=90, help="Zero-based first frame index to export.")
    parser.add_argument("--num-frames", type=int, default=4, help="Number of consecutive frames to export.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for exported PNG frames.")
    parser.add_argument("--prefix", default="occlusion", help="Output filename prefix.")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Ultralytics tracking label directory.")
    parser.add_argument("--compare-labels-dir", type=Path, default=None, help="Optional second tracker label directory.")
    parser.add_argument("--auto-window", action="store_true", help="Select the strongest occlusion window from labels.")
    parser.add_argument("--analysis-md", type=Path, default=None, help="Optional Markdown analysis output path.")
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def import_cv2():
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None
    return cv2


def import_pillow_image():
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on local optional dependency.
        raise RuntimeError(
            "OpenCV is unavailable and Pillow is missing, so MJPEG fallback extraction cannot run."
        ) from exc
    return Image


def find_jpeg_frames(data: bytes) -> list[tuple[int, int]]:
    frames: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = data.find(b"\xff\xd8", cursor)
        if start < 0:
            break
        end = data.find(b"\xff\xd9", start + 2)
        if end < 0:
            break
        frames.append((start, end + 2))
        cursor = end + 2
    return frames


def export_mjpeg_frames(
    video_path: Path,
    start_frame: int,
    num_frames: int,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    Image = import_pillow_image()
    data = video_path.read_bytes()
    frames = find_jpeg_frames(data)
    end_frame = start_frame + num_frames
    if len(frames) < end_frame:
        raise RuntimeError(
            f"OpenCV is unavailable and MJPEG fallback found only {len(frames)} frames; "
            f"cannot export through frame {end_frame - 1}."
        )

    exported: list[Path] = []
    for frame_index in range(start_frame, end_frame):
        start, end = frames[frame_index]
        image = Image.open(BytesIO(data[start:end])).convert("RGB")
        output_path = output_dir / f"{prefix}_f{frame_index:04d}.png"
        image.save(output_path)
        exported.append(output_path)
    return exported


def export_cv2_frames(
    cv2,
    video_path: Path,
    start_frame: int,
    num_frames: int,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    exported: list[Path] = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for offset in range(num_frames):
            frame_index = start_frame + offset
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Unable to read frame {frame_index} from {video_path}")
            output_path = output_dir / f"{prefix}_f{frame_index:04d}.png"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Unable to write frame: {output_path}")
            exported.append(output_path)
    finally:
        capture.release()
    return exported


def export_frames(video_path: Path, start_frame: int, num_frames: int, output_dir: Path, prefix: str) -> list[Path]:
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2 = import_cv2()
    if cv2 is not None:
        return export_cv2_frames(cv2, video_path, start_frame, num_frames, output_dir, prefix)
    return export_mjpeg_frames(video_path, start_frame, num_frames, output_dir, prefix)


def coerce_int_token(value: str) -> int | None:
    try:
        numeric = float(value)
    except ValueError:
        return None
    rounded = round(numeric)
    if abs(numeric - rounded) <= 1e-6:
        return int(rounded)
    return None


def parse_label_line(line: str, frame_index: int) -> TrackBox | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = coerce_int_token(parts[0])
    if class_id is None:
        return None
    try:
        x_center, y_center, width, height = (float(value) for value in parts[1:5])
    except ValueError:
        return None

    conf: float | None = None
    track_id: int | None = None
    if len(parts) >= 7:
        try:
            conf = float(parts[5])
        except ValueError:
            conf = None
        track_id = coerce_int_token(parts[6])
    elif len(parts) == 6:
        maybe_track_id = coerce_int_token(parts[5])
        if maybe_track_id is None:
            try:
                conf = float(parts[5])
            except ValueError:
                conf = None
        else:
            track_id = maybe_track_id

    return TrackBox(
        frame_index=frame_index,
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        conf=conf,
        track_id=track_id,
    )


def label_frame_index(path: Path) -> int | None:
    suffix = path.stem.rsplit("_", maxsplit=1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def load_tracking_labels(labels_dir: Path | None) -> dict[int, list[TrackBox]]:
    if labels_dir is None:
        return {}
    labels_dir = resolve_repo_path(labels_dir)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    frames: dict[int, list[TrackBox]] = {}
    for label_path in sorted(labels_dir.glob("*.txt")):
        frame_index = label_frame_index(label_path)
        if frame_index is None:
            continue
        boxes: list[TrackBox] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            box = parse_label_line(line, frame_index)
            if box is not None:
                boxes.append(box)
        frames[frame_index] = boxes
    return frames


def box_iou(left: TrackBox, right: TrackBox) -> float:
    left_x1, left_y1, left_x2, left_y2 = left.xyxy
    right_x1, right_y1, right_x2, right_y2 = right.xyxy
    inter_w = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    inter_h = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = inter_w * inter_h
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def frame_overlap_stats(boxes: list[TrackBox]) -> FrameOverlapStats:
    ious: list[float] = []
    for left, right in combinations(boxes, 2):
        if left.track_id is not None and left.track_id == right.track_id:
            continue
        ious.append(box_iou(left, right))
    if not ious:
        return FrameOverlapStats(max_iou=0.0, mean_iou=0.0, pair_count=0, box_count=len(boxes))
    return FrameOverlapStats(
        max_iou=max(ious),
        mean_iou=sum(ious) / len(ious),
        pair_count=len(ious),
        box_count=len(boxes),
    )


def choose_occlusion_window(label_frames: dict[int, list[TrackBox]], num_frames: int) -> WindowSelection:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if not label_frames:
        raise ValueError("--auto-window requires non-empty --labels-dir.")

    min_frame = min(label_frames)
    max_frame = max(label_frames)
    if max_frame - min_frame + 1 < num_frames:
        raise ValueError(f"Need at least {num_frames} consecutive frames, found {min_frame}..{max_frame}.")

    best: WindowSelection | None = None
    best_score: tuple[float, float, int, int] | None = None
    for start_frame in range(min_frame, max_frame - num_frames + 2):
        max_iou = 0.0
        pair_count = 0
        pair_iou_sum = 0.0
        box_count = 0
        for frame_index in range(start_frame, start_frame + num_frames):
            stats = frame_overlap_stats(label_frames.get(frame_index, []))
            max_iou = max(max_iou, stats.max_iou)
            pair_iou_sum += stats.mean_iou * stats.pair_count
            pair_count += stats.pair_count
            box_count += stats.box_count

        mean_iou = pair_iou_sum / pair_count if pair_count else 0.0
        score = (max_iou, mean_iou, box_count, pair_count)
        if best_score is None or score > best_score:
            best_score = score
            best = WindowSelection(
                start_frame=start_frame,
                max_iou=max_iou,
                mean_iou=mean_iou,
                pair_count=pair_count,
                box_count=box_count,
            )

    if best is None:
        raise ValueError("Unable to select an occlusion window from labels.")
    return best


def summarize_window(label_frames: dict[int, list[TrackBox]], start_frame: int, num_frames: int) -> WindowSelection:
    max_iou = 0.0
    pair_count = 0
    pair_iou_sum = 0.0
    box_count = 0
    for frame_index in range(start_frame, start_frame + num_frames):
        stats = frame_overlap_stats(label_frames.get(frame_index, []))
        max_iou = max(max_iou, stats.max_iou)
        pair_iou_sum += stats.mean_iou * stats.pair_count
        pair_count += stats.pair_count
        box_count += stats.box_count
    mean_iou = pair_iou_sum / pair_count if pair_count else 0.0
    return WindowSelection(
        start_frame=start_frame,
        max_iou=max_iou,
        mean_iou=mean_iou,
        pair_count=pair_count,
        box_count=box_count,
    )


def summarize_ids(boxes: list[TrackBox], limit: int = 10) -> str:
    if not boxes:
        return "-"
    ordered = sorted(
        boxes,
        key=lambda box: (
            box.track_id is None,
            box.track_id if box.track_id is not None else 10**9,
            -(box.conf or 0.0),
        ),
    )
    pieces: list[str] = []
    for box in ordered[:limit]:
        track = f"ID{box.track_id}" if box.track_id is not None else "no-id"
        conf = f", {box.conf:.2f}" if box.conf is not None else ""
        pieces.append(f"c{box.class_id}:{track}{conf}")
    if len(ordered) > limit:
        pieces.append(f"...(+{len(ordered) - limit})")
    return "<br>".join(pieces)


def display_path(path: Path | None) -> str:
    if path is None:
        return "not provided"
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_analysis_markdown(
    output_path: Path,
    video_path: Path,
    labels_dir: Path | None,
    compare_labels_dir: Path | None,
    start_frame: int,
    num_frames: int,
    exported_frames: list[Path],
    label_frames: dict[int, list[TrackBox]],
    compare_label_frames: dict[int, list[TrackBox]],
    selection: WindowSelection | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    end_frame = start_frame + num_frames - 1
    lines = [
        "# Task 2 Occlusion Analysis",
        "",
        f"Source video: `{display_path(video_path)}`",
        f"Frame window: `{start_frame}`-`{end_frame}`",
        f"Primary tracker labels: `{display_path(labels_dir)}`",
        f"Comparison tracker labels: `{display_path(compare_labels_dir)}`",
        "",
        "Exported frames:",
    ]
    lines.extend(f"- `{display_path(path)}`" for path in exported_frames)
    if selection is not None:
        lines.extend(
            [
                "",
                (
                    "Auto-window score: "
                    f"max IoU={selection.max_iou:.3f}, mean pair IoU={selection.mean_iou:.3f}, "
                    f"pairs={selection.pair_count}, boxes={selection.box_count}."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "| Frame | ByteTrack IDs | BoT-SORT IDs | Max bbox IoU | Mean bbox IoU |",
            "| ---: | --- | --- | ---: | ---: |",
        ]
    )
    for frame_index in range(start_frame, start_frame + num_frames):
        primary_boxes = label_frames.get(frame_index, [])
        compare_boxes = compare_label_frames.get(frame_index, [])
        stats = frame_overlap_stats(primary_boxes)
        lines.append(
            "| "
            f"{frame_index} | {summarize_ids(primary_boxes)} | {summarize_ids(compare_boxes)} | "
            f"{stats.max_iou:.3f} | {stats.mean_iou:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Analysis Paragraph",
            "",
            (
                f"在所选视频的第 {start_frame}-{end_frame} 帧中，车辆框出现连续重叠，"
                f"该窗口内 ByteTrack 标注框的最高 IoU 为 {selection.max_iou if selection else 0.0:.3f}，"
                f"平均成对 IoU 为 {selection.mean_iou if selection else 0.0:.3f}。"
                "遮挡发生时，可见区域缩小会使检测框位置和尺度波动，当前帧检测框与上一帧轨迹预测框的 IoU 下降；"
                "如果低置信度框没有被保留下来，或匹配距离超过阈值，ByteTrack 就可能把短暂被挡住的车辆标成 lost，"
                "甚至在重新出现时分配新 ID。"
            ),
            "",
            (
                "ByteTrack 的优势是会利用高分框和低分框两阶段关联，因而在短时遮挡中仍有机会把同一车辆接回原 ID；"
                "`track_buffer` 决定 lost 轨迹在多少帧内仍可等待重连，适当增大它可以容忍更长的遮挡，"
                "但也会提高旧轨迹误匹配到相邻车辆的风险。`match_thresh` 需要和检测置信度一起调节："
                "过严会在 IoU 快速下降时断 ID，过松则可能在密集交汇时交换 ID。"
            ),
            "",
            (
                "BoT-SORT 在运动/IoU 关联之外还支持相机运动补偿和可选的 ReID 外观特征。"
                "当遮挡导致 bbox IoU 信息不稳定时，启用 ReID 可利用车辆外观相似度辅助重新关联，"
                "所以在这一类遮挡片段中通常比只依赖运动和 IoU 的方案更有机会保持 ID 连续。"
                "最终报告中应结合上表逐帧 ID：若 ByteTrack 出现 ID 跳变而 BoT-SORT 保持一致，"
                "可将原因归结为遮挡造成的 IoU 下降和外观信息缺失；若二者都稳定，则说明当前遮挡持续时间短，"
                "检测框仍足以支持轨迹关联。"
            ),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    video = resolve_repo_path(args.video)
    output_dir = resolve_repo_path(args.output_dir)
    labels_dir = resolve_repo_path(args.labels_dir) if args.labels_dir else None
    compare_labels_dir = resolve_repo_path(args.compare_labels_dir) if args.compare_labels_dir else None

    label_frames = load_tracking_labels(labels_dir)
    compare_label_frames = load_tracking_labels(compare_labels_dir)
    selection: WindowSelection | None = None
    start_frame = args.start_frame
    if args.auto_window:
        selection = choose_occlusion_window(label_frames, args.num_frames)
        start_frame = selection.start_frame
    elif label_frames:
        selection = summarize_window(label_frames, start_frame, args.num_frames)

    exported = export_frames(video, start_frame, args.num_frames, output_dir, args.prefix)
    if args.analysis_md:
        write_analysis_markdown(
            output_path=resolve_repo_path(args.analysis_md),
            video_path=video,
            labels_dir=labels_dir,
            compare_labels_dir=compare_labels_dir,
            start_frame=start_frame,
            num_frames=args.num_frames,
            exported_frames=exported,
            label_frames=label_frames,
            compare_label_frames=compare_label_frames,
            selection=selection,
        )

    print(f"start_frame={start_frame}")
    for path in exported:
        print(path)
    if args.analysis_md:
        print(f"analysis_md={resolve_repo_path(args.analysis_md)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
