"""Drawing and parsing helpers for Task 2 line-crossing count overlay.

OpenCV is imported lazily so this module is importable on machines without
``cv2`` (e.g. the planning workstation). Only the drawing helpers and the
video-writer picker actually require cv2.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import hypot
from pathlib import Path
from typing import Any


Line = tuple[float, float, float, float]
Point = tuple[float, float]


def parse_line_arg(value: str) -> Line:
    """Parse a ``"x1,y1,x2,y2"`` string into four floats.

    Rejects malformed inputs and zero-length lines, which would otherwise blow
    up later in ``LineCounter.side`` (division by zero on the unit normal).
    """

    if value is None:
        raise ValueError("--line must be provided when running count_video.")
    parts = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if len(parts) != 4:
        raise ValueError(
            f"--line expects 'x1,y1,x2,y2'; got {value!r} ({len(parts)} numeric parts)."
        )
    try:
        x1, y1, x2, y2 = (float(piece) for piece in parts)
    except ValueError as exc:
        raise ValueError(f"--line contains non-numeric value: {value!r}") from exc
    if hypot(x2 - x1, y2 - y1) == 0:
        raise ValueError(f"--line endpoints must differ; got {value!r}.")
    return (x1, y1, x2, y2)


def parse_class_filter(value: str | None) -> set[int]:
    """Parse a comma-separated class-id list into a set of ints.

    Empty / ``None`` yields the empty set, meaning "exclude nothing".
    """

    if value is None:
        return set()
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not pieces:
        return set()
    try:
        return {int(piece) for piece in pieces}
    except ValueError as exc:
        raise ValueError(f"class filter expects comma-separated ints; got {value!r}") from exc


def bbox_bottom_center(xyxy: Sequence[float]) -> Point:
    """Return ``((x1+x2)/2, y2)`` for a YOLO bbox.

    The wheels-on-road bottom-center is more stable than the geometric center
    for counting line crossings — vehicle height does not shift this anchor.
    """

    if len(xyxy) != 4:
        raise ValueError("xyxy must have 4 elements.")
    x1, _y1, x2, y2 = (float(v) for v in xyxy)
    return ((x1 + x2) / 2.0, y2)


def draw_counting_line(
    frame: "Any",
    line: Sequence[float],
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> "Any":
    """Draw the counting line with end arrows onto ``frame`` in place."""

    import cv2  # lazy

    x1, y1, x2, y2 = (int(round(v)) for v in line)
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.arrowedLine(frame, (x1, y1), (x2, y2), color, thickness, line_type=cv2.LINE_AA, tipLength=0.04)
    cv2.arrowedLine(frame, (x2, y2), (x1, y1), color, thickness, line_type=cv2.LINE_AA, tipLength=0.04)
    return frame


def draw_count_dots(
    frame: "Any",
    centers: Iterable[Sequence[float]],
    color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 6,
) -> "Any":
    """Highlight track centers that crossed the line on the current frame."""

    import cv2  # lazy

    for cx, cy in centers:
        cv2.circle(frame, (int(round(cx)), int(round(cy))), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (int(round(cx)), int(round(cy))), radius + 2, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return frame


def draw_count_overlay(
    frame: "Any",
    total: int,
    direction_counts: Mapping[str, int],
    per_class_counts: Mapping[int, int],
    class_names: Mapping[int, str],
    top_k: int = 3,
    origin: tuple[int, int] = (12, 12),
) -> "Any":
    """Render a translucent stat panel on the top-left of ``frame``."""

    import cv2  # lazy

    pos_to_neg = int(direction_counts.get("positive_to_negative", 0))
    neg_to_pos = int(direction_counts.get("negative_to_positive", 0))
    top_classes = sorted(per_class_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    lines = [
        f"Total: {int(total)}",
        f"-> (pos->neg): {pos_to_neg}",
        f"<- (neg->pos): {neg_to_pos}",
    ]
    if top_classes:
        lines.append(f"Top-{len(top_classes)} classes:")
        for cls_id, cnt in top_classes:
            name = class_names.get(int(cls_id), str(cls_id))
            lines.append(f"  {name}: {int(cnt)}")
    else:
        lines.append("Top classes: -")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    line_spacing = 8
    pad_x, pad_y = 10, 8

    (_, text_h), baseline = cv2.getTextSize("Hg", font, font_scale, font_thickness)
    row_h = text_h + line_spacing
    text_w = 0
    for text in lines:
        (w, _), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w = max(text_w, w)

    panel_w = text_w + pad_x * 2
    panel_h = row_h * len(lines) + pad_y * 2 + baseline
    x0, y0 = origin
    x1, y1 = x0 + panel_w, y0 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), thickness=1)

    cursor_y = y0 + pad_y + text_h
    for text in lines:
        cv2.putText(
            frame,
            text,
            (x0 + pad_x, cursor_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
        cursor_y += row_h
    return frame


def pick_video_writer(
    path: Path,
    fps: float,
    size: tuple[int, int],
) -> tuple["Any", str, Path]:
    """Open a ``cv2.VideoWriter`` with a portable codec.

    Tries ``mp4v`` first. If the writer fails to open (which on some Linux
    OpenCV builds also manifests as a silently-empty file), falls back to
    ``XVID`` with a ``.avi`` extension.

    Returns ``(writer, codec_label, actual_path)``.
    """

    import cv2  # lazy

    width, height = int(size[0]), int(size[1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    primary_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), primary_fourcc, float(fps), (width, height))
    if writer.isOpened():
        return writer, "mp4v", path

    writer.release()
    fallback_path = path.with_suffix(".avi")
    fallback_fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(fallback_path), fallback_fourcc, float(fps), (width, height))
    if writer.isOpened():
        return writer, "XVID", fallback_path

    writer.release()
    raise RuntimeError(
        f"cv2.VideoWriter failed to open both mp4v ({path}) and XVID ({fallback_path}). "
        "Check the OpenCV FFmpeg build and the output directory permissions."
    )
