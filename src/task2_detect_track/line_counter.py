"""Virtual line crossing counter for Task 2 tracking outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import hypot
from typing import Any


Point = tuple[float, float]
Line = tuple[float, float, float, float]


@dataclass(frozen=True)
class CountEvent:
    """Result returned by one counter update."""

    track_id: int
    counted: bool
    side: int
    direction: str | None
    total_count: int


class LineCounter:
    """Count unique track IDs that cross a directed line segment.

    The line is represented by two endpoints ``(x1, y1, x2, y2)``. A track is
    counted once when its center moves from one non-zero side of the line to the
    other. ``margin`` is a perpendicular distance in pixels; points within that
    band do not update the stored side, which prevents near-line jitter from
    creating accidental crossings.
    """

    POSITIVE_TO_NEGATIVE = "positive_to_negative"
    NEGATIVE_TO_POSITIVE = "negative_to_positive"

    def __init__(
        self,
        line: Sequence[float],
        margin: float = 0.0,
        include_classes: Iterable[int] | None = None,
    ) -> None:
        if len(line) != 4:
            raise ValueError("line must be (x1, y1, x2, y2).")
        x1, y1, x2, y2 = (float(value) for value in line)
        length = hypot(x2 - x1, y2 - y1)
        if length == 0:
            raise ValueError("line endpoints must not be identical.")
        if margin < 0:
            raise ValueError("margin must be non-negative.")

        self.line: Line = (x1, y1, x2, y2)
        self.margin = float(margin)
        self.include_classes = set(include_classes) if include_classes is not None else None
        self.last_side: dict[int, int] = {}
        self.counted_ids: set[int] = set()
        self.total_count = 0
        self.direction_counts = {
            self.POSITIVE_TO_NEGATIVE: 0,
            self.NEGATIVE_TO_POSITIVE: 0,
        }

    def reset(self) -> None:
        """Clear all accumulated counting state."""

        self.last_side.clear()
        self.counted_ids.clear()
        self.total_count = 0
        for key in self.direction_counts:
            self.direction_counts[key] = 0

    def side(self, point: Sequence[float]) -> int:
        """Return the signed side of ``point`` relative to the configured line."""

        x, y = _as_pair(point)
        x1, y1, x2, y2 = self.line
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        distance = cross / hypot(x2 - x1, y2 - y1)
        if abs(distance) <= self.margin:
            return 0
        return 1 if distance > 0 else -1

    def update(
        self,
        track_id: int,
        geometry: Sequence[float] | None = None,
        *,
        center: Sequence[float] | None = None,
        xyxy: Sequence[float] | None = None,
        class_id: int | None = None,
    ) -> CountEvent:
        """Update the counter with one tracked object.

        ``geometry`` may be either a center ``(x, y)`` or a box
        ``(x1, y1, x2, y2)``. The explicit ``center=`` and ``xyxy=`` keyword
        arguments are also supported for clarity at call sites.
        """

        normalized_track_id = int(track_id)
        if self.include_classes is not None and class_id not in self.include_classes:
            return CountEvent(normalized_track_id, False, 0, None, self.total_count)

        point = self._resolve_center(geometry=geometry, center=center, xyxy=xyxy)
        current_side = self.side(point)
        if current_side == 0:
            return CountEvent(normalized_track_id, False, current_side, None, self.total_count)

        previous_side = self.last_side.get(normalized_track_id)
        direction = self._direction(previous_side, current_side)
        counted = direction is not None and normalized_track_id not in self.counted_ids
        if counted:
            self.total_count += 1
            self.counted_ids.add(normalized_track_id)
            self.direction_counts[direction] += 1

        self.last_side[normalized_track_id] = current_side
        return CountEvent(normalized_track_id, counted, current_side, direction if counted else None, self.total_count)

    def update_many(self, tracks: Iterable[Mapping[str, Any] | Sequence[Any]]) -> list[CountEvent]:
        """Update the counter from several track records."""

        events = []
        for record in tracks:
            track_id, geometry, class_id = self._parse_track(record)
            events.append(self.update(track_id, geometry, class_id=class_id))
        return events

    def _resolve_center(
        self,
        *,
        geometry: Sequence[float] | None,
        center: Sequence[float] | None,
        xyxy: Sequence[float] | None,
    ) -> Point:
        provided = sum(value is not None for value in (geometry, center, xyxy))
        if provided != 1:
            raise ValueError("provide exactly one of geometry, center, or xyxy.")
        if center is not None:
            return _as_pair(center)
        if xyxy is not None:
            return _xyxy_center(xyxy)
        if geometry is None:
            raise ValueError("geometry must not be None.")
        if len(geometry) == 2:
            return _as_pair(geometry)
        if len(geometry) == 4:
            return _xyxy_center(geometry)
        raise ValueError("geometry must be either (x, y) or (x1, y1, x2, y2).")

    @classmethod
    def _direction(cls, previous_side: int | None, current_side: int) -> str | None:
        if previous_side is None or previous_side == current_side:
            return None
        if previous_side > 0 and current_side < 0:
            return cls.POSITIVE_TO_NEGATIVE
        if previous_side < 0 and current_side > 0:
            return cls.NEGATIVE_TO_POSITIVE
        return None

    @staticmethod
    def _parse_track(record: Mapping[str, Any] | Sequence[Any]) -> tuple[Any, Any, Any | None]:
        if isinstance(record, Mapping):
            track_id = record.get("track_id", record.get("id"))
            class_id = record.get("class_id", record.get("cls"))
            if "center" in record:
                return track_id, record["center"], class_id
            if "xyxy" in record:
                return track_id, record["xyxy"], class_id
            raise ValueError("track mapping must contain center or xyxy.")
        if len(record) == 2:
            track_id, geometry = record
            return track_id, geometry, None
        if len(record) == 3:
            track_id, geometry, class_id = record
            return track_id, geometry, class_id
        raise ValueError("track sequence must be (track_id, geometry[, class_id]).")


def _as_pair(point: Sequence[float]) -> Point:
    if len(point) != 2:
        raise ValueError("point must be (x, y).")
    return float(point[0]), float(point[1])


def _xyxy_center(box: Sequence[float]) -> Point:
    if len(box) != 4:
        raise ValueError("box must be (x1, y1, x2, y2).")
    x1, y1, x2, y2 = (float(value) for value in box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
