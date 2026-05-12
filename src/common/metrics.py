"""Metrics shared by classification, detection, and segmentation tasks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - lets non-training tools import this module.
    torch = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for metric computations.")
    return torch


def _as_label_prediction(pred: Any, target: Any) -> Any:
    torch_module = _require_torch()
    pred = torch_module.as_tensor(pred)
    target = torch_module.as_tensor(target)
    if pred.ndim == target.ndim + 1:
        pred = pred.argmax(dim=1)
    return pred.long(), target.long()


def topk_accuracy(
    logits: Any,
    target: Any,
    topk: tuple[int, ...] = (1,),
    ignore_index: int | None = None,
) -> list[float]:
    """Compute top-k classification accuracy as floats in [0, 1]."""

    torch_module = _require_torch()
    logits = torch_module.as_tensor(logits)
    target = torch_module.as_tensor(target).long()
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C].")
    if target.ndim != 1:
        target = target.reshape(-1)

    if ignore_index is not None:
        mask = target != ignore_index
        logits = logits[mask]
        target = target[mask]

    if target.numel() == 0:
        return [0.0 for _ in topk]

    max_k = min(max(topk), logits.size(1))
    pred = logits.topk(max_k, dim=1).indices.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [float(correct[: min(k, max_k)].any(dim=0).float().mean().item()) for k in topk]


def confusion_matrix(
    pred: Any,
    target: Any,
    num_classes: int,
    ignore_index: int | None = 255,
) -> Any:
    """Compute a row-target, column-prediction confusion matrix."""

    torch_module = _require_torch()
    pred, target = _as_label_prediction(pred, target)
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    mask = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        mask &= target != ignore_index
    pred = pred[mask]
    target = target[mask]
    pred = pred.clamp(min=0, max=num_classes - 1)

    indices = target * num_classes + pred
    matrix = torch_module.bincount(indices, minlength=num_classes**2)
    return matrix.reshape(num_classes, num_classes).long()


def pixel_accuracy(
    pred: Any,
    target: Any,
    num_classes: int | None = None,
    ignore_index: int | None = 255,
) -> float:
    """Compute semantic segmentation pixel accuracy."""

    torch_module = _require_torch()
    pred, target = _as_label_prediction(pred, target)
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    mask = torch_module.ones_like(target, dtype=torch_module.bool)
    if ignore_index is not None:
        mask &= target != ignore_index
    if num_classes is not None:
        mask &= (target >= 0) & (target < num_classes)
    if mask.sum().item() == 0:
        return 0.0
    return float((pred[mask] == target[mask]).float().mean().item())


def intersection_and_union(
    pred: Any,
    target: Any,
    num_classes: int,
    ignore_index: int | None = 255,
) -> tuple[Any, Any]:
    """Return per-class intersection and union tensors."""

    matrix = confusion_matrix(pred, target, num_classes=num_classes, ignore_index=ignore_index)
    intersection = matrix.diag()
    target_area = matrix.sum(dim=1)
    pred_area = matrix.sum(dim=0)
    union = target_area + pred_area - intersection
    return intersection.long(), union.long()


def mean_iou(
    pred: Any,
    target: Any,
    num_classes: int,
    ignore_index: int | None = 255,
) -> dict[str, Any]:
    """Compute mIoU, class-wise IoU, and pixel accuracy."""

    intersection, union = intersection_and_union(pred, target, num_classes, ignore_index)
    class_iou: list[float | None] = []
    valid_values: list[float] = []
    for inter, denom in zip(intersection.tolist(), union.tolist()):
        if denom == 0:
            class_iou.append(None)
            continue
        value = float(inter) / float(denom)
        class_iou.append(value)
        valid_values.append(value)

    miou = sum(valid_values) / len(valid_values) if valid_values else 0.0
    return {
        "miou": float(miou),
        "class_iou": class_iou,
        "valid_classes": len(valid_values),
        "pixel_acc": pixel_accuracy(pred, target, num_classes=num_classes, ignore_index=ignore_index),
    }


def _parse_scalar(value: str) -> float | str:
    value = value.strip()
    if value == "":
        return value
    try:
        return float(value)
    except ValueError:
        return value


def read_ultralytics_results_csv(path: str | Path) -> list[dict[str, float | str]]:
    """Read Ultralytics ``results.csv`` using the standard library."""

    rows: list[dict[str, float | str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({key.strip(): _parse_scalar(value) for key, value in row.items() if key is not None})
    return rows


def latest_ultralytics_metrics(path: str | Path) -> dict[str, float | str]:
    """Return the final row from an Ultralytics ``results.csv`` file."""

    rows = read_ultralytics_results_csv(path)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows[-1]
