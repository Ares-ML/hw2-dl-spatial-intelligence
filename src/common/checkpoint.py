"""Checkpoint save/load helpers for classification and segmentation training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except ImportError:  # pragma: no cover - lets docs/tools import without torch.
    torch = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for checkpoint operations.")
    return torch


def _state_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(f"Object of type {type(obj)!r} does not provide state_dict().")


def save_checkpoint(
    path: str | Path,
    *,
    model: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Serialize a training checkpoint to ``path``."""

    torch_module = _require_torch()
    checkpoint: dict[str, Any] = {}
    if model is not None:
        checkpoint["model_state"] = _state_dict(model)
    if optimizer is not None:
        checkpoint["optimizer_state"] = _state_dict(optimizer)
    if scheduler is not None:
        checkpoint["scheduler_state"] = _state_dict(scheduler)
    if epoch is not None:
        checkpoint["epoch"] = int(epoch)
    if metrics is not None:
        checkpoint["metrics"] = dict(metrics)
    if config is not None:
        checkpoint["config"] = dict(config)
    if extra is not None:
        checkpoint.update(dict(extra))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(checkpoint, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    map_location: str | Any = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore model/optimizer/scheduler."""

    torch_module = _require_torch()
    checkpoint = torch_module.load(Path(path), map_location=map_location)

    if model is not None and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=strict)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint


class CheckpointManager:
    """Manage ``last.pt`` and metric-driven ``best.pt`` checkpoints."""

    def __init__(
        self,
        directory: str | Path,
        monitor: str,
        mode: str = "max",
        best_filename: str = "best.pt",
        last_filename: str = "last.pt",
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'.")
        self.directory = Path(directory)
        self.monitor = monitor
        self.mode = mode
        self.best_path = self.directory / best_filename
        self.last_path = self.directory / last_filename
        self.best_score: float | None = None

    def is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def save(
        self,
        *,
        model: Any,
        epoch: int,
        metrics: Mapping[str, Any],
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        config: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save last checkpoint and update best checkpoint when monitor improves."""

        self.directory.mkdir(parents=True, exist_ok=True)
        metric_value = metrics.get(self.monitor)
        is_best = metric_value is not None and self.is_better(float(metric_value))

        payload_extra = dict(extra or {})
        payload_extra.update(
            {
                "monitor": self.monitor,
                "mode": self.mode,
                "best_score": float(metric_value) if is_best else self.best_score,
            }
        )

        save_checkpoint(
            self.last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            config=config,
            extra=payload_extra,
        )

        best_path = None
        if is_best:
            self.best_score = float(metric_value)
            payload_extra["best_score"] = self.best_score
            best_path = save_checkpoint(
                self.best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
                extra=payload_extra,
            )

        return {"last": self.last_path, "best": best_path, "is_best": is_best}
