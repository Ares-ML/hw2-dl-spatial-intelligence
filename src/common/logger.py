"""Logging helpers, including a safe SwanLab wrapper."""

from __future__ import annotations

import importlib
import logging
import subprocess
from pathlib import Path
from typing import Any, Mapping


def _normalize_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    return logging._nameToLevel.get(level.upper(), logging.INFO)


def setup_logging(
    name: str = "hw2",
    log_file: str | Path | None = None,
    level: int | str = logging.INFO,
    use_console: bool = True,
    file_mode: str = "a",
) -> logging.Logger:
    """Create a reusable logger without duplicating handlers on repeated calls."""

    logger = logging.getLogger(name)
    logger.setLevel(_normalize_level(level))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if use_console and not any(getattr(h, "_hw2_console", False) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(_normalize_level(level))
        console_handler._hw2_console = True  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        exists = any(getattr(h, "_hw2_log_file", None) == resolved for h in logger.handlers)
        if not exists:
            file_handler = logging.FileHandler(path, mode=file_mode, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(_normalize_level(level))
            file_handler._hw2_log_file = resolved  # type: ignore[attr-defined]
            logger.addHandler(file_handler)

    return logger


def get_git_commit(repo_root: str | Path | None = None, short: bool = True) -> str:
    """Return the current git commit hash, or 'unknown' outside a git checkout."""

    args = ["git", "rev-parse"]
    if short:
        args.append("--short")
    args.append("HEAD")

    result = subprocess.run(
        args,
        cwd=Path(repo_root) if repo_root is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


class SwanLabLogger:
    """Small no-op-safe wrapper around SwanLab logging.

    Use ``enabled=False`` in tests or dry runs to avoid network/login side effects.
    If SwanLab import or initialization fails, this wrapper silently degrades to a
    no-op so training can continue and local smoke tests remain lightweight.
    """

    def __init__(
        self,
        project: str | None = None,
        experiment_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        enabled: bool = True,
        **init_kwargs: Any,
    ) -> None:
        self.enabled = False
        self.run = None
        self._swanlab = None

        if not enabled or project is None:
            return

        try:
            swanlab = importlib.import_module("swanlab")
            kwargs: dict[str, Any] = {"project": project}
            if experiment_name is not None:
                kwargs["experiment_name"] = experiment_name
            if config is not None:
                kwargs["config"] = dict(config)
            kwargs.update(init_kwargs)
            self.run = swanlab.init(**kwargs)
            self._swanlab = swanlab
            self.enabled = True
        except Exception:
            self.enabled = False
            self.run = None
            self._swanlab = None

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._swanlab is None:
            return
        data = dict(metrics)
        if step is None:
            self._swanlab.log(data)
        else:
            self._swanlab.log(data, step=step)

    def update_config(self, values: Mapping[str, Any]) -> None:
        if not self.enabled or self._swanlab is None:
            return
        config = getattr(self._swanlab, "config", None)
        if config is not None and hasattr(config, "update"):
            config.update(dict(values))

    def finish(self) -> None:
        if not self.enabled or self._swanlab is None:
            return
        finish = getattr(self._swanlab, "finish", None)
        if callable(finish):
            finish()
