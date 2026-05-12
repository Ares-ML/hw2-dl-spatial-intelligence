"""Thin SwanLab wrapper used by all HW2 training entrypoints."""

from __future__ import annotations

import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "swanlab.yaml"
DEFAULT_VALUES = {
    "workspace": "Ares-ML",
    "project": "hw2-dl-spatial-intelligence",
    "groups": {key: key for key in ("task1", "task2", "task3")},
    "metric_prefixes": {key: key for key in ("task1", "task2", "task3")},
    "run_name_format": "{model}_{init}_{loss}_{lr}_{epoch}_{seed}",
}
_ACTIVE: "SwanLabRun | None" = None


def _with_env_overrides(values: dict[str, Any]) -> dict[str, Any]:
    for env_key, config_key in (("SWANLAB_WORKSPACE", "workspace"), ("SWANLAB_PROJ_NAME", "project")):
        if os.environ.get(env_key):
            values[config_key] = os.environ[env_key]
    return values


@dataclass
class SwanLabRun:
    enabled: bool = False
    module: Any = None
    run: Any = None
    project_url: str = ""
    experiment_url: str = ""
    error: str = ""


def load_swanlab_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        return _with_env_overrides(dict(DEFAULT_VALUES))
    try:
        import yaml
    except ImportError:
        return _with_env_overrides(dict(DEFAULT_VALUES))
    with config_path.open("r", encoding="utf-8") as file:
        values = {**DEFAULT_VALUES, **(yaml.safe_load(file) or {})}
    return _with_env_overrides(values)


def _lr_text(value: Any) -> str:
    if isinstance(value, (float, int)):
        text = f"{value:.0e}"
    else:
        text = str(value)
    return text if text.startswith("lr") else f"lr{text}"


def format_run_name(values: Mapping[str, Any], swanlab_config: Mapping[str, Any] | None = None) -> str:
    template = (swanlab_config or load_swanlab_config()).get("run_name_format", DEFAULT_VALUES["run_name_format"])
    parts = {
        "model": values.get("model", "model"),
        "init": values.get("init", "init"),
        "loss": values.get("loss", "loss"),
        "lr": _lr_text(values.get("lr", "lr")),
        "epoch": f"e{values.get('epochs', values.get('epoch', 'x'))}",
        "seed": f"s{values.get('seed', 'x')}",
    }
    text = re.sub(r"e([+-])0+(\d+)", r"e\1\2", template.format(**parts))
    return text.replace(".", "p").replace("+", "plus")


def metric_name(task: str, name: str, swanlab_config: Mapping[str, Any] | None = None) -> str:
    prefixes = (swanlab_config or load_swanlab_config()).get("metric_prefixes") or {}
    return f"{prefixes.get(task, task)}/{name}"


def init_run(
    *,
    task: str,
    run_name: str,
    config: Mapping[str, Any] | None = None,
    swanlab_config_path: str | Path = DEFAULT_CONFIG,
    mode: str = "disabled",
    tags: list[str] | None = None,
    enabled: bool = True,
) -> SwanLabRun:
    global _ACTIVE
    swan_cfg = load_swanlab_config(swanlab_config_path)
    if not enabled or mode == "disabled":
        _ACTIVE = SwanLabRun()
        return _ACTIVE
    try:
        import swanlab

        run = swanlab.init(
            workspace=swan_cfg.get("workspace"),
            project=swan_cfg.get("project"),
            group=(swan_cfg.get("groups") or {}).get(task, task),
            experiment_name=run_name,
            config=dict(config or {}),
            mode=mode,
            tags=tags,
            logdir=str(REPO_ROOT / "swanlog"),
        )
        cloud = getattr(getattr(run, "public", None), "cloud", None)
        project_url = getattr(cloud, "project_url", "") or getattr(run, "get_project_url", lambda: "")()
        experiment_url = getattr(cloud, "experiment_url", "") or getattr(run, "get_url", lambda: "")()
        _ACTIVE = SwanLabRun(True, swanlab, run, str(project_url or ""), str(experiment_url or ""))
    except Exception as exc:
        if mode == "cloud":
            print(f"SwanLab init failed: {exc!r}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        _ACTIVE = SwanLabRun(error=repr(exc))
    return _ACTIVE


def log_metrics(metrics: Mapping[str, Any], step: int | None = None, task: str | None = None) -> None:
    if _ACTIVE is None or not _ACTIVE.enabled or _ACTIVE.module is None:
        return
    data = {metric_name(task, key) if task else key: value for key, value in dict(metrics).items()}
    _ACTIVE.module.log(data) if step is None else _ACTIVE.module.log(data, step=step)


def log_image(name: str, image: Any, caption: str | None = None, step: int | None = None, task: str | None = None) -> None:
    if _ACTIVE is None or not _ACTIVE.enabled or _ACTIVE.module is None:
        return
    key = metric_name(task, name) if task else name
    try:
        image = _ACTIVE.module.Image(image, caption=caption) if caption is not None else _ACTIVE.module.Image(image)
    except Exception:
        pass
    log_metrics({key: image}, step=step)


def finish() -> SwanLabRun:
    global _ACTIVE
    run = _ACTIVE or SwanLabRun()
    cloud = getattr(getattr(run.run, "public", None), "cloud", None) if run.run is not None else None
    project_url = getattr(cloud, "project_url", "") or getattr(run.run, "get_project_url", lambda: "")()
    experiment_url = getattr(cloud, "experiment_url", "") or getattr(run.run, "get_url", lambda: "")()
    run.project_url = run.project_url or str(project_url or "")
    run.experiment_url = run.experiment_url or str(experiment_url or "")
    close = getattr(run.module, "finish", None) if run.enabled and run.module is not None else None
    if callable(close):
        close()
    _ACTIVE = None
    return run
