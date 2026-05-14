"""Expand Task 1 grid YAML into per-run training configs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-config", type=Path, required=True, help="Grid YAML with base and experiments sections.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated per-run YAML files.")
    return parser.parse_args()


def sanitize_run_id(value: Any, index: int) -> str:
    run_id = str(value or f"run_{index:02d}")
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("._-")
    if not run_id:
        raise ValueError(f"experiment #{index} has an empty id after sanitization.")
    return run_id


def load_grid(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file) or {}
    base = document.get("base") or {}
    experiments = document.get("experiments") or document.get("grid") or []
    if not isinstance(base, dict):
        raise TypeError(f"{path} must contain a mapping 'base'.")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"{path} must contain a non-empty list 'experiments'.")
    for index, experiment in enumerate(experiments, start=1):
        if not isinstance(experiment, dict):
            raise TypeError(f"experiment #{index} must be a mapping.")
    return base, experiments


def expanded_config(base: dict[str, Any], experiment: dict[str, Any], run_id: str, grid_path: Path) -> dict[str, Any]:
    config = dict(base)
    config.update({key: value for key, value in experiment.items() if key != "id"})
    config["experiment_id"] = run_id
    config["grid_config"] = str(grid_path)
    swanlab_config = config.get("swanlab") if isinstance(config.get("swanlab"), dict) else {}
    swanlab_config.setdefault("tags", ["day2", "grid", "classification"])
    config["swanlab"] = swanlab_config
    return config


def main() -> int:
    args = parse_args()
    grid_path = args.grid_config
    output_dir = args.output_dir
    base, experiments = load_grid(grid_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, experiment in enumerate(experiments, start=1):
        run_id = sanitize_run_id(experiment.get("id"), index)
        output_path = output_dir / f"{run_id}.yaml"
        config = expanded_config(base, experiment, run_id, grid_path)
        with output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False, allow_unicode=False)
        print(f"{run_id}\t{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
