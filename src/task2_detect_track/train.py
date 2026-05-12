"""Train Task 2 YOLOv8 detectors and mirror metrics to SwanLab."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from src.common.logger import setup_logging
from src.common.metrics import latest_ultralytics_metrics
from src.common.seed import set_seed
from src.common.swanlab_logger import finish, format_run_name, init_run, log_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "task2_yolov8n.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="Accepted for a shared smoke CLI; not used by Ultralytics.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--swanlab-mode", default="disabled", choices=["cloud", "offline", "local", "disabled"])
    parser.add_argument("--require-swanlab", action="store_true")
    parser.add_argument("--links-file", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "logs" / "swanlab" / "task2_train.log")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def append_link(path: Path | None, task: str, run_name: str, project_url: str, experiment_url: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"| {task} | {run_name} | {project_url} | {experiment_url} |\n")


def select_weights(configured: str) -> str:
    candidates = [Path(configured), REPO_ROOT / configured, REPO_ROOT / "weights" / configured]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return configured


def map_yolo_metrics(row: dict[str, float | str]) -> dict[str, float]:
    aliases = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50_95",
        "train/box_loss": "train_box_loss",
        "train/cls_loss": "train_cls_loss",
        "train/dfl_loss": "train_dfl_loss",
        "val/box_loss": "val_box_loss",
        "val/cls_loss": "val_cls_loss",
        "val/dfl_loss": "val_dfl_loss",
    }
    metrics: dict[str, float] = {}
    for source, target in aliases.items():
        value = row.get(source)
        if isinstance(value, (int, float)):
            metrics[target] = float(value)
    return metrics


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    epochs = args.epochs or int(cfg.get("epochs", 1))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    lr_value = args.lr if args.lr is not None else cfg.get("lr", "default")
    set_seed(seed)
    logger = setup_logging("hw2.task2", args.log_file, level=logging.INFO, file_mode="w")

    run_values = {**cfg, "epochs": epochs, "seed": seed, "lr": lr_value}
    run_name = format_run_name(run_values)
    run = init_run(task="task2", run_name=run_name, config=run_values, mode=args.swanlab_mode, tags=["day1", "smoke", "detection"])
    if args.require_swanlab and not run.enabled:
        raise RuntimeError("SwanLab cloud run was required but initialization failed.")

    from ultralytics import YOLO

    model_path = select_weights(str(cfg.get("weights", "yolov8n.pt")))
    model = YOLO(model_path)
    project = REPO_ROOT / str(cfg.get("project", "runs/task2"))
    model.train(
        data=str(cfg.get("data", "data/road_vehicle/data.yaml")),
        epochs=epochs,
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("batch", 16)),
        project=str(project),
        name=str(cfg.get("name", "road_vehicle_yolov8n")),
        device=str(args.device or cfg.get("device", "0")),
        workers=int(cfg.get("workers", 4)),
        exist_ok=True,
        seed=seed,
        amp=bool(cfg.get("amp", False)),
        plots=bool(cfg.get("plots", False)),
    )

    results_csv = project / str(cfg.get("name", "road_vehicle_yolov8n")) / "results.csv"
    metrics = map_yolo_metrics(latest_ultralytics_metrics(results_csv))
    logger.info("metrics=%s", metrics)
    log_metrics(metrics, step=epochs, task="task2")
    run = finish()
    append_link(args.links_file, "task2", run_name, run.project_url, run.experiment_url)
    if args.max_batches is not None:
        logger.info("max_batches=%s was ignored because Ultralytics does not expose that train override.", args.max_batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
