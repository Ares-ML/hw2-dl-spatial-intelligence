"""Smoke test one epoch of YOLOv8 training on the Road Vehicle dataset."""

from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "road_vehicle" / "data.yaml"
DEFAULT_LOG_FILE = REPO_ROOT / "logs" / "smoke" / "t2_yolov8n_smoke.log"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "smoke"
DEFAULT_NAME = "road_vehicle_yolov8n_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Ultralytics data.yaml path.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name/path, e.g. yolov8n.pt.")
    parser.add_argument("--epochs", type=int, default=1, help="Smoke-test epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--device", default="0", help="Ultralytics device string.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Ultralytics project directory.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Ultralytics run name.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE, help="Full smoke-test log path.")
    parser.add_argument("--exist-ok", action="store_true", default=True, help="Reuse an existing smoke run directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.project.mkdir(parents=True, exist_ok=True)

    with args.log_file.open("w", encoding="utf-8") as log:
        log.write("Task 2 YOLOv8 Road Vehicle smoke test\n")
        log.write(f"repo_root={REPO_ROOT}\n")
        log.write(
            "command=yolo detect train "
            f"data={args.data} model={args.model} epochs={args.epochs} imgsz={args.imgsz} "
            f"batch={args.batch} project={args.project} name={args.name} "
            f"device={args.device} workers={args.workers} exist_ok={args.exist_ok}\n\n"
        )
        log.flush()

        try:
            with redirect_stdout(log), redirect_stderr(log):
                from ultralytics import YOLO

                model = YOLO(args.model)
                model.train(
                    data=str(args.data),
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    project=str(args.project),
                    name=args.name,
                    device=args.device,
                    workers=args.workers,
                    exist_ok=args.exist_ok,
                )
        except Exception as exc:
            log.write(f"\nFAIL ultralytics smoke train failed: {exc}\n")
            return 1

    run_dir = args.project / args.name
    results_csv = run_dir / "results.csv"
    if not run_dir.is_dir() or not results_csv.is_file():
        with args.log_file.open("a", encoding="utf-8") as log:
            log.write(f"FAIL missing expected run output: {run_dir} / results.csv\n")
        return 1

    with args.log_file.open("a", encoding="utf-8") as log:
        log.write(f"PASS run_dir={run_dir}\n")
        log.write(f"PASS results_csv={results_csv}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
