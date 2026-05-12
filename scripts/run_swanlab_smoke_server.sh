#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-hw2}"
LINKS_FILE="${LINKS_FILE:-logs/swanlab/swanlab_links.md}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

source scripts/cuda_driver_shim.sh
prepare_cuda_driver_shim

mkdir -p "$(dirname "${LINKS_FILE}")"
cat > "${LINKS_FILE}" <<'EOF'
# SwanLab Day 1 Smoke Runs

| Task | Run | Project URL | Experiment URL |
| --- | --- | --- | --- |
EOF

echo "== Verifying CUDA environment in conda env '${ENV_NAME}' =="
conda run -n "${ENV_NAME}" python scripts/verify_env.py --require-cuda

echo "== Task 1 SwanLab smoke =="
conda run -n "${ENV_NAME}" python -m src.task1_cls.train \
  --config configs/task1_resnet18_pretrained.yaml \
  --epochs 1 \
  --max-batches 5 \
  --max-val-batches 2 \
  --device cuda \
  --swanlab-mode cloud \
  --require-swanlab \
  --links-file "${LINKS_FILE}"

echo "== Task 2 SwanLab smoke =="
conda run -n "${ENV_NAME}" python -m src.task2_detect_track.train \
  --config configs/task2_yolov8n.yaml \
  --epochs 1 \
  --device 0 \
  --swanlab-mode cloud \
  --require-swanlab \
  --links-file "${LINKS_FILE}"

echo "== Task 3 SwanLab smoke =="
conda run -n "${ENV_NAME}" python -m src.task3_seg.train \
  --config configs/task3_unet_ce.yaml \
  --epochs 1 \
  --max-batches 5 \
  --max-val-batches 2 \
  --device cuda \
  --swanlab-mode cloud \
  --require-swanlab \
  --links-file "${LINKS_FILE}"

echo "SwanLab links:"
cat "${LINKS_FILE}"
