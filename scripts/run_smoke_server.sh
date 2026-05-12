#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-hw2}"

TASK1_DATA_ROOT="${TASK1_DATA_ROOT:-data/flower102}"
TASK1_DEVICE="${TASK1_DEVICE:-cuda}"
TASK1_LOG_FILE="${TASK1_LOG_FILE:-logs/smoke/t1_resnet18_smoke.log}"

TASK2_DATA="${TASK2_DATA:-data/road_vehicle/data.yaml}"
TASK2_MODEL="${TASK2_MODEL:-yolov8n.pt}"
TASK2_DEVICE="${TASK2_DEVICE:-0}"
TASK2_LOG_FILE="${TASK2_LOG_FILE:-logs/smoke/t2_yolov8n_smoke.log}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

source scripts/cuda_driver_shim.sh
prepare_cuda_driver_shim

mkdir -p logs/smoke

echo "== Verifying CUDA environment in conda env '${ENV_NAME}' =="
conda run -n "${ENV_NAME}" python scripts/verify_env.py --require-cuda

echo "== Running Task 1 ResNet-18 smoke test =="
conda run -n "${ENV_NAME}" python scripts/smoke_task1_resnet18.py \
  --data-root "${TASK1_DATA_ROOT}" \
  --device "${TASK1_DEVICE}" \
  --log-file "${TASK1_LOG_FILE}"

echo "== Running Task 2 YOLOv8 smoke test =="
conda run -n "${ENV_NAME}" python scripts/smoke_task2_yolo.py \
  --data "${TASK2_DATA}" \
  --model "${TASK2_MODEL}" \
  --device "${TASK2_DEVICE}" \
  --log-file "${TASK2_LOG_FILE}"

echo "Smoke tests completed:"
echo "- ${TASK1_LOG_FILE}"
echo "- ${TASK2_LOG_FILE}"
