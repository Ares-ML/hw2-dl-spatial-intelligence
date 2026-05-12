#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-hw2}"

TASK1_DATA_ROOT="${TASK1_DATA_ROOT:-data/flower102}"
TASK1_DEVICE="${TASK1_DEVICE:-cuda}"
TASK1_LOG_FILE="${TASK1_LOG_FILE:-logs/smoke/t1_resnet18_smoke.log}"

TASK2_DATA="${TASK2_DATA:-data/road_vehicle/data.yaml}"
TASK2_MODEL="${TASK2_MODEL:-}"
TASK2_WEIGHT_PATH="${TASK2_WEIGHT_PATH:-weights/yolov8n.pt}"
TASK2_ALLOW_YAML_FALLBACK="${TASK2_ALLOW_YAML_FALLBACK:-1}"
TASK2_DEVICE="${TASK2_DEVICE:-0}"
TASK2_LOG_FILE="${TASK2_LOG_FILE:-logs/smoke/t2_yolov8n_smoke.log}"
TASK2_AMP="${TASK2_AMP:-0}"
TASK2_PLOTS="${TASK2_PLOTS:-0}"

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

TASK2_NOTES=()
if [[ -n "${TASK2_MODEL}" ]]; then
  TASK2_NOTES+=("model_source=${TASK2_MODEL}")
elif [[ -s yolov8n.pt ]]; then
  TASK2_MODEL="yolov8n.pt"
  TASK2_NOTES+=("model_source=repo_root/yolov8n.pt")
elif [[ -s "${TASK2_WEIGHT_PATH}" ]]; then
  TASK2_MODEL="${TASK2_WEIGHT_PATH}"
  TASK2_NOTES+=("model_source=${TASK2_WEIGHT_PATH}")
else
  echo "== Preparing YOLOv8n weight =="
  if conda run -n "${ENV_NAME}" python scripts/fetch_yolo_weight.py --output "${TASK2_WEIGHT_PATH}"; then
    TASK2_MODEL="${TASK2_WEIGHT_PATH}"
    TASK2_NOTES+=("model_source=${TASK2_WEIGHT_PATH}")
  elif [[ "${TASK2_ALLOW_YAML_FALLBACK}" == "1" ]]; then
    TASK2_MODEL="yolov8n.yaml"
    TASK2_NOTES+=("fallback_model=yolov8n.yaml")
    TASK2_NOTES+=("fallback_reason=yolov8n.pt unavailable from local files and mirror downloads")
    echo "WARNING: YOLOv8n weight unavailable; falling back to ${TASK2_MODEL}."
  else
    echo "ERROR: YOLOv8n weight unavailable and TASK2_ALLOW_YAML_FALLBACK=${TASK2_ALLOW_YAML_FALLBACK}."
    exit 1
  fi
fi

TASK2_ARGS=(
  --data "${TASK2_DATA}"
  --model "${TASK2_MODEL}"
  --device "${TASK2_DEVICE}"
  --log-file "${TASK2_LOG_FILE}"
)

if [[ "${TASK2_AMP}" == "1" ]]; then
  TASK2_ARGS+=(--amp)
fi

if [[ "${TASK2_PLOTS}" == "1" ]]; then
  TASK2_ARGS+=(--plots)
fi

for note in "${TASK2_NOTES[@]}"; do
  TASK2_ARGS+=(--note "${note}")
done

echo "== Running Task 2 YOLOv8 smoke test =="
conda run -n "${ENV_NAME}" python scripts/smoke_task2_yolo.py "${TASK2_ARGS[@]}"

echo "Smoke tests completed:"
echo "- ${TASK1_LOG_FILE}"
echo "- ${TASK2_LOG_FILE}"
