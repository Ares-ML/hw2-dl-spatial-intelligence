#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-hw2}"
GRID_CONFIG="${2:-configs/task1_resnet18_pretrained_grid.yaml}"

TASK1_DEVICE="${TASK1_DEVICE:-cuda}"
TASK1_SWANLAB_MODE="${TASK1_SWANLAB_MODE:-cloud}"
TASK1_REQUIRE_SWANLAB="${TASK1_REQUIRE_SWANLAB:-1}"
TASK1_GRID_MAX_RUNS="${TASK1_GRID_MAX_RUNS:-0}"
TASK1_GRID_EPOCHS_OVERRIDE="${TASK1_GRID_EPOCHS_OVERRIDE:-}"
TASK1_GRID_MAX_BATCHES="${TASK1_GRID_MAX_BATCHES:-}"
TASK1_GRID_MAX_VAL_BATCHES="${TASK1_GRID_MAX_VAL_BATCHES:-}"
LINKS_FILE="${LINKS_FILE:-logs/swanlab/swanlab_links.md}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

source scripts/cuda_driver_shim.sh
prepare_cuda_driver_shim

CONFIG_DIR="logs/swanlab/task1_grid_configs"
mkdir -p "${CONFIG_DIR}" logs/swanlab checkpoints/task1/grid

if [[ ! -s "${LINKS_FILE}" ]]; then
  mkdir -p "$(dirname "${LINKS_FILE}")"
  cat > "${LINKS_FILE}" <<'EOF'
# SwanLab Runs

| Task | Run | Project URL | Experiment URL |
| --- | --- | --- | --- |
EOF
fi

echo "== Preparing Task 1 grid configs from ${GRID_CONFIG} =="
RUN_LIST="${CONFIG_DIR}/runs.tsv"
conda run -n "${ENV_NAME}" python scripts/expand_task1_grid.py \
  --grid-config "${GRID_CONFIG}" \
  --output-dir "${CONFIG_DIR}" \
  > "${RUN_LIST}"

mapfile -t RUNS < <(sed '/^[[:space:]]*$/d' "${RUN_LIST}")

if [[ "${#RUNS[@]}" -eq 0 ]]; then
  echo "grid_config=${GRID_CONFIG}" >&2
  echo "config_dir=${CONFIG_DIR}" >&2
  if [[ -f "${RUN_LIST}" ]]; then
    echo "run_list=${RUN_LIST} exists but has no run rows." >&2
  else
    echo "run_list=${RUN_LIST} was not created." >&2
  fi
  echo "ERROR: no Task 1 grid runs were generated." >&2
  exit 1
fi
echo "Generated ${#RUNS[@]} Task 1 grid run configs in ${CONFIG_DIR}."

TOTAL_RUNS="${#RUNS[@]}"
if [[ "${TASK1_GRID_MAX_RUNS}" -gt 0 && "${TASK1_GRID_MAX_RUNS}" -lt "${TOTAL_RUNS}" ]]; then
  TOTAL_RUNS="${TASK1_GRID_MAX_RUNS}"
fi

for ((index = 0; index < TOTAL_RUNS; index++)); do
  IFS=$'\t' read -r RUN_ID CONFIG_PATH <<< "${RUNS[$index]}"
  LOG_FILE="logs/swanlab/task1_grid_${RUN_ID}.log"
  CONSOLE_LOG="logs/swanlab/task1_grid_${RUN_ID}.console.log"
  CHECKPOINT_DIR="checkpoints/task1/grid/${RUN_ID}"

  TRAIN_ARGS=(
    python -m src.task1_cls.train
    --config "${CONFIG_PATH}"
    --device "${TASK1_DEVICE}"
    --swanlab-mode "${TASK1_SWANLAB_MODE}"
    --links-file "${LINKS_FILE}"
    --log-file "${LOG_FILE}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
  )

  if [[ "${TASK1_SWANLAB_MODE}" == "cloud" && "${TASK1_REQUIRE_SWANLAB}" == "1" ]]; then
    TRAIN_ARGS+=(--require-swanlab)
  fi
  if [[ -n "${TASK1_GRID_EPOCHS_OVERRIDE}" ]]; then
    TRAIN_ARGS+=(--epochs "${TASK1_GRID_EPOCHS_OVERRIDE}")
  fi
  if [[ -n "${TASK1_GRID_MAX_BATCHES}" ]]; then
    TRAIN_ARGS+=(--max-batches "${TASK1_GRID_MAX_BATCHES}")
  fi
  if [[ -n "${TASK1_GRID_MAX_VAL_BATCHES}" ]]; then
    TRAIN_ARGS+=(--max-val-batches "${TASK1_GRID_MAX_VAL_BATCHES}")
  fi

  echo "== Task 1 grid $((index + 1))/${TOTAL_RUNS}: ${RUN_ID} =="
  echo "config=${CONFIG_PATH}"
  echo "log=${LOG_FILE}"
  echo "checkpoint_dir=${CHECKPOINT_DIR}"
  conda run -n "${ENV_NAME}" "${TRAIN_ARGS[@]}" 2>&1 | tee "${CONSOLE_LOG}"
done

echo "Task 1 grid completed. SwanLab links: ${LINKS_FILE}"
