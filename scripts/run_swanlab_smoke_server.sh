#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-hw2}"
LINKS_FILE="${LINKS_FILE:-logs/swanlab/swanlab_links.md}"

print_swanlab_login_hint() {
  cat <<'EOF'

SwanLab cloud initialization failed. Check the error above, then authenticate on the server with one of:
  conda run -n hw2 swanlab login -k <YOUR_API_KEY>
  export SWANLAB_API_KEY=<YOUR_API_KEY>

If the account is logged in but still fails, confirm it can write to workspace Ares_ML,
or override the target with SWANLAB_WORKSPACE=<workspace> and SWANLAB_PROJ_NAME=<project>.
EOF
}

trap 'status=$?; if [[ ${status} -ne 0 ]]; then print_swanlab_login_hint; fi' EXIT

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

echo "== SwanLab preflight =="
conda run -n "${ENV_NAME}" python -c "import os, swanlab; from src.common.swanlab_logger import load_swanlab_config; cfg=load_swanlab_config(); print(f'swanlab_version={getattr(swanlab, \"__version__\", \"unknown\")}'); print('SWANLAB_API_KEY=set' if os.environ.get('SWANLAB_API_KEY') else 'SWANLAB_API_KEY=not-set'); print(f'workspace={cfg.get(\"workspace\")}'); print(f'project={cfg.get(\"project\")}')"

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
