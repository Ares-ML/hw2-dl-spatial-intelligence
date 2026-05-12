#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-hw2}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# CUDA forward-compat libraries can trigger Error 804 on GeForce GPUs inside
# containers. Prefer the host driver libcuda exposed by the NVIDIA runtime.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  CLEANED_LD_LIBRARY_PATH=""
  IFS=':' read -r -a LD_PATHS <<< "${LD_LIBRARY_PATH}"
  for path in "${LD_PATHS[@]}"; do
    if [[ "${path}" == *"/cuda/compat"* || "${path}" == *"/cuda-"*"/compat"* ]]; then
      echo "Removing CUDA forward-compat path from LD_LIBRARY_PATH: ${path}"
      continue
    fi
    if [[ -z "${CLEANED_LD_LIBRARY_PATH}" ]]; then
      CLEANED_LD_LIBRARY_PATH="${path}"
    else
      CLEANED_LD_LIBRARY_PATH="${CLEANED_LD_LIBRARY_PATH}:${path}"
    fi
  done
  export LD_LIBRARY_PATH="${CLEANED_LD_LIBRARY_PATH}"
fi

conda run -n "${ENV_NAME}" python scripts/verify_env.py --require-cuda
