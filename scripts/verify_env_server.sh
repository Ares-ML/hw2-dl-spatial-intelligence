#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-hw2}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

source scripts/cuda_driver_shim.sh
prepare_cuda_driver_shim

conda run -n "${ENV_NAME}" python scripts/verify_env.py --require-cuda
