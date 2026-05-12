#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-hw2}"
PYPI_INDEX="${PYPI_INDEX:-https://mirrors.cernet.edu.cn/pypi/web/simple}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
  conda create -n "${ENV_NAME}" python=3.10 pip -y
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip -i "${PYPI_INDEX}"
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url "${PYTORCH_INDEX}"
conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt -i "${PYPI_INDEX}"
conda run -n "${ENV_NAME}" python scripts/verify_env.py --require-cuda
