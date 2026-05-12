#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-hw2}"
PYPI_INDEX="${PYPI_INDEX:-https://mirrors.cernet.edu.cn/pypi/web/simple}"
PYTORCH_FLAVOR="${PYTORCH_FLAVOR:-cu118}"
PIP_RETRIES="${PIP_RETRIES:-10}"
PIP_TIMEOUT="${PIP_TIMEOUT:-120}"

case "${PYTORCH_FLAVOR}" in
  cu118)
    TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
    TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
    TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
    PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu118}"
    ;;
  cu121)
    TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
    TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
    TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
    PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
    ;;
  cu124)
    TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
    TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
    TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
    PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
    ;;
  *)
    echo "Unsupported PYTORCH_FLAVOR='${PYTORCH_FLAVOR}'. Use cu118, cu121, or cu124." >&2
    exit 1
    ;;
esac

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

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
  conda create -n "${ENV_NAME}" python=3.10 pip -y
fi

echo "Using PyTorch ${TORCH_VERSION}/${TORCHVISION_VERSION}/${TORCHAUDIO_VERSION} from ${PYTORCH_INDEX}"

conda run -n "${ENV_NAME}" python -m pip install \
  --upgrade pip \
  --no-cache-dir \
  --retries "${PIP_RETRIES}" \
  --timeout "${PIP_TIMEOUT}" \
  -i "${PYPI_INDEX}"

conda run -n "${ENV_NAME}" python -m pip install \
  --force-reinstall \
  --no-cache-dir \
  --retries "${PIP_RETRIES}" \
  --timeout "${PIP_TIMEOUT}" \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_INDEX}"

conda run -n "${ENV_NAME}" python -m pip install \
  --no-cache-dir \
  --retries "${PIP_RETRIES}" \
  --timeout "${PIP_TIMEOUT}" \
  -r requirements.txt \
  -i "${PYPI_INDEX}"

# Repair common interrupted-download states. On headless servers we want the
# headless OpenCV wheel to be the last cv2 provider installed.
conda run -n "${ENV_NAME}" python -m pip uninstall -y opencv-python opencv-python-headless || true
conda run -n "${ENV_NAME}" python -m pip install \
  --force-reinstall \
  --no-cache-dir \
  --retries "${PIP_RETRIES}" \
  --timeout "${PIP_TIMEOUT}" \
  opencv-python-headless==4.10.0.84 \
  -i "${PYPI_INDEX}"

bash scripts/verify_env_server.sh "${ENV_NAME}"
