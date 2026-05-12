#!/usr/bin/env bash
set -u

ENV_NAME="${1:-hw2}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== System =="
date
uname -a
echo "PWD=$PWD"
echo "USER=$(id)"

echo
echo "== Environment Variables =="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"
echo "PATH=$PATH"

echo
echo "== NVIDIA Devices =="
ls -l /dev/nvidia* 2>/dev/null || true

echo
echo "== nvidia-smi =="
nvidia-smi || true

echo
echo "== Driver Version =="
cat /proc/driver/nvidia/version 2>/dev/null || true

echo
echo "== libcuda candidates =="
ldconfig -p 2>/dev/null | grep -E 'libcuda|libnvidia-ml' || true
find /usr /opt -name 'libcuda.so*' -o -name 'libnvidia-ml.so*' 2>/dev/null | sort || true

echo
echo "== Conda Packages =="
conda run -n "${ENV_NAME}" python -m pip list | grep -E 'torch|nvidia|cuda|ultralytics|opencv|albumentations|swanlab|numpy' || true

echo
echo "== Python CUDA Probe =="
conda run -n "${ENV_NAME}" python - <<'PY'
import ctypes
import ctypes.util
import os
import sys

print("python:", sys.executable)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "<empty>"))
print("ctypes find_library('cuda'):", ctypes.util.find_library("cuda"))
print("ctypes find_library('nvidia-ml'):", ctypes.util.find_library("nvidia-ml"))

for lib in ("libcuda.so.1", "libnvidia-ml.so.1"):
    try:
        handle = ctypes.CDLL(lib)
        print(f"CDLL {lib}: OK ({handle._name})")
    except OSError as exc:
        print(f"CDLL {lib}: FAIL ({exc})")

try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
except Exception as exc:
    print("torch probe failed:", repr(exc))
PY
