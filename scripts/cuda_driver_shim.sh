#!/usr/bin/env bash

# Helpers for forcing PyTorch to use the real host NVIDIA driver library inside
# containers. Some CUDA images place /usr/local/cuda*/compat ahead of the host
# driver, which can make GeForce GPUs fail with CUDA Error 804.

sanitize_cuda_ld_library_path() {
  if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    return 0
  fi

  local cleaned=""
  local path
  IFS=':' read -r -a LD_PATHS <<< "${LD_LIBRARY_PATH}"
  for path in "${LD_PATHS[@]}"; do
    if [[ "${path}" == *"/cuda/compat"* || "${path}" == *"/cuda-"*"/compat"* ]]; then
      echo "Removing CUDA forward-compat path from LD_LIBRARY_PATH: ${path}"
      continue
    fi
    if [[ -z "${cleaned}" ]]; then
      cleaned="${path}"
    else
      cleaned="${cleaned}:${path}"
    fi
  done
  export LD_LIBRARY_PATH="${cleaned}"
}

detect_nvidia_driver_version() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    | head -n 1 \
    | tr -d '[:space:]'
}

find_real_libcuda() {
  local driver_version="${1:-}"
  local roots=(
    /usr/lib/x86_64-linux-gnu
    /usr/lib64
    /usr/lib
    /lib/x86_64-linux-gnu
    /run/nvidia/driver/usr/lib/x86_64-linux-gnu
    /run/nvidia/driver/usr/lib64
  )
  local candidate resolved root

  if [[ -n "${driver_version}" ]]; then
    for root in "${roots[@]}"; do
      candidate="${root}/libcuda.so.${driver_version}"
      if [[ -e "${candidate}" ]]; then
        resolved="$(readlink -f "${candidate}")"
        if [[ -f "${resolved}" && "${resolved}" != *"/compat/"* && "${resolved}" != *"/stubs/"* ]]; then
          echo "${resolved}"
          return 0
        fi
      fi
    done
  fi

  while IFS= read -r candidate; do
    resolved="$(readlink -f "${candidate}")"
    if [[ -f "${resolved}" && "${resolved}" != *"/compat/"* && "${resolved}" != *"/stubs/"* ]]; then
      echo "${resolved}"
      return 0
    fi
  done < <(
    find /usr /lib /run/nvidia/driver \
      -path '*/compat/*' -prune -o \
      -path '*/stubs/*' -prune -o \
      -type f -name 'libcuda.so.*' -print 2>/dev/null \
      | sort -Vr
  )

  return 1
}

prepare_cuda_driver_shim() {
  sanitize_cuda_ld_library_path

  local driver_version real_libcuda shim_dir
  driver_version="$(detect_nvidia_driver_version || true)"
  real_libcuda="$(find_real_libcuda "${driver_version}" || true)"

  if [[ -z "${real_libcuda}" ]]; then
    echo "WARNING: could not find a non-compat libcuda.so for driver '${driver_version:-unknown}'." >&2
    return 0
  fi

  shim_dir="${CUDA_DRIVER_SHIM_DIR:-${PWD}/.cuda_driver_shim}"
  mkdir -p "${shim_dir}"
  ln -sfn "${real_libcuda}" "${shim_dir}/libcuda.so.1"
  ln -sfn "${real_libcuda}" "${shim_dir}/libcuda.so"

  export LD_LIBRARY_PATH="${shim_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "Using CUDA driver shim: ${shim_dir}/libcuda.so.1 -> ${real_libcuda}"
}
