"""Verify the HW2 Python environment without downloading models or logging in."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import re
import platform
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
    ("albumentations", "albumentations"),
    ("cv2", "opencv-python"),
    ("swanlab", "swanlab"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("yaml", "pyyaml"),
]


def run_command(args: list[str]) -> str:
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def nvidia_smi() -> str:
    return run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )


def first_driver_major(nvidia_smi_output: str) -> int | None:
    match = re.search(r",\s*(\d+)\.", nvidia_smi_output)
    if match is None:
        return None
    return int(match.group(1))


def package_version(module_name: str, distribution_name: str) -> tuple[object | None, str]:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return module, str(version)


def print_header(title: str) -> None:
    print(f"\n== {title} ==")


def verify_imports() -> tuple[dict[str, object], list[str]]:
    print_header("Packages")
    modules: dict[str, object] = {}
    failures: list[str] = []

    for module_name, distribution_name in PACKAGES:
        try:
            module, version = package_version(module_name, distribution_name)
        except Exception as exc:  # pragma: no cover - intentionally broad smoke check
            failures.append(f"{module_name}: {exc}")
            print(f"[FAIL] {module_name}: {exc}")
            continue

        modules[module_name] = module
        print(f"[ OK ] {module_name}: {version}")

    return modules, failures


def verify_torch(modules: dict[str, object], require_cuda: bool) -> list[str]:
    failures: list[str] = []
    torch = modules.get("torch")
    if torch is None:
        return ["torch import failed; skipping tensor checks"]

    print_header("PyTorch Smoke Tests")
    print(f"torch.version.cuda: {torch.version.cuda}")
    smi_output = nvidia_smi()
    print(f"nvidia-smi: {smi_output}")

    try:
        x = torch.rand(2, 3)
        y = x @ x.T
        assert tuple(y.shape) == (2, 2)
        print(f"[ OK ] CPU tensor matmul: shape={tuple(y.shape)}, sum={float(y.sum()):.6f}")
    except Exception as exc:  # pragma: no cover - smoke check
        failures.append(f"CPU tensor smoke test failed: {exc}")
        print(f"[FAIL] CPU tensor smoke test: {exc}")

    cuda_available = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available(): {cuda_available}")

    if not cuda_available:
        message = "CUDA is not available in this environment"
        if require_cuda:
            failures.append(message)
            driver_major = first_driver_major(smi_output)
            if driver_major is not None and driver_major < 550 and str(torch.version.cuda).startswith("12.4"):
                failures.append(
                    "Driver is older than 550 while PyTorch uses CUDA 12.4. "
                    "Use the default cu121 setup script path or upgrade the host driver."
                )
            print(f"[FAIL] {message}")
        else:
            print(f"[WARN] {message}; rerun with --require-cuda on the GPU server")
        return failures

    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")
    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        total_gib = props.total_memory / 1024**3
        print(f"[ OK ] GPU {idx}: {props.name}, memory={total_gib:.2f} GiB")

    try:
        device = torch.device("cuda:0")
        x = torch.ones((128, 128), device=device)
        y = x @ x
        torch.cuda.synchronize(device)
        print(f"[ OK ] CUDA tensor matmul: shape={tuple(y.shape)}, sum={float(y.sum().item()):.1f}")
    except Exception as exc:  # pragma: no cover - smoke check
        failures.append(f"CUDA tensor smoke test failed: {exc}")
        print(f"[FAIL] CUDA tensor smoke test: {exc}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the HW2 deep learning environment.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA is not available. Use this on the GPU server.",
    )
    args = parser.parse_args()

    print_header("Runtime")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Git commit: {run_command(['git', 'rev-parse', '--short', 'HEAD'])}")

    modules, failures = verify_imports()
    failures.extend(verify_torch(modules, args.require_cuda))

    print_header("Result")
    if failures:
        print("Environment verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
