# HW2: Deep Learning and Spatial Intelligence

Graduate course homework for Deep Learning and Spatial Intelligence.

## Tasks

- Task 1: Image classification and attention experiments
- Task 2: Vehicle detection, tracking, and counting
- Task 3: Semantic segmentation with U-Net

## Environment

- Python 3.10
- PyTorch 2.5.1 with CUDA 11.8 wheels
- CUDA 12.4 on RTX 4090 server

Create and verify the environment on the GPU server:

```bash
bash scripts/setup_env_server.sh 2>&1 | tee setup_env_server.log
```

The setup script defaults to CUDA 11.8 PyTorch wheels because the course server
currently uses an NVIDIA 535 driver with GeForce RTX 4090 GPUs, and CUDA 12.x
wheels can raise `Error 804: forward compatibility was attempted on non supported
HW` in some containers. CUDA 12.1 or 12.4 wheel sets can be selected explicitly:

```bash
PYTORCH_FLAVOR=cu121 bash scripts/setup_env_server.sh
PYTORCH_FLAVOR=cu124 bash scripts/setup_env_server.sh
```

To verify an existing server environment without reinstalling packages:

```bash
bash scripts/verify_env_server.sh
```

## Smoke tests

Run smoke tests on the GPU server through the CUDA driver shim wrapper. This is
important on the course server: running the Python scripts with plain
`conda run` can trigger CUDA error 804 because the real host NVIDIA driver is
not placed first on `LD_LIBRARY_PATH`.

```bash
bash scripts/run_smoke_server.sh hw2 2>&1 | tee run_smoke_server.log
```

The wrapper first verifies CUDA, then writes:

- `logs/smoke/t1_resnet18_smoke.log`
- `logs/smoke/t2_yolov8n_smoke.log`

For Task 2, the wrapper reuses `yolov8n.pt` from the repo root or
`weights/yolov8n.pt` when present. If the weight is missing, it tries mirror
downloads before falling back to `yolov8n.yaml` for the smoke test. You can
override the mirror list for domestic network environments:

```bash
YOLO_MODEL_URLS="https://your-mirror.example/yolov8n.pt" bash scripts/run_smoke_server.sh hw2
```

Task 2 disables Ultralytics AMP checks and plot generation by default to avoid
extra network downloads during the smoke run. Set `TASK2_AMP=1` or
`TASK2_PLOTS=1` only when those paths need to be tested explicitly.

If CUDA is still unavailable while `nvidia-smi` works, collect diagnostics:

```bash
bash scripts/debug_cuda_server.sh 2>&1 | tee debug_cuda_server.log
```

The local development machine does not need to download CUDA wheels if it has no NVIDIA GPU.

## Notes

Large datasets, logs, and model weights are not stored in this repository.
