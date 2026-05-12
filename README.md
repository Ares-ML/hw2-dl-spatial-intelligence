# HW2: Deep Learning and Spatial Intelligence

Graduate course homework for Deep Learning and Spatial Intelligence.

## Tasks

- Task 1: Image classification and attention experiments
- Task 2: Vehicle detection, tracking, and counting
- Task 3: Semantic segmentation with U-Net

## Environment

- Python 3.10
- PyTorch 2.5.1 with CUDA 12.1 wheels
- CUDA 12.4 on RTX 4090 server

Create and verify the environment on the GPU server:

```bash
bash scripts/setup_env_server.sh 2>&1 | tee setup_env_server.log
```

The setup script defaults to CUDA 12.1 PyTorch wheels because the course server
currently uses an NVIDIA 535 driver with GeForce RTX 4090 GPUs. CUDA 12.4 wheels
can raise `Error 804: forward compatibility was attempted on non supported HW`
on that combination. If the host driver is upgraded to 550.54.14 or newer, the
CUDA 12.4 wheel set can be selected explicitly:

```bash
PYTORCH_FLAVOR=cu124 bash scripts/setup_env_server.sh
```

To verify an existing server environment without reinstalling packages:

```bash
bash scripts/verify_env_server.sh
```

The local development machine does not need to download CUDA wheels if it has no NVIDIA GPU.

## Notes

Large datasets, logs, and model weights are not stored in this repository.
