# HW2: Deep Learning and Spatial Intelligence

Graduate course homework for Deep Learning and Spatial Intelligence.

## Tasks

- Task 1: Image classification and attention experiments
- Task 2: Vehicle detection, tracking, and counting
- Task 3: Semantic segmentation with U-Net

## Environment

- Python 3.10
- PyTorch 2.6.0 with CUDA 12.4 wheels
- CUDA 12.4 on RTX 4090 server

Create and verify the environment on the GPU server:

```bash
conda create -n hw2 python=3.10 pip -y
conda run -n hw2 python -m pip install --upgrade pip
conda run -n hw2 python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n hw2 python -m pip install -r requirements.txt -i https://mirrors.cernet.edu.cn/pypi/web/simple
conda run -n hw2 python scripts/verify_env.py --require-cuda
```

Or run the setup helper:

```bash
bash scripts/setup_env_server.sh
```

The local development machine does not need to download CUDA wheels if it has no NVIDIA GPU.

## Notes

Large datasets, logs, and model weights are not stored in this repository.
