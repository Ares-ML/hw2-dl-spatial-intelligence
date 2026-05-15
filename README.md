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

Run the Task 3 U-Net forward smoke test after preparing Stanford Background:

```bash
source scripts/cuda_driver_shim.sh
prepare_cuda_driver_shim
conda run -n hw2 python scripts/smoke_task3_unet.py \
  --data-root data/stanford_background \
  --device cuda \
  --log-file logs/smoke/t3_unet_smoke.log
```

If CUDA is still unavailable while `nvidia-smi` works, collect diagnostics:

```bash
bash scripts/debug_cuda_server.sh 2>&1 | tee debug_cuda_server.log
```

The local development machine does not need to download CUDA wheels if it has no NVIDIA GPU.

## SwanLab

The Day 1 SwanLab milestone uses one project under the `Ares_ML` workspace:
`hw2-dl-spatial-intelligence`. Runs are grouped as `task1`, `task2`, and
`task3`, and metrics use the `taskN/<metric_name>` convention.

Create the three 1-epoch smoke runs on the GPU server after `swanlab login`:

```bash
conda run -n hw2 swanlab login -k <YOUR_API_KEY>
bash scripts/run_swanlab_smoke_server.sh hw2 2>&1 | tee run_swanlab_smoke_server.log
```

If needed, override the cloud target without editing config files:

```bash
SWANLAB_WORKSPACE=Ares_ML SWANLAB_PROJ_NAME=hw2-dl-spatial-intelligence \
  bash scripts/run_swanlab_smoke_server.sh hw2
```

The runner writes the project and experiment URLs to:

- `logs/swanlab/swanlab_links.md`

## Day 4 reproduction

Run these commands on the GPU server from the repository root. GPU 0 is the
currently recommended free device for the Day 4 reruns.

```bash
cd /root/Zhr/DL/HW2
set -euo pipefail
source scripts/cuda_driver_shim.sh
prepare_cuda_driver_shim
export CUDA_VISIBLE_DEVICES=0
```

Reproduce the full Task 1 hyperparameter grid when needed:

```bash
TASK1_DEVICE=cuda \
TASK1_SWANLAB_MODE=cloud \
TASK1_REQUIRE_SWANLAB=1 \
bash scripts/run_task1_grid_server.sh hw2 configs/task1_resnet18_pretrained_grid.yaml \
  2>&1 | tee logs/swanlab/task1_grid_day4_rerun.console.log
```

Retrain the final Task 1 model selected by the grid
(`lr_backbone=1e-4`, `lr_head=5e-4`, `epochs=100`):

```bash
conda run -n hw2 python -m src.task1_cls.train \
  --config configs/task1_resnet18_final.yaml \
  --device cuda \
  --swanlab-mode cloud \
  --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task1_final_bb1e-4_head5e-4_e100.log \
  --checkpoint-dir checkpoints/task1/final_resnet18_pretrained_bb1e-4_head5e-4_e100 \
  2>&1 | tee logs/swanlab/task1_final_bb1e-4_head5e-4_e100.console.log
```

Run ByteTrack on the second traffic video:

```bash
conda run -n hw2 python -m src.task2_detect_track.track_video \
  --weights runs/task2/road_vehicle_yolov8n_e120/weights/best.pt \
  --source data/videos/second_video_2026-05-15_215310_932.mp4 \
  --tracker bytetrack.yaml \
  --project runs/track \
  --name second_video_bytetrack \
  --device 0 \
  --conf 0.25 \
  --iou 0.7 \
  --imgsz 640 \
  --save-txt \
  --save-conf \
  2>&1 | tee logs/swanlab/task2_second_video_bytetrack.console.log
```

Run BoT-SORT on the same video:

```bash
conda run -n hw2 python -m src.task2_detect_track.track_video \
  --weights runs/task2/road_vehicle_yolov8n_e120/weights/best.pt \
  --source data/videos/second_video_2026-05-15_215310_932.mp4 \
  --tracker botsort.yaml \
  --project runs/track \
  --name second_video_botsort \
  --device 0 \
  --conf 0.25 \
  --iou 0.7 \
  --imgsz 640 \
  --save-txt \
  --save-conf \
  2>&1 | tee logs/swanlab/task2_second_video_botsort.console.log
```

Export the four-frame occlusion sequence and write the report paragraph:

```bash
conda run -n hw2 python -m src.task2_detect_track.analyze_occlusion \
  --video runs/track/second_video_bytetrack/second_video_2026-05-15_215310_932.avi \
  --labels-dir runs/track/second_video_bytetrack/labels \
  --compare-labels-dir runs/track/second_video_botsort/labels \
  --auto-window \
  --num-frames 4 \
  --output-dir report/figures/task2/occlusion_second \
  --prefix second_occlusion \
  --analysis-md report/task2_occlusion_analysis.md
```

Expected Day 4 outputs:

- `checkpoints/task1/final_resnet18_pretrained_bb1e-4_head5e-4_e100/best.pt`
- `runs/track/second_video_bytetrack/*.avi`
- `runs/track/second_video_botsort/*.avi`
- `report/figures/task2/occlusion_second/second_occlusion_f*.png`
- `report/task2_occlusion_analysis.md`

## Notes

Large datasets, logs, and model weights are not stored in this repository.
