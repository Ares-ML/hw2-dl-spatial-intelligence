# HW2: Deep Learning and Spatial Intelligence

Graduate course homework for *Deep Learning and Spatial Intelligence*.

- GitHub: https://github.com/Ares-ML/hw2-dl-spatial-intelligence
- Weights cloud drive: _<待填>_ — 详见 [`WEIGHTS_MANIFEST.md`](WEIGHTS_MANIFEST.md)

## 1. Tasks

- **Task 1**: 102-class flower classification (ResNet-18 主线 + CBAM 消融 + 超参网格；ResNet-34 / ViT-Tiny 加分)
- **Task 2**: Road Vehicle detection (YOLOv8n 主线 / YOLOv8s 加分) + ByteTrack/BoT-SORT 跟踪 + 越线计数
- **Task 3**: Stanford Background U-Net 分割，三种损失 (CE / Dice / CE+Dice) 对比

## 2. Directory structure

```
hw2-dl-spatial-intelligence/
├── README.md
├── WEIGHTS_MANIFEST.md              # 8 份权重云盘清单
├── requirements.txt
├── configs/                         # 训练配置 yaml
│   ├── task1_resnet18_{pretrained,random,cbam,pretrained_grid,final}.yaml
│   ├── task1_resnet34_final.yaml
│   ├── task1_vit_tiny_pretrained.yaml
│   ├── task2_yolov8{n,s}.yaml
│   └── task3_unet_{ce,dice,ce_dice}.yaml
├── data/                            # 数据集（不入库；prepare_*.py 准备）
│   ├── flower102/, road_vehicle/, stanford_background/, videos/
├── scripts/                         # 数据准备、烟雾测试、镜像下载
│   ├── prepare_{flower102,road_vehicle,stanford_background}.py
│   ├── setup_env_server.sh, verify_env_server.sh
│   ├── run_smoke_server.sh, run_swanlab_smoke_server.sh
│   ├── fetch_yolo_weight.py, expand_task1_grid.py
│   └── smoke_*.py
├── src/                             # 训练 / 评估 / 推理
│   ├── common/                      # seed / logger / metrics / checkpoint / swanlab_logger
│   ├── task1_cls/                   # train / models / attention
│   ├── task2_detect_track/          # train / track_video / count_video / line_counter / render_counts / analyze_occlusion
│   └── task3_seg/                   # train / unet / datasets / transforms / losses / eval_compare / visualize_masks / colorize
├── tests/                           # 73 项单元测试
├── checkpoints/                     # 训练权重（部分入库 .gitkeep）
│   ├── task1/, task3/
├── runs/                            # YOLO 训练 + 跟踪 + 越线计数输出（不入库）
│   ├── task2/road_vehicle_yolov8{n,s}_e120/
│   └── track/second_video_*_{bytetrack,botsort,counts_*}/
├── report/                          # 报告与图表
│   ├── HW2_report.md, task{1,2,3}_section_draft.md
│   ├── task2_occlusion_analysis.md, task2_line_counting.md
│   ├── weekend_task_card.md
│   ├── tables/                      # task1_*.md, task3_*.md
│   └── figures/                     # task1/{t1_a,t1_c,t1_cbam,t1_final,t1_grid,t1_resnet34,t1_vit}, task2/{detection,occlusion_*,counting}, task3/{fig5,mask_comparison,curves_*}
└── logs/                            # swanlab / smoke 日志
    └── swanlab/{swanlab_links.md, task*.log, task*.console.log}
```

## 3. Data preparation

第一次训练前各跑一次：

```bash
conda run -n hw2 python scripts/prepare_flower102.py            # Task 1（torchvision 自动下载）
conda run -n hw2 python scripts/prepare_road_vehicle.py         # Task 2（Kaggle CLI 需提前 ~/.kaggle/kaggle.json 配置）
conda run -n hw2 python scripts/prepare_stanford_background.py  # Task 3
```

Task 2 自拍视频请放到 `data/videos/`（不在 repo 内）；本作业使用的是 `data/videos/second_video_2026-05-15_215310_932.mp4`，1280×720 / 30 fps / 28.4 s。

## 4. 环境配置

> 主推 GPU 服务器一键脚本；本地 Mac/Windows 装 Python + `pip install -r requirements.txt` 即可读代码与报告，无需 CUDA。

### 4.1 一键安装（GPU 服务器 conda 环境 `hw2`）

```bash
bash scripts/setup_env_server.sh 2>&1 | tee setup_env_server.log
```

脚本默认装 PyTorch 2.5.1 + CUDA 11.8 wheel（对应课程服务器 NVIDIA 535 驱动）。其它 CUDA 选项：

```bash
PYTORCH_FLAVOR=cu121 bash scripts/setup_env_server.sh
PYTORCH_FLAVOR=cu124 bash scripts/setup_env_server.sh
```

### 4.2 验证已存在环境

```bash
bash scripts/verify_env_server.sh
```

### 4.3 关键依赖

| 包 | 版本 |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1 + CUDA 11.8 |
| ultralytics | 8.4.41 |
| timm | >= 0.9（ViT-Tiny 加分需要） |
| albumentations | 2.0.8 |
| swanlab | 0.7.17 |
| opencv-python-headless | 4.10.0.84 |

### 4.4 CUDA 驱动 shim（容器内）

课程服务器在容器里 CUDA 12.x wheel 偶发 `Error 804: forward compatibility was attempted on non supported HW`；用 shim 把宿主驱动放在最前：

```bash
source scripts/cuda_driver_shim.sh
prepare_cuda_driver_shim
```

诊断脚本：`bash scripts/debug_cuda_server.sh 2>&1 | tee debug_cuda_server.log`。

### 4.5 HuggingFace 镜像（国内自动启用）

`src/task1_cls/models.py::build_vit_tiny_classifier` 在 `import timm` 前调用 `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`，国内服务器无需任何额外配置即可拉 ViT-Tiny 预训练权重；海外服务器若想用 huggingface.co，`export HF_ENDPOINT=https://huggingface.co` 即可覆盖（`setdefault` 不会清掉外部已有值）。

### 4.6 SwanLab 登录（云端日志）

```bash
conda run -n hw2 swanlab login -k <YOUR_API_KEY>
```

项目名：`hw2-dl-spatial-intelligence`（`@Ares_ML` 工作空间），group 用 `task1/task2/task3`，metric 命名 `taskN/<metric_name>`。

## 5. 训练

所有训练命令默认 `--device cuda --swanlab-mode cloud --require-swanlab`。不想上 SwanLab 云端可把 `--swanlab-mode cloud --require-swanlab` 换成 `--swanlab-mode disabled`。

### 5.1 Task 1（Flower102 分类）

```bash
# 主线：差化学习率 100 ep（Val Acc = 0.9138）
conda run -n hw2 python -m src.task1_cls.train \
  --config configs/task1_resnet18_final.yaml \
  --device cuda --swanlab-mode cloud --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task1_final.log \
  --checkpoint-dir checkpoints/task1/final_resnet18_pretrained_bb1e-4_head5e-4_e100

# 消融：50 ep baseline / random init / CBAM
conda run -n hw2 python -m src.task1_cls.train --config configs/task1_resnet18_pretrained.yaml --device cuda --swanlab-mode cloud --require-swanlab --links-file logs/swanlab/swanlab_links.md
conda run -n hw2 python -m src.task1_cls.train --config configs/task1_resnet18_random.yaml     --device cuda --swanlab-mode cloud --require-swanlab --links-file logs/swanlab/swanlab_links.md
conda run -n hw2 python -m src.task1_cls.train --config configs/task1_resnet18_cbam.yaml       --device cuda --swanlab-mode cloud --require-swanlab --links-file logs/swanlab/swanlab_links.md

# 超参网格（8 组：lr_backbone × lr_head × epochs）
TASK1_DEVICE=cuda TASK1_SWANLAB_MODE=cloud TASK1_REQUIRE_SWANLAB=1 \
bash scripts/run_task1_grid_server.sh hw2 configs/task1_resnet18_pretrained_grid.yaml \
  2>&1 | tee logs/swanlab/task1_grid.console.log

# 加分：ResNet-34（与 final ResNet-18 同 recipe）
conda run -n hw2 python -m src.task1_cls.train \
  --config configs/task1_resnet34_final.yaml \
  --device cuda --swanlab-mode cloud --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task1_final_resnet34_pretrained_bb1e-4_head5e-4_e100.log

# 加分：ViT-Tiny（timm，50 ep，HF mirror 自动启用）
conda run -n hw2 python -m src.task1_cls.train \
  --config configs/task1_vit_tiny_pretrained.yaml \
  --device cuda --swanlab-mode cloud --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task1_bonus_vit_tiny_pretrained.log
```

### 5.2 Task 2（Road Vehicle 检测）

YOLO 训练前**必须**预下载权重，绕开 Ultralytics 在 Docker overlayfs 上的假阳性 `Insufficient free disk space` 报错（`shutil.disk_usage` 偶发返回 0）：

```bash
# 预下载到 weights/ + cp 到 cwd（含 ghfast/gh-proxy/hf-mirror/官方 4 个镜像 fallback）
conda run -n hw2 python scripts/fetch_yolo_weight.py --model yolov8n.pt
conda run -n hw2 python scripts/fetch_yolo_weight.py --model yolov8s.pt
cp weights/yolov8n.pt yolov8n.pt
cp weights/yolov8s.pt yolov8s.pt
```

然后用项目的 Python wrapper（与 Day 2 那次成功的 v8n e120 同入口；不要用裸 `yolo detect train ...`，因为 tee pipe 会让 tqdm 静默假死，且 `--swanlab-mode` 等 flag 是 wrapper 的，bare CLI 不接受）：

```bash
# 主线 YOLOv8n e120（mAP@50 best = 0.4436）
conda run -n hw2 python -m src.task2_detect_track.train \
  --config configs/task2_yolov8n.yaml \
  --device cuda --swanlab-mode cloud --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task2_yolov8n_e120.log

# 加分 YOLOv8s e120（mAP@50 best = 0.5857）
conda run -n hw2 python -m src.task2_detect_track.train \
  --config configs/task2_yolov8s.yaml \
  --device cuda --swanlab-mode cloud --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task2_yolov8s_e120.log
```

### 5.3 Task 3（Stanford Background 分割）

三种损失同一 recipe（`AdamW lr=1e-3`, 100 ep, image_size=256, seed=42），单独训：

```bash
conda run -n hw2 python -m src.task3_seg.train --config configs/task3_unet_ce.yaml      --device cuda --swanlab-mode cloud --require-swanlab --links-file logs/swanlab/swanlab_links.md --log-file logs/swanlab/task3_unet_ce_e100.log
conda run -n hw2 python -m src.task3_seg.train --config configs/task3_unet_dice.yaml    --device cuda --swanlab-mode cloud --require-swanlab --links-file logs/swanlab/swanlab_links.md --log-file logs/swanlab/task3_unet_dice.log
conda run -n hw2 python -m src.task3_seg.train --config configs/task3_unet_ce_dice.yaml --device cuda --swanlab-mode cloud --require-swanlab --links-file logs/swanlab/swanlab_links.md --log-file logs/swanlab/task3_unet_ce_dice.log
```

CE+Dice 取得最佳 mIoU=0.6320。

## 6. 测试 / 推理

### 6.1 Task 1：从 checkpoint 读 best metric（无需重跑 val）

```bash
conda run -n hw2 python -c "
import torch
ck = torch.load('checkpoints/task1/final_resnet18_pretrained_bb1e-4_head5e-4_e100/best.pt', map_location='cpu')
print('epoch=', ck['epoch'], 'best_score=', ck['best_score'], 'metrics=', ck['metrics'])
"
```

可读 `final_resnet34_*`、`bonus_vit_tiny_*` 等任意一份 best.pt。

### 6.2 Task 2：跟踪 + 越线计数 + 遮挡分析

```bash
# 跟踪：ByteTrack（主输出）
conda run -n hw2 python -m src.task2_detect_track.track_video \
  --weights runs/task2/road_vehicle_yolov8n_e120/weights/best.pt \
  --source data/videos/second_video_2026-05-15_215310_932.mp4 \
  --tracker bytetrack.yaml \
  --project runs/track --name second_video_bytetrack \
  --device 0 --save-txt --save-conf

# 跟踪：BoT-SORT（遮挡对照）
conda run -n hw2 python -m src.task2_detect_track.track_video \
  --weights runs/task2/road_vehicle_yolov8n_e120/weights/best.pt \
  --source data/videos/second_video_2026-05-15_215310_932.mp4 \
  --tracker botsort.yaml \
  --project runs/track --name second_video_botsort \
  --device 0 --save-txt --save-conf

# 越线计数：含 bbox/class/ID/计数线/统计面板/越线红点的最终演示视频
conda run -n hw2 python -m src.task2_detect_track.count_video \
  --weights runs/task2/road_vehicle_yolov8n_e120/weights/best.pt \
  --source data/videos/second_video_2026-05-15_215310_932.mp4 \
  --line "60,360,1220,360" --line-margin 4.0 --exclude-classes "3,20" \
  --device 0 --require-cuda \
  --project runs/track --name second_video_bytetrack_counts_final

# 遮挡 3-4 帧分析：自动找 max-IoU 窗口，导出 PNG + 写入 Markdown 段
conda run -n hw2 python -m src.task2_detect_track.analyze_occlusion \
  --video runs/track/second_video_bytetrack/second_video_2026-05-15_215310_932.avi \
  --labels-dir runs/track/second_video_bytetrack/labels \
  --compare-labels-dir runs/track/second_video_botsort/labels \
  --auto-window --num-frames 4 \
  --output-dir report/figures/task2/occlusion_second \
  --prefix second_occlusion \
  --analysis-md report/task2_occlusion_analysis.md
```

### 6.3 Task 3：三组损失对比 + Mask 可视化

```bash
# Table 3（loss 总览 + class-wise IoU）+ Fig 5（2×2 train/val loss + val mIoU + val pixel acc）
conda run -n hw2 python -m src.task3_seg.eval_compare \
  --variants ce,dice,ce_dice \
  --data-root data/stanford_background \
  --checkpoints-root checkpoints/task3 \
  --logs-root logs/swanlab \
  --output-dir report \
  --device 0 --require-cuda

# Fig 6（6×5 image / GT / pred-CE / pred-Dice / pred-CE+Dice，按 CE per-image mIoU 选 6 个档位样本）
conda run -n hw2 python -m src.task3_seg.visualize_masks \
  --variants ce,dice,ce_dice \
  --data-root data/stanford_background \
  --checkpoints-root checkpoints/task3 \
  --output-path report/figures/task3/mask_comparison.png \
  --bucket-counts best=1,median=2,hard=2,worst=1 \
  --scoring-variant ce \
  --device 0 --require-cuda
```

### 6.4 单元测试

```bash
conda run -n hw2 python -m unittest discover tests   # 73 项；cv2 / timm 缺失时自动 skip
```

### 6.5 端到端烟雾测试（<5 分钟）

```bash
bash scripts/run_smoke_server.sh hw2 2>&1 | tee run_smoke_server.log
```

写入 `logs/smoke/{t1_resnet18,t2_yolov8n,t3_unet}_smoke.log`。任务 2 默认禁用 Ultralytics AMP 检测与 plot 生成以避免额外网络下载；`TASK2_AMP=1` / `TASK2_PLOTS=1` 可显式开启。

如果 `nvidia-smi` 正常但 CUDA 仍然不可用，先跑 `bash scripts/debug_cuda_server.sh 2>&1 | tee debug_cuda_server.log`。

## 7. 模型权重

完整清单 + 建议云盘文件名 + sha256 + 加载示例都在 [`WEIGHTS_MANIFEST.md`](WEIGHTS_MANIFEST.md)。

- 云盘 base URL：链接：https://pan.baidu.com/s/1HrU0pjPZykTFZ8GlINiB5Q?pwd=213g
- 提取码：213g
- 必交三件（PDF §8.3）：`task1_resnet18_final.pt` / `task2_yolov8n_e120.pt` / `task3_unet_ce_dice.pt`
- 加分 / 消融 5 件：见 manifest Tier 2 / Tier 3

## 8. 报告

主报告 [`report/HW2_report.md`](report/HW2_report.md)；各任务独立草稿 [`task1_section_draft.md`](report/task1_section_draft.md) / [`task2_section_draft.md`](report/task2_section_draft.md) / [`task3_section_draft.md`](report/task3_section_draft.md)；提交日（5/19）导出为 PDF。

## 9. SwanLab

Day 1 SwanLab 里程碑使用 `Ares_ML` workspace 下的单一项目 `hw2-dl-spatial-intelligence`。Runs 分组为 `task1` / `task2` / `task3`，metric 名为 `taskN/<metric_name>`。

```bash
conda run -n hw2 swanlab login -k <YOUR_API_KEY>
bash scripts/run_swanlab_smoke_server.sh hw2 2>&1 | tee run_swanlab_smoke_server.log
```

若要覆盖 workspace / 项目：

```bash
SWANLAB_WORKSPACE=Ares_ML SWANLAB_PROJ_NAME=hw2-dl-spatial-intelligence \
  bash scripts/run_swanlab_smoke_server.sh hw2
```

所有 run 链接维护在 [`logs/swanlab/swanlab_links.md`](logs/swanlab/swanlab_links.md)。

## 10. Day 4 reproduction（保留快捷链）

下面是 Day 4 那次 final 训练 + 跟踪 + 遮挡分析的完整命令快照。`§5/§6` 已涵盖通用语法；本节作为"按当时实际命令"的还原参考。

```bash
cd /root/Zhr/DL/HW2
set -euo pipefail
source scripts/cuda_driver_shim.sh
prepare_cuda_driver_shim
export CUDA_VISIBLE_DEVICES=0

# Task 1 final（最优配置）
conda run -n hw2 python -m src.task1_cls.train \
  --config configs/task1_resnet18_final.yaml \
  --device cuda --swanlab-mode cloud --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task1_final_bb1e-4_head5e-4_e100.log \
  --checkpoint-dir checkpoints/task1/final_resnet18_pretrained_bb1e-4_head5e-4_e100 \
  2>&1 | tee logs/swanlab/task1_final_bb1e-4_head5e-4_e100.console.log
```

ByteTrack / BoT-SORT 跟踪与遮挡分析命令见 `§6.2`。

## 11. 备注

- 大数据集、训练 log、模型权重不入 repo；用 [`WEIGHTS_MANIFEST.md`](WEIGHTS_MANIFEST.md) + 云盘下载
- Windows 本地无 GPU 也可读代码与报告；单元测试中 cv2 / timm 缺失会自动 skip
- 周末（5/16—5/19）任务节奏见 [`report/weekend_task_card.md`](report/weekend_task_card.md)
