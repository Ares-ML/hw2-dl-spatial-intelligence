# HW2: Deep Learning and Spatial Intelligence

Graduate course homework for *Deep Learning and Spatial Intelligence*.

- GitHub: https://github.com/Ares-ML/hw2-dl-spatial-intelligence
- Weights cloud drive: https://pan.baidu.com/s/1HrU0pjPZykTFZ8GlINiB5Q?pwd=213g

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
├── data/                            # 数据集不入库；仅保留 Task 2 报告视频
│   ├── flower102/, road_vehicle/, stanford_background/
│   └── videos/
│       ├── task2_video1_line_count_yolov8n_botsort_h264.mp4
│       └── task2_video9_occlusion_botsort_h264.mp4
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
│   └── track/
├── report/                          # 报告与图表
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

Task 2 报告实际使用的两段最终 H.264 视频已随 repo 提交到 `data/videos/`：

- `data/videos/task2_video1_line_count_yolov8n_botsort_h264.mp4`：视频 1 越线计数演示，1920×1080 / 30 fps / 617 帧 / 20.57 s，YOLOv8n + BoT-SORT，计数线 `(0,720,1920,720)`，自动越线计数 total=41。
- `data/videos/task2_video9_occlusion_botsort_h264.mp4`：视频 9 遮挡与 ID 跳变分析演示，1920×1080 / 30 fps / 853 帧 / 28.43 s，BoT-SORT `track_buffer=240`。

原始数据集、训练输出、权重和其它中间视频仍由 `.gitignore` 排除；如需重跑完整检测/跟踪/计数管线，可将原始自采集视频另放到 `data/videos/` 并替换下方命令中的 `--source`。

## 4. 环境配置

> 主推 GPU 服务器一键脚本；本地 Mac/Windows 装 Python + `pip install -r requirements.txt` 即可读代码与报告，无需 CUDA。

### 4.1 一键安装（GPU 服务器 conda 环境 `hw2`）

```bash
bash scripts/setup_env_server.sh 2>&1 | tee setup_env_server.log
```

脚本默认装 PyTorch 2.5.1 + CUDA 11.8 wheel。其它 CUDA 选项：

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
| timm | 1.0.15 |
| albumentations | 2.0.8 |
| swanlab | 0.7.17 |
| opencv-python-headless | 4.10.0.84 |
| numpy | 2.2.6 |
| Pillow | 12.1.1 |
| PyYAML | 6.0.3 |

## 5. 训练

所有训练命令默认 `--device cuda --swanlab-mode cloud --require-swanlab`。不想上 SwanLab 云端可把 `--swanlab-mode cloud --require-swanlab` 换成 `--swanlab-mode disabled`。

### 5.1 Task 1（Flower102 分类）

```bash
# 主线：差异化学习率 100 ep（Val Acc = 0.9138）
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

# 拓展：ResNet-34（与 final ResNet-18 同 recipe）
conda run -n hw2 python -m src.task1_cls.train \
  --config configs/task1_resnet34_final.yaml \
  --device cuda --swanlab-mode cloud --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task1_final_resnet34_pretrained_bb1e-4_head5e-4_e100.log

# 拓展：ViT-Tiny（timm，50 ep，HF mirror 自动启用）
conda run -n hw2 python -m src.task1_cls.train \
  --config configs/task1_vit_tiny_pretrained.yaml \
  --device cuda --swanlab-mode cloud --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task1_bonus_vit_tiny_pretrained.log
```

### 5.2 Task 2（Road Vehicle 检测）

YOLO 训练前请预下载权重：

```bash
# 预下载到 weights/ + cp 到 cwd（含 ghfast/gh-proxy/hf-mirror/官方 4 个镜像 fallback）
conda run -n hw2 python scripts/fetch_yolo_weight.py --model yolov8n.pt
conda run -n hw2 python scripts/fetch_yolo_weight.py --model yolov8s.pt
cp weights/yolov8n.pt yolov8n.pt
cp weights/yolov8s.pt yolov8s.pt
```

然后用项目的 Python wrapper：

```bash
# 主线 YOLOv8n e120（mAP@50 best = 0.4436）
conda run -n hw2 python -m src.task2_detect_track.train \
  --config configs/task2_yolov8n.yaml \
  --device cuda --swanlab-mode cloud --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task2_yolov8n_e120.log

# 拓展 YOLOv8s e120（mAP@50 best = 0.5857）
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
# 将 RAW_VIDEO_1 / RAW_VIDEO_9 替换为原始自采集视频路径；repo 内保留的是最终 H.264 证据视频。
RAW_VIDEO_1=data/videos/raw_video1.mp4
RAW_VIDEO_9=data/videos/raw_video9.mp4

# 跟踪：ByteTrack（对照输出）
conda run -n hw2 python -m src.task2_detect_track.track_video \
  --weights runs/task2/road_vehicle_yolov8n_e120/weights/best.pt \
  --source "$RAW_VIDEO_9" \
  --tracker bytetrack.yaml \
  --project runs/track --name task2_video9_bytetrack \
  --device 0 --save-txt --save-conf

# 跟踪：BoT-SORT（遮挡对照）
conda run -n hw2 python -m src.task2_detect_track.track_video \
  --weights runs/task2/road_vehicle_yolov8n_e120/weights/best.pt \
  --source "$RAW_VIDEO_9" \
  --tracker botsort.yaml \
  --project runs/track --name task2_video9_botsort \
  --device 0 --save-txt --save-conf

# 越线计数：含 bbox/class/ID/计数线/统计面板/越线红点的最终演示视频
conda run -n hw2 python -m src.task2_detect_track.count_video \
  --weights runs/task2/road_vehicle_yolov8n_e120/weights/best.pt \
  --source "$RAW_VIDEO_1" \
  --line "0,720,1920,720" --line-margin 4.0 \
  --device 0 --require-cuda \
  --project runs/track --name task2_video1_botsort_counts_final

# 遮挡 3-4 帧分析：找 max-IoU 窗口，导出 PNG + 写入 Markdown 段
conda run -n hw2 python -m src.task2_detect_track.analyze_occlusion \
  --video runs/track/task2_video9_bytetrack/raw_video9.avi \
  --labels-dir runs/track/task2_video9_bytetrack/labels \
  --compare-labels-dir runs/track/task2_video9_botsort/labels \
  --auto-window --num-frames 4 \
  --output-dir report/figures/task2/occlusion_video9 \
  --prefix video9_occlusion \
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

### 6.5 端到端 smoke test 

```bash
bash scripts/run_smoke_server.sh hw2 2>&1 | tee run_smoke_server.log
```

写入 `logs/smoke/{t1_resnet18,t2_yolov8n,t3_unet}_smoke.log`。任务 2 默认禁用 Ultralytics AMP 检测与 plot 生成以避免额外网络下载；`TASK2_AMP=1` / `TASK2_PLOTS=1` 可显式开启。

## 7. 模型权重

完整清单 + 建议云盘文件名 + sha256 + 加载示例都在 [`WEIGHTS_MANIFEST.md`](WEIGHTS_MANIFEST.md)。

- 云盘 base URL：链接：https://pan.baidu.com/s/1HrU0pjPZykTFZ8GlINiB5Q?pwd=213g
- 提取码：213g
- 必交三件（PDF §8.3）：`task1_resnet18_final.pt` / `task2_yolov8n_e120.pt` / `task3_unet_ce_dice.pt`
- 拓展 / 消融 5 件：见 manifest Tier 2 / Tier 3

## 8. SwanLab

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

## 9. 备注

- 大数据集、训练 log、模型权重不入 repo；用 [`WEIGHTS_MANIFEST.md`](WEIGHTS_MANIFEST.md) + 云盘下载
