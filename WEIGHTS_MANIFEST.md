# HW2 Model Weights Manifest

> Day 6 deliverable. 上传以下 8 份权重到百度云 / Google Drive；上传完毕后把 **Cloud URL** 与 **SHA256** 列填回本文档，并把云盘 base URL 同步到 [README.md §7](README.md) 和 [report/HW2_report.md](report/HW2_report.md) 首页。

| Cloud base URL | _<待填，建议百度云链接 + 提取码>_ |
| --- | --- |

## 服务器一键计算 sha256 + size

在 `/root/Zhr/DL/HW2` 下跑：

```bash
for f in \
  checkpoints/task1/final_resnet18_pretrained_bb1e-4_head5e-4_e100/best.pt \
  checkpoints/task1/final_resnet34_pretrained_bb1e-4_head5e-4_e100/best.pt \
  checkpoints/task1/bonus_vit_tiny_pretrained_bb1e-4_head5e-4_e50/best.pt \
  runs/task2/road_vehicle_yolov8n_e120/weights/best.pt \
  runs/task2/road_vehicle_yolov8s_e120/weights/best.pt \
  checkpoints/task3/unet_ce/best.pt \
  checkpoints/task3/unet_dice/best.pt \
  checkpoints/task3/unet_ce_dice/best.pt; do
  printf '%s  %s  %s\n' "$(sha256sum "$f" | cut -c1-12)" "$(du -h "$f" | cut -f1)" "$f"
done
```

把每行 `sha256-前 12 位 / 实际大小` 填到下面三个表的 **SHA256** 与 **Size** 列。

## Tier 1 — 主线必交（PDF §8.3 三件）

| Task | Cloud Filename | Server Path | 实验 | 关键数字 | Size (估) | SHA256 |
|---|---|---|---|---|---:|---|
| T1 | `task1_resnet18_final.pt` | `checkpoints/task1/final_resnet18_pretrained_bb1e-4_head5e-4_e100/best.pt` | ResNet-18 final, 100 ep, `bb=1e-4 / head=5e-4` | Best Val Acc **0.9138** @ ep99 | ~45 MB | _填_ |
| T2 | `task2_yolov8n_e120.pt` | `runs/task2/road_vehicle_yolov8n_e120/weights/best.pt` | YOLOv8n e120, batch=16, imgsz=640 | Best mAP@50 **0.4436** @ ep84 | ~6 MB | _填_ |
| T3 | `task3_unet_ce_dice.pt` | `checkpoints/task3/unet_ce_dice/best.pt` | U-Net CE+Dice (λ=1.0), 100 ep | Best mIoU **0.6320** @ ep95 | ~89 MB | _填_ |

## Tier 2 — 加分实验（与正文报告内容呼应）

| Task | Cloud Filename | Server Path | 实验 | 关键数字 | Size (估) | SHA256 |
|---|---|---|---|---|---:|---|
| T1 | `task1_resnet34_final.pt` | `checkpoints/task1/final_resnet34_pretrained_bb1e-4_head5e-4_e100/best.pt` | ResNet-34 final, 100 ep, `bb=1e-4 / head=5e-4` | Best Val Acc **0.9156** @ ep87 | ~85 MB | _填_ |
| T1 | `task1_vit_tiny.pt` | `checkpoints/task1/bonus_vit_tiny_pretrained_bb1e-4_head5e-4_e50/best.pt` | ViT-Tiny (timm `vit_tiny_patch16_224`), 50 ep | Best Val Acc **0.8397** @ ep33 | ~22 MB | _填_ |
| T2 | `task2_yolov8s_e120.pt` | `runs/task2/road_vehicle_yolov8s_e120/weights/best.pt` | YOLOv8s e120, batch=16, imgsz=640 | Best mAP@50 **0.5857** @ ep42 | ~22 MB | _填_ |

## Tier 3 — 损失消融（Task 3 三组对比的另两组）

| Task | Cloud Filename | Server Path | 实验 | 关键数字 | Size (估) | SHA256 |
|---|---|---|---|---|---:|---|
| T3 | `task3_unet_ce.pt` | `checkpoints/task3/unet_ce/best.pt` | U-Net CE only, 100 ep | Best mIoU 0.6253 @ ep98 | ~89 MB | _填_ |
| T3 | `task3_unet_dice.pt` | `checkpoints/task3/unet_dice/best.pt` | U-Net Dice only, 100 ep | Best mIoU 0.5779 @ ep37 | ~89 MB | _填_ |

## 建议的云盘组织结构

```
HW2_weights/
├── README.txt                     ← 把本文档摘要 / SHA256 表粘进去
├── required/
│   ├── task1_resnet18_final.pt
│   ├── task2_yolov8n_e120.pt
│   └── task3_unet_ce_dice.pt
├── bonus/
│   ├── task1_resnet34_final.pt
│   ├── task1_vit_tiny.pt
│   └── task2_yolov8s_e120.pt
└── ablation/
    ├── task3_unet_ce.pt
    └── task3_unet_dice.pt
```

总大小约 **450 MB**——百度云免费空间 / Google Drive 15 GB 免费空间都够装。

## 上传后回填步骤

1. **填 SHA256 + Size**：在服务器跑上面那段 `for f in ...` 命令，把每行 `<sha12> <size> <path>` 对回三个表。
2. **填 Cloud URL**：本文档顶部 `Cloud base URL` 行 + [README.md](README.md) §7 的"云盘 base URL"占位 + [report/HW2_report.md](report/HW2_report.md) 首页"权重云盘"占位三处都填 https://pan.baidu.com/s/xxxxx 形式的链接（带提取码，如有）。
3. **验证**：在 **无痕浏览器** 至少试一次下载 + sha256 校验，确认外部人能拿到。
4. **commit**：`git commit -m "docs: backfill weights cloud URL + sha256"`。

## 加载示例

```python
# Task 1 (ResNet-18 / ResNet-34 / ViT-Tiny)
from src.task1_cls.models import build_classifier
import torch
model = build_classifier("resnet18", num_classes=102, init="random")   # or "resnet34" / "vit_tiny"
ckpt = torch.load("task1_resnet18_final.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()

# Task 2 (YOLOv8n / YOLOv8s)
from ultralytics import YOLO
model = YOLO("task2_yolov8n_e120.pt")          # 直接传 .pt 路径

# Task 3 (U-Net)
from src.task3_seg.unet import UNet
from src.common.checkpoint import load_checkpoint
unet = UNet(num_classes=8, base_channels=32)
load_checkpoint("task3_unet_ce_dice.pt", model=unet, map_location="cpu", strict=True)
unet.eval()
```
