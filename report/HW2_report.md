# HW2: Deep Learning and Spatial Intelligence

> 课程作业 · 主报告（汇总稿）

| 项 | 内容 |
| --- | --- |
| 课程 | 深度学习与空间智能 |
| 作业 | HW2 |
| 作者 / 学号 | _请填_ |
| 分工 | 单人完成 |
| GitHub | https://github.com/Ares-ML/hw2-dl-spatial-intelligence |
| 模型权重云盘 | https://pan.baidu.com/s/1HrU0pjPZykTFZ8GlINiB5Q?pwd=213g |
| 提交日期 | 2026-05-19 |

---

## 摘要

本工作在三个公开数据集上完成视觉理解三项子任务，覆盖图像级分类、对象级检测+跟踪+越线计数、以及像素级语义分割：

- **Task 1（Flower102 分类）**：以 ImageNet 预训练 ResNet-18 为主线，通过差化学习率（`backbone=1e-4, head=5e-4`）训练 100 epoch 获最佳验证准确率 **91.38%**；CBAM 注意力消融与超参网格证明 backbone 学习率提升与训练长度延长是关键改进点。加分实验：ResNet-34 把准确率推到 **0.9156**（+0.18 pp），ViT-Tiny 仅得 **0.8397**（−7.41 pp）——揭示了在 1 k 训练样本 + 同 transform 配方下，ImageNet-pretrained CNN 仍领先同等参数预算的 Transformer。
- **Task 2（Road Vehicle 检测+跟踪+越线计数）**：YOLOv8n 训练 120 epoch（early-stop at e100），final mAP@50=**0.4295** / best mAP@50=0.4436；ByteTrack 与 BoT-SORT 在自拍 28.4 秒路口视频上做轨迹连续性对比；用叉乘符号 + counted_ids 去重 + 4 px margin 的越线计数算法在线 `(60,360,1220,360)` 下统计到 **6 辆机动车**跨线（2 下行 + 4 上行）。加分实验 YOLOv8s 把 best mAP@50 抬到 **0.5857**（+14.2 pp），但同 line / 同阈值下越线计数从 6 → 5、per-class 分布也变——揭示了**检测指标 ≠ 计数指标**，下游链路阈值与关联策略会放大细微误差。
- **Task 3（Stanford Background 分割）**：自实现 U-Net（base_channels=32），三种损失（CE / Dice / CE+Dice，λ=1.0）在严格同条件下对比，**CE+Dice 取得最佳全局 mIoU=0.6320**，Pixel Acc 0.8228；Dice 单独使用在长尾稀有类 `mountain` 上崩塌至 IoU 0.0；mask 可视化按 CE 模型 per-image mIoU 在 6 个档位样本（best / 2×median / 2×hard / worst）展示三种损失的差异。

主要结论：（1）pretraining 在所有三任务上都是首要决定因素；（2）单纯增大 backbone（或换更新架构）在小数据集只有边际收益、有时甚至倒退；（3）下游指标（线越计数）对上游检测器的阈值/类别分布很敏感，应以端到端指标为准；（4）组合损失（CE+Dice）在长尾分割上比单一损失稳健。代码、权重与曲线全部以 SwanLab 链接和 GitHub repo 公开复现。

## 1. 引言

视觉理解任务可按输出粒度分为图像级（分类）、对象级（检测/跟踪）、像素级（分割）三层。本作业以三个对应数据集为基底，要求在统一的工程框架内对每一层级完成「主线 + 消融 + 分析」三段式实验。为此，我们：

1. 搭建统一的 PyTorch + SwanLab 日志框架（`src/common/{logger, swanlab_logger, checkpoint, seed, metrics}.py`），所有任务共享 seed=42、checkpoint manager（monitor best val 指标）、统一日志路径。
2. 每个任务都有完整的训练入口（`src/task{1,2,3}_*/train.py`）、可复现配置（`configs/*.yaml`）、单元测试（`tests/*.py`），并在 SwanLab 云端记录所有 run。
3. 报告主线必做实验（Task 1: pretrained / random / CBAM / 超参网格；Task 2: YOLOv8n / ByteTrack / BoT-SORT / 遮挡 / 越线计数；Task 3: CE / Dice / CE+Dice），并在缓冲日补加分实验。

本报告 §3 实验环境给出统一的硬件 / 软件 / 种子设置；§4—§6 分别详细描述三任务的方法与结果；§7—§8 总结全局发现与共性局限。

## 2. 相关工作

- **ResNet** [1] 提出残差连接以训练深网络，至今仍是图像分类的主流 baseline。
- **CBAM** [2] 将通道注意力与空间注意力级联插入到卷积特征图上，常作为分类 backbone 的"附加增益"模块。
- **YOLOv8** [3] 是 Ultralytics 推出的一阶段检测/跟踪一体化框架，YOLOv8n/s/m/l/x 提供模型容量阶梯。
- **ByteTrack** [4] 利用高置信度 + 低置信度两阶段关联，在短时遮挡上保持 ID 稳定；**BoT-SORT** [5] 增加相机运动补偿与可选 ReID。
- **U-Net** [6] 是医学分割经典工作，使用对称编解码 + 跳跃连接；本作业从零实现。
- **Dice loss** [7]（与 V-Net 论文同期出现）作为软重叠度量，对类不平衡较鲁棒；CE+Dice 是常见组合写法。
- **Stanford Background Dataset** [8] 提供 715 张户外场景的 8 类像素级标注。**Flower 102** [9] 提供 102 类花卉的细粒度分类挑战。

## 3. 实验环境

| 项 | 取值 |
|---|---|
| 硬件 | NVIDIA RTX 4090（24 GB）× 1（device 0；其他卡有其他实验占用） |
| 系统 / 容器 | Linux + Docker，NVIDIA Driver 535.129.03，CUDA 12.2 |
| Python | 3.10 |
| PyTorch | 2.5.1（CUDA 11.8 wheels） |
| 关键依赖 | `ultralytics==8.4.41, albumentations==2.0.8, swanlab==0.7.17, opencv-python-headless==4.10.0.84, timm>=0.9` |
| 随机种子 | 42（全部任务） |
| 数据集本地路径 | `data/flower102/, data/road_vehicle/, data/stanford_background/, data/videos/` |
| 实验追踪 | SwanLab project `hw2-dl-spatial-intelligence`（`@Ares_ML`），group=task{1,2,3} |
| GPU 训练总时长 | ~30 h（Task 1 ~12 h，Task 2 ~6 h，Task 3 ~6 h，bonus ~9 h） |

完整环境创建脚本与诊断流程在 [README.md](../README.md)。

## 4. Task 1 — Flower102 Classification

详见独立章节：[task1_section_draft.md](task1_section_draft.md)

**章节要点**：6 段（实验设置 / Pretraining 消融 / 注意力对比 / 超参网格 / **加分：Backbone Comparison（ResNet-34 + ViT-Tiny vs ResNet-18）** / 结论与局限）。最佳数字 = ResNet-34 bonus **0.9156**（epoch 87，+0.18 pp vs ResNet-18 final 0.9138）；ViT-Tiny bonus **0.8397**（epoch 33，−7.41 pp）。

辅助表格 / 图：

- 主表：[report/tables/task1_grid_summary.md](tables/task1_grid_summary.md)、[task1_t1a_summary.md](tables/task1_t1a_summary.md)、[task1_t1c_summary.md](tables/task1_t1c_summary.md)、[task1_resnet34_vs_resnet18.md](tables/task1_resnet34_vs_resnet18.md)（含 ResNet-18 / ResNet-34 / ViT-Tiny 三行）
- 训练曲线（SwanLab 导出）：[report/figures/task1/](figures/task1/) 下 7 个子目录（t1_a/t1_c/t1_cbam/t1_final/t1_grid/t1_resnet34/t1_vit）
- 加分 checkpoints：`checkpoints/task1/{final_resnet34_pretrained_bb1e-4_head5e-4_e100, bonus_vit_tiny_pretrained_bb1e-4_head5e-4_e50}/best.pt`

## 5. Task 2 — Road Vehicle Detection, Tracking, Counting

详见独立章节：[task2_section_draft.md](task2_section_draft.md)

**章节要点**：8 段（实验设置 / YOLOv8n 训练 / 检测指标 / 跟踪流程 / 遮挡分析 / 越线计数 / **加分：YOLOv8s vs YOLOv8n** / 结论与局限）。主线最终演示视频含 bbox/class/ID/计数线/统计面板/越线红点；总计 6 辆机动车跨线，方向分布 4 上 + 2 下，per-class car=2 motorbike=3 rickshaw=1。加分 YOLOv8s best mAP@50=**0.5857**（+14.2 pp vs YOLOv8n 0.4436），但同 line 下越线计数 6 → 5。

辅助表格 / 图：

- 检测图：[report/figures/task2/detection/](figures/task2/detection/)（results.png + 4 张曲线 + 2 张混淆矩阵 + 3 张 val_batch 预测）
- 遮挡帧：[report/figures/task2/occlusion_second/](figures/task2/occlusion_second/) 与分析 [report/task2_occlusion_analysis.md](task2_occlusion_analysis.md)
- 计数演示帧：[report/figures/task2/counting/](figures/task2/counting/) 3 张代表帧 + 元数据
- 越线计数完整说明：[report/task2_line_counting.md](task2_line_counting.md)
- 加分 artifacts：`runs/task2/road_vehicle_yolov8s_e120/weights/best.pt` + `runs/track/second_video_yolov8s_counts/{*_counted.mp4, counts.json}`

## 6. Task 3 — Stanford Background Segmentation

详见独立章节：[task3_section_draft.md](task3_section_draft.md)

**章节要点**：8 段（实验设置 / U-Net 架构 / 损失函数 / 量化对比 / 训练曲线 / Mask 可视化 / 结论与局限 / 代码索引）。CE+Dice 全局 mIoU 0.6320 > CE 0.6253 > Dice 0.5779。

辅助表格 / 图：

- Table 3：[report/tables/task3_loss_comparison.md](tables/task3_loss_comparison.md)、class-wise IoU [task3_class_iou.md](tables/task3_class_iou.md)
- Fig 5：[report/figures/task3/fig5_loss_curves.png](figures/task3/fig5_loss_curves.png)（2×2 网格 train/val loss + val mIoU + val pixel acc）
- Fig 6：[report/figures/task3/mask_comparison.png](figures/task3/mask_comparison.png)（6×5 image/GT/pred-CE/pred-Dice/pred-CE+Dice 网格 + per-cell mIoU）
- SwanLab 训练曲线（per-task）：[report/figures/task3/curves_{ce,dice,ce_dice}/](figures/task3/)

## 7. 结论

三项任务统一在如下观察上：

1. **预训练是性能基线的第一序贡献**：Task 1 中 pretrained 比 random 提升 45 pp；Task 2 的 YOLOv8 是 Ultralytics 预训练权重微调；Task 3 是唯一的"从零开始"且 mIoU 维持在 0.6 量级，提示如果给 U-Net 配预训练 encoder（如 ImageNet ResNet-34）还有上调空间。
2. **训练 recipe 比 backbone 容量更重要（在小数据规模下）**：Task 1 超参网格找到的 `bb=1e-4, head=5e-4, 100 ep` 比 backbone 升级到 ResNet-34 多带来 +1.4 pp；Task 3 的 CE+Dice 组合损失多带 +0.7 pp。
3. **任务复杂度差异引导工具栈**：分类用 PyTorch 原生；检测/跟踪用 Ultralytics + 自写 ByteTrack 包装；分割用 PyTorch 原生 + 自写 mIoU/Dice。统一的 logger / checkpoint / metrics utility 让三任务的训练循环结构高度一致。

## 8. 局限与改进方向

- **Task 1**：所有模型 train_acc=1.0，validation 之外没有进一步信号；augmentation（RandAugment / mixup）与 label smoothing 可能继续提升 1—2 pp。
- **Task 2**：mAP@50≈0.43 受稀有类（scooter/auto rickshaw 等）拖累；YOLOv8s 加分实验若呈现明显提升，则报告里建议把 s 作为主线；越线计数对单线场景充分，但多车道需要扩展为多线/多边形。
- **Task 3**：715 张样本规模小，class-wise IoU 噪声高（Dice 的 mountain=0 是这种噪声的极端表现）；引入预训练 encoder（如 ImageNet-pretrained ResNet-34 from Task 1）或加更强 augmentation 会有可观提升。

## 附录 A：核心代码片段

将在 Day 6 提取，包括：

- `src/task2_detect_track/line_counter.py::LineCounter.update`（叉乘+counted_ids+margin 的 ~30 行核心逻辑）
- `src/task3_seg/eval_compare.py::evaluate_checkpoint`（全 val 集 intersection/union 累加 → 全局 mIoU 的 ~25 行）
- `src/task1_cls/models.py::build_classifier`（多 backbone 工厂分派 ~15 行）
- `src/task3_seg/losses.py::CEDiceLoss.forward`（CE + Dice 组合的 ~15 行）

## 附录 B：完整超参表

将在 Day 6 由 `configs/task{1,2,3}_*.yaml` 自动导出。要点：

- Task 1: `bb=1e-4, head=5e-4, epochs=100, batch=32, image_size=224, seed=42`
- Task 2: `epochs=120, imgsz=640, batch=16, patience=30, optimizer=auto, seed=42`
- Task 3: `lr=1e-3 (AdamW), epochs=100, image_size=256, ignore_index=255, num_classes=8, seed=42`

## 参考文献

[1] He et al. *Deep Residual Learning for Image Recognition*. CVPR 2016.
[2] Woo et al. *CBAM: Convolutional Block Attention Module*. ECCV 2018.
[3] Ultralytics. *YOLOv8*. 2023. https://github.com/ultralytics/ultralytics
[4] Zhang et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
[5] Aharon et al. *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*. arXiv:2206.14651, 2022.
[6] Ronneberger et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
[7] Milletari et al. *V-Net (Dice loss)*. 3DV 2016.
[8] Gould et al. *Decomposing a Scene into Geometric and Semantically Consistent Regions*. ICCV 2009.
[9] Nilsback & Zisserman. *Automated Flower Classification over a Large Number of Classes*. IVCNZ 2008.
[10] Dosovitskiy et al. *An Image is Worth 16×16 Words (ViT)*. ICLR 2021.
