# HW2 Weekend Task Card (5/16 — 5/19)

> Day 4 daily standup 产出。DDL 2026-05-19 23:59，剩余 ~84 小时（4 个自然日）。

## 总览

| 项 | 状态 |
|---|---|
| DDL | 2026-05-19 23:59 |
| Day 4 验收 | 4/4（内容完成 + 仓库清理本轮 commit 后视为完整 ✅） |
| 主线必做项 | 三任务全部交付（结果 + 报告章节 + 演示视频 + 遮挡分析） |
| 加分项已完成 | 1 项（ResNet-34，+0.18 pp vs ResNet-18） |
| 加分项排期 | 2 项（ViT-Tiny + YOLOv8s，预计 Day 5 fire） |
| 报告 | 三个 task 章节初稿已 push；主报告 `report/HW2_report.md` 骨架本轮入库 |
| 单测 | 73 项（70 ok + 3 skipped 在服务器有 timm 后会全绿） |

## 已交付清单（截至 5/16 早）

- [x] T1：ResNet-18 50 ep / Random / CBAM / Grid (8 configs) / Final 100 ep / ResNet-34 bonus
- [x] T2：YOLOv8n e120 + ByteTrack + BoT-SORT 对照 + 4 帧遮挡分析 + 越线计数终版视频
- [x] T3：U-Net 三组 loss + 6×5 mask 对比图 + 2×2 训练曲线
- [x] 报告：task1/2/3_section_draft.md
- [x] 主报告骨架：report/HW2_report.md
- [x] SwanLab 曲线 84 张归档到 report/figures/task{1,3}/<sub>/
- [x] Day 4 上半段 figures 与 occlusion_analysis.md 入库

## 待办优先级

| 优先级 | 任务 | 责任 | ETA |
|---|---|---|---|
| 🔴 P0 | ViT-Tiny 训练 + 报告回填 | 服务器 + 本地 | 5/16 09:00—12:00 |
| 🔴 P0 | YOLOv8s 训练 + 跟踪/计数对比 | 服务器 | 5/16 11:00—18:00 |
| 🟡 P1 | 主报告 HW2_report.md：摘要 / 引言 / 实验环境补充 → 1.0 稿 | 本地 | 5/16 09:00—13:00 |
| 🟡 P1 | README 扩充：环境配置 / 三任务训练命令 / 测试 / 权重链接 | 本地 | 5/16 下午 |
| 🟡 P1 | 模型权重上传云盘（百度云/Google Drive） | 用户 | 5/16 下午 |
| 🟢 P2 | 结论与局限章节深化 + 附录 A/B | 本地 | 5/17 上午 |
| 🟢 P2 | PDF 编译第一稿 | 本地 | 5/17 下午 |
| 🟢 P2 | 自审 + 互审（数字/图表一致性逐条核对） | 本地 | 5/17 全天 |
| 🟢 P3 | README 最终化（含目录树、复现命令、权重链接） | 本地 | 5/18 上午 |
| 🟢 P3 | 干净环境复现 1-epoch 烟雾测试 | 服务器 | 5/18 下午 |
| 🟢 P3 | 报告所有图重导 300 dpi | 本地 | 5/18 下午 |
| 🟢 P3 | 终稿通读 ×3 | 本地 | 5/18 晚 |
| 🔴 P0 | **正式提交 PDF** | 用户 | 5/19 12:00—14:00 |

## Day 5 — 5/16 Saturday（缓冲 + 加分 + 报告血肉）

| 时段 | 任务 | 类型 | GPU? | Owner | 验收 |
|---|---|---|---|---|---|
| 09:00—11:00 | ViT-Tiny pretrained 训练 50 ep | 加分 | ✅ device 0 | 服务器 | best.pt + swanlab 链接 |
| 09:00—11:00 | 本地：HW2_report.md 摘要 / 引言 / 实验环境精修 | 报告 | ❌ | 本地 | 主报告 1.0 稿 commit |
| 11:00—18:00 | YOLOv8s e120 训练 | 加分 | ✅ device 0 | 服务器 | best.pt + results.csv |
| 11:00—12:00 | 写相关工作章节 references 验证 | 报告 | ❌ | 本地 | section 2 完成 |
| 13:00—14:30 | README 扩充：环境/训练/测试三段 | 文档 | ❌ | 本地 | README diff 可读 |
| 14:30—16:30 | 权重云盘上传：3 个 best.pt + counted.mp4 | 资产 | ❌ | 用户 | 链接验证可下载 |
| 18:00—19:00 | YOLOv8s + ByteTrack 跟踪 + count_video 重跑 | 加分 | ✅ device 0 | 服务器 | counted.mp4 + counts.json |
| 19:00—21:00 | 加分对比段写作：ResNet-34 + ViT-Tiny 进 task1，YOLOv8s 进 task2 | 报告 | ❌ | 本地 | 两段 markdown |
| 21:00 | Day 5 standup：明天 review checklist | 同步 | ❌ | 用户 | Day 6 任务卡定型 |

## Day 6 — 5/17 Sunday（精修 + 互审）

| 时段 | 任务 | 类型 | 验收 |
|---|---|---|---|
| 09:00—12:00 | 写结论与局限章节 + 附录 A（核心代码片段 ~30 行 × 4）+ 附录 B（超参表导出）| 报告 | HW2_report.md §7/§8/Appendix A/B 完整 |
| 13:00—15:00 | 自审：每个数字、每个表、每张图引用一致性逐条核对 | 报告 | review checklist 全打勾 |
| 15:00—17:00 | PDF 编译第一稿（推荐 pandoc：`pandoc HW2_report.md -o HW2_report.pdf --toc -V geometry:margin=2cm --pdf-engine=xelatex --include-in-header=zhfont.tex`）| 报告 | PDF 可正常渲染 |
| 17:00—19:00 | 通读 PDF，记 review 意见到 review.md | 报告 | review.md ≥ 5 条意见 |
| 19:00—21:00 | 按 review 意见修订；统一图表风格（matplotlib seaborn-paper） | 报告 | review 全部 close |

## Day 7 — 5/18 Monday（终化）

| 时段 | 任务 |
|---|---|
| 09:00—12:00 | README 最终化：环境配置 / 三任务训练命令 / 测试 / 权重链接 / 目录结构图 |
| 13:00—15:00 | 干净环境复现 Task 1 1-epoch 烟雾测试（确保陌生人能跑） |
| 15:00—17:00 | 报告所有图重导 300 dpi（matplotlib `savefig(..., dpi=300)`；Ultralytics 图保留原始） |
| 17:00—21:00 | 终稿通读 ×3 |

## Day 8 — 5/19 Tuesday（提交日）

| 时段 | 任务 |
|---|---|
| 09:00—11:00 | 终极通读：首页姓名学号 / GitHub / 权重云盘链接 / 摘要数字 / 表格数字逐项核对 |
| 11:00—12:00 | 提交前 Checklist 逐项打勾（参考执行方案 §8）|
| 12:00—14:00 | **正式提交 PDF**（提早到 13:00 完成，留 10 小时缓冲）|
| 14:00—23:59 | 故障缓冲：系统拥堵 / 链接失效 / 补交 |

## GPU 时间预算（剩余 5/16—5/19）

| 日期 | 可用 GPU h | 已规划 | 余量 | 备注 |
|---:|---:|---:|---:|---|
| 5/16 | 12 | 9（ViT-Tiny 2 + YOLOv8s 6 + track/count 1）| 3 | device 0 单卡串行 |
| 5/17 | 12 | 0 | 12 | 报告 + 互审，留 buffer 防 reruns |
| 5/18 | 6 | 0 | 6 | 1-epoch 烟雾 + 加分回补 buffer |
| 5/19 | 6 | 0 | 6 | 提交日，buffer only |
| **合计** | **36** | **9** | **27** | 余量 75% |

> 27 h 余量足以应付训练失败 / 重跑 / 调超参等 contingency。

## 服务器一键启动指令（5/16 早 9 点）

### 0. 同步 + 装 timm

```bash
cd /root/Zhr/DL/HW2
git fetch --all && git checkout main && git pull --ff-only
conda run -n hw2 pip install "timm>=0.9"
conda run -n hw2 python -c "import timm; print('timm version:', timm.__version__)"
mkdir -p logs/swanlab
```

### 1. ViT-Tiny 训练（~2 h on RTX 4090）

**HF 镜像说明**：`timm.create_model("vit_tiny_patch16_224", pretrained=True)` 内部会向 `huggingface.co` 下载权重；国内服务器无法直连。`build_vit_tiny_classifier` 已经把 `HF_ENDPOINT` 默认值改成 `https://hf-mirror.com`（在 import timm 之前 `os.environ.setdefault`），无需命令行额外设置；若海外环境想用 huggingface.co，外层 `export HF_ENDPOINT=https://huggingface.co` 即可覆盖。

#### 1a. 镜像预热 / 缓存权重（5 秒）

```bash
conda run -n hw2 python -c "
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
print('HF_ENDPOINT=', os.environ['HF_ENDPOINT'])
import timm
m = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=102)
print('ViT-Tiny OK, params=', sum(p.numel() for p in m.parameters())/1e6, 'M')
"
```

预期：`ViT-Tiny OK, params= 5.7 M`。缓存到 `~/.cache/huggingface/hub/models--timm--vit_tiny_patch16_224.augreg_in21k_ft_in1k/`，后续训练不再走网络。

#### 1b. 正式训练

```bash
conda run -n hw2 python -m src.task1_cls.train \
  --config configs/task1_vit_tiny_pretrained.yaml \
  --device cuda --swanlab-mode cloud --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task1_bonus_vit_tiny_pretrained.log \
  2>&1 | tee logs/swanlab/task1_bonus_vit_tiny_pretrained.console.log
```

预期：`checkpoints/task1/bonus_vit_tiny_pretrained_bb1e-4_head5e-4_e50/best.pt` + swanlab 链接。

#### 1c. 兜底（hf-mirror 也不可达时）

按可达性递增：
1. **手动 safetensors**：在外网机器 `huggingface-cli download timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k model.safetensors`，`scp` 到 `~/.cache/huggingface/hub/models--timm--vit_tiny_patch16_224.augreg_in21k_ft_in1k/snapshots/<sha>/`。
2. **ModelScope CDN**：`pip install modelscope && modelscope download --model timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`，再按 HF cache 目录结构重命名。
3. **砍掉 ViT-Tiny 加分**：已有 ResNet-34 覆盖 Task 1 加分需求，直接专注 YOLOv8s。

### 2. YOLOv8s 训练（~5—7 h）

**Disk-check 假阳性说明**：直接 `yolo detect train model=yolov8s.pt ...` 会触发 Ultralytics 内部的 `check_disk_space`，而该函数在部分 Docker overlayfs 环境下读到 `shutil.disk_usage(...).free=0`，**假性**报 `MemoryError: Insufficient free disk space 0.0 MB < 32.3 MB required`。绕开方法：**预下载** `yolov8s.pt` 到 cwd，Ultralytics 的 `attempt_download_asset` 会第一步命中本地文件，跳过整条 `safe_download → check_disk_space` 链。

#### 2a. 预下载 yolov8s.pt（~10 秒，镜像 fallback ghfast → gh-proxy → hf-mirror → 官方）

```bash
# 下载到 weights/yolov8s.pt（脚本默认）
conda run -n hw2 python scripts/fetch_yolo_weight.py --model yolov8s.pt
# 拷一份到 cwd 让 yolo CLI 直接命中
cp weights/yolov8s.pt yolov8s.pt
ls -la yolov8s.pt  # 应 ~22 MB
```

#### 2b. 正式训练（用项目 Python wrapper，不要直接用 bare yolo CLI）

bare `yolo detect train ... 2>&1 | tee ...` 走 pipe 时 `tqdm` 检测到非 TTY 会**几乎不刷新输出**，给人"训练已挂"的错觉；且 bare yolo CLI **不接受** `--swanlab-mode`/`--require-swanlab`/`--links-file` 这些项目 argparse flag（会立刻 `SyntaxError`）。Day 2 那次成功的 YOLOv8n e120 用的是项目 wrapper `src/task2_detect_track/train.py`，本次 v8s 加分同样用它：

```bash
# 启动前若上次半 run 残留，挪走（不删，便于查 results.csv 行数判断到底跑了多少 epoch）
if [ -d runs/task2/road_vehicle_yolov8s_e120 ]; then
  mv runs/task2/road_vehicle_yolov8s_e120 runs/task2/road_vehicle_yolov8s_e120_killed_$(date +%H%M)
fi

# fire（注意是 python -m，不是 yolo）
conda run -n hw2 python -m src.task2_detect_track.train \
  --config configs/task2_yolov8s.yaml \
  --device cuda \
  --swanlab-mode cloud \
  --require-swanlab \
  --links-file logs/swanlab/swanlab_links.md \
  --log-file logs/swanlab/task2_bonus_yolov8s_e120.log \
  2>&1 | tee logs/swanlab/task2_bonus_yolov8s_e120.console.log
```

预期：
- log 头部打印 SwanLab 实验 URL（点开实时看曲线，**不靠终端**判断进度）
- `logs/swanlab/task2_bonus_yolov8s_e120.log` 每 epoch 写一行（Python `logging` 行缓冲，不被 tee 截断）
- `runs/task2/road_vehicle_yolov8s_e120/results.csv` 每 epoch 追加一行
- 5—7 h 完成

中途确认还在跑（不要 kill）：

```bash
tail -n 20 logs/swanlab/task2_bonus_yolov8s_e120.log    # 行数应每 epoch 增长
wc -l runs/task2/road_vehicle_yolov8s_e120/results.csv  # 同上
nvidia-smi | grep -E "MiB|Default"                       # GPU 占用应 >> 0
```

> 若 batch=16 显存超：编辑 `configs/task2_yolov8s.yaml` 改 `batch: 12`。

#### 2c. 兜底（脚本 4 个镜像都不通时）

```bash
# 备用 A：直接 curl ghfast
curl -L -o yolov8s.pt \
  "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt"

# 备用 B：HF 镜像
curl -L -o yolov8s.pt \
  "https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/yolov8s.pt"

ls -la yolov8s.pt  # 验证 ~22 MB
```

若两个 curl 都失败，**砍掉 YOLOv8s 加分**（YOLOv8n e120 主线 + ResNet-34 + ViT-Tiny 已经覆盖加分需求）。

### 3. YOLOv8s + ByteTrack 跟踪 + 越线计数（~10 min）

```bash
conda run -n hw2 python -m src.task2_detect_track.track_video \
  --weights runs/task2/road_vehicle_yolov8s_e120/weights/best.pt \
  --source data/videos/second_video_2026-05-15_215310_932.mp4 \
  --tracker bytetrack.yaml \
  --project runs/track --name second_video_yolov8s_bytetrack \
  --device 0 \
  2>&1 | tee logs/swanlab/task2_bonus_yolov8s_track.log

conda run -n hw2 python -m src.task2_detect_track.count_video \
  --weights runs/task2/road_vehicle_yolov8s_e120/weights/best.pt \
  --source data/videos/second_video_2026-05-15_215310_932.mp4 \
  --line "60,360,1220,360" --line-margin 4.0 --exclude-classes "3,20" \
  --device 0 --project runs/track --name second_video_yolov8s_counts \
  --require-cuda \
  2>&1 | tee logs/swanlab/task2_bonus_yolov8s_counts.log
```

### 4. 拉数字 → 本地回填

```bash
conda run -n hw2 python -c "
import torch
ck = torch.load('checkpoints/task1/bonus_vit_tiny_pretrained_bb1e-4_head5e-4_e50/best.pt', map_location='cpu')
print('vit_tiny epoch=', ck.get('epoch'), 'best_score=', ck.get('best_score'), 'metrics=', ck.get('metrics'))
print('vit_tiny params=', sum(p.numel() for p in ck['model_state'].values())/1e6, 'M')
"

conda run -n hw2 python -c "
import csv
rows=list(csv.DictReader(open('runs/task2/road_vehicle_yolov8s_e120/results.csv')))
best=max(rows, key=lambda r: float(r['metrics/mAP50(B)']))
print('yolov8s best epoch=', best['epoch'], 'mAP50=', best['metrics/mAP50(B)'], 'mAP50-95=', best['metrics/mAP50-95(B)'])
print('yolov8s final epoch=', rows[-1]['epoch'], 'mAP50=', rows[-1]['metrics/mAP50(B)'])
"

cat runs/track/second_video_yolov8s_counts/counts.json | head -30
```

把这些数字发回，我同步到 `report/task1_section_draft.md` / `report/task2_section_draft.md` / `report/HW2_report.md`。

## 风险清单

| 风险 | 概率 | 应对 |
|---|---|---|
| 服务器无法访问 PyPI 装 timm | 低 | 用清华镜像 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "timm>=0.9"` |
| YOLOv8s 显存爆 | 中 | 退 batch=12 或 imgsz=576；或 GPU 抢占失败时改 device=1 |
| ViT-Tiny 用 ResNet 风格 transform 性能略低 | 低 | 接受 — 与 ResNet 同 transform 才公平对比 |
| 加分实验超时 | 中 | 5/17 早晨 9 点前未跑完则砍掉 YOLOv8s，仅保 ViT-Tiny |
| PDF 编译失败 | 中 | 备用：Word + 嵌入 PNG；提前在 5/17 完成首次编译 |
| 5/19 提交平台拥堵 | 低 | 提早到 13:00 完成；预留邮箱补交渠道 |

## 验收（本任务卡的 closing condition）

- [x] 4/4 Day 4 验收口径满足
- [x] 周末任务卡可读 / 可执行
- [x] ViT-Tiny 代码骨架就绪（models.py 工厂 + config + 单测 + requirements.txt）
- [ ] 5/16 11:00 前 ViT-Tiny 跑完并回填数字
- [ ] 5/16 18:00 前 YOLOv8s 跑完并回填数字
- [ ] 5/19 14:00 前 PDF 提交
