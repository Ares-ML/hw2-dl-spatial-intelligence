# Task 2 Line-Crossing Counting

> Day 4 11:00—12:00 交付：在 `best.pt + ByteTrack` 之上叠加越线计数逻辑，生成带计数显示的最终演示视频。
> 本文件落地后由 `report/HW2_report.md` Task 2 章节直接引用。

## 1. 算法

跨线判定采用 **二维叉乘符号 + counted_ids 去重 + perpendicular margin 防抖**。

对线段 $\mathbf{AB}=\{(x_1,y_1)\to (x_2,y_2)\}$ 与目标点 $\mathbf{P}=(x,y)$，定义有符号距离：

$$
\text{side}(\mathbf{P}) = \operatorname{sign}\!\left(\frac{(x_2-x_1)(y-y_1)-(y_2-y_1)(x-x_1)}{\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}}\right),\quad \text{若 } |\,\text{距离}\,| \le \text{margin}, \text{ 则记为 }0.
$$

每个 `track_id` 缓存上一帧的 `side`；当本帧 side 与上帧 side 均非零且符号翻转，即触发一次计数事件，`counted_ids` 集合记录该 ID 防止重复计数。计数参考点取 **bbox 底边中心** $\bigl(\tfrac{x_1+x_2}{2}, y_2\bigr)$（车轮触地点），比几何中心更稳定（卡车/SUV 高度差异不会移动该锚点）。

## 2. 实现位置

- 计数核心：[src/task2_detect_track/line_counter.py](src/task2_detect_track/line_counter.py) — `LineCounter` 类（含 `total_count` / `direction_counts` / `counted_ids` / `last_side`）。
- 视频流水线：[src/task2_detect_track/count_video.py](src/task2_detect_track/count_video.py) — `model.track(stream=True, persist=True, save=False)` 迭代 `Result`，feed 入 `LineCounter`，写自有 `cv2.VideoWriter`。
- 绘制辅助：[src/task2_detect_track/render_counts.py](src/task2_detect_track/render_counts.py) — `draw_counting_line` / `draw_count_dots` / `draw_count_overlay` / `pick_video_writer` / `parse_line_arg` / `parse_class_filter` / `bbox_bottom_center`。
- 单元测试：[tests/test_line_counter.py](tests/test_line_counter.py)（6 项）+ [tests/test_render_counts.py](tests/test_render_counts.py)（14 项）。

## 3. 关键参数

| 项 | 值 | 说明 |
|---|---|---|
| 检测权重 | `runs/task2/road_vehicle_yolov8n_e120/weights/best.pt` | YOLOv8n / 120 epochs / Road Vehicle 21 类 |
| Tracker | `bytetrack.yaml` | Ultralytics 默认 ByteTrack 配置（主输出） |
| 输入视频 | `data/videos/second_video_2026-05-15_215310_932.mp4`（1280×720 / 30 fps / 28.4 s / 853 帧） | 路口、密度高，与遮挡分析同源 |
| 计数线（首版） | `(200, 300, 1100, 300)` | y=300 压在车顶位置；total=6 (neg→pos=6) |
| 计数线（终版） | `(60, 360, 1220, 360)` | y=360 压在远端主车道车轮触地线；x 范围加宽到画面左右各 60 px |
| margin | 4.0 px | perpendicular jitter band |
| 排除类 | `class_id ∈ {3 (bicycle), 20 (wheelbarrow)}` | 仅统计机动车 |
| conf / iou / imgsz | 0.25 / 0.7 / 640 | 与训练评估一致 |

## 4. 结果

### 4.0 首版（默认线 y=300）烟雾跑

| 指标 | 值 |
|---|---|
| Total Count | 6 |
| Positive → Negative | 0 |
| Negative → Positive | 6 |
| Per-class | `car=2, suv=3, motorbike=1` |
| 视频时长 / 帧数 | 28.4 s / 853 帧 |
| 含 Tracks 帧占比 | 853 / 853 |
| 推理耗时 / GPU peak mem | 24.8 s / 32.0 MB（YOLOv8n） |
| 日志 | `logs/swanlab/task2_second_counts_smoke.log` |
| Run 目录 | `runs/track/second_video_bytetrack_counts_smoke/` |

### 4.1 终版（调整线 y=360）总览（待运行后填入）

> 服务器端运行 `count_video.py` final 后，把 `counts.json` 的数据填到下表，并把 `*_counted.mp4` / `*_first_annotated.png` 拷贝到 `report/figures/task2/counting/`。

| 指标 | 值 |
|---|---|
| Total Count | _N_ |
| Positive → Negative | _N1_ |
| Negative → Positive | _N2_ |
| 视频时长 / 帧数 | _T s / F 帧_ |
| 含 Tracks 帧占比 | _frames_with_tracks / frame_count_ |
| 推理耗时 / GPU peak mem | _T_wall s / M MB_ |
| 日志 | `logs/swanlab/task2_second_counts_final.log` |
| Run 目录 | `runs/track/second_video_bytetrack_counts_final/` |

### 4.2 Per-Class 计数（top-N）

| 类别 | 计数 |
|---|---|
| car | _N_ |
| truck | _N_ |
| suv | _N_ |
| ... | ... |

### 4.3 代表帧

- 起始帧（首帧叠加）：`report/figures/task2/counting/second_video_2026-05-15_215310_932_first_annotated.png`
- 计数发生瞬间：_待截取（用 VLC / ffmpeg 抓 counted_centers 出现的帧）_
- 结束帧（最终累计值）：_待截取_

## 5. 最终交付

- 演示视频：`report/figures/task2/counting/second_video_2026-05-15_215310_932_counted.mp4`
- 元数据 JSON：`report/figures/task2/counting/counts.json`
- 服务器原始产物目录：`runs/track/second_video_bytetrack_counts_final/`

## 6. 局限与改进方向

- 单条直线无法覆盖多车道分流场景，可扩展为多线 / 多边形 ROI。
- 极快目标在两帧间可能跨越「margin 带」与「过线」两个动作合并，导致漏计（margin=4 px 在 30 fps 下足以应付常见车速，但更高速度需要降 margin 并依赖更高帧率）。
- `result.plot()` 沿用 Ultralytics 默认配色，多类时颜色相近；如需更清晰的可读性可自定义 palette。
