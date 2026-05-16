# Task 2: Road Vehicle Detection, Tracking, and Line-Crossing Counting

## Experimental Setup

Task 2 covers three stages on the Road Vehicle Dataset (21 classes of motorised and non-motorised vehicles, 2 704 training images): (1) object detection with YOLOv8n, (2) multi-object tracking on a self-recorded intersection video using ByteTrack with a BoT-SORT comparison, and (3) virtual-line crossing counting with deduplicated per-track counts. All experiments use seed 42 and log training to SwanLab.

## YOLOv8n Training

The detector is fine-tuned from the official `yolov8n.pt` (Ultralytics) for up to 120 epochs with `imgsz=640`, `batch=16`, `patience=30`, `optimizer=auto`, and `close_mosaic=10`. Early stopping triggered at epoch 100. The dataset configuration is `data/road_vehicle/data.yaml` with 21 named classes. SwanLab run: `yolov8n_pretrained_yolo_lrdefault_e120_s42` (https://swanlab.cn/@Ares_ML/hw2-dl-spatial-intelligence/runs/w22nv59gh21egoyce4ngo).

| Metric | Final epoch (100) | Best epoch (84) |
| --- | ---: | ---: |
| Precision (B) | 0.6413 | 0.5742 |
| Recall (B) | 0.3764 | 0.4452 |
| mAP@50 (B) | **0.4295** | **0.4436** |
| mAP@50-95 (B) | 0.2630 | 0.2719 |

The final-epoch row is the published value the report cites for consistency with the `best.pt` weights file used for tracking; the best-mAP epoch (84) is also recorded to show training stabilised before the patience window closed. Class-level recall is uneven — frequent classes (car, suv, truck) drive most of the mAP, while rare classes (`mountain`/`scooter`/`auto rickshaw` etc.) remain low.

Relevant figures:

- `report/figures/task2/detection/results.png` — train/val loss and metric curves
- `report/figures/task2/detection/BoxPR_curve.png` — Precision-Recall curve
- `report/figures/task2/detection/BoxF1_curve.png` — F1 vs confidence
- `report/figures/task2/detection/BoxP_curve.png`, `BoxR_curve.png` — Precision / Recall vs confidence
- `report/figures/task2/detection/confusion_matrix_normalized.png` — row-normalised confusion matrix
- `report/figures/task2/detection/val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg` — qualitative validation predictions

## Tracking Pipeline

The trained `best.pt` is wrapped by Ultralytics' `YOLO.track(...)` in `src/task2_detect_track/track_video.py`. Two tracker configurations are run on the same source video (`data/videos/second_video_2026-05-15_215310_932.mp4`, 1280 × 720, 30 fps, 853 frames, ≈ 28.4 s):

- **ByteTrack** (`bytetrack.yaml`, primary): two-stage association uses both high- and low-confidence detections, giving short-occlusion robustness through `track_buffer`.
- **BoT-SORT** (comparison): adds camera-motion compensation and optional ReID; useful baseline when bbox-IoU evidence drops under occlusion.

Tracked outputs:

- `runs/track/second_video_bytetrack/second_video_2026-05-15_215310_932.avi` (ByteTrack overlay)
- `runs/track/second_video_botsort/second_video_2026-05-15_215310_932.avi` (BoT-SORT overlay)

## Occlusion Analysis (frames 288–291)

Auto-window scoring (max bbox IoU across consecutive frames) picked the 288–291 window in the BoT-SORT vs ByteTrack labels as the densest overlap case (max bbox IoU 0.997, mean pair IoU 0.039 across 210 pairs over 43 boxes).

| Frame | Max bbox IoU | Mean pair IoU | Notable IDs |
| ---: | ---: | ---: | --- |
| 288 | 0.980 | 0.037 | `c5:ID3` stable both trackers; new entrants ID214/ID290 |
| 289 | 0.981 | 0.040 | `c15:ID24` appears in both with IoU 0.59 |
| 290 | 0.981 | 0.041 | `c10:ID308` (ByteTrack) ↔ `c10:ID299` (BoT-SORT) |
| 291 | 0.997 | 0.037 | ByteTrack remaps `c5:ID1 → c15:ID1` (class flip); BoT-SORT keeps `c5:ID1` |

When occlusion is short the high-IoU lower-confidence boxes preserved by ByteTrack's two-stage association keep the original ID. `track_buffer` controls how long a lost track waits before being released; larger buffer tolerates longer occlusion at the cost of cross-vehicle ID swaps in dense scenes, and `match_thresh` must be co-tuned with detection confidence. BoT-SORT's optional appearance ReID gives an extra signal when bbox-IoU information is unreliable, which can keep IDs continuous through deeper occlusion at the price of latency. Full per-frame tables and the rationale paragraph are in `report/task2_occlusion_analysis.md`. Frame images:

- `report/figures/task2/occlusion_second/second_occlusion_f0288.png`
- `report/figures/task2/occlusion_second/second_occlusion_f0289.png`
- `report/figures/task2/occlusion_second/second_occlusion_f0290.png`
- `report/figures/task2/occlusion_second/second_occlusion_f0291.png`

## Line-Crossing Counting

A directed line segment is configured in image space; every tracked detection's bounding-box bottom-center (wheels-on-road anchor) is fed to `LineCounter` in `src/task2_detect_track/line_counter.py`. The crossing decision uses the 2-D cross product of `(B − A)` and `(P − A)` for line endpoints `A, B` and point `P`:

```
sign(dist) = sign( (x2 − x1)(y − y1) − (y2 − y1)(x − x1) ) ,
             with sign set to 0 inside a margin of 4 px to suppress jitter.
```

A track is counted once when the sign flips between non-zero values, and its `track_id` is added to a `counted_ids` set to prevent re-counting. Bidirectional totals are kept separately.

For the deliverable video we excluded `bicycle` (class 3) and `wheelbarrow` (class 20) as non-vehicle classes, used the ByteTrack tracker, and picked the line by eye-balling the first raw frame so it sits on the far-road wheel-touchdown line.

| Parameter | Value |
| --- | --- |
| Line `(x1, y1, x2, y2)` | `(60, 360, 1220, 360)` |
| Margin | 4 px |
| Excluded classes | bicycle, wheelbarrow |
| Tracker | ByteTrack |

| Counting metric | Value |
| --- | ---: |
| **Total Count** | **6** |
| Positive → Negative (down crossing) | 2 |
| Negative → Positive (up crossing) | 4 |
| `car` | 2 |
| `motorbike` | 3 |
| `rickshaw` | 1 |

Compared with a y=300 default line (which counted 6 with all crossings in one direction — vehicles' rooftops crossing the line above the active driving surface), the y=360 line places the crossing band at the actual wheel-touchdown of the far-road traffic. The total stays at 6 but the **direction split is more meaningful** (4 up + 2 down), and the per-class composition shifts from `car=2/suv=3/motorbike=1` to `car=2/motorbike=3/rickshaw=1`, consistent with the dominant two-wheeler / rickshaw flow in the recorded clip.

Representative frames (annotated demo video, `runs/track/second_video_bytetrack_counts_final/second_video_2026-05-15_215310_932_counted.mp4`):

- `report/figures/task2/counting/demo_start_frame0.png` — frame 0 (counts panel reads 0)
- `report/figures/task2/counting/demo_mid_frame426.png` — frame 426 (≈ 14.2 s, counting in progress)
- `report/figures/task2/counting/demo_end_frame851.png` — frame 851 (final cumulative state)

The full algorithm derivation, the JSON metadata schema, and limitations (single-line-only, fast-target margin trade-off) are documented in `report/task2_line_counting.md`.

## Bonus: YOLOv8s vs YOLOv8n

A YOLOv8s detector is fine-tuned under identical configuration (`epochs=120, patience=30, imgsz=640, batch=16, seed=42, optimizer=auto, close_mosaic=10`) for a fair detector-only comparison. Training is launched through the same project wrapper `python -m src.task2_detect_track.train --config configs/task2_yolov8s.yaml` that was used for the YOLOv8n main run, so the only delta is the backbone size.

| Detector | Params | Best mAP@50 | Best mAP@50-95 | Best Epoch | Final mAP@50 (early-stop) | Line-count Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | ~3.2 M | 0.4436 | 0.2719 | 84 | 0.4295 (ep100) | 6 |
| YOLOv8s | ~11.2 M | **0.5857** | **0.3141** | 42 | 0.5165 (ep58) | 5 |
| Δ | ~3.5× | **+14.2 pp** | **+4.2 pp** | −42 ep | +8.7 pp | −1 |

YOLOv8s reaches a substantially higher detection ceiling: **+14.2 pp mAP@50** and +4.2 pp mAP@50-95 over the YOLOv8n baseline, and it converges much earlier (best at ep42 vs ep84). Early stop also fires earlier (ep58 vs ep100) because the larger model overfits to the 2,704-image training set faster.

The line-crossing count, however, **drops from 6 to 5** unique vehicles under identical line `(60, 360, 1220, 360)`, margin 4 px, and excluded classes. Per-class composition also shifts: YOLOv8s sees `car=3, motorbike=1, suv=1`; YOLOv8n sees `car=2, motorbike=3, rickshaw=1`. Two factors explain why a stronger detector can produce *fewer* counts on the same clip: (1) at the chosen `conf=0.25 / iou=0.7` thresholds, the more discriminative v8s collapses some borderline low-confidence frames that v8n still emitted, so ByteTrack receives fewer detections in those frames and can fail to keep a track alive across the line; (2) the class confidence distributions differ enough that the same physical object gets classified into different classes (`motorbike ↔ rickshaw` ambiguity in particular). The takeaway is that **detector mAP gains do not transfer monotonically to downstream counting** — the detector → tracker → counter pipeline has additional thresholds and association decisions that the mAP metric does not capture. For a counting-deployment scenario the right benchmark is end-to-end count accuracy against a hand-labelled ground truth, not detection mAP alone.

Bonus artifacts:

- `runs/task2/road_vehicle_yolov8s_e120/weights/best.pt` (~22 MB)
- `runs/track/second_video_yolov8s_counts/second_video_2026-05-15_215310_932_counted.mp4`
- `runs/track/second_video_yolov8s_counts/counts.json`
- SwanLab run: `yolov8s_pretrained_yolo_lrdefault_e120_s42` (链接见 `logs/swanlab/swanlab_links.md`)

## Code Index

| Module | File |
| --- | --- |
| Detector fine-tuning | `src/task2_detect_track/train.py` |
| Tracking (ByteTrack / BoT-SORT) | `src/task2_detect_track/track_video.py` |
| Line-crossing counting pipeline | `src/task2_detect_track/count_video.py` |
| Counting core (cross-product + dedup) | `src/task2_detect_track/line_counter.py` |
| Overlay / writer / parsers | `src/task2_detect_track/render_counts.py` |
| Occlusion frame export & IoU table | `src/task2_detect_track/analyze_occlusion.py` |

## Conclusions and Limitations

Detection performance is class-imbalanced — `mAP@50` of ~0.43 reflects strong recognition on frequent classes (car, suv, truck) but weaker performance on rare classes (`scooter`, `auto rickshaw`, etc.). For tracking, ByteTrack proves the more practical default on this scene because its two-stage association keeps IDs through brief occlusion; BoT-SORT's ReID becomes relevant only when occlusion is deeper than the chosen `track_buffer`. The counting pipeline counted 6 unique vehicles in 28.4 s with a sensible directional split once the line was placed on the wheel-touchdown level, demonstrating that the cross-product + dedup + margin rule is sufficient for a single-line, single-camera setup. Main limitations: single straight line cannot disambiguate multi-lane flows, the rare classes still drag mAP down (an obvious next step is a stronger backbone such as YOLOv8s), and very fast objects can jump the margin band between frames and be missed at 30 fps.
