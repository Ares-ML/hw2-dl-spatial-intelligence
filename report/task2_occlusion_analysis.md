# Task 2 Occlusion Analysis

Source video: `runs/track/second_video_bytetrack/second_video_2026-05-15_215310_932.avi`
Frame window: `288`-`291`
Primary tracker labels: `runs/track/second_video_bytetrack/labels`
Comparison tracker labels: `runs/track/second_video_botsort/labels`

Exported frames:
- `report/figures/task2/occlusion_second/second_occlusion_f0288.png`
- `report/figures/task2/occlusion_second/second_occlusion_f0289.png`
- `report/figures/task2/occlusion_second/second_occlusion_f0290.png`
- `report/figures/task2/occlusion_second/second_occlusion_f0291.png`

Auto-window score: max IoU=0.997, mean pair IoU=0.039, pairs=210, boxes=43.

| Frame | ByteTrack IDs | BoT-SORT IDs | Max bbox IoU | Mean bbox IoU |
| ---: | --- | --- | ---: | ---: |
| 288 | c5:ID1, 0.34<br>c5:ID3, 0.86<br>c17:ID24, 0.25<br>c10:ID214, 0.76<br>c10:ID242, 0.70<br>c10:ID253, 0.54<br>c10:ID265, 0.31<br>c5:ID267, 0.76<br>c15:ID290, 0.41<br>c10:ID308, 0.37<br>...(+1) | c5:ID1, 0.34<br>c5:ID3, 0.86<br>c17:ID23, 0.25<br>c10:ID148, 0.70<br>c10:ID206, 0.76<br>c10:ID243, 0.54<br>c10:ID255, 0.31<br>c5:ID257, 0.76<br>c15:ID278, 0.41<br>c10:ID299, 0.37<br>...(+1) | 0.980 | 0.037 |
| 289 | c5:ID1, 0.34<br>c5:ID3, 0.86<br>c15:ID24, 0.59<br>c10:ID214, 0.74<br>c10:ID242, 0.68<br>c10:ID253, 0.57<br>c10:ID265, 0.28<br>c15:ID290, 0.43<br>c10:ID308, 0.46<br>c3:ID309, 0.47<br>...(+1) | c5:ID1, 0.34<br>c5:ID3, 0.86<br>c15:ID23, 0.59<br>c10:ID148, 0.68<br>c10:ID206, 0.74<br>c10:ID243, 0.57<br>c10:ID255, 0.28<br>c5:ID257, 0.28<br>c15:ID278, 0.43<br>c10:ID299, 0.46<br>...(+2) | 0.981 | 0.040 |
| 290 | c5:ID1, 0.32<br>c5:ID3, 0.86<br>c15:ID24, 0.71<br>c10:ID214, 0.74<br>c10:ID242, 0.69<br>c10:ID253, 0.61<br>c10:ID265, 0.34<br>c15:ID290, 0.52<br>c10:ID308, 0.48<br>c3:ID309, 0.53<br>...(+1) | c5:ID1, 0.32<br>c5:ID3, 0.86<br>c15:ID23, 0.71<br>c10:ID148, 0.69<br>c10:ID206, 0.74<br>c10:ID243, 0.61<br>c10:ID255, 0.34<br>c5:ID257, 0.49<br>c15:ID278, 0.52<br>c10:ID299, 0.48<br>...(+1) | 0.981 | 0.041 |
| 291 | c15:ID1, 0.63<br>c5:ID3, 0.86<br>c15:ID24, 0.69<br>c10:ID214, 0.70<br>c10:ID242, 0.70<br>c10:ID265, 0.59<br>c5:ID290, 0.30<br>c10:ID308, 0.49<br>c10:ID309, 0.38<br>c3:ID316, 0.71 | c5:ID1, 0.30<br>c5:ID3, 0.86<br>c15:ID23, 0.69<br>c10:ID148, 0.70<br>c10:ID206, 0.70<br>c10:ID255, 0.59<br>c5:ID257, 0.26<br>c15:ID278, 0.63<br>c10:ID299, 0.49<br>c10:ID300, 0.38<br>...(+1) | 0.997 | 0.037 |

## Analysis Paragraph

在所选视频的第 288-291 帧中，车辆框出现连续重叠，该窗口内 ByteTrack 标注框的最高 IoU 为 0.997，平均成对 IoU 为 0.039。遮挡发生时，可见区域缩小会使检测框位置和尺度波动，当前帧检测框与上一帧轨迹预测框的 IoU 下降；如果低置信度框没有被保留下来，或匹配距离超过阈值，ByteTrack 就可能把短暂被挡住的车辆标成 lost，甚至在重新出现时分配新 ID。

ByteTrack 的优势是会利用高分框和低分框两阶段关联，因而在短时遮挡中仍有机会把同一车辆接回原 ID；`track_buffer` 决定 lost 轨迹在多少帧内仍可等待重连，适当增大它可以容忍更长的遮挡，但也会提高旧轨迹误匹配到相邻车辆的风险。`match_thresh` 需要和检测置信度一起调节：过严会在 IoU 快速下降时断 ID，过松则可能在密集交汇时交换 ID。

BoT-SORT 在运动/IoU 关联之外还支持相机运动补偿和可选的 ReID 外观特征。当遮挡导致 bbox IoU 信息不稳定时，启用 ReID 可利用车辆外观相似度辅助重新关联，所以在这一类遮挡片段中通常比只依赖运动和 IoU 的方案更有机会保持 ID 连续。最终报告中应结合上表逐帧 ID：若 ByteTrack 出现 ID 跳变而 BoT-SORT 保持一致，可将原因归结为遮挡造成的 IoU 下降和外观信息缺失；若二者都稳定，则说明当前遮挡持续时间短，检测框仍足以支持轨迹关联。
