# Task 1 Hyperparameter Grid Summary

Experiment: ResNet-18 pretrained, Flower102, backbone/head learning-rate grid.

| Rank | Run ID | Epochs | LR Backbone | LR Head | Final Val Acc | Best Epoch | Best Val Acc |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `bb1e-4_head5e-4_e100` | 100 | 1e-4 | 5e-4 | 0.904994 | 99 | 0.913783 |
| 2 | `bb1e-4_head1e-3_e100` | 100 | 1e-4 | 1e-3 | 0.903181 | 68 | 0.911830 |
| 3 | `bb1e-4_head5e-4_e50` | 50 | 1e-4 | 5e-4 | 0.908901 | 49 | 0.909040 |
| 4 | `bb1e-5_head1e-3_e100` | 100 | 1e-5 | 1e-3 | 0.900251 | 57 | 0.909040 |
| 5 | `bb1e-5_head5e-4_e100` | 100 | 1e-5 | 5e-4 | 0.905134 | 51 | 0.909040 |
| 6 | `bb1e-4_head1e-3_e50` | 50 | 1e-4 | 1e-3 | 0.906948 | 46 | 0.908064 |
| 7 | `bb1e-5_head1e-3_e50` | 50 | 1e-5 | 1e-3 | 0.903181 | 34 | 0.908064 |
| 8 | `bb1e-5_head5e-4_e50` | 50 | 1e-5 | 5e-4 | 0.899275 | 49 | 0.906110 |

Best setting: `lr_backbone=1e-4`, `lr_head=5e-4`, `epochs=100`.

Source logs: `logs/swanlab/task1_grid_*.log`.

