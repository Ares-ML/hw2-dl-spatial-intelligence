# Task 3 — Class-wise IoU

Per-class IoU on the validation split (143 images, 8 classes). Best.pt for each loss variant loaded; argmax predictions are accumulated into a confusion matrix and IoU = TP / (TP + FP + FN) per class.

| Class | CE | Dice | CE+Dice |
| --- | ---: | ---: | ---: |
| sky | 0.8764 | 0.8768 | 0.8819 |
| tree | 0.6485 | 0.6576 | 0.6363 |
| road | 0.7783 | 0.7629 | 0.7973 |
| grass | 0.6475 | 0.7109 | 0.6973 |
| water | 0.6099 | 0.4505 | 0.6053 |
| building | 0.7040 | 0.6603 | 0.7033 |
| mountain | 0.1887 | 0.0000 | 0.1591 |
| foreground | 0.5488 | 0.5042 | 0.5758 |
| **mIoU** | **0.6253** | **0.5779** | **0.6320** |
