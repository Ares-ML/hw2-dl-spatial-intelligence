# Task 3 — Three-Loss Comparison Summary

Validation re-evaluation done with `python -m src.task3_seg.eval_compare`, after loading `checkpoints/task3/unet_{ce,dice,ce_dice}/best.pt`. Eval mIoU is computed by accumulating per-class intersection/union over the full validation split (143 images), then averaging IoU across classes — this is the global (not per-batch-mean) mIoU.

| Loss | Best Epoch | Best Val mIoU (training) | Eval mIoU (val) | Eval Pixel Acc | Final Train Loss | Final Val Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 98 | 0.6253 | 0.6253 | 0.8192 | 0.2449 | 0.6512 |
| Dice | 37 | 0.5779 | 0.5779 | 0.8066 | 0.3098 | 0.3251 |
| CE+Dice | 95 | 0.6320 | 0.6320 | 0.8228 | 0.5713 | 1.0683 |
