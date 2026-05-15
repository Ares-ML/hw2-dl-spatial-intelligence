# Task 1 — ResNet-18 vs ResNet-34 (Pretrained, 100 ep, bb=1e-4 / head=5e-4)

Bonus comparison for Task 1. Both runs share the same Flower102 split, optimizer (`AdamW`), differential learning rates (`lr_backbone=1e-4, lr_head=5e-4`), 100 epochs, seed 42, `image_size=224`, `batch=32`. The only change is the backbone family (`resnet18` vs `resnet34`); CBAM is **not** enabled for either run so the comparison stays clean.

| Backbone | Params (M) | Best Val Acc | Best Epoch | Final Val Acc | Train Time | SwanLab |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ResNet-18 (final) | 11.69 | **0.9138** | 99 | 0.9050 | _T_18_ | [`bb1e-4_head5e-4_e100`](https://swanlab.cn/@Ares_ML/hw2-dl-spatial-intelligence/runs/1vwvo8v15fwhvdbjo3jgl) |
| ResNet-34 (bonus) | 21.80 | _N_34_ | _E_34_ | _N_34_final_ | _T_34_ | _url_ |

## How to fill in

After the server-side training finishes:

1. Read `checkpoints/task1/final_resnet34_pretrained_bb1e-4_head5e-4_e100/best.pt`:
   - `epoch` → Best Epoch column
   - `best_score` → Best Val Acc column
   - `metrics['val_acc']` (from `last.pt`) → Final Val Acc column
2. Read training log header for SwanLab URL → SwanLab column
3. Approximate train time from log timestamps (first vs last epoch line).

## Expected discussion (after numbers land)

ResNet-34 doubles backbone depth (3-4-6-3 BasicBlocks vs 2-2-2-2) and nearly doubles parameter count (~21.8 M vs ~11.7 M). On Flower102 with only ~1k training images per class shared across 102 classes, the extra capacity can either help (more representational headroom for fine-grained botanical features) or hurt (overfitting under the same augmentation budget). The actual gap relative to the ResNet-18 baseline (`Best Val Acc = 0.9138`) tells us which regime we are in.

A useful sanity check is whether `train_acc` saturates earlier for ResNet-34 (overfitting symptom) and whether per-epoch val accuracy is more volatile.
