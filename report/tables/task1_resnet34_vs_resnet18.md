# Task 1 — ResNet-18 vs ResNet-34 (Pretrained, 100 ep, bb=1e-4 / head=5e-4)

Bonus comparison for Task 1. Both runs share the same Flower102 split, optimizer (`AdamW`), differential learning rates (`lr_backbone=1e-4, lr_head=5e-4`), 100 epochs, seed 42, `image_size=224`, `batch=32`. The only change is the backbone family (`resnet18` vs `resnet34`); CBAM is **not** enabled for either run so the comparison stays clean.

| Backbone | Params (M) | Best Val Acc | Best Epoch | Final Val Acc | Final Train Acc | Final Train Loss | SwanLab |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ResNet-18 (final) | 11.69 | 0.9138 | 99 | 0.9050 | 1.0000 | — | [`bb1e-4_head5e-4_e100`](https://swanlab.cn/@Ares_ML/hw2-dl-spatial-intelligence/runs/1vwvo8v15fwhvdbjo3jgl) |
| ResNet-34 (bonus) | **21.35** | **0.9156** | **87** | 0.9039 | 1.0000 | 2.9e-4 | `final_resnet34_pretrained_bb1e-4_head5e-4_e100` (链接见 `logs/swanlab/swanlab_links.md`) |

> Params 列为 state-dict 全量参数+缓冲（含 BN running mean/var），与教科书纯权重参数量略有差异；ResNet-18=11.69 M, ResNet-34=21.35 M。

## Observed discussion

ResNet-34 doubles backbone depth (BasicBlocks 3-4-6-3 vs 2-2-2-2) and 1.83 × the parameter footprint. With identical Flower102 training recipe it reaches **Best Val Acc = 0.9156 at epoch 87**, edging the ResNet-18 best (`0.9138 at epoch 99`) by **+0.18 percentage points** while peaking ~12 epochs earlier. The final-epoch validation accuracy (`0.9039`) is slightly below the best — the same post-peak drift shape we see with ResNet-18.

Both backbones saturate `train_acc=1.0` quickly (`train_loss=2.9e-4` for ResNet-34) so the training set provides no further signal — the validation set is the only meaningful selection criterion. The +0.18 pp gap is small enough that it could plausibly sit inside the seed-to-seed noise band of a 143-image-per-class Flower102 validation set.

**Takeaway**: at the Flower102 budget, going from 11.7 M to 21.4 M parameters yields at most a marginal accuracy improvement; training schedule, learning-rate split and augmentation matter more than backbone depth at this scale.
