# Task 1 — Backbone Bonus Comparison (Pretrained, 100 ep, bb=1e-4 / head=5e-4)

Bonus comparison for Task 1 across three backbones on Flower102 under the **same** recipe (`AdamW`, `lr_backbone=1e-4, lr_head=5e-4`, seed 42, `image_size=224`, `batch=32`, no MixUp / no RandAugment). The ViT-Tiny run uses 50 epochs (it saturates earlier), the two ResNet runs use 100 epochs. CBAM is **not** enabled for any of these so the comparison isolates the backbone family.

| Backbone | Params (M) | Best Val Acc | Best Epoch | Final Val Acc | Final Train Acc | Final Train Loss | SwanLab |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ResNet-18 (final, 100 ep) | 11.69 | 0.9138 | 99 | 0.9050 | 1.0000 | — | [`bb1e-4_head5e-4_e100`](https://swanlab.cn/@Ares_ML/hw2-dl-spatial-intelligence/runs/1vwvo8v15fwhvdbjo3jgl) |
| ResNet-34 (bonus, 100 ep) | **21.35** | **0.9156** | **87** | 0.9039 | 1.0000 | 2.9e-4 | `final_resnet34_pretrained_bb1e-4_head5e-4_e100` (链接见 `logs/swanlab/swanlab_links.md`) |
| ViT-Tiny (bonus, 50 ep) | 5.54 | 0.8397 | 33 | — | 1.0000 | 2.7e-3 | `bonus_vit_tiny_pretrained_bb1e-4_head5e-4_e50` (链接见 `logs/swanlab/swanlab_links.md`) |

> Params 列为 state-dict 全量参数+缓冲（含 BN running stats / position embedding 等）。

## Observed discussion

1. **ResNet-34 vs ResNet-18 (+0.18 pp)**：1.83× 的参数膨胀只换来 0.18 pp 的边际收益，best epoch 提前 12 个（87 vs 99）；考虑 Flower102 只有 ~1 k 训练样本，这个差距很可能仍在 seed 噪声内。在同 budget 下加深 CNN 不是显著加分。
2. **ViT-Tiny vs ResNet-18 (−7.41 pp)**：参数量只有 ResNet-18 的 ~47 %，best Val Acc 仅 0.8397，**显著拉跨**。三个原因叠加：(a) Flower102 仅 ~1 k 训练样本，远低于 ViT 在无强增强下追上 CNN 所需的数据规模；(b) 我们沿用 ResNet 风格 transform（Resize / CenterCrop / ImageNet norm），没用 ViT 训练常用的 MixUp / RandAugment / RandomErasing；(c) `train_acc` 在 epoch 33 就触顶 1.0，val 之后不再涨——典型的小数据 + 弱归纳偏置组合下的过拟合现象。
3. **Takeaway**：小数据集 fine-tune 场景下，ImageNet pretrained CNN 仍领先同等参数预算的 ImageNet pretrained Transformer；选 backbone 并非"越新越好"。架构选择应当与数据规模和 augmentation 配方共同决策。
