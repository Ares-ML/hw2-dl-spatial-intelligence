# Task 3: Stanford Background Semantic Segmentation

## Experimental Setup

Task 3 trains a from-scratch U-Net on the Stanford Background dataset and compares three pixel-wise loss functions: standard Cross-Entropy (CE), soft Dice, and a CE + Dice composite with `λ=1.0`. The dataset contains 715 images annotated with 8 semantic classes (`sky / tree / road / grass / water / building / mountain / foreground`). Images and masks are resized to 256 × 256 (bilinear for images, nearest for masks); the ignore label is 255. The 80/20 train/val split is fixed by `seed=42` in `data/stanford_background/splits/{train,val}.txt` (572 train / 143 val). All three loss runs use the identical model, optimizer (`AdamW lr=1e-3`), batch size and 100-epoch schedule — only the loss function changes — so the comparison isolates the loss effect. Per-epoch metrics are logged via SwanLab (see `logs/swanlab/swanlab_links.md` for the three run links: `unet_random_{ce,dice,ce_dice}_lr1e-3_e100_s42`).

## U-Net Architecture

The hand-written U-Net (`src/task3_seg/unet.py`) uses `base_channels=32` with channels `[32, 64, 128, 256, 512]`. The encoder is `DoubleConv → Down → DoubleConv → Down → …` and the decoder mirrors it with `Up → DoubleConv` and skip connections. No pretrained weights are loaded. With strided pooling the input must be divisible by 16; 256 × 256 satisfies this and is small enough to fit a batch comfortably on a single RTX 4090. Forward output is `[B, 8, 256, 256]` logits.

## Loss Functions

Implemented in `src/task3_seg/losses.py` and built via `build_segmentation_loss(name, num_classes, ignore_index, dice_weight, smooth)`:

- **CE** — `nn.CrossEntropyLoss(ignore_index=255)`. Standard per-pixel cross-entropy.
- **Dice** — soft Dice with `smooth=1.0`; an ignore-pixel mask is multiplied into both numerator and denominator so that pad / unlabeled pixels do not contribute.
- **CE + Dice** — `CE + dice_weight × Dice` with `dice_weight=1.0`. Same ignore-pixel handling as the Dice path.

## Quantitative Comparison (Table 3)

The three `best.pt` checkpoints are re-evaluated on the validation split by `src/task3_seg/eval_compare.py`, which accumulates per-class intersection / union over the whole validation set and then averages IoU across classes — this is the global mIoU rather than a per-batch mean. Source: `report/tables/task3_loss_comparison.md`.

| Loss | Best Epoch | Best Val mIoU (training) | Eval mIoU (val) | Eval Pixel Acc |
| --- | ---: | ---: | ---: | ---: |
| CE | 98 | 0.6253 | 0.6253 | 0.8192 |
| Dice | 37 | 0.5779 | 0.5779 | 0.8066 |
| CE+Dice | 95 | **0.6320** | **0.6320** | **0.8228** |

**CE+Dice wins by a small but consistent margin**: +0.67 pp over CE and +5.41 pp over Dice on mIoU; pixel accuracy moves in the same direction. The in-training and re-evaluated mIoU numbers match to the last decimal, which confirms the validation pipeline is deterministic and the published table is reproducible from disk.

### Class-wise IoU

Source: `report/tables/task3_class_iou.md`.

| Class | CE | Dice | CE+Dice |
| --- | ---: | ---: | ---: |
| sky | 0.8764 | 0.8768 | 0.8819 |
| tree | 0.6485 | 0.6576 | 0.6363 |
| road | 0.7783 | 0.7629 | 0.7973 |
| grass | 0.6475 | 0.7109 | 0.6973 |
| water | 0.6099 | 0.4505 | 0.6053 |
| building | 0.7040 | 0.6603 | 0.7033 |
| mountain | 0.1887 | **0.0000** | 0.1591 |
| foreground | 0.5488 | 0.5042 | 0.5758 |
| **mIoU** | **0.6253** | **0.5779** | **0.6320** |

Two patterns stand out:

1. **Dice alone collapses on the rarest class.** `mountain` has so few pixels in the validation split that the global Dice numerator never grows large enough to keep gradient pressure on it, and the resulting IoU is exactly 0.0. CE forces non-zero recall via its per-pixel gradient even on rare classes, and CE+Dice recovers most of that recall (0.1591).
2. **Dice still wins on `grass` (0.7109)** — the only class where Dice leads both other losses — because grass tends to occupy large connected regions where a soft-overlap loss captures region shape better than CE's per-pixel gradient.

## Training Curves (Fig 5)

`report/figures/task3/fig5_loss_curves.png` is a 2 × 2 panel covering the full 100-epoch schedule for all three losses:

- **Train Loss** — CE drops cleanly and smoothly. Dice settles into a low oscillating plateau by ~epoch 30. CE+Dice tracks CE in shape but at a larger absolute value (sum of two loss components).
- **Val Loss** — CE and CE+Dice trend downward stably. Dice exhibits a sharp spike around epoch 60 — a symptom of the rare-class collapse — but recovers.
- **Val mIoU** — CE+Dice gradually overtakes CE around epoch 90. Dice peaks early at epoch 37 (0.5779) and oscillates afterwards without revisiting that peak.
- **Val Pixel Acc** — CE and CE+Dice both climb past 0.82; Dice remains ~0.80.

Takeaway: **Dice converges fastest but plateaus low**; CE is the most stable; CE+Dice combines the two and adds a small mIoU bonus at the cost of needing the full 100 epochs to fully overtake plain CE.

## Qualitative Mask Comparison (Fig 6)

`report/figures/task3/mask_comparison.png` shows 6 validation samples in a 6 × 5 grid. Columns are `Image / GT / pred-CE / pred-Dice / pred-CE+Dice`. Samples are chosen by `src/task3_seg/visualize_masks.py` using the CE model's per-image mIoU, then split into buckets `best=1, median=2, hard=2, worst=1`:

| Bucket | Sample ID | CE mIoU |
| --- | --- | ---: |
| best | `9001034` | 0.917 |
| median | `0000382` | 0.484 |
| median | `0007323` | 0.484 |
| hard | `9001071` | 0.367 |
| hard | `6000186` | 0.365 |
| worst | `3000716` | 0.213 |

Visual observations: in the easy row (`9001034`, a double-decker bus on a street) all three models agree on `road / sky / foreground` with sharp boundaries. In the median rows the building-sky boundary becomes shaky for Dice and is sometimes replaced by `foreground` or `tree`. In the hard rows the small foreground objects are blurred or merged into building / road. In the worst row (`3000716`, a low-contrast cow on grass) CE and CE+Dice still make a reasonable "foreground over grass" prediction while Dice loses the foreground object entirely. The qualitative ranking matches the quantitative one: **CE+Dice ≥ CE > Dice**.

Per-sample per-variant mIoU is logged in `report/figures/task3/mask_comparison_samples.json`.

## Conclusions and Limitations

The Stanford Background segmentation experiments yield two takeaways. First, **the composite CE+Dice loss is the best choice** under this budget — it inherits CE's per-pixel gradient that keeps rare classes alive and gains a small region-aware bonus from Dice, for a final mIoU of **0.6320**. Second, **using Dice alone is risky on a long-tailed dataset**: the rare `mountain` class collapses to IoU 0.0, and the global mIoU drops ~5.4 pp below CE despite Dice's strong performance on big-region classes like `grass`.

The main limitations are:

- **Dataset size** (715 total, 143 val). Class statistics are unstable; rare-class IoU swings are partly seed noise.
- **Fixed 256 × 256 input.** Fine boundary detail is lost, especially for thin classes such as foreground objects.
- **No pretraining.** The U-Net learns features from scratch; with a pretrained encoder (e.g. a ResNet-34 such as the one trained in Task 1's bonus run) the mIoU floor would likely rise several points.

## Code Index

| Module | File |
| --- | --- |
| U-Net architecture | `src/task3_seg/unet.py` |
| Stanford dataset loader | `src/task3_seg/datasets.py` |
| Image / mask transforms | `src/task3_seg/transforms.py` |
| Loss functions | `src/task3_seg/losses.py` |
| Training entry | `src/task3_seg/train.py` |
| Eval + Table 3 + Fig 5 generator | `src/task3_seg/eval_compare.py` |
| Mask grid (Fig 6) generator | `src/task3_seg/visualize_masks.py` |
| Mask colorization helper | `src/task3_seg/colorize.py` |
| Metrics (mIoU / confusion) | `src/common/metrics.py` |
