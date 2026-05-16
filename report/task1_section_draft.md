# Task 1: Flower102 Classification Draft

## Experimental Setup

Task 1 studies image classification on Flower102. The main backbone is ResNet-18 with the final fully connected layer replaced by a 102-class classifier. Images are resized and center-cropped to 224 x 224, normalized with ImageNet statistics, and trained with cross-entropy loss. All reported runs use seed 42 and log train loss, train accuracy, validation loss, and validation accuracy to SwanLab.

The experiments cover three required comparisons: pretrained ResNet-18, random-initialized ResNet-18, and an attention-enhanced ResNet-18 with CBAM. A separate pretrained hyperparameter grid tests backbone learning rate, head learning rate, and training length.

## Pretraining Ablation

The pretrained ResNet-18 baseline reaches a final validation accuracy of **0.904994** after 50 epochs. The random-initialized ResNet-18 reaches **0.458566** validation accuracy under the same 50-epoch budget, despite both models reaching near-perfect training accuracy. This gap shows that ImageNet initialization is crucial for Flower102 when the training set is limited: pretraining improves generalization, while the random model mainly memorizes the training split.

Relevant figures:

- `report/figures/task1/t1_a/T1_A_train_loss.png`
- `report/figures/task1/t1_a/T1_A_val_acc.png`
- `report/figures/task1/t1_c/T1_C_train_loss.png`
- `report/figures/task1/t1_c/T1_C_val_acc.png`

## Attention Module Comparison

The CBAM run uses a pretrained ResNet-18 backbone and inserts channel-spatial attention before global average pooling. Its best validation accuracy is **0.903041** at epoch 49, with final validation accuracy **0.900949** at epoch 50. This is close to the pretrained baseline but does not improve over it in the current setting. The result suggests that CBAM is compatible with the backbone, but the extra attention capacity does not clearly help Flower102 under this training schedule.

## Hyperparameter Grid

The hyperparameter grid tests:

- `lr_backbone` in `{1e-5, 1e-4}`
- `lr_head` in `{1e-3, 5e-4}`
- `epochs` in `{50, 100}`

The best run is **`bb1e-4_head5e-4_e100`**, with `lr_backbone=1e-4`, `lr_head=5e-4`, and 100 epochs. It reaches the best validation accuracy of **0.913783** at epoch 99. The second-best run, `bb1e-4_head1e-3_e100`, reaches **0.911830**, also favoring the larger backbone learning rate and a longer schedule.

The grid results indicate that allowing the pretrained backbone to adapt more actively (`1e-4` rather than `1e-5`) is beneficial. Extending training from 50 to 100 epochs also helps the best configurations, although the final validation accuracy can fluctuate slightly after the best epoch.

Full grid table: `report/tables/task1_grid_summary.md`. The final retraining configuration is fixed in `configs/task1_resnet18_final.yaml`, and the final model should be saved to `checkpoints/task1/final_resnet18_pretrained_bb1e-4_head5e-4_e100/best.pt`.

## Bonus: ResNet-34 vs ResNet-18 Backbone

A ResNet-34 (~21.4 M parameters, 1.83 × the ResNet-18 footprint) is trained under exactly the same recipe as the best ResNet-18 run (`bb=1e-4, head=5e-4`, 100 epochs, seed 42, `batch=32`, no augmentation change; CBAM disabled for both to isolate the backbone effect). The deeper backbone reaches a best validation accuracy of **0.9156** at **epoch 87**, edging the ResNet-18 best (`0.9138` at epoch 99) by **+0.18 pp** while peaking about 12 epochs earlier. The final-epoch validation accuracy is **0.9039**, slightly below the best — a post-peak drift of similar shape to ResNet-18's. Training accuracy saturates at 1.0 with a final training loss of 2.9e-4, so both backbones overfit Flower102 quickly and the validation set is the only meaningful selection signal. Given how small the gap is, the result is consistent with ResNet-34 being a minor improvement that could plausibly sit inside the seed-to-seed noise band rather than a decisive win for the deeper backbone.

Full numbers in `report/tables/task1_resnet34_vs_resnet18.md`; config in `configs/task1_resnet34_final.yaml`; model in `checkpoints/task1/final_resnet34_pretrained_bb1e-4_head5e-4_e100/best.pt`.

## Conclusions And Limitations

Pretraining is the dominant factor in Task 1: it raises validation accuracy from **45.86%** to **90.50%** under the same 50-epoch setup. CBAM remains competitive but does not outperform the plain pretrained baseline. The best hyperparameter grid setting further improves validation accuracy to **91.38%**, so it should be used as the final Task 1 model setting. The ResNet-34 bonus run further confirms that on Flower102, going from 11.7 M to 21.4 M parameters yields at most a ~0.2 pp improvement — the training schedule, learning-rate split, and augmentation matter more than the backbone depth at this scale.

The main limitation is that all models reach very high training accuracy, so the validation set becomes the primary signal for model selection. Future improvements should focus on stronger augmentation, regularization, or a larger pretrained backbone rather than simply adding epochs.
