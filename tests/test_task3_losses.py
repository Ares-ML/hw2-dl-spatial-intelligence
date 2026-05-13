from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - local machines may not have torch.
    torch = None

if torch is not None:
    from torch import nn

    from src.task3_seg.losses import CEDiceLoss, DiceLoss, build_segmentation_loss


@unittest.skipIf(torch is None, "PyTorch is not installed")
class Task3LossesTest(unittest.TestCase):
    def test_dice_loss_is_near_zero_for_perfect_predictions(self) -> None:
        target = torch.tensor([[[0, 1], [2, 255]]])
        logits = torch.full((1, 3, 2, 2), -20.0)
        logits[0, 0, 0, 0] = 20.0
        logits[0, 1, 0, 1] = 20.0
        logits[0, 2, 1, 0] = 20.0

        loss = DiceLoss(num_classes=3, ignore_index=255)(logits, target)

        self.assertLess(float(loss.item()), 1e-4)

    def test_ignore_index_pixels_do_not_affect_dice(self) -> None:
        full_target = torch.tensor([[[0, 255]]])
        full_logits = torch.tensor([[[[8.0, -8.0]], [[-8.0, 8.0]]]])
        cropped_target = full_target[:, :, :1]
        cropped_logits = full_logits[:, :, :, :1]
        loss_fn = DiceLoss(num_classes=2, ignore_index=255)

        full_loss = loss_fn(full_logits, full_target)
        cropped_loss = loss_fn(cropped_logits, cropped_target)

        self.assertAlmostEqual(float(full_loss.item()), float(cropped_loss.item()), places=6)

    def test_ce_dice_loss_matches_ce_plus_weighted_dice(self) -> None:
        target = torch.tensor([[[0, 1], [1, 255]]])
        logits = torch.tensor(
            [
                [
                    [[2.0, -1.0], [-0.5, 0.0]],
                    [[-1.0, 2.0], [0.5, 0.0]],
                ]
            ]
        )
        dice_weight = 0.7
        ce = nn.CrossEntropyLoss(ignore_index=255)(logits, target)
        dice = DiceLoss(num_classes=2, ignore_index=255)(logits, target)
        combined = CEDiceLoss(num_classes=2, ignore_index=255, dice_weight=dice_weight)(logits, target)

        self.assertAlmostEqual(float(combined.item()), float((ce + dice_weight * dice).item()), places=6)

    def test_loss_builder_returns_expected_loss_types(self) -> None:
        self.assertIsInstance(build_segmentation_loss("ce", num_classes=2), nn.CrossEntropyLoss)
        self.assertIsInstance(build_segmentation_loss("dice", num_classes=2), DiceLoss)
        self.assertIsInstance(build_segmentation_loss("ce_dice", num_classes=2), CEDiceLoss)


if __name__ == "__main__":
    unittest.main()
