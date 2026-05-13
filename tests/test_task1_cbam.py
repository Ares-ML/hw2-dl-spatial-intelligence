from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - local machines may not have torch.
    torch = None

if torch is not None:
    from src.task1_cls.attention import CBAM
    from src.task1_cls.models import build_resnet18_classifier


@unittest.skipIf(torch is None, "PyTorch is not installed")
class Task1CBAMTest(unittest.TestCase):
    def test_cbam_preserves_feature_shape(self) -> None:
        module = CBAM(channels=32)
        x = torch.randn(2, 32, 14, 14)

        y = module(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_cbam_handles_channels_smaller_than_reduction(self) -> None:
        module = CBAM(channels=4, reduction=16)
        x = torch.randn(2, 4, 8, 8)

        y = module(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_resnet18_cbam_builder_outputs_classifier_logits(self) -> None:
        model = build_resnet18_classifier(num_classes=102, init="random", attention="cbam")
        model.eval()
        x = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(tuple(logits.shape), (2, 102))


if __name__ == "__main__":
    unittest.main()
