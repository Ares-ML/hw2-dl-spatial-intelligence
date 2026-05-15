from __future__ import annotations

import unittest

import torch

from src.task1_cls.models import (
    build_classifier,
    build_resnet18_classifier,
    build_resnet34_classifier,
)


class BuildClassifierFactoryTest(unittest.TestCase):
    def test_resnet18_dispatch_returns_resnet18_shape(self) -> None:
        model = build_classifier("resnet18", num_classes=102, init="random")
        out = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 102))

    def test_resnet34_dispatch_returns_correct_shape(self) -> None:
        model = build_classifier("resnet34", num_classes=102, init="random")
        out = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 102))

    def test_resnet34_with_cbam_keeps_shape(self) -> None:
        model = build_classifier(
            "resnet34", num_classes=102, init="random", attention="cbam"
        )
        out = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 102))

    def test_default_dispatch_is_resnet18(self) -> None:
        small = build_classifier("", num_classes=102, init="random")
        deep = build_classifier("resnet34", num_classes=102, init="random")
        small_params = sum(p.numel() for p in small.parameters())
        deep_params = sum(p.numel() for p in deep.parameters())
        # ResNet-18 ≈ 11.7M ; ResNet-34 ≈ 21.8M — deep must be meaningfully larger.
        self.assertLess(small_params, 15_000_000)
        self.assertGreater(deep_params, small_params * 1.5)

    def test_unknown_model_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            build_classifier("vit_huge", num_classes=102)


class BuildResNet34ClassifierTest(unittest.TestCase):
    def test_random_init_smaller_than_pretrained_call_path(self) -> None:
        # We do not assert pretrained mode here (would hit the network); just confirm
        # the direct builder still produces a 102-class head with the expected feature dim.
        model = build_resnet34_classifier(num_classes=102, init="random")
        self.assertEqual(model.fc.out_features, 102)
        self.assertEqual(model.fc.in_features, 512)

    def test_resnet18_builder_still_works_after_refactor(self) -> None:
        model = build_resnet18_classifier(num_classes=102, init="random")
        self.assertEqual(model.fc.out_features, 102)
        self.assertEqual(model.fc.in_features, 512)


if __name__ == "__main__":
    unittest.main()
