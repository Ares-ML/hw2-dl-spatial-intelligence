from __future__ import annotations

import importlib.util
import unittest

import torch

from src.task1_cls.models import build_classifier, build_vit_tiny_classifier


_HAS_TIMM = importlib.util.find_spec("timm") is not None


class FactoryRoutingTest(unittest.TestCase):
    def test_factory_accepts_vit_tiny_name(self) -> None:
        if not _HAS_TIMM:
            self.skipTest("timm not installed in this environment")
        model = build_classifier("vit_tiny", num_classes=102, init="random")
        out = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 102))

    def test_factory_unknown_name_raises_includes_vit_tiny_in_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_classifier("vit_huge", num_classes=102)
        self.assertIn("vit_tiny", str(ctx.exception))


class ViTTinyBuilderTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_TIMM, "timm not installed in this environment")
    def test_returns_correct_shape_and_param_count(self) -> None:
        model = build_vit_tiny_classifier(num_classes=102, init="random")
        out = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 102))
        params = sum(p.numel() for p in model.parameters())
        # ViT-Tiny is ~5.5—5.8 M params with a 102-class head.
        self.assertLess(params, 7_000_000)
        self.assertGreater(params, 4_500_000)

    @unittest.skipUnless(_HAS_TIMM, "timm not installed in this environment")
    def test_ignores_extra_attention_kwargs(self) -> None:
        # The factory passes attention / cbam_* through; ViT must swallow them.
        model = build_vit_tiny_classifier(
            num_classes=102, init="random",
            attention="cbam", cbam_reduction=16, cbam_spatial_kernel_size=7,
        )
        out = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (1, 102))

    def test_missing_timm_emits_actionable_runtime_error(self) -> None:
        if _HAS_TIMM:
            self.skipTest("timm is installed; cannot exercise the missing-timm path")
        with self.assertRaises(RuntimeError) as ctx:
            build_vit_tiny_classifier(num_classes=102, init="random")
        self.assertIn("timm", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
