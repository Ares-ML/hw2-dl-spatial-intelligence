from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - local machines may not have torch.
    torch = None

if torch is not None:
    from src.task3_seg.unet import DoubleConv, Down, OutConv, UNet, Up


@unittest.skipIf(torch is None, "PyTorch is not installed")
class Task3UNetTest(unittest.TestCase):
    def test_unet_output_shape_matches_input_spatial_size(self) -> None:
        model = UNet(num_classes=8, base_channels=4)
        model.eval()
        x = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(tuple(logits.shape), (2, 8, 64, 64))

    def test_unet_has_required_building_blocks(self) -> None:
        model = UNet(num_classes=8, base_channels=4)

        self.assertIsInstance(model.inc, DoubleConv)
        self.assertIsInstance(model.down1, Down)
        self.assertIsInstance(model.down4, Down)
        self.assertIsInstance(model.up1, Up)
        self.assertIsInstance(model.up4, Up)
        self.assertIsInstance(model.outc, OutConv)

    def test_up_aligns_skip_connection_spatial_shape(self) -> None:
        up = Up(in_channels=16, skip_channels=8, out_channels=8)
        up.eval()
        x = torch.randn(1, 16, 4, 5)
        skip = torch.randn(1, 8, 9, 11)

        with torch.no_grad():
            y = up(x, skip)

        self.assertEqual(tuple(y.shape), (1, 8, 9, 11))

    def test_unet_handles_odd_input_spatial_size(self) -> None:
        model = UNet(num_classes=8, base_channels=4)
        model.eval()
        x = torch.randn(1, 3, 65, 67)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(tuple(logits.shape), (1, 8, 65, 67))


if __name__ == "__main__":
    unittest.main()
