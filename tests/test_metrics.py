from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - local machines may not have torch.
    torch = None

from src.common.metrics import (
    confusion_matrix,
    latest_ultralytics_metrics,
    mean_iou,
    pixel_accuracy,
    read_ultralytics_results_csv,
    topk_accuracy,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MetricsTest(unittest.TestCase):
    def test_topk_accuracy(self) -> None:
        logits = torch.tensor(
            [
                [0.1, 0.9, 0.0],
                [0.8, 0.1, 0.2],
                [0.1, 0.2, 0.7],
            ]
        )
        target = torch.tensor([1, 2, 0])

        top1, top2 = topk_accuracy(logits, target, topk=(1, 2))

        self.assertAlmostEqual(top1, 1.0 / 3.0)
        self.assertAlmostEqual(top2, 2.0 / 3.0)

    def test_segmentation_metrics_with_ignore_index(self) -> None:
        pred = torch.tensor([[0, 1, 2], [0, 0, 1]])
        target = torch.tensor([[0, 1, 1], [0, 2, 255]])

        matrix = confusion_matrix(pred, target, num_classes=3, ignore_index=255)
        expected = torch.tensor([[2, 0, 0], [0, 1, 1], [1, 0, 0]])
        self.assertTrue(torch.equal(matrix, expected))

        self.assertAlmostEqual(pixel_accuracy(pred, target, num_classes=3), 3.0 / 5.0)

        metrics = mean_iou(pred, target, num_classes=3, ignore_index=255)
        self.assertAlmostEqual(metrics["class_iou"][0], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["class_iou"][1], 1.0 / 2.0)
        self.assertAlmostEqual(metrics["class_iou"][2], 0.0)
        self.assertAlmostEqual(metrics["miou"], (2.0 / 3.0 + 1.0 / 2.0) / 3.0)
        self.assertAlmostEqual(metrics["pixel_acc"], 3.0 / 5.0)

    def test_ultralytics_csv_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "metrics/mAP50(B)", " val/loss "])
                writer.writerow([0, "0.25", "1.5"])
                writer.writerow([1, "0.75", "0.9"])

            rows = read_ultralytics_results_csv(path)
            latest = latest_ultralytics_metrics(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(latest["epoch"], 1.0)
        self.assertEqual(latest["metrics/mAP50(B)"], 0.75)
        self.assertEqual(latest["val/loss"], 0.9)


if __name__ == "__main__":
    unittest.main()
