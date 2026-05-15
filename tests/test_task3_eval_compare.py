from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.task3_seg.colorize import colorize_mask
from src.task3_seg.eval_compare import parse_epoch_log
from src.task3_seg.visualize_masks import parse_bucket_counts, pick_buckets


REPO_ROOT = Path(__file__).resolve().parents[1]


class ParseEpochLogTest(unittest.TestCase):
    def test_extracts_metrics_from_single_line(self) -> None:
        line = (
            "2026-05-14 16:11:05 | INFO | hw2.task3 | epoch=1 metrics={'train_loss': 1.2, "
            "'val_loss': 0.9, 'val_miou': 0.5, 'val_pixel_acc': 0.6}\n"
        )
        tmp = REPO_ROOT / "tests" / "_tmp_one_line.log"
        tmp.write_text(line, encoding="utf-8")
        try:
            rows = parse_epoch_log(tmp)
        finally:
            tmp.unlink()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["epoch"], 1)
        self.assertAlmostEqual(rows[0]["train_loss"], 1.2)
        self.assertAlmostEqual(rows[0]["val_miou"], 0.5)

    def test_skips_unrelated_lines(self) -> None:
        content = (
            "2026-05-14 | INFO | hw2.task3 | checkpoint_dir=...\n"
            "2026-05-14 | INFO | hw2.task3 | epoch=2 metrics={'train_loss': 0.5, 'val_miou': 0.7}\n"
            "garbage line with no epoch info\n"
            "2026-05-14 | INFO | hw2.task3 | epoch=2 checkpoint last=... best=None is_best=False\n"
        )
        tmp = REPO_ROOT / "tests" / "_tmp_mixed.log"
        tmp.write_text(content, encoding="utf-8")
        try:
            rows = parse_epoch_log(tmp)
        finally:
            tmp.unlink()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["epoch"], 2)
        self.assertEqual(rows[0]["val_miou"], 0.7)

    def test_handles_real_log_file_smoke(self) -> None:
        log_path = REPO_ROOT / "logs" / "swanlab" / "task3_unet_ce_e100.log"
        if not log_path.is_file():
            self.skipTest("real training log not available in this environment")
        rows = parse_epoch_log(log_path)
        self.assertGreaterEqual(len(rows), 80, "expected at least 80 epoch rows")
        self.assertIn("train_loss", rows[0])
        self.assertIn("val_miou", rows[0])
        self.assertEqual(rows[0]["epoch"], 1)
        epochs = [r["epoch"] for r in rows]
        self.assertEqual(epochs, sorted(epochs), "epochs must be ascending")


class ParseBucketCountsTest(unittest.TestCase):
    def test_default_parses_to_six(self) -> None:
        counts = parse_bucket_counts("best=1,median=2,hard=2,worst=1")
        self.assertEqual(counts, {"best": 1, "median": 2, "hard": 2, "worst": 1})

    def test_missing_buckets_default_to_zero(self) -> None:
        counts = parse_bucket_counts("best=1,worst=1")
        self.assertEqual(counts, {"best": 1, "median": 0, "hard": 0, "worst": 1})

    def test_rejects_unknown_bucket(self) -> None:
        with self.assertRaises(ValueError):
            parse_bucket_counts("super=1")

    def test_rejects_no_samples(self) -> None:
        with self.assertRaises(ValueError):
            parse_bucket_counts("best=0,median=0,hard=0,worst=0")


class PickBucketsTest(unittest.TestCase):
    def _scored(self, n: int) -> list[tuple[str, float]]:
        # ascending scores from 0.0..1.0
        return [(f"s{i:03d}", i / (n - 1)) for i in range(n)]

    def test_returns_correct_count(self) -> None:
        scored = self._scored(30)
        chosen = pick_buckets(scored, {"best": 1, "median": 2, "hard": 2, "worst": 1})
        self.assertEqual(len(chosen), 6)
        self.assertEqual(len({c["sample_id"] for c in chosen}), 6)

    def test_orders_best_first_worst_last(self) -> None:
        scored = self._scored(30)
        chosen = pick_buckets(scored, {"best": 1, "median": 2, "hard": 2, "worst": 1})
        buckets = [c["bucket"] for c in chosen]
        self.assertEqual(buckets[0], "best")
        self.assertEqual(buckets[-1], "worst")
        # best score must be the global max
        self.assertEqual(chosen[0]["sample_id"], scored[-1][0])
        # worst score must be the global min
        self.assertEqual(chosen[-1]["sample_id"], scored[0][0])

    def test_no_duplicate_ids(self) -> None:
        scored = self._scored(6)
        chosen = pick_buckets(scored, {"best": 1, "median": 2, "hard": 2, "worst": 1})
        ids = [c["sample_id"] for c in chosen]
        self.assertEqual(len(set(ids)), len(ids), "expected distinct sample ids even on tiny datasets")

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(pick_buckets([], {"best": 1}), [])

    def test_rejects_negative_counts(self) -> None:
        scored = self._scored(10)
        with self.assertRaises(ValueError):
            pick_buckets(scored, {"best": -1})


class ColorizeMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.palette = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]

    def test_shape_and_dtype(self) -> None:
        mask = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
        out = colorize_mask(mask, self.palette)
        self.assertEqual(out.shape, (2, 3, 3))
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(out[1, 1], [40, 50, 60])

    def test_ignore_pixels_use_ignore_color(self) -> None:
        mask = np.array([[0, 255], [255, 1]], dtype=np.int64)
        out = colorize_mask(mask, self.palette, ignore_index=255, ignore_color=(255, 255, 255))
        np.testing.assert_array_equal(out[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(out[0, 1], [255, 255, 255])
        np.testing.assert_array_equal(out[1, 0], [255, 255, 255])
        np.testing.assert_array_equal(out[1, 1], [40, 50, 60])

    def test_out_of_range_classes_fall_back_to_ignore_color(self) -> None:
        mask = np.array([[0, 9]], dtype=np.int64)
        out = colorize_mask(mask, self.palette, ignore_color=(1, 2, 3))
        np.testing.assert_array_equal(out[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(out[0, 1], [1, 2, 3])

    def test_rejects_3d_mask(self) -> None:
        mask = np.zeros((1, 2, 3), dtype=np.int64)
        with self.assertRaises(ValueError):
            colorize_mask(mask, self.palette)


if __name__ == "__main__":
    unittest.main()
