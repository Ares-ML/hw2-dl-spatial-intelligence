from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from src.task2_detect_track.render_counts import (
    bbox_bottom_center,
    parse_class_filter,
    parse_line_arg,
)


_HAS_CV2 = importlib.util.find_spec("cv2") is not None


class ParseLineArgTest(unittest.TestCase):
    def test_parses_four_numeric_values(self) -> None:
        self.assertEqual(parse_line_arg("200,300,1100,300"), (200.0, 300.0, 1100.0, 300.0))

    def test_tolerates_whitespace(self) -> None:
        self.assertEqual(parse_line_arg(" 1, 2 , 3 ,4 "), (1.0, 2.0, 3.0, 4.0))

    def test_rejects_three_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_line_arg("1,2,3")

    def test_rejects_zero_length_line(self) -> None:
        with self.assertRaises(ValueError):
            parse_line_arg("5,5,5,5")

    def test_rejects_non_numeric(self) -> None:
        with self.assertRaises(ValueError):
            parse_line_arg("1,2,a,4")


class ParseClassFilterTest(unittest.TestCase):
    def test_empty_string_yields_empty_set(self) -> None:
        self.assertEqual(parse_class_filter(""), set())

    def test_none_yields_empty_set(self) -> None:
        self.assertEqual(parse_class_filter(None), set())

    def test_handles_whitespace_and_dedups(self) -> None:
        self.assertEqual(parse_class_filter("3, 20 ,3"), {3, 20})

    def test_rejects_non_integer(self) -> None:
        with self.assertRaises(ValueError):
            parse_class_filter("3,foo")


class BboxBottomCenterTest(unittest.TestCase):
    def test_returns_midpoint_and_bottom(self) -> None:
        self.assertEqual(bbox_bottom_center((10.0, 20.0, 40.0, 80.0)), (25.0, 80.0))

    def test_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            bbox_bottom_center((1.0, 2.0, 3.0))


@unittest.skipUnless(_HAS_CV2, "cv2 not installed in this environment")
class DrawingSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((120, 200, 3), dtype=np.uint8)

    def test_overlay_preserves_shape_and_dtype(self) -> None:
        from src.task2_detect_track.render_counts import draw_count_overlay

        out = draw_count_overlay(
            self.frame.copy(),
            total=5,
            direction_counts={"positive_to_negative": 3, "negative_to_positive": 2},
            per_class_counts={5: 4, 18: 1},
            class_names={5: "car", 18: "truck"},
        )
        self.assertEqual(out.shape, self.frame.shape)
        self.assertEqual(out.dtype, self.frame.dtype)
        self.assertTrue(out.any(), "overlay should write some non-zero pixels")

    def test_counting_line_modifies_pixels(self) -> None:
        from src.task2_detect_track.render_counts import draw_counting_line

        out = draw_counting_line(self.frame.copy(), (10.0, 60.0, 190.0, 60.0))
        self.assertTrue(out.any(), "drawing the line should leave non-zero pixels")

    def test_count_dots_modify_pixels(self) -> None:
        from src.task2_detect_track.render_counts import draw_count_dots

        out = draw_count_dots(self.frame.copy(), [(50.0, 60.0), (120.0, 90.0)])
        self.assertTrue(out.any(), "drawing dots should leave non-zero pixels")


if __name__ == "__main__":
    unittest.main()
