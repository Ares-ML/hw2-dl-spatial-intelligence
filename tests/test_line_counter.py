from __future__ import annotations

import unittest

from src.task2_detect_track.line_counter import LineCounter


class LineCounterTest(unittest.TestCase):
    def test_single_track_crossing_counts_once(self) -> None:
        counter = LineCounter((0, 0, 10, 0))

        first = counter.update(7, center=(5, -3))
        second = counter.update(7, center=(5, 3))
        third = counter.update(7, center=(5, -4))

        self.assertFalse(first.counted)
        self.assertTrue(second.counted)
        self.assertEqual(counter.total_count, 1)
        self.assertEqual(second.direction, LineCounter.NEGATIVE_TO_POSITIVE)
        self.assertFalse(third.counted)
        self.assertEqual(counter.counted_ids, {7})

    def test_near_line_jitter_with_margin_does_not_count(self) -> None:
        counter = LineCounter((0, 0, 10, 0), margin=2.0)

        counter.update(1, center=(5, -5))
        for y in (-1.0, 0.0, 1.5, -1.5):
            event = counter.update(1, center=(5, y))
            self.assertFalse(event.counted)

        self.assertEqual(counter.total_count, 0)
        self.assertEqual(counter.last_side[1], -1)

    def test_multiple_tracks_count_independently(self) -> None:
        counter = LineCounter((0, 0, 10, 0))

        counter.update(1, center=(3, -2))
        counter.update(2, center=(7, 2))
        event_one = counter.update(1, center=(3, 2))
        event_two = counter.update(2, center=(7, -2))

        self.assertTrue(event_one.counted)
        self.assertTrue(event_two.counted)
        self.assertEqual(counter.total_count, 2)
        self.assertEqual(counter.direction_counts[LineCounter.NEGATIVE_TO_POSITIVE], 1)
        self.assertEqual(counter.direction_counts[LineCounter.POSITIVE_TO_NEGATIVE], 1)

    def test_non_crossing_track_remains_zero(self) -> None:
        counter = LineCounter((0, 0, 10, 0))

        counter.update(3, center=(1, -5))
        counter.update(3, center=(3, -4))
        counter.update(3, xyxy=(4, -6, 6, -2))

        self.assertEqual(counter.total_count, 0)
        self.assertEqual(counter.counted_ids, set())

    def test_class_filter_excludes_unwanted_classes(self) -> None:
        counter = LineCounter((0, 0, 10, 0), include_classes={5})

        counter.update(10, center=(5, -4), class_id=3)
        counter.update(10, center=(5, 4), class_id=3)
        self.assertEqual(counter.total_count, 0)
        self.assertNotIn(10, counter.last_side)

        counter.update(10, center=(5, -4), class_id=5)
        event = counter.update(10, center=(5, 4), class_id=5)
        self.assertTrue(event.counted)
        self.assertEqual(counter.total_count, 1)

    def test_update_many_accepts_mapping_and_tuple_records(self) -> None:
        counter = LineCounter((0, 0, 10, 0))

        events = counter.update_many(
            [
                {"track_id": 1, "center": (5, -1), "class_id": 5},
                (1, (5, 1), 5),
                {"id": 2, "xyxy": (4, 1, 6, 3), "cls": 5},
                (2, (5, -1)),
            ]
        )

        self.assertEqual([event.counted for event in events], [False, True, False, True])
        self.assertEqual(counter.total_count, 2)


if __name__ == "__main__":
    unittest.main()
