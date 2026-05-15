from pathlib import Path

from src.task2_detect_track.analyze_occlusion import (
    TrackBox,
    box_iou,
    choose_occlusion_window,
    parse_label_line,
    write_analysis_markdown,
)


def test_parse_ultralytics_track_label_with_conf_and_id():
    box = parse_label_line("4 0.5 0.5 0.2 0.4 0.91 17", frame_index=12)

    assert box is not None
    assert box.class_id == 4
    assert box.track_id == 17
    assert box.conf == 0.91
    assert box.xyxy == (0.4, 0.3, 0.6, 0.7)


def test_parse_ultralytics_track_label_without_conf():
    box = parse_label_line("5 0.25 0.75 0.1 0.2 9", frame_index=3)

    assert box is not None
    assert box.class_id == 5
    assert box.track_id == 9
    assert box.conf is None


def test_choose_occlusion_window_prefers_high_iou_sequence():
    frames = {
        10: [
            TrackBox(10, 0, 0.10, 0.10, 0.10, 0.10, track_id=1),
            TrackBox(10, 0, 0.90, 0.90, 0.10, 0.10, track_id=2),
        ],
        11: [
            TrackBox(11, 0, 0.50, 0.50, 0.40, 0.40, track_id=1),
            TrackBox(11, 0, 0.55, 0.50, 0.40, 0.40, track_id=2),
        ],
        12: [
            TrackBox(12, 0, 0.52, 0.50, 0.40, 0.40, track_id=1),
            TrackBox(12, 0, 0.56, 0.50, 0.40, 0.40, track_id=2),
        ],
        13: [TrackBox(13, 0, 0.90, 0.90, 0.10, 0.10, track_id=3)],
    }

    selected = choose_occlusion_window(frames, num_frames=2)

    assert selected.start_frame == 11
    assert selected.max_iou > 0.75
    assert box_iou(frames[11][0], frames[11][1]) > 0.75


def test_write_analysis_markdown_contains_tracker_terms(tmp_path: Path):
    analysis_path = tmp_path / "analysis.md"
    exported = [tmp_path / "frame0.png", tmp_path / "frame1.png"]
    frames = {
        20: [
            TrackBox(20, 4, 0.50, 0.50, 0.30, 0.30, conf=0.80, track_id=1),
            TrackBox(20, 4, 0.55, 0.50, 0.30, 0.30, conf=0.75, track_id=2),
        ],
        21: [TrackBox(21, 4, 0.60, 0.50, 0.30, 0.30, conf=0.70, track_id=1)],
    }

    write_analysis_markdown(
        output_path=analysis_path,
        video_path=tmp_path / "video.avi",
        labels_dir=tmp_path / "bytetrack",
        compare_labels_dir=tmp_path / "botsort",
        start_frame=20,
        num_frames=2,
        exported_frames=exported,
        label_frames=frames,
        compare_label_frames=frames,
        selection=choose_occlusion_window(frames, num_frames=2),
    )

    text = analysis_path.read_text(encoding="utf-8")
    assert "ByteTrack" in text
    assert "BoT-SORT" in text
    assert "track_buffer" in text
    assert "ReID" in text
