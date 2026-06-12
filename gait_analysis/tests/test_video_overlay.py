from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from modules.data_loader.landmarks import GAIT_LANDMARKS
from modules.visualization.video_overlay import (
    POSE_POINT_IDS,
    draw_overlay,
    frame_marks,
    load_xy,
    produce_marked_video,
    produce_marked_videos,
)

_REC = Path(__file__).resolve().parents[1] / "data" / "caliscope_project" / "recordings" / "p1_3"


def test_pose_point_ids_cover_the_twelve_gait_joints():
    assert set(POSE_POINT_IDS.values()) == set(GAIT_LANDMARKS)


def test_frame_marks_maps_ids_skips_other_ports_frames_and_nan():
    xy = pd.DataFrame({
        "port": [1, 1, 1, 1, 2],
        "frame_index": [0, 0, 0, 1, 0],
        "point_id": [23, 999, 25, 23, 24],     # 23=left_hip 25=left_knee 999=non-gait
        "img_loc_x": [100, 50, 110, 5, 700],
        "img_loc_y": [200, 50, 260, 5, 200],
    })
    marks = frame_marks(xy, port=1, frame_index=0)
    assert marks["left_hip"] == (100, 200)
    assert marks["left_knee"] == (110, 260)
    assert "right_hip" not in marks            # port 2 / other frame excluded
    assert len(marks) == 2                     # point_id 999 ignored


def test_frame_marks_skips_nan_coords():
    xy = pd.DataFrame({
        "port": [1], "frame_index": [0], "point_id": [23],
        "img_loc_x": [np.nan], "img_loc_y": [np.nan],
    })
    assert frame_marks(xy, port=1, frame_index=0) == {}


def test_draw_overlay_draws_bones_and_joints_preserving_shape():
    frame = np.full((300, 300, 3), 128, dtype=np.uint8)   # gray bg: black bones show as a change
    marks = {"left_hip": (100, 150), "left_knee": (100, 250)}
    out = draw_overlay(frame, marks, edges=[("left_hip", "left_knee")])
    assert out.shape == (300, 300, 3)
    assert tuple(int(c) for c in out[150, 100]) != (128, 128, 128)   # joint drawn
    assert tuple(int(c) for c in out[200, 100]) != (128, 128, 128)   # bone midpoint drawn


@pytest.mark.skipif(not (_REC / "port_1.mp4").exists(), reason="p1_3 data not present")
def test_produce_marked_video_writes_playable_mp4(tmp_path):
    xy = load_xy(str(_REC), "SIMPLE_HOLISTIC")
    out = tmp_path / "port_1_marked.mp4"
    produce_marked_video(_REC / "port_1.mp4", xy, port=1, out_path=out)
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n > 0


@pytest.mark.skipif(not (_REC / "port_1.mp4").exists(), reason="p1_3 data not present")
def test_produce_marked_videos_writes_one_per_camera(tmp_path):
    outs = produce_marked_videos(str(_REC), "SIMPLE_HOLISTIC", str(tmp_path))
    assert len(outs) == len(sorted(_REC.glob("port_*.mp4")))
    assert all(Path(o).exists() and Path(o).stat().st_size > 0 for o in outs)


def test_produce_marked_videos_raises_when_no_videos(tmp_path):
    with pytest.raises(FileNotFoundError):
        produce_marked_videos(str(tmp_path), "SIMPLE_HOLISTIC", str(tmp_path / "out"))
