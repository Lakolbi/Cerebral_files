import numpy as np
import pandas as pd

from modules.data_loader.landmarks import GAIT_LANDMARKS
from modules.visualization.skeleton_3d import (
    SKELETON_EDGES,
    SKELETON_LANDMARKS,
    bounds,
    frame_points,
    segment_lines,
)


def test_frame_points_extracts_xyz_and_skips_nan():
    df = pd.DataFrame({
        "left_hip_x": [0.0], "left_hip_y": [1.0], "left_hip_z": [2.0],
        "left_knee_x": [np.nan], "left_knee_y": [1.0], "left_knee_z": [0.0],
    })
    pts = frame_points(df, 0)
    assert pts["left_hip"] == (0.0, 1.0, 2.0)
    assert "left_knee" not in pts          # NaN endpoint dropped


def test_segment_lines_skips_edges_with_missing_endpoint():
    points = {"left_hip": (0, 0, 0), "left_knee": (0, 0, 1)}
    lines = segment_lines(points, edges=[("left_hip", "left_knee"),
                                         ("left_knee", "left_ankle")])
    assert lines == [((0, 0, 0), (0, 0, 1))]


def test_skeleton_edges_reference_known_landmarks():
    names = set(GAIT_LANDMARKS)
    for a, b in SKELETON_EDGES:
        assert a in names and b in names


def test_frame_points_names_filter_excludes_others():
    df = pd.DataFrame({
        "left_hip_x": [0.0], "left_hip_y": [0.0], "left_hip_z": [0.0],
        "nose_tip_x": [9.0], "nose_tip_y": [9.0], "nose_tip_z": [9.0],
    })
    pts = frame_points(df, 0, names=["left_hip"])
    assert "left_hip" in pts and "nose_tip" not in pts


def test_skeleton_landmarks_are_edge_endpoints():
    assert set(SKELETON_LANDMARKS) == {n for e in SKELETON_EDGES for n in e}


def test_bounds_spans_all_frames_of_a_walk():
    # left_hip translates x 0->4 across frames; bounds must cover the whole walk,
    # not just frame 0 (the bug that let the skeleton leave the view).
    n = 5
    df = pd.DataFrame({
        "left_hip_x": np.linspace(0, 4, n), "left_hip_y": np.zeros(n),
        "left_hip_z": np.full(n, 1.0),
        "left_knee_x": np.linspace(0, 4, n), "left_knee_y": np.zeros(n),
        "left_knee_z": np.full(n, 0.5),
    })
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds(df)
    assert xmin == 0.0 and xmax == 4.0     # full horizontal travel captured
    assert zmin == 0.5 and zmax == 1.0


def test_bounds_returns_none_when_no_skeleton_columns():
    df = pd.DataFrame({"nose_tip_x": [1.0], "nose_tip_y": [1.0], "nose_tip_z": [1.0]})
    assert bounds(df) is None
