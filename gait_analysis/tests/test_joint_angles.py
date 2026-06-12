import numpy as np
import pandas as pd
import pytest

from modules.kinematics.joint_angles import calc_angle_3d, calc_joint_angles_timeseries


def test_calc_angle_90_degrees():
    p1 = np.array([1.0, 0.0, 0.0])
    vertex = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    assert calc_angle_3d(p1, vertex, p2) == pytest.approx(90.0, abs=1e-6)


def test_calc_angle_180_degrees_collinear():
    assert calc_angle_3d(np.array([1.0, 0, 0]), np.array([0.0, 0, 0]),
                         np.array([-1.0, 0, 0])) == pytest.approx(180.0, abs=1e-6)


def test_calc_angle_45_degrees():
    assert calc_angle_3d(np.array([1.0, 0, 0]), np.array([0.0, 0, 0]),
                         np.array([1.0, 1.0, 0])) == pytest.approx(45.0, abs=1e-6)


def test_calc_angle_nan_when_point_missing():
    out = calc_angle_3d(np.array([np.nan, 0, 0]), np.array([0.0, 0, 0]),
                        np.array([0.0, 1, 0]))
    assert np.isnan(out)


def test_joint_angles_timeseries_adds_expected_columns():
    # One frame: straight left leg (hip, knee, ankle collinear) -> knee angle 180.
    row = {
        "frame": [0], "timestamp": [0.0],
        "left_hip_x": [0.0], "left_hip_y": [0.0], "left_hip_z": [1.0],
        "left_knee_x": [0.0], "left_knee_y": [0.0], "left_knee_z": [0.5],
        "left_ankle_x": [0.0], "left_ankle_y": [0.0], "left_ankle_z": [0.0],
        "left_foot_index_x": [0.2], "left_foot_index_y": [0.0], "left_foot_index_z": [0.0],
        "left_shoulder_x": [0.0], "left_shoulder_y": [0.0], "left_shoulder_z": [1.5],
    }
    df = pd.DataFrame(row)
    out = calc_joint_angles_timeseries(df)
    assert "left_knee_angle" in out.columns
    assert "left_hip_angle" in out.columns
    assert "left_ankle_angle" in out.columns
    assert out["left_knee_angle"].iloc[0] == pytest.approx(180.0, abs=1e-6)


def test_joint_angles_timeseries_handles_missing_landmark_as_nan():
    df = pd.DataFrame({
        "frame": [0], "timestamp": [0.0],
        "left_hip_x": [0.0], "left_hip_y": [0.0], "left_hip_z": [1.0],
        "left_knee_x": [np.nan], "left_knee_y": [np.nan], "left_knee_z": [np.nan],
        "left_ankle_x": [0.0], "left_ankle_y": [0.0], "left_ankle_z": [0.0],
    })
    out = calc_joint_angles_timeseries(df)
    assert np.isnan(out["left_knee_angle"].iloc[0])
