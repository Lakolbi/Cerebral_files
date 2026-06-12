import numpy as np
import pandas as pd
import pytest

from modules.kinematics.spatiotemporal import calc_spatiotemporal


def _constant_velocity_walk(fps=50.0):
    t = np.arange(0, 4.0, 1 / fps)
    x = 1.0 * t  # +x at 1 m/s
    df = pd.DataFrame({
        "frame": range(len(t)), "timestamp": t,
        "left_heel_x": x, "left_heel_y": np.zeros_like(t), "left_heel_z": np.zeros_like(t),
        "right_heel_x": x, "right_heel_y": np.full_like(t, 0.2), "right_heel_z": np.zeros_like(t),
    })
    return df, fps


def test_cadence_and_speed_are_positive_and_reasonable():
    df, fps = _constant_velocity_walk()
    events = {
        "left_HS": [0, 50, 100, 150], "left_TO": [30, 80, 130],
        "right_HS": [25, 75, 125], "right_TO": [55, 105, 155],
    }
    out = calc_spatiotemporal(df, events, fps=fps)
    assert out["cadence_steps_per_min"] > 0
    assert out["speed_m_per_s"] > 0
    assert out["stride_length_m"] == pytest.approx(1.0, abs=0.2)


def test_stance_swing_sum_to_about_100():
    df, fps = _constant_velocity_walk()
    events = {
        "left_HS": [0, 50, 100, 150], "left_TO": [30, 80, 130],
        "right_HS": [25, 75, 125], "right_TO": [55, 105, 155],
    }
    out = calc_spatiotemporal(df, events, fps=fps)
    assert out["stance_pct"] + out["swing_pct"] == pytest.approx(100.0, abs=1.0)


def test_stride_rejection_drops_implausible_stride():
    # 3 left HS -> 2 strides; second stride is a 5 m artifact and must be dropped.
    t = np.arange(0, 4.0, 1 / 50.0)
    zeros = np.zeros_like(t)
    df = pd.DataFrame({
        "frame": range(len(t)), "timestamp": t,
        "left_heel_x": zeros.copy(), "left_heel_y": zeros.copy(), "left_heel_z": zeros.copy(),
        "right_heel_x": zeros.copy(), "right_heel_y": zeros.copy(), "right_heel_z": zeros.copy(),
    })
    # place left heel x: HS0 at frame0 (x=0), HS1 at frame50 (x=0.8 -> stride 0.8),
    # HS2 at frame100 (x=5.8 -> stride 5.0, implausible)
    df.loc[0, "left_heel_x"] = 0.0
    df.loc[50, "left_heel_x"] = 0.8
    df.loc[100, "left_heel_x"] = 5.8
    events = {
        "left_HS": [0, 50, 100], "left_TO": [30, 80],
        "right_HS": [25, 75], "right_TO": [55, 105],
    }
    out = calc_spatiotemporal(df, events, fps=50.0, max_stride_m=1.5, max_step_m=1.0)
    # only the 0.8 m stride survives
    assert out["stride_length_m"] == pytest.approx(0.8, abs=0.05)
    assert out["n_strides_used"] == 1
    assert out["n_strides_total"] == 2


def test_stride_rejection_keeps_all_when_plausible():
    df, fps = _constant_velocity_walk()
    events = {"left_HS": [0, 50, 100, 150], "left_TO": [30, 80, 130],
              "right_HS": [25, 75, 125], "right_TO": [55, 105, 155]}
    out = calc_spatiotemporal(df, events, fps=fps, max_stride_m=1.5, max_step_m=1.0)
    assert out["n_strides_used"] == out["n_strides_total"] == 3


def test_step_rejection_drops_length_and_width_together():
    # Travel axis is +x (left heel ramps forward). Two right-HS steps:
    # frame 25 = normal (0.4 m ahead, 0.1 m lateral); frame 75 = artifact
    # (3 m ahead -> length > max_step_m). The artifact step must drop BOTH
    # its length and its width, leaving only the normal step's values.
    fps = 50.0
    t = np.arange(0, 3.0, 1 / fps)
    zeros = np.zeros_like(t)
    df = pd.DataFrame({
        "frame": range(len(t)), "timestamp": t,
        "left_heel_x": 1.0 * t, "left_heel_y": zeros.copy(), "left_heel_z": zeros.copy(),
        "right_heel_x": 1.0 * t, "right_heel_y": zeros.copy(), "right_heel_z": zeros.copy(),
    })
    df.loc[25, "right_heel_x"] = df.loc[25, "left_heel_x"] + 0.4
    df.loc[25, "right_heel_y"] = 0.1
    df.loc[75, "right_heel_x"] = df.loc[75, "left_heel_x"] + 3.0
    df.loc[75, "right_heel_y"] = 5.0
    events = {"left_HS": [0, 50, 100], "left_TO": [30, 80],
              "right_HS": [25, 75], "right_TO": [55, 105]}
    out = calc_spatiotemporal(df, events, fps=fps, max_stride_m=1.5, max_step_m=1.0)
    assert out["step_length_m"] == pytest.approx(0.4, abs=0.1)
    assert out["step_width_m"] == pytest.approx(0.1, abs=0.1)
