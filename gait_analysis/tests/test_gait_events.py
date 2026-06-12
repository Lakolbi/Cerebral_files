import numpy as np
import pandas as pd

from modules.kinematics.gait_events import detect_gait_events


def _walking_df(fps=50.0, stride_hz=1.0, n_strides=4):
    t = np.arange(0, n_strides / stride_hz, 1 / fps)
    heel_z = -np.cos(2 * np.pi * stride_hz * t)   # minima (contact) once per stride
    toe_z = np.cos(2 * np.pi * stride_hz * t)     # maxima offset by half a stride
    return pd.DataFrame({
        "frame": range(len(t)), "timestamp": t,
        "left_heel_z": heel_z, "left_foot_index_z": toe_z,
        "left_heel_x": np.zeros_like(t), "left_foot_index_x": np.zeros_like(t),
        "left_heel_y": np.zeros_like(t), "left_foot_index_y": np.zeros_like(t),
    })


def test_detect_finds_expected_number_of_heel_strikes():
    fps = 50.0
    df = _walking_df(fps=fps, stride_hz=1.0, n_strides=4)
    events = detect_gait_events(df, fps=fps, side="left", method="velocity")
    assert 3 <= len(events["left_HS"]) <= 5


def test_detect_heel_strikes_located_near_minima():
    fps = 50.0
    df = _walking_df(fps=fps, stride_hz=1.0, n_strides=3)
    events = detect_gait_events(df, fps=fps, side="left", method="velocity")
    times = df["timestamp"].to_numpy()[events["left_HS"]]
    for tt in times:
        assert min(abs(tt - k) for k in range(4)) < 0.1


def test_detect_both_sides_returns_all_keys():
    fps = 50.0
    df = _walking_df(fps=fps)
    for c in list(df.columns):
        if c.startswith("left_"):
            df[c.replace("left_", "right_")] = df[c]
    events = detect_gait_events(df, fps=fps, side="both", method="velocity")
    assert set(events) == {"left_HS", "left_TO", "right_HS", "right_TO"}
