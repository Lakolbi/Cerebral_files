import shutil
from pathlib import Path

import numpy as np
import pytest

from modules.data_loader.caliscope_reader import derive_fps, list_landmarks, load_caliscope_session
from modules.data_loader.landmarks import GAIT_LANDMARKS

FIX = Path(__file__).parent / "fixtures"


def test_list_landmarks_returns_gait_set_for_known_model():
    lm = list_landmarks("SIMPLE_HOLISTIC")
    assert "left_knee" in lm and "right_foot_index" in lm
    assert lm == GAIT_LANDMARKS


def test_list_landmarks_rejects_unknown_model():
    with pytest.raises(ValueError):
        list_landmarks("NOT_A_MODEL")


def test_derive_fps_from_even_timestamps():
    ts = np.arange(0, 1.0, 1 / 20.0)  # 20 fps
    assert derive_fps(ts) == pytest.approx(20.0, abs=0.01)


def test_derive_fps_robust_to_a_dropped_frame():
    ts = np.array([0.0, 0.05, 0.10, 0.20, 0.25])  # one gap at 0.15
    assert derive_fps(ts) == pytest.approx(20.0, abs=0.01)


def _make_session(tmp_path, model="SIMPLE_HOLISTIC"):
    model_dir = tmp_path / "sess" / model
    model_dir.mkdir(parents=True)
    shutil.copy(FIX / "mini_labelled.csv", model_dir / f"xyz_{model}_labelled.csv")
    shutil.copy(FIX / "mini_frame_time_history.csv", model_dir / "frame_time_history.csv")
    return tmp_path / "sess"


def test_load_session_shape_and_columns(tmp_path):
    sess = _make_session(tmp_path)
    df = load_caliscope_session(str(sess), model="SIMPLE_HOLISTIC")
    assert list(df.columns[:2]) == ["frame", "timestamp"]
    assert "left_hip_x" in df.columns and "left_heel_z" in df.columns
    assert len(df) == 4
    assert list(df["frame"]) == [0, 1, 2, 3]


def test_load_session_timestamps_zero_based_and_monotonic(tmp_path):
    sess = _make_session(tmp_path)
    df = load_caliscope_session(str(sess), model="SIMPLE_HOLISTIC")
    assert df["timestamp"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert df["timestamp"].is_monotonic_increasing
    # sync-6 port-mean (1000.0505) minus sync-5 port-mean (1000.0005) = 0.0500 s
    assert df["timestamp"].iloc[1] == pytest.approx(0.0500, abs=1e-4)


def test_load_session_sets_fps_and_model_attrs(tmp_path):
    sess = _make_session(tmp_path)
    df = load_caliscope_session(str(sess), model="SIMPLE_HOLISTIC")
    assert df.attrs["model"] == "SIMPLE_HOLISTIC"
    assert df.attrs["fps"] == pytest.approx(20.0, abs=0.5)
