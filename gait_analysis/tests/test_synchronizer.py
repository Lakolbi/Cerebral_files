import numpy as np
import pandas as pd
import pytest

from modules.data_loader.synchronizer import synchronize


def _df(fps, duration, phase=0.0):
    t = np.arange(0, duration, 1 / fps)
    return pd.DataFrame({
        "frame": range(len(t)),
        "timestamp": t,
        "left_hip_z": np.sin(2 * np.pi * 1.0 * t + phase),
    })


def test_synchronize_equal_length_and_grid():
    a = _df(30.0, 2.0)
    b = _df(100.0, 2.0)
    sa, sb = synchronize(a, b, target_fps=100.0)
    assert len(sa) == len(sb)
    assert np.allclose(sa["timestamp"].to_numpy(), sb["timestamp"].to_numpy())
    dt = np.diff(sa["timestamp"].to_numpy())
    assert np.allclose(dt, 1 / 100.0, atol=1e-9)


def test_synchronize_clips_to_overlap():
    a = _df(50.0, 2.0)
    b = _df(50.0, 3.0)
    b["timestamp"] = b["timestamp"] + 0.5   # overlap [0.5, 2.0]
    sa, sb = synchronize(a, b, target_fps=50.0)
    assert sa["timestamp"].iloc[0] >= 0.5 - 1e-9
    assert sa["timestamp"].iloc[-1] <= 2.0 + 1e-9


def test_synchronize_short_streams_use_linear_without_error():
    a = pd.DataFrame({"frame": range(3), "timestamp": [0.0, 0.1, 0.2],
                      "left_hip_z": [0.0, 1.0, 2.0]})
    b = pd.DataFrame({"frame": range(3), "timestamp": [0.0, 0.1, 0.2],
                      "left_hip_z": [0.0, 2.0, 4.0]})
    sa, sb = synchronize(a, b, target_fps=50.0)
    assert len(sa) == len(sb) and len(sa) > 0
    assert not sa["left_hip_z"].isna().all()


def test_synchronize_raises_on_single_sample():
    a = pd.DataFrame({"frame": [0], "timestamp": [0.0], "left_hip_z": [1.0]})
    b = pd.DataFrame({"frame": range(5), "timestamp": np.arange(5) / 50.0,
                      "left_hip_z": np.arange(5.0)})
    with pytest.raises(ValueError):
        synchronize(a, b, target_fps=50.0)
