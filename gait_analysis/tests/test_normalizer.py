import numpy as np
import pytest

from modules.kinematics.normalizer import get_mean_std_cycle, normalize_gait_cycle


def test_normalize_returns_101_points_per_cycle():
    signal = np.linspace(0, 30, 31)
    events = {"left_HS": [0, 10, 20, 30]}  # 3 cycles
    mat = normalize_gait_cycle(signal, events, side="left", n_points=101)
    assert mat.shape == (3, 101)


def test_normalize_preserves_endpoints_of_linear_cycle():
    signal = np.arange(0, 21, dtype=float)   # ramp 0..20
    events = {"left_HS": [0, 10, 20]}
    mat = normalize_gait_cycle(signal, events, side="left")
    assert mat[0, 0] == pytest.approx(0.0)
    assert mat[0, -1] == pytest.approx(10.0)


def test_get_mean_std_cycle_shapes_and_values():
    mat = np.vstack([np.full(101, 2.0), np.full(101, 4.0)])  # mean 3, std 1
    mean, std = get_mean_std_cycle(mat)
    assert mean.shape == (101,) and std.shape == (101,)
    assert mean[0] == pytest.approx(3.0)
    assert std[0] == pytest.approx(1.0)
