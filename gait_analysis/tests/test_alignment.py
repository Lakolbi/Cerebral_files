import numpy as np

from modules.comparison.alignment import (
    apply_transform,
    detect_sync_event,
    estimate_rigid_transform,
    estimate_time_shift_xcorr,
)


def _rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0, 0, 1.0]])


def test_umeyama_recovers_known_transform():
    rng = np.random.default_rng(42)
    src = rng.normal(size=(50, 3))
    R_true, s_true, T_true = _rot_z(30.0), 2.5, np.array([1.0, -2.0, 0.5])
    dst = s_true * (src @ R_true.T) + T_true

    R, T, s = estimate_rigid_transform(src, dst)
    assert np.isclose(s, s_true, atol=1e-6)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(T, T_true, atol=1e-6)
    assert np.linalg.det(R) > 0          # reflection-free
    assert np.allclose(apply_transform(src, R, T, s), dst, atol=1e-6)


def test_xcorr_recovers_injected_time_shift():
    fs = 100.0
    t = np.arange(0, 4, 1 / fs)
    # a moving pelvis with a sharp speed burst (a "jump"): position ramp + bump
    pos = np.zeros((len(t), 3))
    pos[:, 0] = np.cumsum(np.exp(-((t - 1.0) ** 2) / 0.01)) / fs   # burst near t=1s
    shift = 0.20
    t_cal = t + shift                                              # caliscope lags by 0.2 s
    s = estimate_time_shift_xcorr(t, pos, t_cal, pos, fs_grid=fs, max_shift=0.5)
    assert abs(s - shift) < 0.03


def test_detect_sync_event_finds_speed_peak():
    fs = 100.0
    t = np.arange(0, 3, 1 / fs)
    pos = np.zeros((len(t), 3))
    pos[:, 2] = np.exp(-((t - 0.5) ** 2) / 0.005)                 # Z bump at 0.5 s
    idx = detect_sync_event(t, pos, mode="zmax")
    assert abs(t[idx] - 0.5) < 0.05
