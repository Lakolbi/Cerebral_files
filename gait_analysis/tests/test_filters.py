import numpy as np
import pandas as pd
import pytest

from modules.kinematics.filters import butterworth_filter, fill_gaps


def test_fill_gaps_fills_short_interior_gap_on_linear_ramp():
    # Cubic interpolation through collinear points reproduces the line exactly.
    df = pd.DataFrame({"left_heel_z": [0.0, 2.0, 4.0, np.nan, 8.0, 10.0]})
    out = fill_gaps(df, max_gap_frames=10)
    assert out["left_heel_z"].isna().sum() == 0
    assert out["left_heel_z"].iloc[3] == pytest.approx(6.0)


def test_fill_gaps_leaves_long_gap_untouched():
    col = [0.0, 1.0, np.nan, np.nan, np.nan, 5.0, 6.0]  # 3-long gap
    df = pd.DataFrame({"left_heel_z": col})
    out = fill_gaps(df, max_gap_frames=2)  # 3 > 2 -> stays fully NaN
    assert out["left_heel_z"].iloc[2:5].isna().all()


def test_fill_gaps_ignores_non_coordinate_columns():
    # left_hip_x is the line y = x + 1 with a one-frame hole at index 1.
    df = pd.DataFrame({"frame": [0, 1, 2, 3, 4],
                       "timestamp": [0.0, 0.1, 0.2, 0.3, 0.4],
                       "left_hip_x": [1.0, np.nan, 3.0, 4.0, 5.0]})
    out = fill_gaps(df, max_gap_frames=5)
    assert list(out["frame"]) == [0, 1, 2, 3, 4]
    assert out["left_hip_x"].iloc[1] == pytest.approx(2.0)


def test_butterworth_attenuates_high_freq_keeps_low_freq():
    fs = 100.0
    t = np.arange(0, 2.0, 1 / fs)
    low = np.sin(2 * np.pi * 1.0 * t)          # 1 Hz (pass)
    high = 0.5 * np.sin(2 * np.pi * 20.0 * t)  # 20 Hz (stop)
    out = butterworth_filter(low + high, cutoff_hz=6.0, fs=fs, order=4)
    assert out.shape == low.shape
    err_low = np.sqrt(np.mean((out - low) ** 2))
    assert err_low < 0.1


def test_butterworth_zero_phase_no_lag_on_symmetric_pulse():
    fs = 100.0
    x = np.zeros(200)
    x[100] = 1.0  # symmetric impulse
    out = butterworth_filter(x, cutoff_hz=10.0, fs=fs, order=4, zero_phase=True)
    assert int(np.argmax(out)) == 100


def test_butterworth_causal_differs_from_zero_phase():
    fs = 100.0
    t = np.arange(0, 2.0, 1 / fs)
    x = np.sin(2 * np.pi * 2.0 * t)
    out_zp = butterworth_filter(x, cutoff_hz=6.0, fs=fs, zero_phase=True)
    out_causal = butterworth_filter(x, cutoff_hz=6.0, fs=fs, zero_phase=False)
    assert out_causal.shape == x.shape
    assert not np.allclose(out_zp, out_causal)


def test_butterworth_raises_when_cutoff_at_or_above_nyquist():
    with pytest.raises(ValueError):
        butterworth_filter(np.zeros(50), cutoff_hz=60.0, fs=100.0)


def test_butterworth_raises_on_too_short_signal_for_filtfilt():
    with pytest.raises(ValueError):
        butterworth_filter(np.zeros(10), cutoff_hz=6.0, fs=100.0, order=4)
