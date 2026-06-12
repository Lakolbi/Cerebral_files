from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.comparison.events import detect_heel_strikes, gait_cycle_events

_GAIT = Path(__file__).resolve().parents[1]
_CALP1 = _GAIT / "data" / "caliscope_project" / "recordings" / "p1_1" / "SIMPLE_HOLISTIC"
_VIC1 = _GAIT / "data" / "Vicon_10_series" / "1.xlsx"


def _walker_df(stride=30, n_strides=10):
    """Synthetic one-side walker whose heel anterior-posterior signal is a realistic
    stance(slow posterior)+swing(fast anterior) sawtooth peaking (=heel-strike) at the
    end of each stride. Pelvis fixed at origin; only the AP (x) axis carries the signal.
    """
    n = stride * n_strides
    ap = np.zeros(n)
    stance = int(stride * 0.66)
    for k in range(n_strides):
        b = k * stride
        ap[b:b + stance] = np.linspace(1.0, -1.0, stance)          # stance: drift posterior
        ap[b + stance:b + stride] = np.linspace(-1.0, 1.0, stride - stance)  # swing: fast anterior
    df = pd.DataFrame({
        "left_hip_x": np.zeros(n), "left_hip_y": np.zeros(n),
        "right_hip_x": np.zeros(n), "right_hip_y": np.zeros(n),
        "left_heel_x": ap, "left_heel_y": np.zeros(n),
    })
    return df, stride, n_strides


def test_detect_heel_strikes_finds_anterior_peaks():
    df, stride, n_strides = _walker_df()
    hs = detect_heel_strikes(df, fps=50.0, side="left", min_stride_sec=stride / 50.0 * 0.8)
    # one heel-strike per stride (+/- the boundary one), evenly spaced ~stride apart
    assert n_strides - 1 <= len(hs) <= n_strides + 1
    gaps = np.diff(hs)
    assert np.all(np.abs(gaps - stride) <= 2)


def test_detect_heel_strikes_orientation_invariant():
    """A sign-flipped dominant axis must yield the same heel-strikes (stance-orientation)."""
    df, stride, _ = _walker_df()
    hs_pos = detect_heel_strikes(df, fps=50.0, side="left", min_stride_sec=stride / 50.0 * 0.8)
    df2 = df.copy()
    df2["left_heel_x"] = -df2["left_heel_x"]            # reversed AP sign
    hs_neg = detect_heel_strikes(df2, fps=50.0, side="left", min_stride_sec=stride / 50.0 * 0.8)
    assert hs_pos == hs_neg


def test_detect_heel_strikes_missing_marker_returns_empty():
    df = pd.DataFrame({"left_hip_x": [0.0, 1.0], "left_hip_y": [0.0, 0.0],
                       "right_hip_x": [0.0, 1.0], "right_hip_y": [0.0, 0.0]})
    assert detect_heel_strikes(df, fps=50.0, side="left") == []


def test_detect_heel_strikes_missing_hip_returns_empty():
    df = pd.DataFrame({"left_heel_x": np.arange(40.0), "left_heel_y": np.zeros(40)})
    assert detect_heel_strikes(df, fps=50.0, side="left") == []


def test_detect_heel_strikes_too_short_returns_empty():
    # heel + hip present but too few frames for zero-phase filtering -> [] (not a raise)
    n = 8
    df = pd.DataFrame({
        "left_hip_x": np.zeros(n), "left_hip_y": np.zeros(n),
        "right_hip_x": np.zeros(n), "right_hip_y": np.zeros(n),
        "left_heel_x": np.linspace(0, 1, n), "left_heel_y": np.zeros(n),
    })
    assert detect_heel_strikes(df, fps=50.0, side="left") == []


def test_gait_cycle_events_returns_both_sides():
    df, stride, _ = _walker_df()
    ev = gait_cycle_events(df, fps=50.0, min_stride_sec=stride / 50.0 * 0.8)
    assert "left_HS" in ev and "right_HS" in ev
    assert len(ev["left_HS"]) >= 1
    assert ev["right_HS"] == []                          # no right_heel columns in fixture


@pytest.mark.skipif(not (_CALP1.exists() and _VIC1.exists()),
                    reason="real caliscope/Vicon data not present")
def test_real_data_knee_cycles_are_physiological():
    """Regression for the anchoring fix: both systems' mean knee cycle must peak in
    flexion (argmin of the 180-flexion included angle) in the physiological swing band
    (~60-90%). Before the fix, caliscope peaked at ~20-95% (mis-anchored)."""
    import yaml

    from compare_pipeline import _joint_curves
    from modules.data_loader.caliscope_reader import load_caliscope_session
    from modules.data_loader.vicon_reader import load_vicon_xlsx, map_vicon_to_caliscope

    cfg = yaml.safe_load(open(_GAIT / "settings.yaml"))
    cal = load_caliscope_session(str(_CALP1.parent), model="SIMPLE_HOLISTIC")
    vic = map_vicon_to_caliscope(load_vicon_xlsx(str(_VIC1)), cfg["landmark_mapping"])
    cal_curves, _ = _joint_curves(cal, cfg, cal.attrs["fps"])
    vic_curves, _ = _joint_curves(vic, cfg, 100.0)
    for side in ("left", "right"):
        cpk = int(np.asarray(cal_curves[f"{side}_knee"]).argmin())
        vpk = int(np.asarray(vic_curves[f"{side}_knee"]).argmin())
        assert 55 <= cpk <= 95, f"caliscope {side} knee peak-flexion @{cpk}% not physiological"
        assert 60 <= vpk <= 90, f"Vicon {side} knee peak-flexion @{vpk}% not physiological"
