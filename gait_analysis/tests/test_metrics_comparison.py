import numpy as np
import pytest

from modules.comparison.metrics import (
    angle_comparison_report,
    calc_icc,
    calc_mae,
    calc_pearson,
    calc_rmse,
    position_comparison_report,
    verdict_for_rmse,
)


def test_identity_arrays_give_perfect_scores():
    a = np.linspace(0, 60, 101)
    assert calc_rmse(a, a) == pytest.approx(0.0, abs=1e-12)
    assert calc_mae(a, a) == pytest.approx(0.0, abs=1e-12)
    assert calc_pearson(a, a) == pytest.approx(1.0, abs=1e-9)
    assert calc_icc(a, a) == pytest.approx(1.0, abs=1e-6)


def test_rmse_mae_known_values():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert calc_rmse(a, b) == pytest.approx(np.sqrt((9 + 16 + 0) / 3))
    assert calc_mae(a, b) == pytest.approx((3 + 4 + 0) / 3)


def test_nan_aware():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, 5.0, 3.0])
    assert calc_rmse(a, b) == pytest.approx(0.0, abs=1e-12)


def test_icc_constant_input_is_nan():
    a = np.ones(50)
    b = np.linspace(0, 1, 50)
    assert np.isnan(calc_icc(a, b))


def test_icc_high_for_strongly_agreeing_raters():
    rng = np.random.default_rng(0)
    a = rng.normal(size=80)
    b = a + rng.normal(scale=0.05, size=80)     # near-identical raters
    assert calc_icc(a, b) > 0.9


def test_verdict_thresholds():
    assert verdict_for_rmse(3.0, 5.0, 10.0) == "good"
    assert verdict_for_rmse(7.0, 5.0, 10.0) == "acceptable"
    assert verdict_for_rmse(12.0, 5.0, 10.0) == "poor"


def test_angle_report_uses_joint_intersection_and_verdicts():
    cal = {"left_knee": np.linspace(0, 60, 101), "left_ankle": np.linspace(0, 20, 101)}
    vic = {"left_knee": np.linspace(0, 60, 101)}          # ankle missing on one side
    rep = angle_comparison_report(cal, vic, good=5.0, acceptable=10.0)
    assert set(rep["joint"]) == {"left_knee"}              # intersection only
    row = rep.set_index("joint").loc["left_knee"]
    assert row["rmse_deg"] == 0.0 and row["verdict"] == "good"
    assert row["icc"] > 0.99


def test_position_report_zero_error_for_identical_points():
    pts = {"LKNE": np.tile([0.1, 0.2, 0.9], (10, 1))}
    rep = position_comparison_report(pts, pts)
    row = rep.set_index("joint").loc["LKNE"]
    assert row["rmse_m"] == 0.0 and row["mae_m"] == 0.0


def test_verdict_boundaries_and_nan():
    # boundary == good -> not "good" (strict <), == acceptable -> "acceptable" (<=)
    assert verdict_for_rmse(5.0, 5.0, 10.0) == "acceptable"
    assert verdict_for_rmse(10.0, 5.0, 10.0) == "acceptable"
    assert verdict_for_rmse(float("nan"), 5.0, 10.0) == "n/a"


def test_empty_reports_keep_columns():
    # no common joints -> empty frame but with the documented columns (no KeyError downstream)
    angle = angle_comparison_report({}, {})
    assert list(angle.columns) == ["joint", "rmse_deg", "mae_deg", "pearson", "icc", "verdict"]
    assert len(angle) == 0
    pos = position_comparison_report({}, {})
    assert "joint" in pos.columns and "rmse_m" in pos.columns and len(pos) == 0


def test_reports_skip_joint_list_entries_missing_from_either_dict():
    cal = {"left_knee": np.linspace(0, 60, 101)}
    vic = {"left_knee": np.linspace(0, 60, 101)}
    # explicit joint_list naming an absent joint must not raise, just skip it
    rep = angle_comparison_report(cal, vic, joint_list=["left_knee", "right_ankle"])
    assert set(rep["joint"]) == {"left_knee"}
