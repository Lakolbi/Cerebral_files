"""Accuracy metrics for Vicon comparison: RMSE/MAE/Pearson (NaN-aware) + ICC."""
import warnings

import numpy as np
import pandas as pd

_ICC_TYPE = {"1,1": "ICC(1,1)", "2,1": "ICC(A,1)", "3,1": "ICC(C,1)"}


def _paired(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    m = ~(np.isnan(a) | np.isnan(b))
    return a[m], b[m]


def calc_rmse(a, b) -> float:
    a, b = _paired(a, b)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def calc_mae(a, b) -> float:
    a, b = _paired(a, b)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def calc_pearson(a, b) -> float:
    a, b = _paired(a, b)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def calc_icc(a, b, icc_type: str = "3,1") -> float:
    """ICC between two raters a, b (1-D, paired). Uses pingouin; NaN if degenerate."""
    import pingouin as pg

    a, b = _paired(a, b)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    n = a.size
    long = pd.DataFrame({
        "target": list(range(n)) * 2,
        "rater": ["a"] * n + ["b"] * n,
        "rating": np.concatenate([a, b]),
    })
    with warnings.catch_warnings():
        # pingouin divides by a zero F-stat denominator when raters agree perfectly
        warnings.simplefilter("ignore", RuntimeWarning)
        res = pg.intraclass_corr(data=long, targets="target", raters="rater",
                                 ratings="rating").set_index("Type")
    return float(res.loc[_ICC_TYPE[icc_type], "ICC"])


def verdict_for_rmse(rmse_deg: float, good: float, acceptable: float) -> str:
    if np.isnan(rmse_deg):
        return "n/a"
    if rmse_deg < good:
        return "good"
    if rmse_deg <= acceptable:
        return "acceptable"
    return "poor"


def angle_comparison_report(cal_curves: dict, vic_curves: dict,
                            good: float = 5.0, acceptable: float = 10.0,
                            icc_type: str = "3,1", joint_list=None) -> pd.DataFrame:
    """Per-joint angle agreement over matched 101-pt curves.

    cal_curves / vic_curves: {"<side>_<joint>": 101-array}. Only joints present in
    BOTH (and non-degenerate) are reported — this naturally drops hip on the Vicon
    side (no shoulder marker -> all-NaN curve).
    """
    joints = joint_list if joint_list is not None else sorted(set(cal_curves) & set(vic_curves))
    rows = []
    for j in joints:
        if j not in cal_curves or j not in vic_curves:
            continue
        a, b = np.asarray(cal_curves[j], float), np.asarray(vic_curves[j], float)
        if np.isnan(a).all() or np.isnan(b).all():
            continue
        rmse = calc_rmse(a, b)
        rows.append({
            "joint": j,
            "rmse_deg": rmse,
            "mae_deg": calc_mae(a, b),
            "pearson": calc_pearson(a, b),
            "icc": calc_icc(a, b, icc_type=icc_type),
            "verdict": verdict_for_rmse(rmse, good, acceptable),
        })
    return pd.DataFrame(rows, columns=["joint", "rmse_deg", "mae_deg",
                                       "pearson", "icc", "verdict"])


_POS_COLS = ["joint", "n_samples", "rmse_m", "mae_m", "max_m", "median_m",
             "rmse_x_m", "rmse_y_m", "rmse_z_m"]


def position_comparison_report(cal_pts: dict, vic_pts: dict, joint_list=None) -> pd.DataFrame:
    """Per-joint positional error (meters) between matched (N,3) arrays.

    cal_pts / vic_pts: {"JOINT": (N,3)} already on a common, aligned grid.
    """
    joints = joint_list if joint_list is not None else sorted(set(cal_pts) & set(vic_pts))
    rows = []
    for j in joints:
        if j not in cal_pts or j not in vic_pts:
            continue
        v, c = np.asarray(vic_pts[j], float), np.asarray(cal_pts[j], float)
        diff = v - c
        mask = ~np.isnan(diff).any(axis=1)
        if not mask.any():
            continue
        d = diff[mask]
        dist = np.linalg.norm(d, axis=1)
        rows.append({
            "joint": j,
            "n_samples": int(dist.size),
            "rmse_m": float(np.sqrt(np.mean(dist ** 2))),
            "mae_m": float(np.mean(dist)),
            "max_m": float(np.max(dist)),
            "median_m": float(np.median(dist)),
            "rmse_x_m": float(np.sqrt(np.mean(d[:, 0] ** 2))),
            "rmse_y_m": float(np.sqrt(np.mean(d[:, 1] ** 2))),
            "rmse_z_m": float(np.sqrt(np.mean(d[:, 2] ** 2))),
        })
    return pd.DataFrame(rows, columns=_POS_COLS)
