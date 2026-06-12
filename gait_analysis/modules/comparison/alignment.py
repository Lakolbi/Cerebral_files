"""Coordinate-system registration (Umeyama) and temporal sync for Vicon comparison.

Ported and cleaned from the VvsC.py baseline (compute_similarity_transform,
estimate_time_shift_by_xcorr, detect_event_index).
"""
import numpy as np


def estimate_rigid_transform(src: np.ndarray, dst: np.ndarray):
    """Umeyama (1991) similarity: find (R, T, s) with dst ≈ s * (src @ R.T) + T.

    Reflection-free (det(R) = +1), uniform scale. src/dst are (N, 3).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - mu_s, dst - mu_d
    H = s0.T @ d0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_s = (s0 ** 2).sum()
    s = S.sum() / var_s if var_s > 0 else 1.0
    T = mu_d - s * (R @ mu_s)
    return R, T, s


def apply_transform(points: np.ndarray, R: np.ndarray, T: np.ndarray, s: float) -> np.ndarray:
    """Apply (R, T, s): s * (points @ R.T) + T."""
    return s * (np.asarray(points, float) @ R.T) + T


def detect_sync_event(time: np.ndarray, point: np.ndarray, mode: str = "speed") -> int:
    """Index of the sync event: 'speed' = max 3D speed, 'zmax' = max Z (jump apex)."""
    time = np.asarray(time, float)
    point = np.asarray(point, float)
    if len(time) < 2:
        return 0
    if mode == "zmax":
        return int(np.nanargmax(point[:, 2]))
    if mode == "speed":
        dt = np.diff(time)
        dt[dt == 0] = np.min(dt[dt > 0]) if np.any(dt > 0) else 1.0
        speed = np.linalg.norm(np.diff(point, axis=0) / dt[:, None], axis=1)
        return int(np.nanargmax(speed)) + 1
    raise ValueError(f"unknown sync mode {mode!r}")


def _speed(time: np.ndarray, point: np.ndarray):
    dt = np.diff(time)
    dt[dt == 0] = np.min(dt[dt > 0]) if np.any(dt > 0) else 1.0
    speed = np.linalg.norm(np.diff(point, axis=0) / dt[:, None], axis=1)
    t_mid = 0.5 * (time[:-1] + time[1:])
    return t_mid, speed


def _pelvis(df):
    cols = [("left_hip", "right_hip")]
    for lh, rh in cols:
        if all(f"{lh}_{a}" in df.columns for a in "xyz") and \
           all(f"{rh}_{a}" in df.columns for a in "xyz"):
            lpts = df[[f"{lh}_x", f"{lh}_y", f"{lh}_z"]].to_numpy(float)
            rpts = df[[f"{rh}_x", f"{rh}_y", f"{rh}_z"]].to_numpy(float)
            return 0.5 * (lpts + rpts)
    return None


def align_streams(cal_df, vic_df, joints, fs_grid=100.0, max_shift=0.5):
    """Time-align (xcorr on pelvis speed) + rigid+scale register Vicon onto caliscope.

    joints: {"JOINT": "canonical_landmark"} to compare, e.g. {"LKNE": "left_knee"}.
    Returns (cal_pts, vic_pts_aligned, info) where each *_pts is {"JOINT": (N,3)} on a
    shared grid, and info has time_shift_s + scale.
    """
    import numpy as np

    from modules.data_loader.synchronizer import synchronize

    cp, vp = _pelvis(cal_df), _pelvis(vic_df)
    shift = 0.0
    if cp is not None and vp is not None:
        shift = estimate_time_shift_xcorr(
            cal_df["timestamp"].to_numpy(), cp,
            vic_df["timestamp"].to_numpy(), vp, fs_grid=fs_grid, max_shift=max_shift)
    vic_shifted = vic_df.copy()
    vic_shifted["timestamp"] = vic_shifted["timestamp"] - shift
    cal_g, vic_g = synchronize(cal_df, vic_shifted, target_fps=fs_grid)

    def pts(df, landmark):
        cols = [f"{landmark}_{a}" for a in "xyz"]
        if not all(c in df.columns for c in cols):
            return None
        return df[cols].to_numpy(float)

    cal_pts, vic_pts = {}, {}
    for jname, landmark in joints.items():
        c, v = pts(cal_g, landmark), pts(vic_g, landmark)
        if c is None or v is None:
            continue
        cal_pts[jname], vic_pts[jname] = c, v

    # estimate one transform on all common, NaN-free samples (concatenated)
    A, B = [], []
    common = sorted(set(cal_pts) & set(vic_pts))
    for j in common:
        m = ~(np.isnan(cal_pts[j]).any(1) | np.isnan(vic_pts[j]).any(1))
        A.append(vic_pts[j][m])
        B.append(cal_pts[j][m])
    scale = None
    if A and sum(len(a) for a in A) >= 3:
        R, T, scale = estimate_rigid_transform(np.concatenate(A), np.concatenate(B))
        vic_pts = {j: apply_transform(vic_pts[j], R, T, scale) for j in vic_pts}
    return cal_pts, vic_pts, {"time_shift_s": float(shift),
                              "scale": None if scale is None else float(scale)}


def estimate_time_shift_xcorr(
    t_a,
    pts_a,
    t_b,
    pts_b,
    fs_grid: float = 100.0,
    max_shift: float = 0.5,
) -> float:
    """Lag (seconds) such that t_b - shift aligns stream B onto stream A.

    Cross-correlates the 3D speed of a shared point (e.g. pelvis). Ported from
    VvsC.estimate_time_shift_by_xcorr.
    """
    t_a, t_b = np.asarray(t_a, float), np.asarray(t_b, float)
    ta_s, sa = _speed(t_a, np.asarray(pts_a, float))
    tb_s, sb = _speed(t_b, np.asarray(pts_b, float))
    start, end = max(ta_s[0], tb_s[0]), min(ta_s[-1], tb_s[-1])
    if end - start <= 2.0 / fs_grid:
        raise ValueError("overlap too small to estimate time shift")
    grid = np.arange(start, end, 1.0 / fs_grid)
    ga = np.interp(grid, ta_s, sa)
    gb = np.interp(grid, tb_s, sb)
    ga = (ga - ga.mean()) / (ga.std() or 1.0)
    gb = (gb - gb.mean()) / (gb.std() or 1.0)
    corr = np.correlate(gb, ga, mode="full")
    lags = np.arange(-len(ga) + 1, len(gb))
    mlag = int(max_shift * fs_grid)
    mask = (lags >= -mlag) & (lags <= mlag)
    best = lags[mask][int(np.argmax(corr[mask]))]
    return best / fs_grid
