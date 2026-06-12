"""Joint-angle computation from 3D landmark triples."""
import numpy as np
import pandas as pd


def calc_angle_3d(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    """Included angle (degrees, 0-180) between vectors (vertex->p1) and (vertex->p2)."""
    p1 = np.asarray(p1, float)
    vertex = np.asarray(vertex, float)
    p2 = np.asarray(p2, float)
    if np.isnan(np.concatenate([p1, vertex, p2])).any():
        return float("nan")
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# (proximal, vertex, distal) landmark triples per joint angle column.
_ANGLE_DEFS = {
    "left_hip_angle":    ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle":   ("right_shoulder", "right_hip", "right_knee"),
    "left_knee_angle":   ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle":  ("right_hip", "right_knee", "right_ankle"),
    "left_ankle_angle":  ("left_knee", "left_ankle", "left_foot_index"),
    "right_ankle_angle": ("right_knee", "right_ankle", "right_foot_index"),
}


def _point(df: pd.DataFrame, name: str) -> np.ndarray | None:
    cols = [f"{name}_x", f"{name}_y", f"{name}_z"]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].to_numpy()


def calc_joint_angles_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Append per-frame sagittal joint angles (included angle, degrees 0-180).

    Columns added: ``<side>_{hip,knee,ankle}_angle``. An angle is NaN for any
    frame where a required landmark is missing or its triple is unavailable.
    Pelvis angles (tilt/obliquity/rotation) are out of scope for Phase 1.
    """
    out = df.copy()
    n = len(df)
    for col, (a, v, b) in _ANGLE_DEFS.items():
        pa, pv, pb = _point(df, a), _point(df, v), _point(df, b)
        if pa is None or pv is None or pb is None:
            out[col] = np.full(n, np.nan)
            continue
        out[col] = np.array([calc_angle_3d(pa[i], pv[i], pb[i]) for i in range(n)])
    return out
