"""Detect heel-strike (HS) and toe-off (TO) gait events."""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .filters import butterworth_filter


def _sides(side: str) -> list[str]:
    return ["left", "right"] if side == "both" else [side]


def _clean(a: np.ndarray) -> np.ndarray:
    """Linear-interpolate residual NaN (both directions) so filtfilt can run."""
    return pd.Series(np.asarray(a, dtype=float)).interpolate(
        limit_direction="both").to_numpy()


def _detect_one_side(
    df: pd.DataFrame,
    fps: float,
    side: str,
    method: str,
    heel: str,
    toe: str,
    vertical: str,
    min_stride_sec: float,
    cutoff_hz: float,
) -> tuple[list, list]:
    heel_col = f"{side}_{heel}_{vertical}"
    toe_col = f"{side}_{toe}_{vertical}"
    heel_v = butterworth_filter(_clean(df[heel_col].to_numpy()), cutoff_hz=cutoff_hz, fs=fps)
    toe_v = butterworth_filter(_clean(df[toe_col].to_numpy()), cutoff_hz=cutoff_hz, fs=fps)
    min_dist = max(1, int(min_stride_sec * fps))

    # Heel strike = local minima of heel vertical = peaks of -heel_v.
    hs, _ = find_peaks(-heel_v, distance=min_dist)
    # Toe off = local maxima of toe vertical.
    to, _ = find_peaks(toe_v, distance=min_dist)
    return hs.tolist(), to.tolist()


def detect_gait_events(
    df: pd.DataFrame,
    fps: float,
    side: str = "both",
    method: str = "velocity",
    heel: str = "heel",
    toe: str = "foot_index",
    vertical: str = "z",
    min_stride_sec: float = 0.8,
    cutoff_hz: float = 6.0,
) -> dict:
    """Return ``{<side>_HS, <side>_TO}`` frame-index lists.

    ``velocity``/``height`` both reduce to vertical-trajectory extrema for the
    Phase-1 lower-body landmarks; ``method`` is accepted for forward
    compatibility. Same-event detections closer than ``min_stride_sec`` are
    suppressed via the ``distance`` constraint in ``find_peaks``. NOTE:
    ``min_stride_sec`` is a full same-foot stride interval (HS->HS), typically
    ~0.8-1.2 s for normal walking -- NOT a step interval. Too small a value
    (e.g. 0.3 s) lets ``find_peaks`` fire multiple spurious events per stride.
    """
    result: dict = {}
    for s in _sides(side):
        hs, to = _detect_one_side(df, fps, s, method, heel, toe, vertical,
                                  min_stride_sec, cutoff_hz)
        result[f"{s}_HS"] = hs
        result[f"{s}_TO"] = to
    return result
