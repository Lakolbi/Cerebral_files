"""Resample two unified DataFrames onto a common time grid."""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def _coord_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]


def _resample(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    if len(df) < 2:
        raise ValueError(f"need at least 2 samples to synchronize, got {len(df)}")
    kind = "cubic" if len(df) >= 4 else "linear"
    t = df["timestamp"].to_numpy()
    out = {"frame": range(len(grid)), "timestamp": grid}
    for c in _coord_cols(df):
        y = df[c].to_numpy()
        f = interp1d(t, y, kind=kind, bounds_error=False, fill_value=np.nan)
        out[c] = f(grid)
    return pd.DataFrame(out)


def synchronize(df_a: pd.DataFrame, df_b: pd.DataFrame,
                target_fps: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interpolate both DataFrames onto a shared grid over their overlap.

    Precondition: each ``timestamp`` column is in seconds and monotonic.
    The grid is half-open [start, end); the final overlap instant is excluded.
    Streams with fewer than 4 samples use linear interpolation.
    """
    start = max(df_a["timestamp"].iloc[0], df_b["timestamp"].iloc[0])
    end = min(df_a["timestamp"].iloc[-1], df_b["timestamp"].iloc[-1])
    if end <= start:
        raise ValueError("DataFrames do not overlap in time")
    n_pts = int(np.floor((end - start) * target_fps))
    grid = start + np.arange(n_pts) / target_fps
    return _resample(df_a, grid), _resample(df_b, grid)
