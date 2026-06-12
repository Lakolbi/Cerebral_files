"""Signal conditioning: gap filling and zero-phase Butterworth filtering."""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

_COORD_SUFFIXES = ("_x", "_y", "_z")


def _is_coord_column(name: str) -> bool:
    return name.endswith(_COORD_SUFFIXES)


def _nan_run_bounds(isna: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) index ranges of consecutive-NaN runs."""
    runs = []
    i, n = 0, len(isna)
    while i < n:
        if isna[i]:
            j = i
            while j < n and isna[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def fill_gaps(df: pd.DataFrame, max_gap_frames: int = 10) -> pd.DataFrame:
    """Cubic-interpolate interior NaN runs no longer than ``max_gap_frames``.

    All-or-nothing per gap: a NaN run longer than ``max_gap_frames`` stays
    fully NaN. Leading/trailing NaNs are left untouched. Only coordinate
    columns (``*_x/_y/_z``) are processed. Call this BEFORE filtering.
    """
    out = df.copy()
    for col in out.columns:
        if not _is_coord_column(col):
            continue
        s = out[col]
        isna = s.isna().to_numpy()
        if s.notna().sum() < 4:  # cubic needs >=4 valid points
            continue
        # Fully interpolate interior gaps, then restore NaN over over-long runs.
        filled = s.interpolate(method="cubic", limit_area="inside").to_numpy().copy()
        for start, end in _nan_run_bounds(isna):
            if (end - start) > max_gap_frames:
                filled[start:end] = np.nan
        out[col] = filled
    return out


def butterworth_filter(signal: np.ndarray, *, cutoff_hz: float = 6.0,
                       fs: float, order: int = 4,
                       zero_phase: bool = True) -> np.ndarray:
    """Low-pass Butterworth filter. Default ``zero_phase`` uses ``filtfilt``
    (no phase lag). ``signal`` must be NaN-free (run ``fill_gaps`` first).

    Raises ``ValueError`` if ``cutoff_hz`` is not below the Nyquist frequency,
    or (zero-phase only) if the signal is too short for ``filtfilt`` padding.
    """
    signal = np.asarray(signal, dtype=float)
    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
        raise ValueError(
            f"cutoff_hz ({cutoff_hz}) must be below the Nyquist frequency ({nyq} Hz)"
        )
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    if not zero_phase:
        from scipy.signal import lfilter
        return lfilter(b, a, signal)
    min_len = 3 * (order + 1)  # scipy filtfilt requires len(x) > padlen
    if len(signal) <= min_len:
        raise ValueError(
            f"signal length {len(signal)} too short for zero-phase filtfilt "
            f"with order={order}; need > {min_len} samples"
        )
    return filtfilt(b, a, signal, padtype="odd")
