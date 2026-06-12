"""Normalize signals to the 0-100% gait cycle (101 points)."""
import numpy as np


def normalize_gait_cycle(signal: np.ndarray, events: dict, side: str,
                         n_points: int = 101) -> np.ndarray:
    """Split ``signal`` into HS->HS cycles and resample each to ``n_points``.

    Returns a ``[n_cycles x n_points]`` matrix.
    """
    signal = np.asarray(signal, dtype=float)
    hs = sorted(events.get(f"{side}_HS", []))
    target = np.linspace(0.0, 1.0, n_points)
    cycles = []
    for start, end in zip(hs[:-1], hs[1:]):
        seg = signal[start:end + 1]
        if len(seg) < 2:
            continue
        src = np.linspace(0.0, 1.0, len(seg))
        cycles.append(np.interp(target, src, seg))
    if not cycles:
        return np.empty((0, n_points))
    return np.vstack(cycles)


def get_mean_std_cycle(cycles_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation across cycles (NaN-aware), per cycle point."""
    if cycles_matrix.size == 0:
        n = cycles_matrix.shape[1] if cycles_matrix.ndim == 2 else 101
        return np.full(n, np.nan), np.full(n, np.nan)
    mean = np.nanmean(cycles_matrix, axis=0)
    std = np.nanstd(cycles_matrix, axis=0)
    return mean, std
