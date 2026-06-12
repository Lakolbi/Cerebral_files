"""Read caliscope 3D output into the unified pose DataFrame."""
from pathlib import Path

import numpy as np
import pandas as pd

from .landmarks import GAIT_LANDMARKS, MODELS


def list_landmarks(model: str) -> list[str]:
    """Gait-relevant canonical landmarks for a caliscope model."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
    return list(GAIT_LANDMARKS)


def derive_fps(timestamps) -> float:
    """Frame rate = 1 / median positive inter-frame interval."""
    ts = np.sort(np.asarray(timestamps, dtype=float))
    d = np.diff(ts)
    d = d[d > 0]
    if d.size == 0:
        raise ValueError("Cannot derive fps from constant/empty timestamps")
    return float(1.0 / np.median(d))


def _frame_timestamps(frame_time_csv: Path) -> pd.DataFrame:
    """Mean ``frame_time`` per ``sync_index`` (collapsing the per-port rows).

    Not zero-based here: the file may contain sync indices that never made it
    into the labelled 3D output, so zero-basing must happen after the join to
    the labelled frames (in ``load_caliscope_session``).
    """
    fth = pd.read_csv(frame_time_csv)
    per_sync = fth.groupby("sync_index", as_index=False)["frame_time"].mean()
    return per_sync.rename(columns={"frame_time": "timestamp"})


def load_caliscope_session(session_dir: str, model: str = "SIMPLE_HOLISTIC") -> pd.DataFrame:
    """Load ``xyz_<model>_labelled.csv`` + real timestamps into a unified DataFrame.

    Returns columns ``frame, timestamp, <landmark>_{x,y,z}``. Sets
    ``df.attrs['fps']`` (derived) and ``df.attrs['model']``.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
    model_dir = Path(session_dir) / model
    labelled = pd.read_csv(model_dir / f"xyz_{model}_labelled.csv")
    times = _frame_timestamps(model_dir / "frame_time_history.csv")

    merged = labelled.merge(times, on="sync_index", how="left")
    merged = merged.sort_values("sync_index").reset_index(drop=True)
    # Every labelled (triangulated) frame must have a capture-clock entry;
    # a missing one means corrupt input, so fail loud rather than emit NaN time.
    if merged["timestamp"].isna().any():
        bad = merged.loc[merged["timestamp"].isna(), "sync_index"].tolist()
        raise ValueError(f"No frame_time entry for sync_index values: {bad}")
    # Zero-base relative to the frames actually present in this session.
    merged["timestamp"] = merged["timestamp"] - merged["timestamp"].min()

    coord_cols = [c for c in merged.columns if c.endswith(("_x", "_y", "_z"))]
    result = pd.DataFrame({"frame": range(len(merged)),
                           "timestamp": merged["timestamp"].to_numpy()})
    result = pd.concat([result, merged[coord_cols].reset_index(drop=True)], axis=1)

    result.attrs["fps"] = derive_fps(result["timestamp"].to_numpy())
    result.attrs["model"] = model
    return result
