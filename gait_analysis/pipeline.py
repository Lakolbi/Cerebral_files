"""Shared analysis pipeline: one implementation for both cli.py and the GUI worker."""
import datetime as dt

import numpy as np

from modules.kinematics.filters import butterworth_filter, fill_gaps
from modules.kinematics.gait_events import detect_gait_events
from modules.kinematics.joint_angles import calc_joint_angles_timeseries
from modules.kinematics.normalizer import get_mean_std_cycle, normalize_gait_cycle
from modules.kinematics.spatiotemporal import calc_spatiotemporal


def _filter_coords(df, cutoff_hz, order, fps):
    """Filter coordinate columns; skip columns with residual NaN or too-short for filtfilt."""
    out = df.copy()
    min_len = 3 * (order + 1)
    for c in out.columns:
        if not c.endswith(("_x", "_y", "_z")):
            continue
        arr = out[c].to_numpy()
        if np.isnan(arr).any() or len(arr) <= min_len:
            continue
        out[c] = butterworth_filter(arr, cutoff_hz=cutoff_hz, fs=fps, order=order)
    return out


def run_pipeline(df, cfg, *, model, session_id, progress_cb=None):
    """Run the full kinematics pipeline on a loaded pose DataFrame.

    df          : unified pose DataFrame from load_caliscope_session (carries fps in df.attrs).
    cfg         : settings dict (processing / gait_events / spatiotemporal sections).
    model       : model name, stamped into results.
    session_id  : session identifier (e.g. folder name), stamped into results.
    progress_cb : optional callable(fraction: float, stage: str); no-op if None.

    Returns (results_dict, df_processed). results_dict is the canonical gait_results.json schema.
    """
    proc = cfg["processing"]
    gcfg = cfg["gait_events"]
    scfg = cfg.get("spatiotemporal", {})

    def report(frac, stage):
        if progress_cb is not None:
            progress_cb(frac, stage)

    fps = df.attrs["fps"]

    report(0.10, "Filling gaps")
    df = fill_gaps(df, max_gap_frames=proc["max_gap_frames"])
    report(0.25, "Filtering")
    df = _filter_coords(df, proc["filter_cutoff_hz"], proc["filter_order"], fps)

    report(0.45, "Detecting gait events")
    events = detect_gait_events(
        df, fps=fps, method=gcfg["method"], heel=gcfg["heel_landmark"],
        toe=gcfg["toe_landmark"], vertical=gcfg["vertical_axis"],
        min_stride_sec=proc["min_stride_duration_sec"],
        cutoff_hz=proc["filter_cutoff_hz"],
    )
    report(0.60, "Joint angles")
    df = calc_joint_angles_timeseries(df)
    report(0.80, "Spatiotemporal")
    spatiotemporal = calc_spatiotemporal(
        df, events, fps=fps, vertical=gcfg["vertical_axis"],
        max_stride_m=scfg.get("max_stride_m", 1.5),
        max_step_m=scfg.get("max_step_m", 1.0),
    )

    report(0.90, "Normalizing cycles")
    angles_mean, angles_std = {}, {}
    for side in ("left", "right"):
        for joint in ("hip", "knee", "ankle"):
            col = f"{side}_{joint}_angle"
            if col not in df.columns or not events.get(f"{side}_HS"):
                continue
            mat = normalize_gait_cycle(df[col].to_numpy(), events, side=side)
            mean, std = get_mean_std_cycle(mat)
            if mat.shape[0] > 0:
                angles_mean[f"{side}_{joint}"] = np.nan_to_num(mean, nan=0.0).round(3).tolist()
                angles_std[f"{side}_{joint}"] = np.nan_to_num(std, nan=0.0).round(3).tolist()

    results = {
        "session_id": session_id,
        "model": model,
        "fps": round(float(fps), 3),
        "n_frames": int(len(df)),
        "processed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "gait_events": {k: list(map(int, v)) for k, v in events.items()},
        "spatiotemporal": {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                               else v if isinstance(v, int)
                               else round(float(v), 3))
                           for k, v in spatiotemporal.items()},
        "joint_angles_mean": angles_mean,
        "joint_angles_std": angles_std,
    }
    report(1.0, "Done")
    return results, df
