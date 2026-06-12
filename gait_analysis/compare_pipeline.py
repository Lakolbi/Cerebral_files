"""Shared Vicon-comparison pipeline: one implementation for cli.py and the GUI worker."""
import logging

import numpy as np

from modules.comparison.events import gait_cycle_events
from modules.comparison.metrics import angle_comparison_report, position_comparison_report
from modules.comparison.report import build_report
from modules.kinematics.filters import fill_gaps
from modules.kinematics.joint_angles import calc_joint_angles_timeseries
from modules.kinematics.normalizer import get_mean_std_cycle, normalize_gait_cycle
from pipeline import _filter_coords  # reuse the Phase-1 coord filter

logger = logging.getLogger(__name__)

_JOINTS = ("hip", "knee", "ankle")


def _joint_curves(df, cfg, fps):
    """Return {"<side>_<joint>": mean 101-pt curve} for joints with valid data.

    All-NaN angle columns (e.g. Vicon hip, which needs a shoulder marker) are skipped.
    """
    proc, g = cfg["processing"], cfg["gait_events"]
    df = fill_gaps(df, max_gap_frames=proc["max_gap_frames"])
    df = _filter_coords(df, proc["filter_cutoff_hz"], proc["filter_order"], fps)
    # Coordinate-based (Zeni) heel-strike anchoring -- robust on low-fps markerless data
    # where vertical-minima detection mis-anchors the cycle (see modules.comparison.events).
    events = gait_cycle_events(
        df, fps, heel=g["heel_landmark"],
        min_stride_sec=proc["min_stride_duration_sec"], cutoff_hz=proc["filter_cutoff_hz"],
    )
    df = calc_joint_angles_timeseries(df)
    curves, cycles = {}, {}
    for side in ("left", "right"):
        if not events.get(f"{side}_HS"):
            continue
        for joint in _JOINTS:
            col = f"{side}_{joint}_angle"
            if col not in df.columns:
                continue
            arr = df[col].to_numpy()
            if np.isnan(arr).all():
                continue
            mat = normalize_gait_cycle(arr, events, side=side)
            if mat.shape[0] == 0:
                continue
            mean, _ = get_mean_std_cycle(mat)
            if np.isnan(mean).all():
                continue
            curves[f"{side}_{joint}"] = mean
            cycles[f"{side}_{joint}"] = mat
    return curves, cycles


def run_comparison(cal_df, vic_df, cfg, *, model, pair_id, progress_cb=None):
    """Compare one caliscope stream vs one Vicon stream → (report, artifacts).

    Angle layer (primary): same Module-2 kinematics on both → matched 101-pt curves.
    Position layer is added in a later task; here it is reported empty when unavailable.
    artifacts carries per-cycle matrices so callers (validate-vicon) can pool across clips.
    """
    import pandas as pd

    ccmp = cfg["comparison"]

    def report(frac, stage):
        if progress_cb is not None:
            progress_cb(frac, stage)

    cal_fps = cal_df.attrs.get("fps")
    vic_fps = vic_df.attrs.get("fps", ccmp.get("vicon_fps", 100.0))

    report(0.15, "Caliscope kinematics")
    cal_curves, cal_cycles = _joint_curves(cal_df, cfg, cal_fps)
    report(0.45, "Vicon kinematics")
    vic_curves, vic_cycles = _joint_curves(vic_df, cfg, vic_fps)

    report(0.70, "Angle metrics")
    angle_df = angle_comparison_report(
        cal_curves, vic_curves,
        good=ccmp["good_rmse_threshold_deg"],
        acceptable=ccmp["acceptable_rmse_threshold_deg"],
        icc_type=ccmp["icc_type"],
    )
    overlay = {
        j: {"caliscope": list(map(float, cal_curves[j])),
            "vicon": list(map(float, vic_curves[j]))}
        for j in set(cal_curves) & set(vic_curves)
    }

    report(0.85, "Position metrics")
    from modules.comparison.alignment import align_streams

    pos_joints = {"LHIP": "left_hip", "RHIP": "right_hip", "LKNE": "left_knee",
                  "RKNE": "right_knee", "LANK": "left_ankle", "RANK": "right_ankle"}
    try:
        cal_pts, vic_pts, align_info = align_streams(
            cal_df, vic_df, pos_joints,
            fs_grid=ccmp.get("vicon_fps", 100.0), max_shift=0.5)
        position_df = position_comparison_report(cal_pts, vic_pts)
    except (ValueError, KeyError) as exc:
        logger.warning("position layer skipped (%s): %s", type(exc).__name__, exc)
        position_df, align_info = pd.DataFrame(), {"time_shift_s": None, "scale": None}
    # scale is None when no rigid+scale transform could be fit (too few NaN-free
    # samples, or the except path above). The Vicon points are then un-registered,
    # so positional meters are meaningless -> drop them and flag low confidence.
    low_confidence = align_info.get("scale") is None
    if low_confidence:
        position_df = pd.DataFrame()

    report(0.95, "Building report")
    rep = build_report(angle_df, position_df, overlay, meta={
        "pair_id": pair_id, "model": model,
        "caliscope_fps": None if cal_fps is None else round(float(cal_fps), 3),
        "vicon_fps": round(float(vic_fps), 3),
        "time_shift_s": align_info.get("time_shift_s"),
        "scale": align_info.get("scale"),
        "low_confidence": low_confidence,
    })
    report(1.0, "Done")
    artifacts = {"cal_cycles": cal_cycles, "vic_cycles": vic_cycles}
    return rep, artifacts
