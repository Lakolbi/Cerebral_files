"""Robust gait-event (heel-strike) detection for cross-system comparison.

The Phase-1 detector (``modules.kinematics.gait_events``) keys heel-strike off heel
*vertical* minima. On low-frame-rate markerless caliscope data that mis-anchors the
gait cycle: cycles came out non-physiological (peak knee flexion at ~20-95% of the
cycle instead of the textbook ~73%), which collapsed the caliscope-vs-Vicon angle
agreement.

This module uses the coordinate-based method of Zeni et al. (2008): heel-strike is
the instant the heel is most *anterior* relative to the pelvis. It is robust across
frame rates and yields physiological cycles for both caliscope (overground) and Vicon
(treadmill). Vicon's vertical-minima cycles were already physiological; this method
keeps them so while fixing caliscope. It is used only by the comparison pipeline; the
Phase-1 analysis pipeline keeps its own detector (its spatiotemporal results depend on
that detector's timing and are validated separately).
"""
import numpy as np
from scipy.signal import find_peaks

from modules.kinematics.filters import butterworth_filter


def _pelvis_xy(df):
    left = df[["left_hip_x", "left_hip_y"]].to_numpy()
    right = df[["right_hip_x", "right_hip_y"]].to_numpy()
    return 0.5 * (left + right)


def detect_heel_strikes(df, fps, side, heel="heel", min_stride_sec=0.8,
                        cutoff_hz=6.0):
    """Heel-strike frame indices for one side (Zeni 2008 anterior-extremum method).

    The heel-minus-pelvis horizontal vector is projected onto its dominant
    (anterior-posterior) axis. Stance (~60% of the stride) lasts longer than swing and
    the planted heel drifts posterior relative to the advancing pelvis, so the AP signal
    spends most of its time decreasing; we orient the axis to satisfy that (resolving
    its arbitrary sign), then heel-strikes are the positive peaks (heel most anterior).

    Returns a sorted list of frame indices ([] if the heel marker is absent or flat).
    """
    cols = [f"{side}_{heel}_x", f"{side}_{heel}_y"]
    hip_cols = ["left_hip_x", "left_hip_y", "right_hip_x", "right_hip_y"]
    if not all(c in df.columns for c in cols + hip_cols):
        return []
    rel = df[cols].to_numpy() - _pelvis_xy(df)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(rel, axis=0)
    if not np.isfinite(sd).any() or np.nanmax(sd) == 0:
        return []
    ap = rel[:, int(np.nanargmax(sd))]
    finite = np.isfinite(ap)
    if finite.sum() < 4:
        return []
    ap = np.nan_to_num(ap, nan=float(np.nanmean(ap[finite])))
    try:
        ap = butterworth_filter(ap, cutoff_hz=cutoff_hz, fs=fps)
    except ValueError:
        return []          # too few samples for zero-phase filtering
    # Orient so stance (the longer, decreasing phase) dominates; flip a sign-reversed axis.
    if np.mean(np.diff(ap) > 0) > 0.5:
        ap = -ap
    distance = max(1, int(min_stride_sec * fps))
    hs, _ = find_peaks(ap, distance=distance)
    return sorted(hs.tolist())


def gait_cycle_events(df, fps, sides=("left", "right"), heel="heel",
                      min_stride_sec=0.8, cutoff_hz=6.0):
    """``{f'{side}_HS': [...]}`` heel-strike anchors for the requested sides."""
    return {
        f"{s}_HS": detect_heel_strikes(df, fps, s, heel=heel,
                                       min_stride_sec=min_stride_sec, cutoff_hz=cutoff_hz)
        for s in sides
    }
