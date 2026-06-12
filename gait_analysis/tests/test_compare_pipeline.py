import numpy as np
import pandas as pd
import yaml

from compare_pipeline import run_comparison


def _walking_df(n=240, fps=60.0):
    """A synthetic lower-body walker: oscillating knee/ankle so events + angles exist."""
    t = np.arange(n) / fps
    df = pd.DataFrame({"frame": range(n), "timestamp": t})
    phase = 2 * np.pi * 1.0 * t                      # ~1 Hz stride
    # vertical (z) bounce drives heel-strike detection; x advances forward
    for side, ph in (("left", 0.0), ("right", np.pi)):
        df[f"{side}_hip_x"] = 0.4 * t
        df[f"{side}_hip_y"] = 0.0
        df[f"{side}_hip_z"] = 0.9 + 0.02 * np.sin(phase + ph)
        df[f"{side}_knee_x"] = 0.4 * t + 0.05 * np.sin(phase + ph)
        df[f"{side}_knee_y"] = 0.0
        df[f"{side}_knee_z"] = 0.5 + 0.05 * np.cos(phase + ph)
        df[f"{side}_ankle_x"] = 0.4 * t
        df[f"{side}_ankle_y"] = 0.0
        df[f"{side}_ankle_z"] = 0.1 + 0.04 * np.sin(phase + ph)
        df[f"{side}_heel_z"] = 0.08 + 0.05 * (np.sin(phase + ph) ** 2)
        df[f"{side}_foot_index_z"] = 0.05 + 0.05 * (np.cos(phase + ph) ** 2)
        # heel swings anterior-posterior relative to the pelvis each stride (drives the
        # Zeni coordinate-based heel-strike detector used by the comparison pipeline)
        df[f"{side}_heel_x"] = 0.4 * t + 0.25 * np.sin(phase + ph)
        df[f"{side}_heel_y"] = 0.0
        df[f"{side}_foot_index_x"] = 0.4 * t + 0.25 * np.sin(phase + ph)
        df[f"{side}_foot_index_y"] = 0.0
    df.attrs["fps"] = fps
    return df


def test_run_comparison_identical_streams_perfect_angle_agreement():
    cfg = yaml.safe_load(open("settings.yaml"))
    cal = _walking_df()
    vic = cal.copy()
    vic.attrs["fps"] = cal.attrs["fps"]               # true copy (df.copy may drop attrs)
    report, _ = run_comparison(cal, vic, cfg, model="SIMPLE_HOLISTIC", pair_id="synthetic")
    assert report["angle"], "expected at least one comparable joint"
    # knee/ankle present in both; hip excluded (no shoulder markers here)
    assert all(j in ("left_knee", "right_knee", "left_ankle", "right_ankle")
               for j in report["angle"])
    knee = report["angle"].get("left_knee") or report["angle"].get("right_knee")
    assert knee["rmse_deg"] < 1.0                     # same motion -> near-zero angle error


def test_run_comparison_flags_low_confidence_when_alignment_fails():
    cfg = yaml.safe_load(open("settings.yaml"))
    cal = _walking_df()
    vic = _walking_df()
    vic["timestamp"] = vic["timestamp"] + 100.0       # no temporal overlap -> xcorr fails
    report, _ = run_comparison(cal, vic, cfg, model="SIMPLE_HOLISTIC", pair_id="noalign")
    # registration could not be fit -> no (garbage) position metrics, flagged low-confidence
    assert report["position"]["joints"] == {}
    assert report["position"]["low_confidence"] is True
    assert report["angle"]                            # angle layer is unaffected
