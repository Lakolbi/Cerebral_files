import numpy as np
import pandas as pd

from pipeline import run_pipeline

CFG = {
    "processing": {"max_gap_frames": 10, "filter_cutoff_hz": 6.0,
                   "filter_order": 4, "min_stride_duration_sec": 0.8},
    "gait_events": {"method": "velocity", "heel_landmark": "heel",
                    "toe_landmark": "foot_index", "vertical_axis": "z"},
    "spatiotemporal": {"max_stride_m": 1.5, "max_step_m": 1.0},
}


def _synthetic_walk(n=120, fps=20.0):
    """A few synthetic gait cycles: oscillating heel/toe verticals + moving hips."""
    t = np.arange(n) / fps
    z = 0.05 * np.sin(2 * np.pi * 1.0 * t)        # ~1 Hz step cadence
    df = pd.DataFrame({"timestamp": t})
    for side in ("left", "right"):
        phase = 0.0 if side == "left" else np.pi
        for name in ("hip", "knee", "ankle", "heel", "foot_index", "shoulder"):
            df[f"{side}_{name}_x"] = 0.2 * t
            df[f"{side}_{name}_y"] = 0.1 if side == "left" else -0.1
            df[f"{side}_{name}_z"] = z * np.cos(phase) + {"hip": 0.9, "knee": 0.5,
                "ankle": 0.1, "heel": 0.05, "foot_index": 0.0, "shoulder": 1.4}[name]
    df.attrs["fps"] = fps
    return df


def test_run_pipeline_returns_canonical_schema():
    df = _synthetic_walk()
    results, df_out = run_pipeline(df, CFG, model="SIMPLE_HOLISTIC", session_id="synth")
    for key in ("session_id", "model", "fps", "n_frames", "processed_at",
                "gait_events", "spatiotemporal", "joint_angles_mean", "joint_angles_std"):
        assert key in results
    assert results["session_id"] == "synth"
    assert results["model"] == "SIMPLE_HOLISTIC"
    assert results["n_frames"] == len(df)
    assert "left_knee_angle" in df_out.columns      # angles appended to returned df


def test_run_pipeline_calls_progress_callback_to_completion():
    df = _synthetic_walk()
    seen = []
    run_pipeline(df, CFG, model="SIMPLE_HOLISTIC", session_id="synth",
                 progress_cb=lambda frac, stage: seen.append((frac, stage)))
    fracs = [f for f, _ in seen]
    assert fracs == sorted(fracs)        # monotonically non-decreasing
    assert fracs[-1] == 1.0              # reaches completion


def test_spatiotemporal_values_are_json_serializable():
    import json
    df = _synthetic_walk()
    results, _ = run_pipeline(df, CFG, model="SIMPLE_HOLISTIC", session_id="synth")
    serialized = json.dumps(results)          # must not raise — nan must never reach JSON
    loaded = json.loads(serialized)
    for v in loaded["spatiotemporal"].values():
        assert v is None or isinstance(v, (int, float))
