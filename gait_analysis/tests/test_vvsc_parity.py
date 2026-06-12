import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

GAIT = Path(__file__).resolve().parents[1]
REPO = GAIT.parent
VICON1 = GAIT / "data" / "Vicon_10_series" / "1.xlsx"
CAL1 = (GAIT / "data" / "caliscope_project" / "recordings" / "p1_1"
        / "SIMPLE_HOLISTIC" / "xyz_SIMPLE_HOLISTIC_labelled.csv")


@pytest.mark.skipif(not (REPO / "VvsC.py").exists(), reason="VvsC.py not present")
def test_our_umeyama_matches_vvsc_similarity_exactly():
    """Core-algorithm parity (criterion #1): same (R, T, s) as VvsC, no real data."""
    sys.path.insert(0, str(REPO))
    import VvsC

    from modules.comparison.alignment import estimate_rigid_transform

    rng = np.random.default_rng(7)
    A = rng.normal(size=(60, 3))
    B = rng.normal(size=(60, 3))
    s_v, R_v, t_v = VvsC.compute_similarity_transform(A, B)   # B ≈ s*A@R.T + t
    R, T, s = estimate_rigid_transform(A, B)                  # B ≈ s*(A@R.T) + T
    assert np.isclose(s, s_v, atol=1e-9)
    assert np.allclose(R, R_v, atol=1e-9)
    assert np.allclose(T, t_v, atol=1e-9)


@pytest.mark.skipif(
    not (VICON1.exists() and CAL1.exists() and (REPO / "VvsC.py").exists()),
    reason="real Vicon/caliscope data or VvsC.py not present",
)
def test_position_layer_rmse_in_vvsc_ballpark():
    """End-to-end positional RMSE is the same order as VvsC (approximate: our
    interp/window choices differ; the exact-parity test above is the criterion-#1 proof)."""
    sys.path.insert(0, str(REPO))
    import VvsC

    baseline = VvsC.compute_errors_for_trial(
        str(VICON1), str(CAL1), vicon_fps=100.0, cal_fps=20.0, event_mode="xcorr")
    expected = baseline["global_metrics"]["rmse_m"]

    from compare_pipeline import run_comparison
    from modules.data_loader.caliscope_reader import load_caliscope_session
    from modules.data_loader.vicon_reader import load_vicon_xlsx, map_vicon_to_caliscope

    cfg = yaml.safe_load(open(GAIT / "settings.yaml"))
    cal = load_caliscope_session(str(CAL1.parents[1]), model="SIMPLE_HOLISTIC")
    vic = map_vicon_to_caliscope(load_vicon_xlsx(str(VICON1)), cfg["landmark_mapping"])
    rep, _ = run_comparison(cal, vic, cfg, model="SIMPLE_HOLISTIC", pair_id="p1_1__1")

    joints = rep["position"]["joints"]
    assert joints, "position layer produced no joints"
    got = np.sqrt(np.mean([j["rmse_m"] ** 2 for j in joints.values()]))
    assert got == pytest.approx(expected, rel=0.5)        # same order of magnitude
