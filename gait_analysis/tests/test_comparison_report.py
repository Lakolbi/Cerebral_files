import numpy as np
import pandas as pd

from modules.comparison.report import build_report


def test_build_report_schema():
    angle = pd.DataFrame([
        {"joint": "left_knee", "rmse_deg": 3.0, "mae_deg": 2.0, "pearson": 0.99,
         "icc": 0.9, "verdict": "good"},
    ])
    position = pd.DataFrame([
        {"joint": "LKNE", "n_samples": 100, "rmse_m": 0.02, "mae_m": 0.01,
         "max_m": 0.05, "median_m": 0.01, "rmse_x_m": 0.01, "rmse_y_m": 0.01,
         "rmse_z_m": 0.01},
    ])
    overlay = {"left_knee": {"caliscope": list(np.zeros(101)),
                             "vicon": list(np.zeros(101))}}
    rep = build_report(angle, position, overlay,
                       meta={"pair_id": "p1_1__vicon1", "model": "SIMPLE_HOLISTIC",
                             "caliscope_fps": 19.0, "vicon_fps": 100.0,
                             "time_shift_s": 0.1, "scale": 1.0, "low_confidence": False})
    assert rep["pair_id"] == "p1_1__vicon1"
    assert rep["angle"]["left_knee"]["verdict"] == "good"
    assert rep["position"]["joints"]["LKNE"]["rmse_m"] == 0.02
    assert "processed_at" in rep
    assert rep["angle_overlay"]["left_knee"]["caliscope"][0] == 0.0
    assert rep["verdict_summary"]                      # non-empty string
