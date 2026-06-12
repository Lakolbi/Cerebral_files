from pathlib import Path

import numpy as np
import pytest

from modules.data_loader.config_reader import load_camera_config

FIX = Path(__file__).parent / "fixtures"


def test_load_camera_config_parses_intrinsics_and_extrinsics():
    cfg = load_camera_config(str(FIX / "mini_config.toml"))
    assert "cam_1" in cfg
    intr = cfg["cam_1"]["intrinsics"]
    assert intr["fx"] == pytest.approx(1129.0)
    assert intr["fy"] == pytest.approx(1125.6)
    assert intr["cx"] == pytest.approx(949.7)
    assert intr["cy"] == pytest.approx(515.0)
    assert len(intr["distortion"]) == 5
    extr = cfg["cam_1"]["extrinsics"]
    assert extr["R"].shape == (3, 3)
    assert extr["T"].shape == (3,)
    assert np.allclose(extr["R"] @ extr["R"].T, np.eye(3), atol=1e-6)
