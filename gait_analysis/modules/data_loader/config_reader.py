"""Parse caliscope config.toml camera intrinsics/extrinsics."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_camera_config(config_path: str) -> dict:
    """Return ``{cam_key: {'intrinsics': {...}, 'extrinsics': {'R','T'}}}``.

    ``rotation`` in config.toml is a Rodrigues (axis-angle) vector; it is
    converted to a 3x3 rotation matrix.
    """
    with open(Path(config_path), "rb") as f:
        raw = tomllib.load(f)
    out: dict = {}
    for key, val in raw.items():
        if not key.startswith("cam_") or not isinstance(val, dict):
            continue
        try:
            m = val["matrix"]
            out[key] = {
                "intrinsics": {
                    "fx": m[0][0], "fy": m[1][1], "cx": m[0][2], "cy": m[1][2],
                    "distortion": list(val["distortions"]),
                },
                "extrinsics": {
                    "R": Rotation.from_rotvec(val["rotation"]).as_matrix(),
                    "T": np.asarray(val["translation"], dtype=float),
                },
            }
        except KeyError as exc:
            raise KeyError(f"Camera {key!r} is missing config field {exc}") from exc
    return out
