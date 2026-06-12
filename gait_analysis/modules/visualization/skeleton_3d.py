"""GL-free skeleton geometry: connectivity + per-frame point/segment extraction."""
import numpy as np
import pandas as pd

# Drawable skeleton connectivity (canonical lowercase landmark names).
SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"),
]

# The landmarks that actually form the skeleton (edge endpoints). Markers and the
# camera bounding box use this set so stray face/hand landmarks don't clutter the
# view or skew the framing.
SKELETON_LANDMARKS = sorted({name for edge in SKELETON_EDGES for name in edge})


def frame_points(df, frame_idx, names=None):
    """Return {landmark: (x, y, z)} for one frame, omitting missing/NaN landmarks.

    ``names`` restricts the landmarks considered (default: every ``*_x`` column).
    """
    row = df.iloc[frame_idx]
    if names is None:
        names = [c[:-2] for c in df.columns if c.endswith("_x")]
    points = {}
    for name in names:
        cols = (f"{name}_x", f"{name}_y", f"{name}_z")
        if not all(c in df.columns for c in cols):
            continue
        xyz = tuple(row.get(c) for c in cols)
        if any(v is None for v in xyz) or any(pd.isna(v) for v in xyz):
            continue
        points[name] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return points


def bounds(df, names=SKELETON_LANDMARKS):
    """Axis-aligned bounding box of ``names`` over ALL frames (NaN-ignoring).

    Returns ``((xmin, ymin, zmin), (xmax, ymax, zmax))`` or ``None`` if no finite
    points exist. Used to frame the whole walk at a stable scale, so the skeleton
    does not translate out of view as the subject moves across the capture volume.
    """
    arrays = []
    for name in names:
        cols = [f"{name}_x", f"{name}_y", f"{name}_z"]
        if all(c in df.columns for c in cols):
            arrays.append(df[cols].to_numpy(dtype=float))
    if not arrays:
        return None
    pts = np.vstack(arrays)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) == 0:
        return None
    return tuple(pts.min(axis=0)), tuple(pts.max(axis=0))


def segment_lines(points, edges=SKELETON_EDGES):
    """Return [((x,y,z),(x,y,z)), ...] for edges whose endpoints both exist in points."""
    lines = []
    for a, b in edges:
        if a in points and b in points:
            lines.append((points[a], points[b]))
    return lines
