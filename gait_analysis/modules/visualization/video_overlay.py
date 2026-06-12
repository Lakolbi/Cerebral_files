"""Draw the gait skeleton onto raw camera videos from caliscope 2D detections (cv2)."""
from pathlib import Path

import cv2
import pandas as pd

from modules.visualization.skeleton_3d import SKELETON_EDGES

# MediaPipe Pose landmark index -> canonical gait landmark name (the 12 skeleton joints).
POSE_POINT_IDS = {
    11: "left_shoulder", 12: "right_shoulder",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

_JOINT_COLOR = (0, 0, 255)   # red (BGR)
_BONE_COLOR = (0, 0, 0)      # black


def load_xy(session_dir, model):
    """Read the per-camera 2D detections CSV for a model."""
    path = Path(session_dir) / model / f"xy_{model}.csv"
    if not path.exists():
        raise FileNotFoundError(f"2D detections not found: {path}")
    return pd.read_csv(path)


def frame_marks(xy_df, port, frame_index):
    """Return {gait_landmark: (x, y)} for one camera frame; missing/NaN landmarks omitted."""
    rows = xy_df[(xy_df["port"] == port) & (xy_df["frame_index"] == frame_index)]
    marks = {}
    for _, r in rows.iterrows():
        name = POSE_POINT_IDS.get(int(r["point_id"]))
        if name is None:
            continue
        x, y = r["img_loc_x"], r["img_loc_y"]
        if pd.isna(x) or pd.isna(y):
            continue
        marks[name] = (int(round(x)), int(round(y)))
    return marks


def draw_overlay(frame, marks, edges=SKELETON_EDGES):
    """Draw bones (lines) + joints (filled circles) onto a BGR frame in place; return it."""
    for a, b in edges:
        if a in marks and b in marks:
            cv2.line(frame, marks[a], marks[b], _BONE_COLOR, 3)
    for (x, y) in marks.values():
        cv2.circle(frame, (x, y), 7, _JOINT_COLOR, -1)
        cv2.circle(frame, (x, y), 7, _BONE_COLOR, 1)
    return frame


def produce_marked_video(video_path, xy_df, port, out_path, progress_cb=None):
    """Draw the skeleton on every frame of one camera video; write an annotated mp4.

    Reads the raw video frame-by-frame; frame N gets the marks for (port, frame_index=N).
    Source fps + resolution are preserved. Returns the output Path.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video writer: {out_path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            draw_overlay(frame, frame_marks(xy_df, port, idx))
            writer.write(frame)
            idx += 1
            if progress_cb is not None and total:
                progress_cb(idx / total, f"port {port}: frame {idx}/{total}")
    finally:
        cap.release()
        writer.release()
    return Path(out_path)


def produce_marked_videos(session_dir, model, out_dir, progress_cb=None):
    """Produce one marked mp4 per camera (each port_*.mp4 at the session root)."""
    session = Path(session_dir)
    videos = sorted(session.glob("port_*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no raw camera videos (port_*.mp4) in {session}")
    xy_df = load_xy(session_dir, model)
    outputs = []
    n = len(videos)
    for i, video in enumerate(videos):
        port = int(video.stem.split("_")[1])          # "port_2" -> 2
        out_path = Path(out_dir) / f"{video.stem}_marked.mp4"

        def port_cb(frac, stage, _i=i):
            if progress_cb is not None:
                progress_cb((_i + frac) / n, stage)

        outputs.append(produce_marked_video(video, xy_df, port, out_path, port_cb))
    return outputs
