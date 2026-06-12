# Design Spec — Video Overlay Producer (skeleton-on-video)

**Version:** 1.0
**Date:** 2026-05-31
**Builds on:** Phase 2 GUI (`gait_analysis`, merged to `main`).
**Status:** Approved for planning

---

## 1. Overview & Goal

Produce annotated videos: take each camera's **raw recording** and draw the gait skeleton
(detected landmarks + bones) directly on the footage, yielding **one marked `.mp4` per camera**
(3 for this dataset). The landmarks come from caliscope's per-camera 2D detections, so the marks
land exactly where each camera saw them — no 3D→2D projection or calibration math.

This complements the abstract 3D skeleton view (Phase 2): the 3D view shows the reconstructed
motion in space; the marked videos show the same skeleton overlaid on the real video, which is the
natural way to eyeball detection quality against the footage.

---

## 2. Decisions (resolved in brainstorming)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Marks source = raw per-camera 2D detections** (`xy_<MODEL>.csv`, `img_loc_x/y`). | Pixel coords already in each video's space → draw directly. No calibration/distortion/mirroring risk. Faithful to what the camera saw. |
| D2 | **Delivery = save N `.mp4` files** (one per camera). No live in-app player. | Matches "we'd have 3 videos"; dead-simple; play in any viewer. |
| D3 | **Stack = OpenCV (`opencv-python-headless`)** for read + draw + write. | Single dependency does everything; verified to install + run on Python 3.14; `headless` avoids GUI libs that could clash with Qt. |
| D4 | **Draw our gait skeleton** — the 12 skeleton joints + `SKELETON_EDGES` bones (reused from `skeleton_3d`). | Consistent with the 3D view; caliscope's own annotated videos are dots-only, cluttered with face/hand points, and downscaled. |
| D5 | **Trigger = GUI button in the Анализ tab + a CLI subcommand**, run in a background thread. | Project principle: everything runnable headless; GUI must never block. Independent of "Run analysis" (uses raw xy + videos directly). |

**Out of scope:** reprojecting the processed/smoothed 3D skeleton (route B); a live embedded video
player; gait-event highlighting (HS/TO markers) on the video; re-encoding/codec options beyond a
sane default.

---

## 3. Feasibility validation (done in brainstorming)

- **`opencv-python-headless` 4.13.0.92** ships a `cp37-abi3` wheel → installed in the project venv,
  `import cv2` works on Python 3.14.4. ✓
- **Proof-of-concept rendered on real data:** read frame 70 of `p1_3/port_1.mp4` (1920×1080),
  mapped `point_id`→landmark, drew bones + joints — the skeleton aligned exactly on the subject. ✓
- System `ffmpeg`/`ffprobe` present (not required by the cv2 path, but available).

---

## 4. Data contracts (verified against `p1_3`)

### 4.1 Raw videos
- Location: `<session>/port_<N>.mp4` (e.g. `port_1.mp4`, `port_2.mp4`, `port_3.mp4`) — the **clean**
  source recordings at the session root. (Do NOT use the per-model `<session>/<MODEL>/port_N_<MODEL>.mp4`,
  which are caliscope's own dot overlays.)
- Observed: 1920×1080, 20 fps, ~131 frames. Resolution/fps are read from the file at runtime, never
  hardcoded.
- Camera count is **discovered** by globbing `port_*.mp4` (don't assume exactly 3).

### 4.2 2D detections — `<session>/<MODEL>/xy_<MODEL>.csv`
Long format, one row per (sync_index, port, point_id):
```
sync_index, port, frame_index, frame_time, point_id, img_loc_x, img_loc_y, obj_loc_x, obj_loc_y
```
- `port` ∈ {1,2,3}; `frame_index` = **the frame number within that port's raw video** (validated: the
  POC read `frame_index` directly and it aligned). `img_loc_x/img_loc_y` = pixel coords in that video.
- `obj_loc_x/obj_loc_y` are unused here.
- A landmark absent in a frame simply has no row → that joint/bone is skipped for that frame.

### 4.3 `point_id` → canonical landmark (MediaPipe Pose indices)
```
11 left_shoulder   12 right_shoulder
23 left_hip        24 right_hip
25 left_knee       26 right_knee
27 left_ankle      28 right_ankle
29 left_heel       30 right_heel
31 left_foot_index 32 right_foot_index
```
Only these 12 (the skeleton joints) are drawn; higher `point_id`s (face/hand) are ignored.

### 4.4 Output
- `settings.paths.output_dir/<session_id>_marked/port_<N>_marked.mp4` by default (output dir
  overridable from the GUI). Source resolution + fps preserved; container `.mp4`, codec fourcc
  `mp4v`.

---

## 5. Architecture

```
Анализ tab  ──[Produce marked videos]──▶ OverlayWorker (QThread)
                                              │  progress(frac, "port 2: 40%")
CLI: produce-videos ──────────────────────────┤
                                              ▼
                       modules/visualization/video_overlay.py
                       produce_marked_videos(session, model, out_dir, cb)
                          ├─ load_xy(session, model)            (pandas)
                          ├─ discover port_*.mp4
                          └─ per port: produce_marked_video(...)  (cv2 read→draw→write)
                                         frame_marks() + draw_overlay()
```

### `modules/visualization/video_overlay.py` (new) — pure logic + cv2 IO
| Function | Responsibility |
|----------|----------------|
| `POSE_POINT_IDS: dict[int,str]` | §4.3 mapping (the 12 gait joints). |
| `load_xy(session_dir, model) -> DataFrame` | Read `<session>/<MODEL>/xy_<MODEL>.csv`. |
| `frame_marks(xy_df, port, frame_index) -> dict[str,(int,int)]` | Gait landmarks present in that camera frame (missing skipped). |
| `draw_overlay(frame, marks, edges=SKELETON_EDGES) -> frame` | Draw bones (lines) + joints (filled circles) in place; returns the frame. Reuses `SKELETON_EDGES` from `skeleton_3d`. |
| `produce_marked_video(video_path, xy_df, port, out_path, progress_cb=None) -> Path` | Iterate the raw video's frames via `cv2.VideoCapture`; for each `frame_index` draw `frame_marks`; write via `cv2.VideoWriter` (source fps + size, fourcc `mp4v`). |
| `produce_marked_videos(session_dir, model, out_dir, progress_cb=None) -> list[Path]` | Discover `port_*.mp4`; load xy once; produce one annotated video per camera; return output paths. **The producer.** |

`progress_cb(fraction: float, stage: str)` — same convention as `run_pipeline`; no-op if `None`.

### `cli.py` (modify)
Add `produce-videos` subcommand: `--session`, `--model` (default from settings), `--out`
(default `settings.paths.output_dir`). Calls `produce_marked_videos`, prints the written paths.

### `gui/overlay_worker.py` (new)
`OverlayWorker(QObject)` mirroring `PipelineWorker`: `progress(float,str)`, `finished(list)`,
`error(str)`; `run()` calls `produce_marked_videos(...)` wrapped in try/except.

### `gui/panels/analyze_panel.py` (modify)
Add a **"Produce marked videos"** `QPushButton` (enabled once a folder is chosen). On click:
spawn `OverlayWorker` on a `QThread` (same lifecycle pattern as `_run`: re-enable on
`thread.finished`; covered by `MainWindow.closeEvent`'s quit+wait — extend it to also wait on the
overlay thread). On `finished`: status/dialog "Saved N videos to <dir>". On `error`: warning dialog.
Reuses the existing `QProgressBar`.

---

## 6. Data flow
Pick session folder + model → click **Produce marked videos** → worker loads `xy_<MODEL>.csv`,
discovers `port_*.mp4`, and for each camera streams frames (read → `frame_marks` → `draw_overlay`
→ write), emitting progress → on done, the N `port_N_marked.mp4` files are written to the output
dir and the UI reports the location.

---

## 7. Error handling
- Missing `xy_<MODEL>.csv` or no `port_*.mp4` at the session root → raised with a clear message,
  surfaced as a dialog (GUI) / non-zero exit (CLI). The UI never blocks (work on the worker thread)
  and never crashes (worker try/except → `error`).
- A frame with missing landmarks → partial skeleton (skip absent joints/bones), not an error.
- `cv2.VideoWriter` failing to open (codec/permission) → raised with the target path in the message.
- Output directory created if absent (`mkdir parents`).

---

## 8. Testing
| Tier | Scope | How |
|------|-------|-----|
| **Unit (headless)** | `POSE_POINT_IDS` completeness; `frame_marks` (synthetic xy DataFrame → expected dict, missing skipped); `draw_overlay` on a small `np.zeros` frame → asserts shape preserved + pixels set at a known joint/bone location. | plain pytest; `cv2` headless needs no display. |
| **Integration (data-gated)** | `produce_marked_video` on `p1_3/port_1.mp4` (optionally a short frame cap) → assert the output `.mp4` exists, opens in `cv2.VideoCapture`, and reports frame count > 0. | `skipif` p1_3 absent (like `test_cli_integration`). |
| **GUI smoke (offscreen)** | `OverlayWorker` error path (bad folder → `error` emitted); AnalyzePanel has the button and it's disabled until a folder is set. | pytest-qt offscreen. |

`ruff` clean; coverage maintained.

**Dependency:** add `opencv-python-headless>=4.10` to `requirements.txt` (installed + verified).

---

## 9. Summary
A small, well-bounded `video_overlay` module (+ CLI subcommand + one GUI button/worker) reads each
raw camera video, draws the gait skeleton from caliscope's 2D detections, and writes one marked
`.mp4` per camera. Route validated end-to-end with a real-frame POC; the only new dependency
(`opencv-python-headless`) is confirmed working on Python 3.14. No projection, no live player, no
new data contracts beyond reading the already-present `xy_<MODEL>.csv`.
