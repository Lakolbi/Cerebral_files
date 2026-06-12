# Video Overlay Producer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw the gait skeleton (caliscope's 2D detections) onto each raw camera video and save one annotated `.mp4` per camera, via a CLI subcommand and a button in the Анализ tab.

**Architecture:** A new `modules/visualization/video_overlay.py` reads `xy_<MODEL>.csv` (per-camera pixel coords), maps MediaPipe `point_id`s to our 12 gait landmarks, and uses OpenCV to read each `port_*.mp4`, draw the skeleton (reusing `SKELETON_EDGES`), and write a marked video. A `cli.py` subcommand and a background `OverlayWorker` (same QThread pattern as the pipeline worker) drive it.

**Tech Stack:** Python 3.14, `opencv-python-headless` (read/draw/write), pandas, PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-05-31-video-overlay-design.md`

---

## Conventions (every task)
- Work from `gait_analysis/`. Use `.venv/bin/python`, `.venv/bin/python -m pytest`, `.venv/bin/ruff`.
- Tests import `from modules...` / `from gui...` / `from cli import ...` (gait_analysis is pytest's rootdir).
- GUI tests rely on `tests/conftest.py` (forces `QT_QPA_PLATFORM=offscreen`).
- Commit with EXACT pathspecs — never `git add -A`/`.` (an unrelated `.docx` + `.idea/` sit untracked).
- `cv2` is `opencv-python-headless` (no GUI funcs needed). Colors are **BGR**.

---

## Task 1: Dependency

**Files:** Modify `gait_analysis/requirements.txt`

- [ ] **Step 1: Add the dependency**

Append to `requirements.txt`:
```
opencv-python-headless>=4.10
```

- [ ] **Step 2: Install + verify**

Run:
```bash
.venv/bin/python -m pip install --only-binary=:all: opencv-python-headless
.venv/bin/python -c "import cv2; print('cv2', cv2.__version__)"
```
Expected: prints `cv2 4.13.0` (or newer). (Already installed during brainstorming; this confirms.)

- [ ] **Step 3: Commit**
```bash
git add gait_analysis/requirements.txt
git commit -m "build: add opencv-python-headless for video overlay"
```

---

## Task 2: Overlay core — mapping, xy loader, frame marks, drawing

**Files:**
- Create: `gait_analysis/modules/visualization/video_overlay.py`
- Test: `gait_analysis/tests/test_video_overlay.py`

- [ ] **Step 1: Write the failing test** — create `gait_analysis/tests/test_video_overlay.py`:

```python
import numpy as np
import pandas as pd

from modules.data_loader.landmarks import GAIT_LANDMARKS
from modules.visualization.video_overlay import (
    POSE_POINT_IDS,
    draw_overlay,
    frame_marks,
)


def test_pose_point_ids_cover_the_twelve_gait_joints():
    assert set(POSE_POINT_IDS.values()) == set(GAIT_LANDMARKS)


def test_frame_marks_maps_ids_skips_other_ports_frames_and_nan():
    xy = pd.DataFrame({
        "port": [1, 1, 1, 1, 2],
        "frame_index": [0, 0, 0, 1, 0],
        "point_id": [23, 999, 25, 23, 24],     # 23=left_hip 25=left_knee 999=non-gait
        "img_loc_x": [100, 50, 110, 5, 700],
        "img_loc_y": [200, 50, 260, 5, 200],
    })
    marks = frame_marks(xy, port=1, frame_index=0)
    assert marks["left_hip"] == (100, 200)
    assert marks["left_knee"] == (110, 260)
    assert "right_hip" not in marks            # port 2 / other frame excluded
    assert len(marks) == 2                     # point_id 999 ignored


def test_frame_marks_skips_nan_coords():
    xy = pd.DataFrame({
        "port": [1], "frame_index": [0], "point_id": [23],
        "img_loc_x": [np.nan], "img_loc_y": [np.nan],
    })
    assert frame_marks(xy, port=1, frame_index=0) == {}


def test_draw_overlay_sets_pixels_and_preserves_shape():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    marks = {"left_hip": (100, 150), "left_knee": (100, 250)}
    out = draw_overlay(frame, marks, edges=[("left_hip", "left_knee")])
    assert out.shape == (300, 300, 3)
    assert out[150, 100].sum() > 0             # joint drawn
    assert out[200, 100].sum() > 0             # bone midpoint drawn
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_video_overlay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modules.visualization.video_overlay'`.

- [ ] **Step 3: Create `gait_analysis/modules/visualization/video_overlay.py`**:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_video_overlay.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**
```bash
git add gait_analysis/modules/visualization/video_overlay.py gait_analysis/tests/test_video_overlay.py
git commit -m "feat(video): overlay core — point-id mapping, xy loader, frame marks, drawing"
```

---

## Task 3: Producers — one video + all cameras

**Files:**
- Modify: `gait_analysis/modules/visualization/video_overlay.py`
- Test: `gait_analysis/tests/test_video_overlay.py` (append)

- [ ] **Step 1: Append the failing tests** to `gait_analysis/tests/test_video_overlay.py`:

```python
from pathlib import Path

import cv2
import pytest

from modules.visualization.video_overlay import (
    load_xy,
    produce_marked_video,
    produce_marked_videos,
)

_REC = Path(__file__).resolve().parents[1] / "data" / "caliscope_project" / "recordings" / "p1_3"


@pytest.mark.skipif(not (_REC / "port_1.mp4").exists(), reason="p1_3 data not present")
def test_produce_marked_video_writes_playable_mp4(tmp_path):
    xy = load_xy(str(_REC), "SIMPLE_HOLISTIC")
    out = tmp_path / "port_1_marked.mp4"
    produce_marked_video(_REC / "port_1.mp4", xy, port=1, out_path=out)
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n > 0


@pytest.mark.skipif(not (_REC / "port_1.mp4").exists(), reason="p1_3 data not present")
def test_produce_marked_videos_writes_one_per_camera(tmp_path):
    outs = produce_marked_videos(str(_REC), "SIMPLE_HOLISTIC", str(tmp_path))
    assert len(outs) == len(sorted(_REC.glob("port_*.mp4")))
    assert all(Path(o).exists() and Path(o).stat().st_size > 0 for o in outs)


def test_produce_marked_videos_raises_when_no_videos(tmp_path):
    with pytest.raises(FileNotFoundError):
        produce_marked_videos(str(tmp_path), "SIMPLE_HOLISTIC", str(tmp_path / "out"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_video_overlay.py -k "produce" -v`
Expected: FAIL — `ImportError: cannot import name 'produce_marked_video'`.

- [ ] **Step 3: Append the producers** to `gait_analysis/modules/visualization/video_overlay.py`:

```python
def produce_marked_video(video_path, xy_df, port, out_path, progress_cb=None):
    """Draw the skeleton on every frame of one camera video; write an annotated mp4.

    Reads the raw video frame-by-frame; frame N gets the marks for (port, frame_index=N).
    Source fps + resolution are preserved. Returns the output Path.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_video_overlay.py -v`
Expected: PASS (7 tests; the two data-gated ones run if `p1_3` is present, else SKIP — report which).

- [ ] **Step 5: Commit**
```bash
git add gait_analysis/modules/visualization/video_overlay.py gait_analysis/tests/test_video_overlay.py
git commit -m "feat(video): produce marked mp4 per camera"
```

---

## Task 4: CLI subcommand `produce-videos`

**Files:**
- Modify: `gait_analysis/cli.py`
- Test: `gait_analysis/tests/test_cli_integration.py` (append)

- [ ] **Step 1: Append the failing test** to `gait_analysis/tests/test_cli_integration.py`:

```python
@pytest.mark.skipif(not (SESSION / "port_1.mp4").exists(),
                    reason="p1_3 raw videos not present")
def test_produce_videos_cli_writes_marked_mp4s(tmp_path):
    cmd = [sys.executable, str(GAIT / "cli.py"), "produce-videos",
           "--session", str(SESSION), "--model", "SIMPLE_HOLISTIC",
           "--out", str(tmp_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GAIT))
    assert proc.returncode == 0, proc.stderr
    written = list(tmp_path.glob("port_*_marked.mp4"))
    assert len(written) >= 1
```
(`SESSION`, `GAIT`, `sys`, `subprocess`, `pytest` are already defined/imported at the top of this file.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_integration.py::test_produce_videos_cli_writes_marked_mp4s -v`
Expected: FAIL — non-zero return code (`invalid choice: 'produce-videos'`).

- [ ] **Step 3: Wire the subcommand in `gait_analysis/cli.py`**

Add the import after the existing `from pipeline import run_pipeline` line:
```python
from modules.visualization.video_overlay import produce_marked_videos
```

In `main()`, after the `reproducibility` subparser block (the `r.add_argument(... "--out" ...)` lines), add:
```python
    v = sub.add_parser("produce-videos", help="Draw the skeleton on each camera video")
    v.add_argument("--session", required=True)
    v.add_argument("--model", default=default_model)
    v.add_argument("--out", required=True)
```

In `main()`, after the `elif args.command == "reproducibility":` block, add:
```python
    elif args.command == "produce-videos":
        outs = produce_marked_videos(args.session, args.model, args.out)
        print(f"Wrote {len(outs)} marked videos:")
        for o in outs:
            print(f"  {o}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cli_integration.py -v`
Expected: PASS (existing CLI tests + the new one; new one SKIPs if no `p1_3` videos).

- [ ] **Step 5: Lint + commit**
```bash
.venv/bin/ruff check .
git add gait_analysis/cli.py gait_analysis/tests/test_cli_integration.py
git commit -m "feat(cli): produce-videos subcommand"
```

---

## Task 5: GUI background worker

**Files:**
- Create: `gait_analysis/gui/overlay_worker.py`
- Test: `gait_analysis/tests/test_gui_smoke.py` (append)

- [ ] **Step 1: Append the failing test** to `gait_analysis/tests/test_gui_smoke.py`:

```python
def test_overlay_worker_error_path_emits_error():
    from gui.overlay_worker import OverlayWorker
    worker = OverlayWorker("/nonexistent/session/folder", "SIMPLE_HOLISTIC", "/tmp/none")
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors          # missing videos/xy -> error signal, no crash
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_overlay_worker_error_path_emits_error -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui.overlay_worker'`.

- [ ] **Step 3: Create `gait_analysis/gui/overlay_worker.py`**:

```python
"""Background worker: produce marked videos off the GUI thread."""
from PyQt6.QtCore import QObject, pyqtSignal

from modules.visualization.video_overlay import produce_marked_videos


class OverlayWorker(QObject):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(list)          # list[Path]
    error = pyqtSignal(str)

    def __init__(self, session_dir, model, out_dir):
        super().__init__()
        self._session_dir = session_dir
        self._model = model
        self._out_dir = out_dir

    def run(self):
        try:
            outs = produce_marked_videos(
                self._session_dir, self._model, self._out_dir,
                progress_cb=lambda frac, stage: self.progress.emit(frac, stage),
            )
            self.finished.emit(outs)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, must never crash
            self.error.emit(str(exc))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_overlay_worker_error_path_emits_error -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add gait_analysis/gui/overlay_worker.py gait_analysis/tests/test_gui_smoke.py
git commit -m "feat(gui): overlay worker"
```

---

## Task 6: Анализ-tab button + close-event guard

**Files:**
- Modify: `gait_analysis/gui/panels/analyze_panel.py`
- Modify: `gait_analysis/gui/main_window.py`
- Test: `gait_analysis/tests/test_gui_smoke.py` (append)

- [ ] **Step 1: Append the failing test** to `gait_analysis/tests/test_gui_smoke.py`:

```python
def test_analyze_panel_has_produce_button_disabled_initially(qtbot):
    from gui.panels.analyze_panel import AnalyzePanel
    panel = AnalyzePanel()
    qtbot.addWidget(panel)
    assert hasattr(panel, "produce_btn")
    assert not panel.produce_btn.isEnabled()       # disabled until a folder is chosen
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_analyze_panel_has_produce_button_disabled_initially -v`
Expected: FAIL — `AttributeError: 'AnalyzePanel' object has no attribute 'produce_btn'`.

- [ ] **Step 3a: Add the import** in `gait_analysis/gui/panels/analyze_panel.py` — after the line `from gui.worker import PipelineWorker`:

```python
from gui.overlay_worker import OverlayWorker
```

- [ ] **Step 3b: Create the button** — in `AnalyzePanel.__init__`, right after the `self.xlsx_btn` block (the two lines creating `xlsx_btn`), add:

```python
        self.produce_btn = QPushButton("Produce marked videos")
        self.produce_btn.setEnabled(False)
```

- [ ] **Step 3c: Add it to the layout** — replace the existing export-row + layout block:

```python
        export_row = QHBoxLayout()
        export_row.addWidget(self.csv_btn)
        export_row.addWidget(self.xlsx_btn)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.quality)
        layout.addWidget(self.table)
        layout.addLayout(export_row)
```
with:
```python
        export_row = QHBoxLayout()
        export_row.addWidget(self.csv_btn)
        export_row.addWidget(self.xlsx_btn)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.quality)
        layout.addWidget(self.table)
        layout.addLayout(export_row)
        layout.addWidget(self.produce_btn)
```

- [ ] **Step 3d: Wire the click** — in `__init__`, after `self.xlsx_btn.clicked.connect(self._export_xlsx)`, add:
```python
        self.produce_btn.clicked.connect(self._produce_videos)
```

- [ ] **Step 3e: Enable on folder pick** — in `_choose_folder`, replace:
```python
        if path:
            self._folder = path
            self.folder_label.setText(Path(path).name)
            self.run_btn.setEnabled(True)
```
with:
```python
        if path:
            self._folder = path
            self.folder_label.setText(Path(path).name)
            self.run_btn.setEnabled(True)
            self.produce_btn.setEnabled(True)
```

- [ ] **Step 3f: Add the producer handlers** — after the `_export_xlsx` method, add:
```python
    def _produce_videos(self):
        self.produce_btn.setEnabled(False)
        self.progress.setValue(0)
        out_dir = (GAIT_DIR / self._cfg["paths"]["output_dir"]
                   / f"{Path(self._folder).name}_marked")
        self._overlay_thread = QThread()
        self._overlay_worker = OverlayWorker(self._folder, self.model.currentText(),
                                             str(out_dir))
        self._overlay_worker.moveToThread(self._overlay_thread)
        self._overlay_thread.started.connect(self._overlay_worker.run)
        self._overlay_worker.progress.connect(self._on_progress)
        self._overlay_worker.finished.connect(self._on_videos_done)
        self._overlay_worker.error.connect(self._on_error)
        self._overlay_worker.finished.connect(self._overlay_thread.quit)
        self._overlay_worker.error.connect(self._overlay_thread.quit)
        self._overlay_thread.finished.connect(lambda: self.produce_btn.setEnabled(True))
        self._overlay_thread.start()

    def _on_videos_done(self, paths):
        where = paths[0].parent if paths else "(none)"
        QMessageBox.information(self, "Marked videos",
                               f"Saved {len(paths)} videos to:\n{where}")
```

- [ ] **Step 3g: Guard the overlay thread on close** — in `gait_analysis/gui/main_window.py`, replace the `closeEvent` method:
```python
    def closeEvent(self, event):
        thread = getattr(self.analyze, "_thread", None)
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait()
        super().closeEvent(event)
```
with:
```python
    def closeEvent(self, event):
        for attr in ("_thread", "_overlay_thread"):
            thread = getattr(self.analyze, attr, None)
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        super().closeEvent(event)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py -v`
Expected: PASS (all, incl. the new produce-button test).

- [ ] **Step 5: Lint + commit**
```bash
.venv/bin/ruff check .
git add gait_analysis/gui/panels/analyze_panel.py gait_analysis/gui/main_window.py gait_analysis/tests/test_gui_smoke.py
git commit -m "feat(gui): Produce marked videos button + overlay-thread close guard"
```

---

## Task 7: Integration — full suite, lint, real-data smoke

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all PASS (video data-gated tests run if `p1_3` present, else SKIP — report which). No regressions.

- [ ] **Step 2: Lint**

Run: `.venv/bin/ruff check .`
Expected: clean (use `ruff check --fix .` for any import-sort, then re-run).

- [ ] **Step 3: Real-data smoke (if `p1_3` present)**

Run:
```bash
.venv/bin/python cli.py produce-videos --session data/caliscope_project/recordings/p1_3 --model SIMPLE_HOLISTIC --out /tmp/marked_demo
ls -la /tmp/marked_demo/
```
Expected: 3 `port_N_marked.mp4` files written. Open one and confirm the skeleton tracks the subject.

(No commit — verification only.)

---

## Self-Review

**Spec coverage:**
- §2 D1 raw 2D → `frame_marks` reads `img_loc_x/y` (Task 2). ✅
- §2 D2 save N mp4 → `produce_marked_videos` (Task 3). ✅
- §2 D3 cv2 → Task 1 dep + Task 2/3. ✅
- §2 D4 our gait skeleton (joints + `SKELETON_EDGES`) → `draw_overlay` (Task 2). ✅
- §2 D5 GUI button + CLI, background thread → Tasks 4/5/6. ✅
- §4.1 discover `port_*.mp4`; §4.2 xy format; §4.3 point-id map; §4.4 output path/codec → Tasks 2/3 + Task 6 out_dir. ✅
- §5 module/CLI/worker/panel + closeEvent extension → Tasks 2–6. ✅
- §7 error handling (missing files raise; partial frames; worker try/except) → Tasks 2/3/5. ✅
- §8 testing tiers (unit / data-gated integration / offscreen smoke) → Tasks 2/3/4/5/6. ✅

**Placeholder scan:** none — every step has complete code/commands.

**Type/name consistency:** `produce_marked_videos(session_dir, model, out_dir, progress_cb=None)` and `produce_marked_video(video_path, xy_df, port, out_path, progress_cb=None)` identical across Tasks 3/4/5; `OverlayWorker(session_dir, model, out_dir)` signals `progress/finished/error` consistent (Tasks 5/6); `produce_btn`, `_produce_videos`, `_on_videos_done` consistent (Task 6); `POSE_POINT_IDS`, `frame_marks`, `draw_overlay`, `load_xy` consistent (Tasks 2/3).
