# Gait Analysis Phase 2 — Visualization + Dead-Simple PyQt6 GUI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a visualization module and a minimal two-tab PyQt6 desktop GUI on top of the finished Phase 1 pipeline, so a user can load a caliscope session, run the analysis, and view the animated 3D skeleton + joint-angle plots, with CSV/XLSX/PNG export.

**Architecture:** Extract the analysis pipeline buried in `cli.analyze` into one shared `run_pipeline` function used by both the CLI and a background GUI worker thread. Add a headless-testable `modules/visualization` package (GL-free skeleton geometry, matplotlib angle plots, exporters). Wrap it in a `QTabWidget` with an `Анализ` tab (load + params + run + results + export) and a `Визуализация` tab (vispy skeleton + matplotlib plots). All heavy work runs off the UI thread.

**Tech Stack:** Python 3.14, PyQt6 6.11, vispy 0.16.2 (OpenGL skeleton), matplotlib (angle plots), openpyxl (XLSX), pytest + pytest-qt (offscreen smoke tests), ruff.

**Spec:** `docs/superpowers/specs/2026-05-30-gait-phase2-gui-design.md`

---

## Conventions for every task

- **Working directory:** all commands run from `gait_analysis/` (the dir containing `pyproject.toml`). That dir is on `sys.path` for pytest, so `import pipeline`, `from modules...`, and `from gui...` all resolve.
- **Python/tools:** use the project venv: `.venv/bin/python`, `.venv/bin/python -m pytest`, `.venv/bin/ruff`.
- **Run a single test:** `.venv/bin/python -m pytest tests/test_x.py::test_name -v`
- **Run all tests:** `.venv/bin/python -m pytest`
- **Lint:** `.venv/bin/ruff check .` (use `.venv/bin/ruff check --fix .` to auto-sort imports / fix trivial issues).
- **Headless GUI tests** rely on `tests/conftest.py` setting `QT_QPA_PLATFORM=offscreen` (created in Task 6). Tasks 3–5 are pure-logic and need no display.
- Landmark names are canonical lowercase (`left_hip`, `right_knee`, `left_foot_index`, …). The gait landmark set lives in `modules/data_loader/landmarks.py::GAIT_LANDMARKS`.

---

## Task 1: Dependencies

**Files:**
- Modify: `gait_analysis/requirements.txt`

- [ ] **Step 1: Add the four Phase-2 GUI dependencies to `requirements.txt`**

Append these lines (matplotlib is already listed from Phase 1; do not duplicate it):

```
PyQt6>=6.11
vispy>=0.16.2
PyOpenGL>=3.1
pytest-qt>=4.5
```

- [ ] **Step 2: Install everything the phase needs into the venv**

Run:
```bash
.venv/bin/python -m pip install --only-binary=:all: "matplotlib>=3.7" PyQt6 vispy PyOpenGL pytest-qt
```
Expected: `Successfully installed ...` (PyQt6/vispy/PyOpenGL/pytest-qt may already be present; matplotlib is the new one). `plotly`, `scikit-learn`, and `pingouin` are NOT needed for Phase 2 — leave them uninstalled.

- [ ] **Step 3: Verify the imports + headless capability**

Run:
```bash
.venv/bin/python -c "import PyQt6, OpenGL, matplotlib, vispy; print('matplotlib', matplotlib.__version__, '| vispy', vispy.__version__)"
QT_QPA_PLATFORM=offscreen .venv/bin/python -c "import vispy; vispy.use(app='pyqt6'); from PyQt6 import QtWidgets; app=QtWidgets.QApplication([]); from vispy import scene; c=scene.SceneCanvas(show=False); print('canvas OK')"
```
Expected: prints versions and `canvas OK` (a `QOpenGLWidget is not supported` warning under offscreen is expected and harmless).

- [ ] **Step 4: Commit**

```bash
git add gait_analysis/requirements.txt
git commit -m "build(phase2): add PyQt6/vispy/PyOpenGL/pytest-qt deps"
```

---

## Task 2: Extract shared `run_pipeline`

Move the analysis pipeline out of `cli.analyze` into a new top-level `pipeline.py` so the GUI worker and the CLI share one implementation. The CLI's `gait_results.json` output must stay **byte-identical** — the existing 41 tests are the regression guard.

**Files:**
- Create: `gait_analysis/pipeline.py`
- Modify: `gait_analysis/cli.py`
- Test: `gait_analysis/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `gait_analysis/tests/test_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline'`.

- [ ] **Step 3: Create `pipeline.py`**

Create `gait_analysis/pipeline.py`:

```python
"""Shared analysis pipeline: one implementation for both cli.py and the GUI worker."""
import datetime as dt

import numpy as np

from modules.kinematics.filters import butterworth_filter, fill_gaps
from modules.kinematics.gait_events import detect_gait_events
from modules.kinematics.joint_angles import calc_joint_angles_timeseries
from modules.kinematics.normalizer import get_mean_std_cycle, normalize_gait_cycle
from modules.kinematics.spatiotemporal import calc_spatiotemporal


def _filter_coords(df, cutoff_hz, order, fps):
    """Filter coordinate columns; skip columns with residual NaN or too-short for filtfilt."""
    out = df.copy()
    min_len = 3 * (order + 1)
    for c in out.columns:
        if not c.endswith(("_x", "_y", "_z")):
            continue
        arr = out[c].to_numpy()
        if np.isnan(arr).any() or len(arr) <= min_len:
            continue
        out[c] = butterworth_filter(arr, cutoff_hz=cutoff_hz, fs=fps, order=order)
    return out


def run_pipeline(df, cfg, *, model, session_id, progress_cb=None):
    """Run the full kinematics pipeline on a loaded pose DataFrame.

    df          : unified pose DataFrame from load_caliscope_session (carries fps in df.attrs).
    cfg         : settings dict (processing / gait_events / spatiotemporal sections).
    model       : model name, stamped into results.
    session_id  : session identifier (e.g. folder name), stamped into results.
    progress_cb : optional callable(fraction: float, stage: str); no-op if None.

    Returns (results_dict, df_processed). results_dict is the canonical gait_results.json schema.
    """
    proc = cfg["processing"]
    gcfg = cfg["gait_events"]
    scfg = cfg.get("spatiotemporal", {})

    def report(frac, stage):
        if progress_cb is not None:
            progress_cb(frac, stage)

    fps = df.attrs["fps"]

    report(0.10, "Filling gaps")
    df = fill_gaps(df, max_gap_frames=proc["max_gap_frames"])
    report(0.25, "Filtering")
    df = _filter_coords(df, proc["filter_cutoff_hz"], proc["filter_order"], fps)

    report(0.45, "Detecting gait events")
    events = detect_gait_events(
        df, fps=fps, method=gcfg["method"], heel=gcfg["heel_landmark"],
        toe=gcfg["toe_landmark"], vertical=gcfg["vertical_axis"],
        min_stride_sec=proc["min_stride_duration_sec"],
        cutoff_hz=proc["filter_cutoff_hz"],
    )
    report(0.60, "Joint angles")
    df = calc_joint_angles_timeseries(df)
    report(0.80, "Spatiotemporal")
    spatiotemporal = calc_spatiotemporal(
        df, events, fps=fps, vertical=gcfg["vertical_axis"],
        max_stride_m=scfg.get("max_stride_m", 1.5),
        max_step_m=scfg.get("max_step_m", 1.0),
    )

    report(0.90, "Normalizing cycles")
    angles_mean, angles_std = {}, {}
    for side in ("left", "right"):
        for joint in ("hip", "knee", "ankle"):
            col = f"{side}_{joint}_angle"
            if col not in df.columns or not events.get(f"{side}_HS"):
                continue
            mat = normalize_gait_cycle(df[col].to_numpy(), events, side=side)
            mean, std = get_mean_std_cycle(mat)
            if mat.shape[0] > 0:
                angles_mean[f"{side}_{joint}"] = np.nan_to_num(mean, nan=0.0).round(3).tolist()
                angles_std[f"{side}_{joint}"] = np.nan_to_num(std, nan=0.0).round(3).tolist()

    results = {
        "session_id": session_id,
        "model": model,
        "fps": round(float(fps), 3),
        "n_frames": int(len(df)),
        "processed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "gait_events": {k: list(map(int, v)) for k, v in events.items()},
        "spatiotemporal": {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                               else v if isinstance(v, int)
                               else round(float(v), 3))
                           for k, v in spatiotemporal.items()},
        "joint_angles_mean": angles_mean,
        "joint_angles_std": angles_std,
    }
    report(1.0, "Done")
    return results, df
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Refactor `cli.py` to call `run_pipeline`**

Replace the import block and the `_filter_coords` + `analyze` definitions. The new top of `cli.py` (imports) becomes exactly:

```python
"""Command-line entry point for the gait analysis pipeline."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from modules.data_loader.caliscope_reader import load_caliscope_session
from pipeline import run_pipeline

HERE = Path(__file__).resolve().parent


def _load_settings() -> dict:
    with open(HERE / "settings.yaml") as f:
        return yaml.safe_load(f)


def analyze(session_dir: str, model: str, out_path: str) -> dict:
    cfg = _load_settings()
    df = load_caliscope_session(session_dir, model=model)
    results, _ = run_pipeline(df, cfg, model=model, session_id=Path(session_dir).name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    return results
```

Delete the old `_filter_coords` function and the old body of `analyze` (the inline pipeline). **Keep `reproducibility()` and `main()` unchanged** — `reproducibility` calls `analyze()`, which still works. `np` stays imported (used in `reproducibility`).

- [ ] **Step 6: Run the FULL suite — regression guard**

Run: `.venv/bin/python -m pytest`
Expected: PASS — all prior tests (incl. `test_cli_integration.py`, which asserts the JSON schema/values) plus the two new `test_pipeline` tests. If `test_cli_integration` is skipped (no `p1_3` data), that's fine; the schema is also covered by `test_pipeline`.

- [ ] **Step 7: Lint**

Run: `.venv/bin/ruff check .`
Expected: no errors (remove any now-unused imports in `cli.py` that ruff flags).

- [ ] **Step 8: Commit**

```bash
git add gait_analysis/pipeline.py gait_analysis/cli.py gait_analysis/tests/test_pipeline.py
git commit -m "refactor(pipeline): extract shared run_pipeline from cli.analyze"
```

---

## Task 3: Visualization core — `skeleton_3d.py` (GL-free)

Pure geometry: skeleton connectivity + per-frame point/segment extraction. No Qt, no OpenGL — fully unit-testable.

**Files:**
- Create: `gait_analysis/modules/visualization/__init__.py` (empty)
- Create: `gait_analysis/modules/visualization/skeleton_3d.py`
- Test: `gait_analysis/tests/test_skeleton_3d.py`

- [ ] **Step 1: Write the failing test**

Create `gait_analysis/tests/test_skeleton_3d.py`:

```python
import numpy as np
import pandas as pd

from modules.data_loader.landmarks import GAIT_LANDMARKS
from modules.visualization.skeleton_3d import SKELETON_EDGES, frame_points, segment_lines


def test_frame_points_extracts_xyz_and_skips_nan():
    df = pd.DataFrame({
        "left_hip_x": [0.0], "left_hip_y": [1.0], "left_hip_z": [2.0],
        "left_knee_x": [np.nan], "left_knee_y": [1.0], "left_knee_z": [0.0],
    })
    pts = frame_points(df, 0)
    assert pts["left_hip"] == (0.0, 1.0, 2.0)
    assert "left_knee" not in pts          # NaN endpoint dropped


def test_segment_lines_skips_edges_with_missing_endpoint():
    points = {"left_hip": (0, 0, 0), "left_knee": (0, 0, 1)}
    lines = segment_lines(points, edges=[("left_hip", "left_knee"),
                                         ("left_knee", "left_ankle")])
    assert lines == [((0, 0, 0), (0, 0, 1))]


def test_skeleton_edges_reference_known_landmarks():
    names = set(GAIT_LANDMARKS)
    for a, b in SKELETON_EDGES:
        assert a in names and b in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_skeleton_3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modules.visualization'`.

- [ ] **Step 3: Create the package + implementation**

Create empty `gait_analysis/modules/visualization/__init__.py`.

Create `gait_analysis/modules/visualization/skeleton_3d.py`:

```python
"""GL-free skeleton geometry: connectivity + per-frame point/segment extraction."""
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


def frame_points(df, frame_idx):
    """Return {landmark: (x, y, z)} for one frame, omitting missing/NaN landmarks."""
    row = df.iloc[frame_idx]
    points = {}
    for col in df.columns:
        if not col.endswith("_x"):
            continue
        name = col[:-2]
        xyz = (row.get(f"{name}_x"), row.get(f"{name}_y"), row.get(f"{name}_z"))
        if any(v is None for v in xyz) or any(pd.isna(v) for v in xyz):
            continue
        points[name] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return points


def segment_lines(points, edges=SKELETON_EDGES):
    """Return [((x,y,z),(x,y,z)), ...] for edges whose endpoints both exist in points."""
    lines = []
    for a, b in edges:
        if a in points and b in points:
            lines.append((points[a], points[b]))
    return lines
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_skeleton_3d.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add gait_analysis/modules/visualization/__init__.py gait_analysis/modules/visualization/skeleton_3d.py gait_analysis/tests/test_skeleton_3d.py
git commit -m "feat(viz): GL-free skeleton geometry core"
```

---

## Task 4: Visualization core — `angle_plots.py`

Build a matplotlib `Figure` (object-oriented API, no pyplot, no backend dependency) of joint-angle curves over the 0–100% gait cycle, mean ± STD corridor, one subplot per joint, left/right overlaid.

**Files:**
- Create: `gait_analysis/modules/visualization/angle_plots.py`
- Test: `gait_analysis/tests/test_angle_plots.py`

- [ ] **Step 1: Write the failing test**

Create `gait_analysis/tests/test_angle_plots.py`:

```python
import numpy as np

from modules.visualization.angle_plots import plot_joint_angles


def test_returns_one_axis_per_joint():
    fig = plot_joint_angles({}, {}, joints=("hip", "knee", "ankle"))
    assert len(fig.axes) == 3


def test_line_ydata_matches_mean():
    vals = list(np.linspace(0, 60, 101))
    fig = plot_joint_angles({"left_knee": vals}, {"left_knee": [0.0] * 101}, joints=("knee",))
    line = fig.axes[0].lines[0]
    assert line.get_ydata()[-1] == vals[-1]


def test_reuses_provided_figure():
    from matplotlib.figure import Figure
    fig = Figure()
    out = plot_joint_angles({"left_knee": [0.0] * 101}, {}, joints=("knee",), fig=fig)
    assert out is fig                       # draws onto the given figure, returns it
    assert len(fig.axes) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_angle_plots.py -v`
Expected: FAIL — `ModuleNotFoundError` / `cannot import name 'plot_joint_angles'`.

- [ ] **Step 3: Create the implementation**

Create `gait_analysis/modules/visualization/angle_plots.py`:

```python
"""Matplotlib figures for joint-angle curves over the gait cycle (OO API, no pyplot)."""
import numpy as np
from matplotlib.figure import Figure


def plot_joint_angles(angles_mean, angles_std, joints=("hip", "knee", "ankle"),
                      norm_corridor=None, fig=None):
    """Build/refresh a Figure: one subplot per joint, left/right mean +- STD over 0-100% cycle.

    angles_mean / angles_std : {"<side>_<joint>": [101 values]} (e.g. "left_knee").
    norm_corridor : optional {"<joint>": (low[101], high[101])} normative band.
    fig : optional existing Figure to draw onto (cleared first); a new one is created if None.
    Returns the Figure.
    """
    if fig is None:
        fig = Figure(figsize=(4 * len(joints), 3.2))
    else:
        fig.clear()
    axes = fig.subplots(1, len(joints), squeeze=False)[0]
    x = np.linspace(0, 100, 101)
    for ax, joint in zip(axes, joints):
        for side, color in (("left", "tab:blue"), ("right", "tab:red")):
            mean = angles_mean.get(f"{side}_{joint}")
            if mean is None:
                continue
            mean = np.asarray(mean, float)
            ax.plot(x, mean, color=color, label=side)
            std = angles_std.get(f"{side}_{joint}")
            if std is not None:
                std = np.asarray(std, float)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        if norm_corridor and joint in norm_corridor:
            low, high = norm_corridor[joint]
            ax.fill_between(x, low, high, color="tab:green", alpha=0.15, label="norm")
        ax.set_title(joint)
        ax.set_xlabel("% gait cycle")
        ax.set_ylabel("angle (deg)")
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_angle_plots.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add gait_analysis/modules/visualization/angle_plots.py gait_analysis/tests/test_angle_plots.py
git commit -m "feat(viz): joint-angle cycle plots (matplotlib OO figure)"
```

---

## Task 5: Visualization core — `export.py`

Export spatiotemporal results to CSV / multi-sheet XLSX, and a matplotlib figure to PNG.

**Files:**
- Create: `gait_analysis/modules/visualization/export.py`
- Test: `gait_analysis/tests/test_export.py`

- [ ] **Step 1: Write the failing test**

Create `gait_analysis/tests/test_export.py`:

```python
import openpyxl
from matplotlib.figure import Figure

from modules.visualization.export import (export_figure_png, export_results_csv,
                                          export_results_xlsx)

RESULTS = {
    "spatiotemporal": {"cadence_steps_per_min": 110.0, "speed_m_per_s": 1.2},
    "joint_angles_mean": {"left_knee": [0.0, 30.0, 60.0]},
}


def test_export_csv_roundtrip(tmp_path):
    p = tmp_path / "r.csv"
    export_results_csv(RESULTS, p)
    text = p.read_text()
    assert "cadence_steps_per_min" in text and "110.0" in text


def test_export_xlsx_has_expected_sheets(tmp_path):
    p = tmp_path / "r.xlsx"
    export_results_xlsx(RESULTS, p)
    wb = openpyxl.load_workbook(p)
    assert "spatiotemporal" in wb.sheetnames
    assert "joint_angles_mean" in wb.sheetnames


def test_export_png_creates_nonempty_file(tmp_path):
    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    p = tmp_path / "f.png"
    export_figure_png(fig, p)
    assert p.exists() and p.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_export.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Create the implementation**

Create `gait_analysis/modules/visualization/export.py`:

```python
"""Export gait results to CSV / XLSX and figures to PNG."""
import csv

import openpyxl


def export_results_csv(results, path):
    """Write spatiotemporal parameters as a (parameter, value) CSV."""
    st = results.get("spatiotemporal", {})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for k, v in st.items():
            writer.writerow([k, v])


def export_results_xlsx(results, path):
    """Write spatiotemporal + per-joint mean angle curves to a multi-sheet XLSX."""
    wb = openpyxl.Workbook()
    st_sheet = wb.active
    st_sheet.title = "spatiotemporal"
    st_sheet.append(["parameter", "value"])
    for k, v in results.get("spatiotemporal", {}).items():
        st_sheet.append([k, v])

    ang = wb.create_sheet("joint_angles_mean")
    means = results.get("joint_angles_mean", {})
    keys = list(means.keys())
    ang.append(["percent_cycle"] + keys)
    n = max((len(v) for v in means.values()), default=0)
    for i in range(n):
        ang.append([i] + [means[k][i] for k in keys])
    wb.save(path)


def export_figure_png(fig, path):
    """Save a matplotlib Figure to PNG."""
    fig.savefig(path, dpi=120)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_export.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add gait_analysis/modules/visualization/export.py gait_analysis/tests/test_export.py
git commit -m "feat(viz): CSV/XLSX/PNG exporters"
```

---

## Task 6: GUI worker + headless test config

The worker loads a session and runs the pipeline **off the UI thread**, emitting Qt signals. Add `conftest.py` to force a headless Qt platform for all GUI tests.

**Files:**
- Create: `gait_analysis/tests/conftest.py`
- Create: `gait_analysis/gui/__init__.py` (empty)
- Create: `gait_analysis/gui/worker.py`
- Test: `gait_analysis/tests/test_gui_smoke.py` (created here; grows in later tasks)

- [ ] **Step 1: Create the headless test config**

Create `gait_analysis/tests/conftest.py`:

```python
"""Force a headless Qt platform before pytest-qt creates the QApplication."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
```

- [ ] **Step 2: Write the failing test**

Create `gait_analysis/tests/test_gui_smoke.py`:

```python
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")


def fixture_results_df():
    """A tiny processed-style df + results pair for GUI smoke tests."""
    n = 20
    df = pd.DataFrame({"timestamp": np.linspace(0, 1, n)})
    for name, base in (("left_hip", 0.9), ("left_knee", 0.5), ("left_ankle", 0.1)):
        df[f"{name}_x"] = np.zeros(n)
        df[f"{name}_y"] = np.zeros(n)
        df[f"{name}_z"] = base + np.linspace(0, 0.1, n)
    results = {
        "fps": 19.0, "n_frames": n,
        "spatiotemporal": {"cadence_steps_per_min": 110.0, "speed_m_per_s": 1.2},
        "joint_angles_mean": {"left_knee": list(np.linspace(0, 60, 101))},
        "joint_angles_std": {"left_knee": [2.0] * 101},
    }
    return results, df


def test_worker_error_path_emits_error():
    from gui.worker import PipelineWorker
    worker = PipelineWorker("/nonexistent/session/folder", "SIMPLE_HOLISTIC", {})
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors          # bad folder -> error signal, no crash, no exception escapes
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui'`.

- [ ] **Step 4: Create the package + worker**

Create empty `gait_analysis/gui/__init__.py`.

Create `gait_analysis/gui/worker.py`:

```python
"""Background worker: load a session + run the pipeline off the GUI thread."""
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from modules.data_loader.caliscope_reader import load_caliscope_session
from pipeline import run_pipeline


class PipelineWorker(QObject):
    progress = pyqtSignal(float, str)        # (fraction 0..1, stage label)
    finished = pyqtSignal(dict, object)      # (results, df_processed)
    error = pyqtSignal(str)

    def __init__(self, folder, model, cfg):
        super().__init__()
        self._folder = folder
        self._model = model
        self._cfg = cfg

    def run(self):
        try:
            self.progress.emit(0.0, "Loading session")
            df = load_caliscope_session(self._folder, model=self._model)
            results, df_out = run_pipeline(
                df, self._cfg, model=self._model,
                session_id=Path(self._folder).name,
                progress_cb=lambda frac, stage: self.progress.emit(frac, stage),
            )
            self.finished.emit(results, df_out)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, must never crash
            self.error.emit(str(exc))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py -v`
Expected: PASS — the bad folder makes `load_caliscope_session` raise, which is caught and re-emitted as `error`.

- [ ] **Step 6: Commit**

```bash
git add gait_analysis/tests/conftest.py gait_analysis/gui/__init__.py gait_analysis/gui/worker.py gait_analysis/tests/test_gui_smoke.py
git commit -m "feat(gui): pipeline worker + headless test config"
```

---

## Task 7: Widget — `plot_widget.py`

A Qt canvas hosting the joint-angle figure, re-rendered in place.

**Files:**
- Create: `gait_analysis/gui/widgets/__init__.py` (empty)
- Create: `gait_analysis/gui/widgets/plot_widget.py`
- Test: append to `gait_analysis/tests/test_gui_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `gait_analysis/tests/test_gui_smoke.py`:

```python
def test_plot_widget_renders_angles(qtbot):
    from gui.widgets.plot_widget import PlotWidget
    w = PlotWidget()
    qtbot.addWidget(w)
    results, _ = fixture_results_df()
    w.render_angles(results["joint_angles_mean"], results["joint_angles_std"])
    assert w.current_figure.axes                     # axes exist after render
    assert w.current_figure.axes[0].lines            # at least one curve drawn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_plot_widget_renders_angles -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui.widgets'`.

- [ ] **Step 3: Create the package + widget**

Create empty `gait_analysis/gui/widgets/__init__.py`.

Create `gait_analysis/gui/widgets/plot_widget.py`:

```python
"""Qt canvas hosting the joint-angle matplotlib figure."""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from modules.visualization.angle_plots import plot_joint_angles


class PlotWidget(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(figsize=(8, 3.2))
        super().__init__(fig)
        plot_joint_angles({}, {}, fig=self.figure)

    def render_angles(self, angles_mean, angles_std):
        plot_joint_angles(angles_mean, angles_std, fig=self.figure)
        self.draw()

    @property
    def current_figure(self):
        return self.figure
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_plot_widget_renders_angles -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gait_analysis/gui/widgets/__init__.py gait_analysis/gui/widgets/plot_widget.py gait_analysis/tests/test_gui_smoke.py
git commit -m "feat(gui): matplotlib plot widget"
```

---

## Task 8: Widget — `skeleton_widget.py` (vispy)

vispy `SceneCanvas` wrapper; `.native` is the embeddable Qt widget. Uses the GL-free core from Task 3 for geometry.

**Files:**
- Create: `gait_analysis/gui/widgets/skeleton_widget.py`
- Test: append to `gait_analysis/tests/test_gui_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `gait_analysis/tests/test_gui_smoke.py`:

```python
def test_skeleton_widget_sets_frames(qtbot):
    try:
        from gui.widgets.skeleton_widget import SkeletonWidget
        w = SkeletonWidget()
    except Exception as exc:                 # vispy/GL unavailable in this env
        pytest.skip(f"vispy widget unavailable: {exc}")
    _, df = fixture_results_df()
    w.set_data(df)
    assert w.n_frames == len(df)
    w.set_frame(5)                            # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_skeleton_widget_sets_frames -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui.widgets.skeleton_widget'`.

- [ ] **Step 3: Create the widget**

Create `gait_analysis/gui/widgets/skeleton_widget.py`:

```python
"""vispy 3D skeleton widget embedded in Qt (geometry from the GL-free core)."""
import numpy as np
import vispy

vispy.use(app="pyqt6")
from vispy import scene  # noqa: E402

from modules.visualization.skeleton_3d import frame_points, segment_lines  # noqa: E402


class SkeletonWidget:
    """Wraps a vispy SceneCanvas. Embed `self.native` (a QWidget) into a layout."""

    def __init__(self):
        self.canvas = scene.SceneCanvas(keys="interactive", show=False, bgcolor="white")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45)
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.lines = scene.visuals.Line(parent=self.view.scene, color="black",
                                        width=2, connect="segments")
        self._df = None

    @property
    def native(self):
        return self.canvas.native

    @property
    def n_frames(self):
        return 0 if self._df is None else len(self._df)

    def set_data(self, df):
        self._df = df
        self.set_frame(0)
        self.view.camera.set_range()

    def set_frame(self, i):
        if self._df is None or not (0 <= i < len(self._df)):
            return
        pts = frame_points(self._df, i)
        if pts:
            coords = np.array(list(pts.values()), dtype=np.float32)
            self.markers.set_data(coords, size=8, face_color="red")
        segs = segment_lines(pts)
        if segs:
            flat = np.array([p for seg in segs for p in seg], dtype=np.float32)
            self.lines.set_data(pos=flat)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_skeleton_widget_sets_frames -v`
Expected: PASS (or SKIP if GL is unavailable in the runner — the manual-visual tier in Task 12 covers real rendering).

- [ ] **Step 5: Commit**

```bash
git add gait_analysis/gui/widgets/skeleton_widget.py gait_analysis/tests/test_gui_smoke.py
git commit -m "feat(gui): vispy 3D skeleton widget"
```

---

## Task 9: Panel — `analyze_panel.py`

Tab «Анализ»: folder picker, model dropdown, two param fields, Run (spawns the worker on a `QThread`), progress bar, data-quality line, results table, CSV/XLSX export. Emits `analysis_done(results, df)` for the Viz tab.

**Files:**
- Create: `gait_analysis/gui/panels/__init__.py` (empty)
- Create: `gait_analysis/gui/panels/analyze_panel.py`
- Test: append to `gait_analysis/tests/test_gui_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `gait_analysis/tests/test_gui_smoke.py`:

```python
def test_analyze_panel_fills_table_and_quality_on_finished(qtbot):
    from gui.panels.analyze_panel import AnalyzePanel
    panel = AnalyzePanel()
    qtbot.addWidget(panel)
    results, df = fixture_results_df()
    panel._on_finished(results, df)
    assert panel.table.rowCount() == len(results["spatiotemporal"])
    assert panel.csv_btn.isEnabled() and panel.xlsx_btn.isEnabled()
    assert "frames 20" in panel.quality.text()


def test_analyze_panel_emits_analysis_done(qtbot):
    from gui.panels.analyze_panel import AnalyzePanel
    panel = AnalyzePanel()
    qtbot.addWidget(panel)
    results, df = fixture_results_df()
    with qtbot.waitSignal(panel.analysis_done, timeout=1000):
        panel._on_finished(results, df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_analyze_panel_fills_table_and_quality_on_finished -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui.panels'`.

- [ ] **Step 3: Create the package + panel**

Create empty `gait_analysis/gui/panels/__init__.py`.

Create `gait_analysis/gui/panels/analyze_panel.py`:

```python
"""Tab «Анализ»: load + params + run + results table + data-quality line + export."""
from pathlib import Path

import numpy as np
import yaml
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                             QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from gui.worker import PipelineWorker
from modules.data_loader.landmarks import MODELS

GAIT_DIR = Path(__file__).resolve().parents[2]      # gait_analysis/


def _load_settings():
    with open(GAIT_DIR / "settings.yaml") as f:
        return yaml.safe_load(f)


class AnalyzePanel(QWidget):
    analysis_done = pyqtSignal(dict, object)        # (results, df) -> MainWindow -> VizPanel

    def __init__(self):
        super().__init__()
        self._cfg = _load_settings()
        self._folder = None
        self._results = None

        proc = self._cfg["processing"]
        self.folder_btn = QPushButton("Choose session folder…")
        self.folder_label = QLabel("(none)")
        self.model = QComboBox()
        self.model.addItems(MODELS)
        self.model.setCurrentText(proc["default_model"])
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0.5, 20.0)
        self.cutoff.setValue(proc["filter_cutoff_hz"])
        self.min_stride = QDoubleSpinBox()
        self.min_stride.setRange(0.1, 3.0)
        self.min_stride.setSingleStep(0.1)
        self.min_stride.setValue(proc["min_stride_duration_sec"])
        self.run_btn = QPushButton("▶ Run analysis")
        self.run_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.quality = QLabel("")
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["parameter", "value"])
        self.csv_btn = QPushButton("Export CSV")
        self.csv_btn.setEnabled(False)
        self.xlsx_btn = QPushButton("Export XLSX")
        self.xlsx_btn.setEnabled(False)

        form = QFormLayout()
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_btn)
        folder_row.addWidget(self.folder_label)
        form.addRow(folder_row)
        form.addRow("Model", self.model)
        form.addRow("Filter cutoff (Hz)", self.cutoff)
        form.addRow("Min stride (s)", self.min_stride)
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

        self.folder_btn.clicked.connect(self._choose_folder)
        self.run_btn.clicked.connect(self._run)
        self.csv_btn.clicked.connect(self._export_csv)
        self.xlsx_btn.clicked.connect(self._export_xlsx)

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose caliscope session folder")
        if path:
            self._folder = path
            self.folder_label.setText(Path(path).name)
            self.run_btn.setEnabled(True)

    def _build_cfg(self):
        cfg = dict(self._cfg)
        cfg["processing"] = dict(self._cfg["processing"])
        cfg["processing"]["filter_cutoff_hz"] = self.cutoff.value()
        cfg["processing"]["min_stride_duration_sec"] = self.min_stride.value()
        return cfg

    def _run(self):
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self._thread = QThread()
        self._worker = PipelineWorker(self._folder, self.model.currentText(),
                                      self._build_cfg())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, frac, stage):
        self.progress.setValue(int(frac * 100))

    def _on_finished(self, results, df):
        self._results = results
        st = results.get("spatiotemporal", {})
        self.table.setRowCount(len(st))
        for row, (k, v) in enumerate(st.items()):
            self.table.setItem(row, 0, QTableWidgetItem(k))
            self.table.setItem(row, 1, QTableWidgetItem("" if v is None else str(v)))
        self._fill_quality(results, df)
        self.run_btn.setEnabled(True)
        self.csv_btn.setEnabled(True)
        self.xlsx_btn.setEnabled(True)
        self.analysis_done.emit(results, df)

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        QMessageBox.warning(self, "Analysis failed", msg)

    def _fill_quality(self, results, df):
        fps = results.get("fps")
        n = results.get("n_frames", len(df))
        duration = round(n / fps, 1) if fps else 0
        bad = []
        for col in df.columns:
            if col.endswith("_x"):
                frac = float(np.isnan(df[col].to_numpy()).mean())
                if frac > 0.05:
                    bad.append(f"{col[:-2]} ({frac * 100:.0f}%)")
        warn = ("  ⚠ >5% NaN: " + ", ".join(bad)) if bad else ""
        self.quality.setText(f"frames {n} · fps {fps} · {duration}s{warn}")

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "gait.csv", "CSV (*.csv)")
        if path:
            from modules.visualization.export import export_results_csv
            export_results_csv(self._results, path)

    def _export_xlsx(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export XLSX", "gait.xlsx", "XLSX (*.xlsx)")
        if path:
            from modules.visualization.export import export_results_xlsx
            export_results_xlsx(self._results, path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py -k analyze_panel -v`
Expected: PASS (both panel tests).

- [ ] **Step 5: Commit**

```bash
git add gait_analysis/gui/panels/__init__.py gait_analysis/gui/panels/analyze_panel.py gait_analysis/tests/test_gui_smoke.py
git commit -m "feat(gui): Анализ panel (load/params/run/results/export)"
```

---

## Task 10: Panel — `viz_panel.py`

Tab «Визуализация»: skeleton widget on top, playback controls (frame slider, Play/Pause, speed), angle plots below, Export PNG.

**Files:**
- Create: `gait_analysis/gui/panels/viz_panel.py`
- Test: append to `gait_analysis/tests/test_gui_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `gait_analysis/tests/test_gui_smoke.py`:

```python
def test_viz_panel_set_data_wires_slider_and_plots(qtbot):
    try:
        from gui.panels.viz_panel import VizPanel
        panel = VizPanel()
    except Exception as exc:                 # vispy/GL unavailable
        pytest.skip(f"vispy widget unavailable: {exc}")
    qtbot.addWidget(panel)
    results, df = fixture_results_df()
    panel.set_data(results, df)
    assert panel.slider.maximum() == len(df) - 1
    assert panel.plots.current_figure.axes        # plots rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_viz_panel_set_data_wires_slider_and_plots -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui.panels.viz_panel'`.

- [ ] **Step 3: Create the panel**

Create `gait_analysis/gui/panels/viz_panel.py`:

```python
"""Tab «Визуализация»: 3D skeleton + playback controls + angle plots."""
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QPushButton,
                             QSlider, QVBoxLayout, QWidget)

from gui.widgets.plot_widget import PlotWidget
from gui.widgets.skeleton_widget import SkeletonWidget
from modules.visualization.export import export_figure_png

_SPEEDS = {"0.25×": 0.25, "0.5×": 0.5, "1×": 1.0}


class VizPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.skeleton = SkeletonWidget()
        self.plots = PlotWidget()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.play_btn = QPushButton("▶")
        self.speed = QComboBox()
        self.speed.addItems(list(_SPEEDS))
        self.speed.setCurrentText("1×")
        self.export_btn = QPushButton("Export PNG")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        controls = QHBoxLayout()
        controls.addWidget(self.play_btn)
        controls.addWidget(self.slider)
        controls.addWidget(self.speed)
        controls.addWidget(self.export_btn)
        layout = QVBoxLayout(self)
        layout.addWidget(self.skeleton.native, stretch=3)
        layout.addLayout(controls)
        layout.addWidget(self.plots, stretch=2)

        self.slider.valueChanged.connect(self.skeleton.set_frame)
        self.play_btn.clicked.connect(self._toggle_play)
        self.export_btn.clicked.connect(self._export_png)

    def set_data(self, results, df):
        self.skeleton.set_data(df)
        self.slider.setRange(0, max(0, len(df) - 1))
        self.slider.setValue(0)
        self.plots.render_angles(results.get("joint_angles_mean", {}),
                                 results.get("joint_angles_std", {}))

    def _toggle_play(self):
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("▶")
        else:
            factor = _SPEEDS[self.speed.currentText()]
            self._timer.start(int(1000 / (30 * factor)))
            self.play_btn.setText("⏸")

    def _advance(self):
        nxt = self.slider.value() + 1
        self.slider.setValue(0 if nxt > self.slider.maximum() else nxt)

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "angles.png", "PNG (*.png)")
        if path:
            export_figure_png(self.plots.current_figure, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_viz_panel_set_data_wires_slider_and_plots -v`
Expected: PASS (or SKIP if GL unavailable).

- [ ] **Step 5: Commit**

```bash
git add gait_analysis/gui/panels/viz_panel.py gait_analysis/tests/test_gui_smoke.py
git commit -m "feat(gui): Визуализация panel (skeleton + playback + plots)"
```

---

## Task 11: Main window + app entry point

Two-tab `QMainWindow` wiring `AnalyzePanel.analysis_done → VizPanel.set_data`, plus the `app.py` launcher.

**Files:**
- Create: `gait_analysis/gui/main_window.py`
- Create: `gait_analysis/app.py`
- Test: append to `gait_analysis/tests/test_gui_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `gait_analysis/tests/test_gui_smoke.py`:

```python
def test_main_window_has_two_tabs_and_routes_results(qtbot):
    try:
        from gui.main_window import MainWindow
        win = MainWindow()
    except Exception as exc:                 # vispy/GL unavailable (VizPanel)
        pytest.skip(f"vispy widget unavailable: {exc}")
    qtbot.addWidget(win)
    assert win.centralWidget().count() == 2
    results, df = fixture_results_df()
    win.analyze._on_finished(results, df)            # emits analysis_done
    assert win.viz.slider.maximum() == len(df) - 1   # routed into the Viz tab
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_main_window_has_two_tabs_and_routes_results -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui.main_window'`.

- [ ] **Step 3: Create the main window + app**

Create `gait_analysis/gui/main_window.py`:

```python
"""Main window: two-tab shell wiring AnalyzePanel -> VizPanel."""
from PyQt6.QtWidgets import QMainWindow, QTabWidget

from gui.panels.analyze_panel import AnalyzePanel
from gui.panels.viz_panel import VizPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Analysis")
        self.resize(1000, 750)
        self.analyze = AnalyzePanel()
        self.viz = VizPanel()
        tabs = QTabWidget()
        tabs.addTab(self.analyze, "Анализ")
        tabs.addTab(self.viz, "Визуализация")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready")
        self.analyze.analysis_done.connect(self.viz.set_data)
        self.analyze.analysis_done.connect(
            lambda *_: self.statusBar().showMessage("Analysis complete"))
```

Create `gait_analysis/app.py`:

```python
"""GUI entry point for the gait analysis application."""
import sys

import vispy

vispy.use(app="pyqt6")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_main_window_has_two_tabs_and_routes_results -v`
Expected: PASS (or SKIP if GL unavailable).

- [ ] **Step 5: Commit**

```bash
git add gait_analysis/gui/main_window.py gait_analysis/app.py gait_analysis/tests/test_gui_smoke.py
git commit -m "feat(gui): two-tab main window + app entry point"
```

---

## Task 12: Integration — full suite, lint, manual-visual checklist

**Files:**
- Modify: none (verification + a short doc note)
- Create: `gait_analysis/docs/manual_visual_checklist.md`

- [ ] **Step 1: Run the entire test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests PASS (GUI tests touching vispy may SKIP if the runner lacks GL; that is acceptable — they pass on a GL-capable machine). The Phase-1 suite must remain green.

- [ ] **Step 2: Coverage check**

Run: `.venv/bin/python -m pytest --cov=modules --cov=pipeline --cov-report=term-missing`
Expected: overall coverage > 70% (the pure-logic cores carry it).

- [ ] **Step 3: Lint**

Run: `.venv/bin/ruff check .`
Expected: clean. If imports need sorting, run `.venv/bin/ruff check --fix .` and re-run.

- [ ] **Step 4: Write the manual-visual checklist**

Create `gait_analysis/docs/manual_visual_checklist.md`:

```markdown
# Phase 2 Manual-Visual Checklist (real GL render)

Run on a machine with a display + working OpenGL:

    cd gait_analysis
    .venv/bin/python app.py

1. **Анализ tab:** click "Choose session folder…", pick `data/caliscope_project/recordings/p1_3`.
   - Run button enables.
2. Click "▶ Run analysis".
   - Progress bar fills to 100%; quality line shows `frames … · fps … · …s` (and any >5% NaN warning).
   - Results table fills with cadence / speed / stride / step / stance / swing.
3. Click "Export CSV" and "Export XLSX" → files open in a spreadsheet, values match the table.
4. **Визуализация tab:**
   - 3D skeleton renders; drag to orbit (turntable camera).
   - Drag the frame slider → skeleton pose updates.
   - Click ▶ → skeleton animates; change speed to 0.5× / 0.25× → playback slows; ⏸ stops.
   - Angle plots show knee/hip/ankle curves with the ±STD corridor.
5. Click "Export PNG" → saved figure matches the on-screen plots.
6. Trigger an error: pick a folder with no labelled CSV → a warning dialog appears, the app does
   not crash, and the Run button re-enables.
```

- [ ] **Step 5: Run the app once against real data (manual)**

Run: `cd gait_analysis && .venv/bin/python app.py` and walk the checklist above on `p1_3`.
Expected: every checklist item passes. (This is the only tier that exercises real GL pixels.)

- [ ] **Step 6: Commit**

```bash
git add gait_analysis/docs/manual_visual_checklist.md
git commit -m "docs(phase2): manual-visual verification checklist"
```

---

## Self-Review

**Spec coverage** (against `2026-05-30-gait-phase2-gui-design.md`):
- §1 prerequisite `run_pipeline` → Task 2. ✅
- §6.1 `skeleton_3d` core → Task 3. ✅
- §6.2 `angle_plots` → Task 4. ✅
- §6.3 `export` (CSV/XLSX/PNG) → Task 5. ✅
- §7.1 worker → Task 6. ✅
- §7.5 plot widget → Task 7; §7.4 skeleton widget → Task 8. ✅
- §7.2 Анализ panel (incl. D5 data-quality line) → Task 9. ✅
- §7.3 Визуализация panel → Task 10. ✅
- main window + app → Task 11. ✅
- §8 three test tiers: unit (Tasks 2–5), offscreen smoke (Tasks 6–11), manual-visual (Task 12). ✅
- §3 deps → Task 1. ✅
- §9 error handling → worker try/except (Task 6) + `_on_error` dialog (Task 9) + checklist item 6 (Task 12). ✅
- Acceptance #7 (skeleton+plots) → Tasks 8/10/12; #8 (CSV+XLSX) → Tasks 5/9; #9/#10 (tests/ruff) → Task 12. ✅
- Out of scope (Vicon tab, matplotlib-3D fallback) → correctly absent. ✅

**Placeholder scan:** none — every code/test step contains complete code; every command has expected output.

**Type/name consistency:** `run_pipeline(df, cfg, *, model, session_id, progress_cb)` identical in Tasks 2/6. `plot_joint_angles(..., fig=None)` identical in Tasks 4/7. `PlotWidget.render_angles` / `.current_figure`, `SkeletonWidget.set_data` / `.set_frame` / `.n_frames` / `.native`, `AnalyzePanel.analysis_done` / `._on_finished`, `VizPanel.set_data` — all consistent across the tasks that use them.
