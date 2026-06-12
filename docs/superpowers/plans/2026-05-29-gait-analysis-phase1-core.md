# Gait Analysis — Phase 1 (Core Spine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Vicon-free analysis spine — Module 1 (data loader/sync) + Module 2 (kinematics) + a CLI — that ingests caliscope output and produces gait events, joint angles, and spatiotemporal parameters end-to-end on the `p1_*` sessions, plus a Level-B reproducibility run.

**Architecture:** Pure-Python packages under `modules/`, no GUI imports. A unified wide-format pose DataFrame (`frame, timestamp, <landmark>_{x,y,z}`) flows between functions. `cli.py` orchestrates the fixed pipeline `fill_gaps → butterworth → detect_events → joint_angles → normalize → spatiotemporal` and writes `gait_results.json`. Everything tunable lives in `settings.yaml`.

**Tech Stack:** Python 3.14, numpy, pandas, scipy, PyYAML, openpyxl, pytest, pytest-cov, ruff. (Module 3's `pingouin`/PyQt6/vispy belong to later phases but are listed in `requirements.txt`.)

**Reference spec:** `docs/superpowers/specs/2026-05-29-gait-kinematics-analysis-design.md`

---

## File Structure (Phase 1)

```
gait_analysis/
├── cli.py                              # CLI entry point (analyze, reproducibility)
├── settings.yaml                       # paths, thresholds, landmark_mapping
├── requirements.txt
├── pyproject.toml                      # pytest + ruff config
├── modules/
│   ├── __init__.py
│   ├── data_loader/
│   │   ├── __init__.py
│   │   ├── landmarks.py                # MODELS, GAIT_LANDMARKS constants
│   │   ├── caliscope_reader.py         # load_caliscope_session, list_landmarks, derive_fps
│   │   ├── config_reader.py            # load_camera_config
│   │   ├── vicon_reader.py             # load_vicon_xlsx, map_vicon_to_caliscope
│   │   └── synchronizer.py             # synchronize
│   └── kinematics/
│       ├── __init__.py
│       ├── filters.py                  # fill_gaps, butterworth_filter
│       ├── gait_events.py              # detect_gait_events
│       ├── joint_angles.py             # calc_angle_3d, calc_joint_angles_timeseries
│       ├── normalizer.py               # normalize_gait_cycle, get_mean_std_cycle
│       └── spatiotemporal.py           # calc_spatiotemporal
├── tests/
│   ├── conftest.py                     # synthetic DataFrame fixtures
│   ├── fixtures/
│   │   ├── mini_labelled.csv
│   │   ├── mini_frame_time_history.csv
│   │   └── mini_config.toml
│   ├── test_caliscope_reader.py
│   ├── test_config_reader.py
│   ├── test_synchronizer.py
│   ├── test_vicon_reader.py
│   ├── test_filters.py
│   ├── test_gait_events.py
│   ├── test_joint_angles.py
│   ├── test_normalizer.py
│   ├── test_spatiotemporal.py
│   └── test_cli_integration.py
└── data/
    └── caliscope_project -> ../../caliscope_project_271025   # symlink
```

All work happens inside a new `gait_analysis/` directory at the repo root. Paths below are relative to the repo root (`/home/grivin/Workspace/Cerebral`).

---

## Conventions locked for this plan (do not deviate)

- **Canonical landmark names:** lowercase, side-prefixed — `left_hip`, `right_knee`, `left_heel`, `left_foot_index`, `left_ankle`, `right_ankle`, `left_shoulder`, etc.
- **Unified DataFrame:** columns `frame` (int 0..N-1), `timestamp` (float seconds, monotonic), then `<landmark>_x|_y|_z` (meters, NaN if missing). `df.attrs['fps']` and `df.attrs['model']` are set by the loader.
- **Vertical axis:** `z` (default, configurable).
- **Joint angle output:** `calc_joint_angles_timeseries` writes the **included angle in degrees, range 0–180** in columns `<side>_<joint>_angle` (e.g. `left_knee_angle`). Clinical flexion-sign conversion is deferred to the Vicon-comparison phase (documented in the docstring).
- **fps is always derived** from data, never assumed.

---

## Task 0: Project scaffold, dependencies, and venv

**Files:**
- Create: `gait_analysis/requirements.txt`
- Create: `gait_analysis/pyproject.toml`
- Create: `gait_analysis/settings.yaml`
- Create: `gait_analysis/modules/__init__.py`, `gait_analysis/modules/data_loader/__init__.py`, `gait_analysis/modules/kinematics/__init__.py`
- Create: `gait_analysis/tests/__init__.py`

- [ ] **Step 1: Create directory tree and empty packages**

Run:
```bash
cd /home/grivin/Workspace/Cerebral
mkdir -p gait_analysis/modules/data_loader gait_analysis/modules/kinematics \
         gait_analysis/tests/fixtures gait_analysis/data
touch gait_analysis/modules/__init__.py \
      gait_analysis/modules/data_loader/__init__.py \
      gait_analysis/modules/kinematics/__init__.py \
      gait_analysis/tests/__init__.py
ln -s ../../caliscope_project_271025 gait_analysis/data/caliscope_project
```

- [ ] **Step 2: Write `gait_analysis/requirements.txt`**

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
PyYAML>=6.0
openpyxl>=3.1
pingouin>=0.5
matplotlib>=3.7
plotly>=5.15
pytest>=7.4
pytest-cov>=4.1
ruff>=0.1
```

- [ ] **Step 3: Write `gait_analysis/pyproject.toml`**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

- [ ] **Step 4: Write `gait_analysis/settings.yaml`**

```yaml
paths:
  caliscope_root: data/caliscope_project
  vicon_root: data/Vicon_10_series
  output_dir: results/
  camera_config: data/caliscope_project/config.toml

processing:
  default_model: SIMPLE_HOLISTIC      # POSE | SIMPLE_HOLISTIC | HOLISTIC
  target_fps: 100                     # sync grid for Vicon comparison only
  filter_cutoff_hz: 6.0
  filter_order: 4
  max_gap_frames: 10
  min_stride_duration_sec: 0.3

gait_events:
  method: velocity                    # velocity | height | combined
  heel_landmark: heel
  toe_landmark: foot_index
  vertical_axis: z

landmark_mapping:                     # Vicon marker -> canonical landmark
  LASI: left_hip
  RASI: right_hip
  LKNE: left_knee
  RKNE: right_knee
  LANK: left_ankle
  RANK: right_ankle
  LHEE: left_heel
  RHEE: right_heel
  LTOE: left_foot_index
  RTOE: right_foot_index

comparison:
  icc_type: '3,1'
  good_rmse_threshold_deg: 5.0
  acceptable_rmse_threshold_deg: 10.0
```

- [ ] **Step 5: Create venv and install dependencies**

Run:
```bash
cd /home/grivin/Workspace/Cerebral/gait_analysis
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import numpy, pandas, scipy, yaml, openpyxl, pytest; print('deps OK')"
```
Expected: `deps OK`. If a wheel is missing for Python 3.14, relax the pin (e.g. install latest available) and note it.

- [ ] **Step 6: Verify pytest collects nothing yet (sanity)**

Run: `.venv/bin/pytest`
Expected: `no tests ran` (exit code 5) — confirms the harness works.

- [ ] **Step 7: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/requirements.txt gait_analysis/pyproject.toml gait_analysis/settings.yaml \
        gait_analysis/modules gait_analysis/tests/__init__.py
echo "gait_analysis/.venv/" >> .gitignore
git add .gitignore
git commit -m "chore: scaffold gait_analysis package, deps, settings"
```

> All subsequent `pytest`/`python` commands use `gait_analysis/.venv/bin/...` and run from `gait_analysis/`.

---

## Task 1: `filters.fill_gaps`

**Files:**
- Create: `gait_analysis/modules/kinematics/filters.py`
- Test: `gait_analysis/tests/test_filters.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_filters.py
import numpy as np
import pandas as pd
import pytest
from modules.kinematics.filters import fill_gaps


def test_fill_gaps_fills_short_interior_gap_on_linear_ramp():
    # Cubic interpolation through collinear points reproduces the line exactly.
    df = pd.DataFrame({"left_heel_z": [0.0, 2.0, 4.0, np.nan, 8.0, 10.0]})
    out = fill_gaps(df, max_gap_frames=10)
    assert out["left_heel_z"].isna().sum() == 0
    assert out["left_heel_z"].iloc[3] == pytest.approx(6.0)


def test_fill_gaps_leaves_long_gap_untouched():
    col = [0.0, 1.0, np.nan, np.nan, np.nan, 5.0, 6.0]  # 3-long gap
    df = pd.DataFrame({"left_heel_z": col})
    out = fill_gaps(df, max_gap_frames=2)  # 3 > 2 -> stays fully NaN
    assert out["left_heel_z"].iloc[2:5].isna().all()


def test_fill_gaps_ignores_non_coordinate_columns():
    # left_hip_x is the line y = x + 1 with a one-frame hole at index 1.
    df = pd.DataFrame({"frame": [0, 1, 2, 3, 4],
                       "timestamp": [0.0, 0.1, 0.2, 0.3, 0.4],
                       "left_hip_x": [1.0, np.nan, 3.0, 4.0, 5.0]})
    out = fill_gaps(df, max_gap_frames=5)
    assert list(out["frame"]) == [0, 1, 2, 3, 4]
    assert out["left_hip_x"].iloc[1] == pytest.approx(2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_filters.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modules.kinematics.filters'`

- [ ] **Step 3: Write minimal implementation**

```python
# modules/kinematics/filters.py
"""Signal conditioning: gap filling and zero-phase Butterworth filtering."""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

_COORD_SUFFIXES = ("_x", "_y", "_z")


def _is_coord_column(name: str) -> bool:
    return name.endswith(_COORD_SUFFIXES)


def _nan_run_bounds(isna: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) index ranges of consecutive-NaN runs."""
    runs = []
    i, n = 0, len(isna)
    while i < n:
        if isna[i]:
            j = i
            while j < n and isna[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def fill_gaps(df: pd.DataFrame, max_gap_frames: int = 10) -> pd.DataFrame:
    """Cubic-interpolate interior NaN runs no longer than ``max_gap_frames``.

    All-or-nothing per gap: a NaN run longer than ``max_gap_frames`` stays
    fully NaN. Leading/trailing NaNs are left untouched. Only coordinate
    columns (``*_x/_y/_z``) are processed. Call this BEFORE filtering.
    """
    out = df.copy()
    for col in out.columns:
        if not _is_coord_column(col):
            continue
        s = out[col]
        isna = s.isna().to_numpy()
        if s.notna().sum() < 4:  # cubic needs >=4 valid points
            continue
        # Fully interpolate interior gaps, then restore NaN over over-long runs.
        filled = s.interpolate(method="cubic", limit_area="inside").to_numpy()
        for start, end in _nan_run_bounds(isna):
            if (end - start) > max_gap_frames:
                filled[start:end] = np.nan
        out[col] = filled
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_filters.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/kinematics/filters.py gait_analysis/tests/test_filters.py
git commit -m "feat(kinematics): fill_gaps cubic interpolation of short tracking gaps"
```

---

## Task 2: `filters.butterworth_filter`

**Files:**
- Modify: `gait_analysis/modules/kinematics/filters.py`
- Test: `gait_analysis/tests/test_filters.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_filters.py
from modules.kinematics.filters import butterworth_filter


def test_butterworth_attenuates_high_freq_keeps_low_freq():
    fs = 100.0
    t = np.arange(0, 2.0, 1 / fs)
    low = np.sin(2 * np.pi * 1.0 * t)      # 1 Hz (pass)
    high = 0.5 * np.sin(2 * np.pi * 20.0 * t)  # 20 Hz (stop)
    out = butterworth_filter(low + high, cutoff_hz=6.0, fs=fs, order=4)
    # Output length preserved.
    assert out.shape == low.shape
    # Low-frequency component largely preserved, high-frequency removed.
    err_low = np.sqrt(np.mean((out - low) ** 2))
    assert err_low < 0.1


def test_butterworth_zero_phase_no_lag_on_symmetric_pulse():
    fs = 100.0
    x = np.zeros(200)
    x[100] = 1.0  # symmetric impulse
    out = butterworth_filter(x, cutoff_hz=10.0, fs=fs, order=4, zero_phase=True)
    # Zero-phase filtering keeps the peak at the same index.
    assert int(np.argmax(out)) == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_filters.py -k butterworth -v`
Expected: FAIL — `ImportError: cannot import name 'butterworth_filter'`

- [ ] **Step 3: Write minimal implementation (append to `filters.py`)**

```python
def butterworth_filter(signal: np.ndarray, cutoff_hz: float = 6.0,
                       fs: float = 30.0, order: int = 4,
                       zero_phase: bool = True) -> np.ndarray:
    """Low-pass Butterworth filter. Default ``zero_phase`` uses ``filtfilt``
    (no phase lag). ``signal`` must be NaN-free (run ``fill_gaps`` first).
    """
    signal = np.asarray(signal, dtype=float)
    nyq = 0.5 * fs
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = butter(order, wn, btype="low")
    if not zero_phase:
        from scipy.signal import lfilter
        return lfilter(b, a, signal)
    return filtfilt(b, a, signal, padtype="odd")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_filters.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/kinematics/filters.py gait_analysis/tests/test_filters.py
git commit -m "feat(kinematics): zero-phase Butterworth low-pass filter"
```

---

## Task 3: Landmark constants + `caliscope_reader.list_landmarks` + `derive_fps`

**Files:**
- Create: `gait_analysis/modules/data_loader/landmarks.py`
- Create: `gait_analysis/modules/data_loader/caliscope_reader.py`
- Test: `gait_analysis/tests/test_caliscope_reader.py`

- [ ] **Step 1: Write `landmarks.py`** (no test needed — pure constants)

```python
# modules/data_loader/landmarks.py
"""Model names and the gait-relevant canonical landmark set."""

MODELS = ["POSE", "SIMPLE_HOLISTIC", "HOLISTIC"]

# Lower-body + shoulders: the landmarks Module 2 needs. Present in all 3 models.
GAIT_LANDMARKS = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
```

- [ ] **Step 2: Write the failing test for `list_landmarks` and `derive_fps`**

```python
# tests/test_caliscope_reader.py
import numpy as np
import pytest
from modules.data_loader.caliscope_reader import list_landmarks, derive_fps
from modules.data_loader.landmarks import GAIT_LANDMARKS


def test_list_landmarks_returns_gait_set_for_known_model():
    lm = list_landmarks("SIMPLE_HOLISTIC")
    assert "left_knee" in lm and "right_foot_index" in lm
    assert lm == GAIT_LANDMARKS


def test_list_landmarks_rejects_unknown_model():
    with pytest.raises(ValueError):
        list_landmarks("NOT_A_MODEL")


def test_derive_fps_from_even_timestamps():
    ts = np.arange(0, 1.0, 1 / 20.0)  # 20 fps
    assert derive_fps(ts) == pytest.approx(20.0, abs=0.01)


def test_derive_fps_robust_to_a_dropped_frame():
    ts = np.array([0.0, 0.05, 0.10, 0.20, 0.25])  # one gap at 0.15
    assert derive_fps(ts) == pytest.approx(20.0, abs=0.01)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_caliscope_reader.py -v`
Expected: FAIL — `ImportError` (functions not defined)

- [ ] **Step 4: Write minimal implementation**

```python
# modules/data_loader/caliscope_reader.py
"""Read caliscope 3D output into the unified pose DataFrame."""
from pathlib import Path

import numpy as np
import pandas as pd

from .landmarks import GAIT_LANDMARKS, MODELS


def list_landmarks(model: str) -> list[str]:
    """Gait-relevant canonical landmarks for a caliscope model."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
    return list(GAIT_LANDMARKS)


def derive_fps(timestamps) -> float:
    """Frame rate = 1 / median positive inter-frame interval."""
    ts = np.sort(np.asarray(timestamps, dtype=float))
    d = np.diff(ts)
    d = d[d > 0]
    if d.size == 0:
        raise ValueError("Cannot derive fps from constant/empty timestamps")
    return float(1.0 / np.median(d))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_caliscope_reader.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/data_loader/landmarks.py \
        gait_analysis/modules/data_loader/caliscope_reader.py \
        gait_analysis/tests/test_caliscope_reader.py
git commit -m "feat(data_loader): landmark constants, list_landmarks, derive_fps"
```

---

## Task 4: Test fixtures for the caliscope reader

**Files:**
- Create: `gait_analysis/tests/fixtures/mini_labelled.csv`
- Create: `gait_analysis/tests/fixtures/mini_frame_time_history.csv`

These mimic the real caliscope format at small scale: a labelled wide CSV keyed by `sync_index` (non-contiguous, starts > 0) and a per-port `frame_time_history.csv` whose `sync_index` aligns with it.

- [ ] **Step 1: Write `mini_labelled.csv`** (4 frames, sync 5,6,8,9 — note the gap at 7)

```
sync_index,left_hip_x,left_hip_y,left_hip_z,left_knee_x,left_knee_y,left_knee_z,left_heel_z,left_foot_index_z
5,0.10,2.40,0.42,0.11,2.41,0.00,-0.74,-0.79
6,0.11,2.41,0.42,0.12,2.42,0.01,-0.70,-0.70
8,0.13,2.43,0.42,0.14,2.44,0.02,-0.60,-0.50
9,0.14,2.44,0.42,0.15,2.45,0.03,-0.55,-0.40
```

- [ ] **Step 2: Write `mini_frame_time_history.csv`** (3 ports per sync_index; base clock 1000.0 s, 0.05 s steps → 20 fps)

```
sync_index,port,frame_index,frame_time
5,1,5,1000.0000
5,2,5,1000.0010
5,3,5,1000.0005
6,1,6,1000.0500
6,2,6,1000.0510
6,3,6,1000.0505
8,1,8,1000.1500
8,2,8,1000.1510
8,3,8,1000.1505
9,1,9,1000.2000
9,2,9,1000.2010
9,3,9,1000.2005
```

- [ ] **Step 3: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/tests/fixtures/mini_labelled.csv \
        gait_analysis/tests/fixtures/mini_frame_time_history.csv
git commit -m "test(data_loader): mini caliscope fixtures with sync gap"
```

---

## Task 5: `caliscope_reader.load_caliscope_session`

**Files:**
- Modify: `gait_analysis/modules/data_loader/caliscope_reader.py`
- Test: `gait_analysis/tests/test_caliscope_reader.py`

This task uses the fixtures via a temp session directory built in the test.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_caliscope_reader.py
import shutil
from pathlib import Path
import pandas as pd
from modules.data_loader.caliscope_reader import load_caliscope_session

FIX = Path(__file__).parent / "fixtures"


def _make_session(tmp_path, model="SIMPLE_HOLISTIC"):
    model_dir = tmp_path / "sess" / model
    model_dir.mkdir(parents=True)
    shutil.copy(FIX / "mini_labelled.csv", model_dir / f"xyz_{model}_labelled.csv")
    shutil.copy(FIX / "mini_frame_time_history.csv", model_dir / "frame_time_history.csv")
    return tmp_path / "sess"


def test_load_session_shape_and_columns(tmp_path):
    sess = _make_session(tmp_path)
    df = load_caliscope_session(str(sess), model="SIMPLE_HOLISTIC")
    assert list(df.columns[:2]) == ["frame", "timestamp"]
    assert "left_hip_x" in df.columns and "left_heel_z" in df.columns
    assert len(df) == 4
    assert list(df["frame"]) == [0, 1, 2, 3]


def test_load_session_timestamps_zero_based_and_monotonic(tmp_path):
    sess = _make_session(tmp_path)
    df = load_caliscope_session(str(sess), model="SIMPLE_HOLISTIC")
    assert df["timestamp"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert df["timestamp"].is_monotonic_increasing
    # mean of the three ports at sync 6 = ~0.0505 s after the sync-5 base
    assert df["timestamp"].iloc[1] == pytest.approx(0.0505, abs=1e-3)


def test_load_session_sets_fps_and_model_attrs(tmp_path):
    sess = _make_session(tmp_path)
    df = load_caliscope_session(str(sess), model="SIMPLE_HOLISTIC")
    assert df.attrs["model"] == "SIMPLE_HOLISTIC"
    assert df.attrs["fps"] == pytest.approx(20.0, abs=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_caliscope_reader.py -k load_session -v`
Expected: FAIL — `ImportError: cannot import name 'load_caliscope_session'`

- [ ] **Step 3: Write minimal implementation (append to `caliscope_reader.py`)**

```python
def _frame_timestamps(frame_time_csv: Path) -> pd.DataFrame:
    """Collapse per-port frame_time to one zero-based timestamp per sync_index."""
    fth = pd.read_csv(frame_time_csv)
    per_sync = fth.groupby("sync_index", as_index=False)["frame_time"].mean()
    per_sync["timestamp"] = per_sync["frame_time"] - per_sync["frame_time"].min()
    return per_sync[["sync_index", "timestamp"]]


def load_caliscope_session(session_dir: str, model: str = "SIMPLE_HOLISTIC") -> pd.DataFrame:
    """Load ``xyz_<model>_labelled.csv`` + real timestamps into a unified DataFrame.

    Returns columns ``frame, timestamp, <landmark>_{x,y,z}``. Sets
    ``df.attrs['fps']`` (derived) and ``df.attrs['model']``.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
    model_dir = Path(session_dir) / model
    labelled = pd.read_csv(model_dir / f"xyz_{model}_labelled.csv")
    times = _frame_timestamps(model_dir / "frame_time_history.csv")

    df = labelled.merge(times, on="sync_index", how="left")
    df = df.sort_values("sync_index").reset_index(drop=True)
    df.insert(0, "frame", range(len(df)))

    coord_cols = [c for c in df.columns
                  if c.endswith(("_x", "_y", "_z")) and c not in ("frame",)]
    df = df[["frame", "timestamp"] + coord_cols]

    df.attrs["fps"] = derive_fps(df["timestamp"].to_numpy())
    df.attrs["model"] = model
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_caliscope_reader.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/data_loader/caliscope_reader.py gait_analysis/tests/test_caliscope_reader.py
git commit -m "feat(data_loader): load_caliscope_session with real timestamps + derived fps"
```

---

## Task 6: `config_reader.load_camera_config`

**Files:**
- Create: `gait_analysis/modules/data_loader/config_reader.py`
- Create: `gait_analysis/tests/fixtures/mini_config.toml`
- Test: `gait_analysis/tests/test_config_reader.py`

- [ ] **Step 1: Write `mini_config.toml`**

```toml
camera_count = 1

[cam_1]
port = 1
size = [1920, 1080]
matrix = [[1129.0, 0.0, 949.7], [0.0, 1125.6, 515.0], [0.0, 0.0, 1.0]]
distortions = [0.148, 0.067, -0.0018, 0.0037, -0.62]
translation = [-0.463, -0.126, 1.874]
rotation = [1.557, 0.690, -0.659]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config_reader.py
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
    # Rotation matrix is orthonormal.
    assert np.allclose(extr["R"] @ extr["R"].T, np.eye(3), atol=1e-6)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: modules.data_loader.config_reader`

- [ ] **Step 4: Write minimal implementation**

```python
# modules/data_loader/config_reader.py
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
    return out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config_reader.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/data_loader/config_reader.py \
        gait_analysis/tests/fixtures/mini_config.toml \
        gait_analysis/tests/test_config_reader.py
git commit -m "feat(data_loader): load_camera_config (Rodrigues -> R matrix)"
```

---

## Task 7: `synchronizer.synchronize`

**Files:**
- Create: `gait_analysis/modules/data_loader/synchronizer.py`
- Test: `gait_analysis/tests/test_synchronizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synchronizer.py
import numpy as np
import pandas as pd
import pytest
from modules.data_loader.synchronizer import synchronize


def _df(fps, duration, phase=0.0):
    t = np.arange(0, duration, 1 / fps)
    return pd.DataFrame({
        "frame": range(len(t)),
        "timestamp": t,
        "left_hip_z": np.sin(2 * np.pi * 1.0 * t + phase),
    })


def test_synchronize_equal_length_and_grid():
    a = _df(30.0, 2.0)        # 30 fps
    b = _df(100.0, 2.0)       # 100 fps
    sa, sb = synchronize(a, b, target_fps=100.0)
    assert len(sa) == len(sb)
    assert np.allclose(sa["timestamp"].to_numpy(), sb["timestamp"].to_numpy())
    # Grid step matches target fps.
    dt = np.diff(sa["timestamp"].to_numpy())
    assert np.allclose(dt, 1 / 100.0, atol=1e-9)


def test_synchronize_clips_to_overlap():
    a = _df(50.0, 2.0)                 # 0 .. ~2.0
    b = _df(50.0, 3.0)                 # 0 .. ~3.0
    b["timestamp"] = b["timestamp"] + 0.5   # shift -> overlap is [0.5, 2.0]
    sa, sb = synchronize(a, b, target_fps=50.0)
    assert sa["timestamp"].iloc[0] >= 0.5 - 1e-9
    assert sa["timestamp"].iloc[-1] <= 2.0 + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_synchronizer.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# modules/data_loader/synchronizer.py
"""Resample two unified DataFrames onto a common time grid."""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def _coord_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]


def _resample(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    t = df["timestamp"].to_numpy()
    out = {"frame": range(len(grid)), "timestamp": grid}
    for c in _coord_cols(df):
        y = df[c].to_numpy()
        f = interp1d(t, y, kind="cubic", bounds_error=False, fill_value=np.nan)
        out[c] = f(grid)
    return pd.DataFrame(out)


def synchronize(df_a: pd.DataFrame, df_b: pd.DataFrame,
                target_fps: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interpolate both DataFrames onto a shared grid over their overlap.

    Precondition: each ``timestamp`` column is in seconds and monotonic.
    """
    start = max(df_a["timestamp"].iloc[0], df_b["timestamp"].iloc[0])
    end = min(df_a["timestamp"].iloc[-1], df_b["timestamp"].iloc[-1])
    if end <= start:
        raise ValueError("DataFrames do not overlap in time")
    grid = np.arange(start, end, 1.0 / target_fps)
    return _resample(df_a, grid), _resample(df_b, grid)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_synchronizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/data_loader/synchronizer.py gait_analysis/tests/test_synchronizer.py
git commit -m "feat(data_loader): synchronize two pose streams onto a common grid"
```

---

## Task 8: `vicon_reader` (documented contract, synthetic-tested)

**Files:**
- Create: `gait_analysis/modules/data_loader/vicon_reader.py`
- Test: `gait_analysis/tests/test_vicon_reader.py`

The test builds a synthetic Vicon-style XLSX (header not in row 1, marker columns `NAME:X|Y|Z`, millimetre units) with openpyxl.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vicon_reader.py
import numpy as np
import pytest
from openpyxl import Workbook
from modules.data_loader.vicon_reader import load_vicon_xlsx, map_vicon_to_caliscope


def _write_vicon_xlsx(path):
    wb = Workbook()
    ws = wb.active
    ws.append(["Trajectories", "100"])          # junk row 1
    ws.append(["LKNE", "", "", "RKNE", "", ""])  # marker-name row
    ws.append(["X", "Y", "Z", "X", "Y", "Z"])    # axis row
    # two frames, millimetres
    ws.append([100.0, 200.0, 300.0, -100.0, 200.0, 300.0])
    ws.append([110.0, 210.0, 310.0, -110.0, 210.0, 310.0])
    wb.save(path)


def test_load_vicon_finds_header_and_converts_mm_to_m(tmp_path):
    p = tmp_path / "vicon.xlsx"
    _write_vicon_xlsx(p)
    df = load_vicon_xlsx(str(p))
    assert "LKNE_x" in df.columns and "RKNE_z" in df.columns
    assert len(df) == 2
    # 100 mm -> 0.1 m
    assert df["LKNE_x"].iloc[0] == pytest.approx(0.1, abs=1e-6)


def test_map_vicon_to_caliscope_renames_markers(tmp_path):
    p = tmp_path / "vicon.xlsx"
    _write_vicon_xlsx(p)
    df = load_vicon_xlsx(str(p))
    mapping = {"LKNE": "left_knee", "RKNE": "right_knee"}
    out = map_vicon_to_caliscope(df, mapping)
    assert "left_knee_x" in out.columns and "right_knee_z" in out.columns
    assert "LKNE_x" not in out.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_vicon_reader.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# modules/data_loader/vicon_reader.py
"""Read Vicon XLSX into the unified pose DataFrame (documented contract)."""
import numpy as np
import pandas as pd
from openpyxl import load_workbook


def _read_rows(path: str) -> list[list]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    return [list(r) for r in ws.iter_rows(values_only=True)]


def _find_axis_header(rows: list[list]) -> int:
    """Index of the row whose cells are X/Y/Z axis labels."""
    for i, row in enumerate(rows):
        cells = [str(c).strip().upper() if c is not None else "" for c in row]
        non_empty = [c for c in cells if c]
        if non_empty and all(c in ("X", "Y", "Z") for c in non_empty):
            return i
    raise ValueError("Could not locate the X/Y/Z axis header row in Vicon XLSX")


def load_vicon_xlsx(filepath: str) -> pd.DataFrame:
    """Auto-detect the header, parse marker columns, convert mm->m if needed.

    Marker names live one row above the X/Y/Z axis row, spanning three columns
    each (NAME, "", "").
    """
    rows = _read_rows(filepath)
    axis_i = _find_axis_header(rows)
    name_row = rows[axis_i - 1]
    axis_row = rows[axis_i]

    # Forward-fill marker names across their three axis columns.
    names: list[str] = []
    last = None
    for c in name_row:
        if c is not None and str(c).strip():
            last = str(c).strip()
        names.append(last)

    columns: list[str] = []
    for name, axis in zip(names, axis_row):
        if axis is None or name is None:
            columns.append(None)
            continue
        columns.append(f"{name}_{str(axis).strip().lower()}")

    data_rows = rows[axis_i + 1:]
    records = []
    for r in data_rows:
        if all(c is None for c in r):
            continue
        rec = {}
        for col, val in zip(columns, r):
            if col is None or val is None:
                continue
            rec[col] = float(val)
        if rec:
            records.append(rec)

    df = pd.DataFrame(records)

    # mm -> m heuristic: if typical coordinate magnitude > 10, assume mm.
    coord = df[[c for c in df.columns]].to_numpy()
    if np.nanmedian(np.abs(coord)) > 10.0:
        df = df / 1000.0

    df.insert(0, "frame", range(len(df)))
    return df


def map_vicon_to_caliscope(vicon_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename ``MARKER_{x,y,z}`` columns to ``canonical_{x,y,z}`` via ``mapping``."""
    rename = {}
    for marker, landmark in mapping.items():
        for axis in ("x", "y", "z"):
            rename[f"{marker}_{axis}"] = f"{landmark}_{axis}"
    return vicon_df.rename(columns=rename)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_vicon_reader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/data_loader/vicon_reader.py gait_analysis/tests/test_vicon_reader.py
git commit -m "feat(data_loader): vicon_reader (header autodetect, mm->m, marker mapping)"
```

---

## Task 9: `joint_angles.calc_angle_3d`

**Files:**
- Create: `gait_analysis/modules/kinematics/joint_angles.py`
- Test: `gait_analysis/tests/test_joint_angles.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_joint_angles.py
import numpy as np
import pytest
from modules.kinematics.joint_angles import calc_angle_3d


def test_calc_angle_90_degrees():
    p1 = np.array([1.0, 0.0, 0.0])
    vertex = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    assert calc_angle_3d(p1, vertex, p2) == pytest.approx(90.0, abs=1e-6)


def test_calc_angle_180_degrees_collinear():
    assert calc_angle_3d(np.array([1.0, 0, 0]), np.array([0.0, 0, 0]),
                         np.array([-1.0, 0, 0])) == pytest.approx(180.0, abs=1e-6)


def test_calc_angle_45_degrees():
    assert calc_angle_3d(np.array([1.0, 0, 0]), np.array([0.0, 0, 0]),
                         np.array([1.0, 1.0, 0])) == pytest.approx(45.0, abs=1e-6)


def test_calc_angle_nan_when_point_missing():
    out = calc_angle_3d(np.array([np.nan, 0, 0]), np.array([0.0, 0, 0]),
                        np.array([0.0, 1, 0]))
    assert np.isnan(out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_joint_angles.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# modules/kinematics/joint_angles.py
"""Joint-angle computation from 3D landmark triples."""
import numpy as np
import pandas as pd


def calc_angle_3d(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    """Included angle (degrees, 0-180) between vectors (vertex->p1) and (vertex->p2)."""
    p1 = np.asarray(p1, float); vertex = np.asarray(vertex, float); p2 = np.asarray(p2, float)
    if np.isnan(np.concatenate([p1, vertex, p2])).any():
        return float("nan")
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_joint_angles.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/kinematics/joint_angles.py gait_analysis/tests/test_joint_angles.py
git commit -m "feat(kinematics): calc_angle_3d included-angle computation"
```

---

## Task 10: `joint_angles.calc_joint_angles_timeseries`

**Files:**
- Modify: `gait_analysis/modules/kinematics/joint_angles.py`
- Test: `gait_analysis/tests/test_joint_angles.py`

Computes hip/knee/ankle sagittal included angles for both sides, frame by frame. (Pelvis angles are deferred to Phase 2/3 per the spec; documented in the docstring.)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_joint_angles.py
import pandas as pd
from modules.kinematics.joint_angles import calc_joint_angles_timeseries


def test_joint_angles_timeseries_adds_expected_columns():
    # One frame: straight left leg (hip, knee, ankle collinear) -> knee angle 180.
    row = {
        "frame": [0], "timestamp": [0.0],
        "left_hip_x": [0.0], "left_hip_y": [0.0], "left_hip_z": [1.0],
        "left_knee_x": [0.0], "left_knee_y": [0.0], "left_knee_z": [0.5],
        "left_ankle_x": [0.0], "left_ankle_y": [0.0], "left_ankle_z": [0.0],
        "left_foot_index_x": [0.2], "left_foot_index_y": [0.0], "left_foot_index_z": [0.0],
        "left_shoulder_x": [0.0], "left_shoulder_y": [0.0], "left_shoulder_z": [1.5],
    }
    df = pd.DataFrame(row)
    out = calc_joint_angles_timeseries(df)
    assert "left_knee_angle" in out.columns
    assert "left_hip_angle" in out.columns
    assert "left_ankle_angle" in out.columns
    assert out["left_knee_angle"].iloc[0] == pytest.approx(180.0, abs=1e-6)


def test_joint_angles_timeseries_handles_missing_landmark_as_nan():
    df = pd.DataFrame({
        "frame": [0], "timestamp": [0.0],
        "left_hip_x": [0.0], "left_hip_y": [0.0], "left_hip_z": [1.0],
        "left_knee_x": [np.nan], "left_knee_y": [np.nan], "left_knee_z": [np.nan],
        "left_ankle_x": [0.0], "left_ankle_y": [0.0], "left_ankle_z": [0.0],
    })
    out = calc_joint_angles_timeseries(df)
    assert np.isnan(out["left_knee_angle"].iloc[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_joint_angles.py -k timeseries -v`
Expected: FAIL — `ImportError: cannot import name 'calc_joint_angles_timeseries'`

- [ ] **Step 3: Write minimal implementation (append to `joint_angles.py`)**

```python
# (proximal, vertex, distal) landmark triples per joint angle column.
_ANGLE_DEFS = {
    "left_hip_angle":    ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle":   ("right_shoulder", "right_hip", "right_knee"),
    "left_knee_angle":   ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle":  ("right_hip", "right_knee", "right_ankle"),
    "left_ankle_angle":  ("left_knee", "left_ankle", "left_foot_index"),
    "right_ankle_angle": ("right_knee", "right_ankle", "right_foot_index"),
}


def _point(df: pd.DataFrame, name: str) -> np.ndarray | None:
    cols = [f"{name}_x", f"{name}_y", f"{name}_z"]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].to_numpy()


def calc_joint_angles_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Append per-frame sagittal joint angles (included angle, degrees 0-180).

    Columns added: ``<side>_{hip,knee,ankle}_angle``. An angle is NaN for any
    frame where a required landmark is missing or its triple is unavailable.
    Pelvis angles (tilt/obliquity/rotation) are out of scope for Phase 1.
    """
    out = df.copy()
    n = len(df)
    for col, (a, v, b) in _ANGLE_DEFS.items():
        pa, pv, pb = _point(df, a), _point(df, v), _point(df, b)
        if pa is None or pv is None or pb is None:
            out[col] = np.full(n, np.nan)
            continue
        out[col] = np.array([calc_angle_3d(pa[i], pv[i], pb[i]) for i in range(n)])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_joint_angles.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/kinematics/joint_angles.py gait_analysis/tests/test_joint_angles.py
git commit -m "feat(kinematics): per-frame hip/knee/ankle sagittal angle timeseries"
```

---

## Task 11: `gait_events.detect_gait_events`

**Files:**
- Create: `gait_analysis/modules/kinematics/gait_events.py`
- Test: `gait_analysis/tests/test_gait_events.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gait_events.py
import numpy as np
import pandas as pd
import pytest
from modules.kinematics.gait_events import detect_gait_events


def _walking_df(fps=50.0, stride_hz=1.0, n_strides=4):
    t = np.arange(0, n_strides / stride_hz, 1 / fps)
    # Heel vertical: minima (ground contact) once per stride at cos = -1.
    heel_z = -np.cos(2 * np.pi * stride_hz * t)
    # Toe vertical: maxima offset by half a stride.
    toe_z = np.cos(2 * np.pi * stride_hz * t)
    return pd.DataFrame({
        "frame": range(len(t)), "timestamp": t,
        "left_heel_z": heel_z, "left_foot_index_z": toe_z,
        "left_heel_x": np.zeros_like(t), "left_foot_index_x": np.zeros_like(t),
        "left_heel_y": np.zeros_like(t), "left_foot_index_y": np.zeros_like(t),
    })


def test_detect_finds_expected_number_of_heel_strikes():
    fps = 50.0
    df = _walking_df(fps=fps, stride_hz=1.0, n_strides=4)
    events = detect_gait_events(df, fps=fps, side="left", method="velocity")
    # heel_z = -cos has minima at t = 0,1,2,3 s -> ~4 heel strikes.
    assert 3 <= len(events["left_HS"]) <= 5


def test_detect_heel_strikes_located_near_minima():
    fps = 50.0
    df = _walking_df(fps=fps, stride_hz=1.0, n_strides=3)
    events = detect_gait_events(df, fps=fps, side="left", method="velocity")
    times = df["timestamp"].to_numpy()[events["left_HS"]]
    # Each detected HS time is close to an integer second.
    for tt in times:
        assert min(abs(tt - k) for k in range(4)) < 0.1


def test_detect_both_sides_returns_all_keys():
    fps = 50.0
    df = _walking_df(fps=fps)
    # add right-side columns mirroring left
    for c in list(df.columns):
        if c.startswith("left_"):
            df[c.replace("left_", "right_")] = df[c]
    events = detect_gait_events(df, fps=fps, side="both", method="velocity")
    assert set(events) == {"left_HS", "left_TO", "right_HS", "right_TO"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_gait_events.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# modules/kinematics/gait_events.py
"""Detect heel-strike (HS) and toe-off (TO) gait events."""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .filters import butterworth_filter


def _sides(side: str) -> list[str]:
    return ["left", "right"] if side == "both" else [side]


def _clean(a: np.ndarray) -> np.ndarray:
    """Linear-interpolate residual NaN (both directions) so filtfilt can run."""
    return pd.Series(np.asarray(a, dtype=float)).interpolate(
        limit_direction="both").to_numpy()


def _detect_one_side(df: pd.DataFrame, fps: float, side: str, method: str,
                     heel: str, toe: str, vertical: str,
                     min_stride_sec: float, cutoff_hz: float) -> tuple[list, list]:
    heel_col = f"{side}_{heel}_{vertical}"
    toe_col = f"{side}_{toe}_{vertical}"
    heel_v = butterworth_filter(_clean(df[heel_col].to_numpy()), cutoff_hz=cutoff_hz, fs=fps)
    toe_v = butterworth_filter(_clean(df[toe_col].to_numpy()), cutoff_hz=cutoff_hz, fs=fps)
    min_dist = max(1, int(min_stride_sec * fps))

    # Heel strike = local minima of heel vertical = peaks of -heel_v.
    hs, _ = find_peaks(-heel_v, distance=min_dist)
    # Toe off = local maxima of toe vertical.
    to, _ = find_peaks(toe_v, distance=min_dist)
    return hs.tolist(), to.tolist()


def detect_gait_events(df: pd.DataFrame, fps: float, side: str = "both",
                       method: str = "velocity", heel: str = "heel",
                       toe: str = "foot_index", vertical: str = "z",
                       min_stride_sec: float = 0.3,
                       cutoff_hz: float = 6.0) -> dict:
    """Return ``{<side>_HS, <side>_TO}`` frame-index lists.

    ``velocity``/``height`` both reduce to vertical-trajectory extrema for the
    Phase-1 lower-body landmarks; ``method`` is accepted for forward
    compatibility. Events closer than ``min_stride_sec`` are suppressed via the
    ``distance`` constraint in ``find_peaks``.
    """
    result: dict = {}
    for s in _sides(side):
        hs, to = _detect_one_side(df, fps, s, method, heel, toe, vertical,
                                  min_stride_sec, cutoff_hz)
        result[f"{s}_HS"] = hs
        result[f"{s}_TO"] = to
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_gait_events.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/kinematics/gait_events.py gait_analysis/tests/test_gait_events.py
git commit -m "feat(kinematics): heel-strike/toe-off detection via vertical extrema"
```

---

## Task 12: `normalizer.normalize_gait_cycle` + `get_mean_std_cycle`

**Files:**
- Create: `gait_analysis/modules/kinematics/normalizer.py`
- Test: `gait_analysis/tests/test_normalizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_normalizer.py
import numpy as np
import pytest
from modules.kinematics.normalizer import normalize_gait_cycle, get_mean_std_cycle


def test_normalize_returns_101_points_per_cycle():
    signal = np.linspace(0, 30, 31)          # 31 samples
    events = {"left_HS": [0, 10, 20, 30]}     # 3 cycles
    mat = normalize_gait_cycle(signal, events, side="left", n_points=101)
    assert mat.shape == (3, 101)


def test_normalize_preserves_endpoints_of_linear_cycle():
    signal = np.arange(0, 21, dtype=float)    # ramp 0..20
    events = {"left_HS": [0, 10, 20]}
    mat = normalize_gait_cycle(signal, events, side="left")
    # First cycle spans values 0..10; endpoints exact.
    assert mat[0, 0] == pytest.approx(0.0)
    assert mat[0, -1] == pytest.approx(10.0)


def test_get_mean_std_cycle_shapes_and_values():
    mat = np.vstack([np.full(101, 2.0), np.full(101, 4.0)])  # mean 3, std 1
    mean, std = get_mean_std_cycle(mat)
    assert mean.shape == (101,) and std.shape == (101,)
    assert mean[0] == pytest.approx(3.0)
    assert std[0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_normalizer.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# modules/kinematics/normalizer.py
"""Normalize signals to the 0-100% gait cycle (101 points)."""
import numpy as np


def normalize_gait_cycle(signal: np.ndarray, events: dict, side: str,
                         n_points: int = 101) -> np.ndarray:
    """Split ``signal`` into HS->HS cycles and resample each to ``n_points``.

    Returns a ``[n_cycles x n_points]`` matrix.
    """
    signal = np.asarray(signal, dtype=float)
    hs = sorted(events[f"{side}_HS"])
    target = np.linspace(0.0, 1.0, n_points)
    cycles = []
    for start, end in zip(hs[:-1], hs[1:]):
        seg = signal[start:end + 1]
        if len(seg) < 2:
            continue
        src = np.linspace(0.0, 1.0, len(seg))
        cycles.append(np.interp(target, src, seg))
    if not cycles:
        return np.empty((0, n_points))
    return np.vstack(cycles)


def get_mean_std_cycle(cycles_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation across cycles (NaN-aware), per cycle point."""
    if cycles_matrix.size == 0:
        n = cycles_matrix.shape[1] if cycles_matrix.ndim == 2 else 101
        return np.full(n, np.nan), np.full(n, np.nan)
    mean = np.nanmean(cycles_matrix, axis=0)
    std = np.nanstd(cycles_matrix, axis=0)
    return mean, std
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_normalizer.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/kinematics/normalizer.py gait_analysis/tests/test_normalizer.py
git commit -m "feat(kinematics): gait-cycle normalization to 101 points + mean/std"
```

---

## Task 13: `spatiotemporal.calc_spatiotemporal`

**Files:**
- Create: `gait_analysis/modules/kinematics/spatiotemporal.py`
- Test: `gait_analysis/tests/test_spatiotemporal.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spatiotemporal.py
import numpy as np
import pandas as pd
import pytest
from modules.kinematics.spatiotemporal import calc_spatiotemporal


def _constant_velocity_walk(fps=50.0):
    # Subject moves +x at 1 m/s for 4 s; heel strikes every 1 s on the left.
    t = np.arange(0, 4.0, 1 / fps)
    x = 1.0 * t
    df = pd.DataFrame({
        "frame": range(len(t)), "timestamp": t,
        "left_heel_x": x, "left_heel_y": np.zeros_like(t), "left_heel_z": np.zeros_like(t),
        "right_heel_x": x, "right_heel_y": np.full_like(t, 0.2), "right_heel_z": np.zeros_like(t),
    })
    return df, fps


def test_cadence_and_speed_are_positive_and_reasonable():
    df, fps = _constant_velocity_walk()
    events = {
        "left_HS": [0, 50, 100, 150], "left_TO": [30, 80, 130],
        "right_HS": [25, 75, 125], "right_TO": [55, 105, 155],
    }
    out = calc_spatiotemporal(df, events, fps=fps)
    assert out["cadence_steps_per_min"] > 0
    assert out["speed_m_per_s"] > 0
    # stride length = distance between consecutive left heel strikes ~ 1 m.
    assert out["stride_length_m"] == pytest.approx(1.0, abs=0.2)


def test_stance_swing_sum_to_about_100():
    df, fps = _constant_velocity_walk()
    events = {
        "left_HS": [0, 50, 100, 150], "left_TO": [30, 80, 130],
        "right_HS": [25, 75, 125], "right_TO": [55, 105, 155],
    }
    out = calc_spatiotemporal(df, events, fps=fps)
    assert out["stance_pct"] + out["swing_pct"] == pytest.approx(100.0, abs=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_spatiotemporal.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# modules/kinematics/spatiotemporal.py
"""Spatiotemporal gait parameters from events + heel positions."""
import numpy as np
import pandas as pd


def _heel_xy(df: pd.DataFrame, side: str, vertical: str) -> np.ndarray:
    axes = [a for a in ("x", "y", "z") if a != vertical]
    return df[[f"{side}_heel_{axes[0]}", f"{side}_heel_{axes[1]}"]].to_numpy()


def _travel_axis(positions: np.ndarray) -> np.ndarray:
    """Principal horizontal direction of motion (unit vector) via PCA."""
    centered = positions - np.nanmean(positions, axis=0)
    centered = centered[~np.isnan(centered).any(axis=1)]
    if len(centered) < 2:
        return np.array([1.0, 0.0])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def calc_spatiotemporal(df: pd.DataFrame, events: dict, fps: float,
                        vertical: str = "z") -> dict:
    """Compute cadence, speed, stride/step length, step width, stance/swing %."""
    timestamps = df["timestamp"].to_numpy()
    left_hs = sorted(events.get("left_HS", []))
    right_hs = sorted(events.get("right_HS", []))
    all_hs = sorted(left_hs + right_hs)

    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else np.nan
    n_steps = max(len(all_hs) - 1, 0)
    cadence = (n_steps / duration) * 60.0 if duration and n_steps else np.nan

    heel_left = _heel_xy(df, "left", vertical)
    travel = _travel_axis(heel_left)

    # Stride length: mean distance between consecutive same-foot heel strikes.
    strides = []
    for a, b in zip(left_hs[:-1], left_hs[1:]):
        strides.append(np.linalg.norm(heel_left[b] - heel_left[a]))
    stride_length = float(np.mean(strides)) if strides else np.nan

    speed = (stride_length * cadence / 120.0
             if not np.isnan(stride_length) and not np.isnan(cadence) else np.nan)

    # Step length/width: project L->R heel vector at contralateral HS onto axes.
    heel_right = _heel_xy(df, "right", vertical)
    perp = np.array([-travel[1], travel[0]])
    step_lengths, step_widths = [], []
    for hs in right_hs:
        if hs < len(heel_right) and hs < len(heel_left):
            vec = heel_right[hs] - heel_left[hs]
            step_lengths.append(abs(np.dot(vec, travel)))
            step_widths.append(abs(np.dot(vec, perp)))
    step_length = float(np.mean(step_lengths)) if step_lengths else np.nan
    step_width = float(np.mean(step_widths)) if step_widths else np.nan

    # Stance/swing %: from left HS->TO->next HS.
    left_to = sorted(events.get("left_TO", []))
    stance_fracs = []
    for i in range(len(left_hs) - 1):
        hs0, hs1 = left_hs[i], left_hs[i + 1]
        tos = [t for t in left_to if hs0 < t < hs1]
        if tos:
            stance_fracs.append((tos[0] - hs0) / (hs1 - hs0))
    stance_pct = float(np.mean(stance_fracs) * 100.0) if stance_fracs else np.nan
    swing_pct = 100.0 - stance_pct if not np.isnan(stance_pct) else np.nan
    double_support_pct = max(2 * stance_pct - 100.0, 0.0) if not np.isnan(stance_pct) else np.nan

    return {
        "cadence_steps_per_min": cadence,
        "speed_m_per_s": speed,
        "stride_length_m": stride_length,
        "step_length_m": step_length,
        "step_width_m": step_width,
        "stance_pct": stance_pct,
        "swing_pct": swing_pct,
        "double_support_pct": double_support_pct,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_spatiotemporal.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/modules/kinematics/spatiotemporal.py gait_analysis/tests/test_spatiotemporal.py
git commit -m "feat(kinematics): spatiotemporal parameters (cadence, speed, stride, stance/swing)"
```

---

## Task 14: CLI `analyze` command + end-to-end integration on real data

**Files:**
- Create: `gait_analysis/cli.py`
- Test: `gait_analysis/tests/test_cli_integration.py`

`analyze` runs the full pipeline on one session and writes `gait_results.json`.

- [ ] **Step 1: Write the failing integration test** (runs on the real symlinked `p1_3` session)

```python
# tests/test_cli_integration.py
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]          # .../Cerebral
GAIT = REPO / "gait_analysis"
SESSION = GAIT / "data" / "caliscope_project" / "recordings" / "p1_3"

pytestmark = pytest.mark.skipif(not SESSION.exists(),
                                reason="real caliscope data not present")


def test_analyze_writes_gait_results(tmp_path):
    out = tmp_path / "p1_3_results.json"
    cmd = [sys.executable, str(GAIT / "cli.py"), "analyze",
           "--session", str(SESSION), "--model", "SIMPLE_HOLISTIC",
           "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GAIT))
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["session_id"] == "p1_3"
    assert data["model"] == "SIMPLE_HOLISTIC"
    assert data["fps"] > 0
    assert "gait_events" in data and "spatiotemporal" in data
    assert "left_HS" in data["gait_events"]
    # Pipeline produced at least the knee angle mean curve (101 points) when cycles exist.
    if data["joint_angles_mean"].get("left_knee"):
        assert len(data["joint_angles_mean"]["left_knee"]) == 101
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_integration.py -v`
Expected: FAIL — `cli.py` does not exist (non-zero return code)

- [ ] **Step 3: Write minimal implementation**

```python
# cli.py
"""Command-line entry point for the gait analysis pipeline."""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import yaml

from modules.data_loader.caliscope_reader import load_caliscope_session
from modules.kinematics.filters import fill_gaps, butterworth_filter
from modules.kinematics.gait_events import detect_gait_events
from modules.kinematics.joint_angles import calc_joint_angles_timeseries
from modules.kinematics.normalizer import normalize_gait_cycle, get_mean_std_cycle
from modules.kinematics.spatiotemporal import calc_spatiotemporal

HERE = Path(__file__).resolve().parent


def _load_settings() -> dict:
    with open(HERE / "settings.yaml") as f:
        return yaml.safe_load(f)


def _filter_coords(df, cutoff_hz, order, fps):
    """Filter coordinate columns. Skip columns that still contain NaN after
    gap-filling (filtfilt would otherwise propagate NaN across the column);
    those keep their gap-filled values and are handled NaN-wise downstream.
    """
    out = df.copy()
    for c in out.columns:
        if not c.endswith(("_x", "_y", "_z")):
            continue
        arr = out[c].to_numpy()
        if np.isnan(arr).any() or len(arr) <= 3 * order:
            continue
        out[c] = butterworth_filter(arr, cutoff_hz=cutoff_hz, fs=fps, order=order)
    return out


def analyze(session_dir: str, model: str, out_path: str) -> dict:
    cfg = _load_settings()
    proc = cfg["processing"]
    gcfg = cfg["gait_events"]

    df = load_caliscope_session(session_dir, model=model)
    fps = df.attrs["fps"]

    df = fill_gaps(df, max_gap_frames=proc["max_gap_frames"])
    df = _filter_coords(df, proc["filter_cutoff_hz"], proc["filter_order"], fps)

    events = detect_gait_events(
        df, fps=fps, method=gcfg["method"], heel=gcfg["heel_landmark"],
        toe=gcfg["toe_landmark"], vertical=gcfg["vertical_axis"],
        min_stride_sec=proc["min_stride_duration_sec"],
        cutoff_hz=proc["filter_cutoff_hz"],
    )
    df = calc_joint_angles_timeseries(df)
    spatiotemporal = calc_spatiotemporal(df, events, fps=fps,
                                         vertical=gcfg["vertical_axis"])

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
        "session_id": Path(session_dir).name,
        "model": model,
        "fps": round(float(fps), 3),
        "n_frames": int(len(df)),
        "processed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "gait_events": {k: list(map(int, v)) for k, v in events.items()},
        "spatiotemporal": {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                               else round(float(v), 3))
                           for k, v in spatiotemporal.items()},
        "joint_angles_mean": angles_mean,
        "joint_angles_std": angles_std,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser(description="Gait analysis CLI")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Run the full pipeline on one session")
    a.add_argument("--session", required=True)
    a.add_argument("--model", default="SIMPLE_HOLISTIC")
    a.add_argument("--out", required=True)

    args = p.parse_args()
    if args.command == "analyze":
        res = analyze(args.session, args.model, args.out)
        print(f"Wrote {args.out}: {res['n_frames']} frames @ {res['fps']} fps, "
              f"{len(res['gait_events'].get('left_HS', []))} left HS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli_integration.py -v`
Expected: PASS (1 test). Then run manually and eyeball the output:
```bash
.venv/bin/python cli.py analyze --session data/caliscope_project/recordings/p1_3 \
    --model SIMPLE_HOLISTIC --out results/p1_3_results.json
```
Expected: prints frame count, fps (~19–20), and a non-zero left-HS count.

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/cli.py gait_analysis/tests/test_cli_integration.py
git commit -m "feat(cli): analyze command runs full pipeline -> gait_results.json"
```

---

## Task 15: CLI `reproducibility` command (Level B — primary validation)

**Files:**
- Modify: `gait_analysis/cli.py`
- Test: `gait_analysis/tests/test_cli_integration.py`

Runs `analyze` across `p1_1…p1_5` and computes the coefficient of variation (CV = std/mean) per spatiotemporal parameter, with the per-session cycle counts reported alongside (spec §13 risk 1).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_cli_integration.py
RECORDINGS = GAIT / "data" / "caliscope_project" / "recordings"
HAVE_P1 = all((RECORDINGS / f"p1_{i}").exists() for i in range(1, 6))


@pytest.mark.skipif(not HAVE_P1, reason="p1_1..p1_5 not present")
def test_reproducibility_computes_cv(tmp_path):
    out = tmp_path / "repro.json"
    cmd = [sys.executable, str(GAIT / "cli.py"), "reproducibility",
           "--recordings", str(RECORDINGS), "--model", "SIMPLE_HOLISTIC",
           "--sessions", "p1_1,p1_2,p1_3,p1_4,p1_5", "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GAIT))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert "per_session" in data and "cv_percent" in data
    assert len(data["per_session"]) == 5
    # CV reported for cadence and includes a cycle-count annotation per session.
    assert "cadence_steps_per_min" in data["cv_percent"]
    assert all("n_left_cycles" in s for s in data["per_session"].values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_integration.py -k reproducibility -v`
Expected: FAIL — `reproducibility` is not a known subcommand (non-zero return code)

- [ ] **Step 3: Write minimal implementation (append `reproducibility` + wire into `main`)**

```python
# add near the top-level functions in cli.py
def reproducibility(recordings_dir: str, sessions: list[str], model: str,
                    out_path: str) -> dict:
    per_session = {}
    params = ["cadence_steps_per_min", "speed_m_per_s", "stride_length_m",
              "step_length_m", "step_width_m", "stance_pct", "swing_pct"]
    collected = {p: [] for p in params}

    for sess in sessions:
        sess_dir = str(Path(recordings_dir) / sess)
        tmp_out = str(Path(out_path).parent / f"_{sess}_results.json")
        res = analyze(sess_dir, model, tmp_out)
        st = res["spatiotemporal"]
        n_cycles = max(len(res["gait_events"].get("left_HS", [])) - 1, 0)
        per_session[sess] = {**st, "n_left_cycles": n_cycles}
        for p in params:
            v = st.get(p)
            if v is not None:
                collected[p].append(v)

    cv_percent = {}
    for p, vals in collected.items():
        arr = np.asarray(vals, dtype=float)
        if len(arr) >= 2 and np.mean(arr) != 0:
            cv_percent[p] = round(float(np.std(arr) / np.mean(arr) * 100.0), 2)
        else:
            cv_percent[p] = None

    out = {"model": model, "sessions": sessions,
           "per_session": per_session, "cv_percent": cv_percent}
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out
```

```python
# in main(), after the `analyze` subparser block, add:
    r = sub.add_parser("reproducibility", help="Level-B CV across sessions")
    r.add_argument("--recordings", required=True)
    r.add_argument("--sessions", required=True, help="comma-separated session names")
    r.add_argument("--model", default="SIMPLE_HOLISTIC")
    r.add_argument("--out", required=True)
```

```python
# in main(), extend the dispatch:
    elif args.command == "reproducibility":
        res = reproducibility(args.recordings, args.sessions.split(","),
                              args.model, args.out)
        print("CV% by parameter:")
        for p, v in res["cv_percent"].items():
            print(f"  {p}: {v}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli_integration.py -v`
Expected: PASS. Then run the real Level-B analysis:
```bash
.venv/bin/python cli.py reproducibility \
    --recordings data/caliscope_project/recordings \
    --sessions p1_1,p1_2,p1_3,p1_4,p1_5 --model SIMPLE_HOLISTIC \
    --out results/reproducibility.json
```
Expected: prints CV% per parameter; `results/reproducibility.json` contains per-session values + cycle counts.

- [ ] **Step 5: Commit**

```bash
cd /home/grivin/Workspace/Cerebral
git add gait_analysis/cli.py gait_analysis/tests/test_cli_integration.py
git commit -m "feat(cli): reproducibility (Level-B CV across sessions) with cycle counts"
```

---

## Task 16: Full suite, coverage gate, and lint

**Files:** none (verification + fixups only)

- [ ] **Step 1: Run the whole suite with coverage**

Run:
```bash
cd /home/grivin/Workspace/Cerebral/gait_analysis
.venv/bin/pytest --cov=modules --cov=cli --cov-report=term-missing
```
Expected: all tests PASS; coverage on `modules/` + `cli.py` > 70%. If below, add focused unit tests for the lowest-covered functions (e.g. `synchronize` overlap error path, `_travel_axis` degenerate input).

- [ ] **Step 2: Run ruff and fix any findings**

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!` (fix imports/unused names inline if reported; re-run until clean).

- [ ] **Step 3: Commit any fixups**

```bash
cd /home/grivin/Workspace/Cerebral
git add -A gait_analysis
git commit -m "test: coverage >70% on core spine; ruff clean"
```

---

## Phase 1 acceptance check (maps to spec §12)

After Task 16, confirm against the spec's acceptance criteria:

- **#2** `fill_gaps` + `butterworth_filter` work on NaN data → Tasks 1–2 (unit) ✅
- **#3** `gait_events` finds HS/TO on `p1_*` sessions → Task 14/15 run produces non-zero HS per session ✅
- **#6** CV across `p1_1…p1_5` < 15% (primary) → Task 15 `reproducibility.json`; report the actual CV%, annotated with cycle counts ✅
- **#9** pytest passes, coverage > 70% → Task 16 ✅
- **#10** `ruff check .` clean → Task 16 ✅

Criteria **#1/#4/#5** (Vicon/`VvsC.py`) remain **gated** → Phase 3.
Criteria **#7/#8** (GUI, export) → Phase 2.

---

## Out of scope for this plan (subsequent phases, separate plans)

- **Pelvis angles** (`pelvic_tilt`, `pelvic_obliquity`, `pelvic_rotation`, spec §6.3): deferred to a Phase-2 kinematics extension. They need a documented pelvis-plane definition from the available hip (± shoulder) landmarks; the lower-limb sagittal angles (hip/knee/ankle) are the Phase-1 core. Tracked so the spec requirement is not dropped.
- **Phase 2:** Module 4 (`skeleton_3d`, `angle_plots`, `export`) + PyQt6 GUI (`app.py`, panels, widgets, QThread worker) + CSV/XLSX/PNG export. Criteria #7/#8.
- **Phase 3 (gated on Vicon XLSX + `VvsC.py`):** Module 3 (`alignment` Umeyama, `metrics` RMSE/MAE/Pearson/ICC, `report`) real-data validation, `VvsC.py` subprocess wrapper, apparatus Levels A & C. Criteria #1/#4/#5. (Module 3 code + synthetic tests may be pulled forward, but live validation is gated.)
