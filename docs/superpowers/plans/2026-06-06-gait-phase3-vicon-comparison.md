# Phase 3 — Vicon Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Module 3 (Vicon comparison) — a two-layer comparison (joint angles in degrees, primary; VvsC positional baseline in meters) wired into a `compare` CLI subcommand and a new GUI "Сравнение" tab — and unlock validation Levels A/C.

**Architecture:** A single `run_comparison(cal_df, vic_df, cfg, ...)` in `compare_pipeline.py` assembles one `comparison_report.json`, called by both `cli.py` and a GUI `ComparisonWorker`. The angle layer reuses the existing Module 2 kinematics on both streams and compares 101-pt gait-cycle curves (rotation/scale-invariant, so no alignment needed). The position layer ports VvsC's similarity transform + xcorr sync and is proven equivalent to `VvsC.py` by a parity test. The real Vicon XLSX schema requires fixing `vicon_reader` first.

**Tech Stack:** Python 3.14, numpy, pandas, scipy, openpyxl, pingouin (ICC), matplotlib, PyQt6, pytest/pytest-qt, ruff.

**Spec:** `docs/superpowers/specs/2026-06-06-gait-phase3-vicon-comparison-design.md`

**Conventions (every task):**
- Work from `gait_analysis/`. Run python/pytest as `.venv/bin/python -m pytest …`.
- Tests import `from modules…`, `from pipeline import…`, `from compare_pipeline import…`.
- Commit with **explicit pathspecs** (never `git add -A`). Append the trailer
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` to commit messages.
- After each task: `.venv/bin/python -m ruff check .` must be clean for touched files.
- Do **not** push (separate user decision).

---

## Task 0: Environment + data access

**Files:**
- Modify: `gait_analysis/requirements.txt` (verify `pingouin` present)
- Create (symlink): `gait_analysis/data/Vicon_10_series -> ../../Vicon_10_series`

- [ ] **Step 1: Install pingouin into the venv**

Run:
```bash
cd gait_analysis && .venv/bin/python -m pip install pingouin
```
Expected: installs pingouin + statsmodels + scikit-learn.

- [ ] **Step 2: Verify import and that requirements.txt lists it**

Run:
```bash
.venv/bin/python -c "import pingouin; print(pingouin.__version__)"
grep -i pingouin requirements.txt || echo "pingouin" >> requirements.txt
```
Expected: a version string prints; `pingouin` is a line in `requirements.txt`.

- [ ] **Step 3: Symlink the Vicon data so `paths.vicon_root` resolves**

`settings.yaml` has `paths.vicon_root: data/Vicon_10_series`, but the files live at repo root. Mirror the existing caliscope symlink:
```bash
ln -snf ../../Vicon_10_series data/Vicon_10_series
ls data/Vicon_10_series/1.xlsx
```
Expected: `data/Vicon_10_series/1.xlsx` lists (symlink resolves).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build(phase3): install pingouin for ICC

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
(The symlink under `data/` is not tracked — `data/` holds only symlinks; leave it untracked like `data/caliscope_project`.)

---

## Task 1: Fix `vicon_reader` for the real XLSX schema

The real header is: row0 `Trajectories`, row1 `100`, row2 marker names colon-prefixed with the subject (`Derevesnikova Darya Alexandrovna:LASI`) **plus** a duplicate `| … |` block and a `Trajectory Count` column, row3 `Frame, Sub Frame, X, Y, Z, …`, row4 units `mm`, data from row5 with frame numbers starting at 225. The current `_find_axis_header` requires a row of *only* X/Y/Z and so raises `ValueError`.

**Files:**
- Modify: `gait_analysis/modules/data_loader/vicon_reader.py`
- Modify: `gait_analysis/tests/test_vicon_reader.py`

- [ ] **Step 1: Write the failing tests (real-schema fixture + timestamp)**

Replace the body of `tests/test_vicon_reader.py` with:
```python
import pytest
from openpyxl import Workbook

from modules.data_loader.vicon_reader import load_vicon_xlsx, map_vicon_to_caliscope


def _write_real_schema_xlsx(path):
    """Mirror the real Vicon export: subject-prefixed names, Frame/Sub Frame,
    a duplicate | ... | block, a Trajectory Count column, units row, mm data."""
    wb = Workbook()
    ws = wb.active
    ws.append(["Trajectories"])                                   # row0
    ws.append(["100"])                                            # row1
    # row2: marker-name row. Names span 3 cols each (NAME,"",""); Frame/SubFrame blank.
    ws.append([None, None,
               "Subj:LKNE", "", "", "Subj:RKNE", "", "",
               "| Subj:LKNE |", "", "", "Trajectory Count"])      # dup block + count
    ws.append(["Frame", "Sub Frame", "X", "Y", "Z", "X", "Y", "Z",
               "X", "Y", "Z", ""])                                # row3 axis row
    ws.append([None, None, "mm", "mm", "mm", "mm", "mm", "mm",
               "mm", "mm", "mm", ""])                             # row4 units
    ws.append([225, 0, 100.0, 200.0, 300.0, -100.0, 200.0, 300.0,
               100.0, 200.0, 300.0, 1])                           # row5 data (mm)
    ws.append([226, 0, 110.0, 210.0, 310.0, -110.0, 210.0, 310.0,
               110.0, 210.0, 310.0, 1])
    wb.save(path)


def test_load_vicon_real_schema_parses_colon_names_and_mm(tmp_path):
    p = tmp_path / "vicon.xlsx"
    _write_real_schema_xlsx(p)
    df = load_vicon_xlsx(str(p), vicon_fps=100.0)
    # subject prefix stripped; clean marker columns present
    assert "LKNE_x" in df.columns and "RKNE_z" in df.columns
    # the | ... | duplicate block must NOT collide with the clean LKNE columns
    assert "LKNE_x" in df.columns and df["LKNE_x"].notna().all()
    # mm -> m
    assert df["LKNE_x"].iloc[0] == pytest.approx(0.1, abs=1e-6)
    assert len(df) == 2


def test_load_vicon_adds_timestamp_from_frame_and_fps(tmp_path):
    p = tmp_path / "vicon.xlsx"
    _write_real_schema_xlsx(p)
    df = load_vicon_xlsx(str(p), vicon_fps=100.0)
    assert "timestamp" in df.columns
    # first frame zero-based; second frame is 1/100 s later
    assert df["timestamp"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert df["timestamp"].iloc[1] == pytest.approx(0.01, abs=1e-9)


def test_map_vicon_to_caliscope_renames(tmp_path):
    p = tmp_path / "vicon.xlsx"
    _write_real_schema_xlsx(p)
    df = load_vicon_xlsx(str(p), vicon_fps=100.0)
    out = map_vicon_to_caliscope(df, {"LKNE": "left_knee", "RKNE": "right_knee"})
    assert "left_knee_x" in out.columns and "right_knee_z" in out.columns
    assert "LKNE_x" not in out.columns
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_vicon_reader.py -v`
Expected: FAIL (current reader raises `ValueError` / no `timestamp` / no `vicon_fps` kwarg).

- [ ] **Step 3: Rewrite `vicon_reader.py`**

Replace the whole file with:
```python
"""Read Vicon XLSX into the unified pose DataFrame (real export schema)."""
import numpy as np
import pandas as pd
from openpyxl import load_workbook


def _read_rows(path: str, max_rows: int | None = None) -> list[list]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = []
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        rows.append(list(r))
        if max_rows is not None and i + 1 >= max_rows:
            break
    wb.close()
    return rows


def _find_axis_row(rows: list[list]) -> int:
    """Index of the row that contains the X/Y/Z axis labels.

    The real axis row also holds 'Frame'/'Sub Frame', so we look for a row where
    X/Y/Z dominate rather than one that is exclusively X/Y/Z.
    """
    for i, row in enumerate(rows):
        cells = [str(c).strip().upper() if c is not None else "" for c in row]
        xyz = sum(c in ("X", "Y", "Z") for c in cells)
        if xyz >= 3:
            return i
    raise ValueError("Could not locate the X/Y/Z axis row in Vicon XLSX")


def _marker_name(cell) -> str | None:
    """Clean marker token: strip subject prefix ('Subject:LKNE' -> 'LKNE')."""
    if cell is None or not str(cell).strip():
        return None
    name = str(cell).strip()
    if ":" in name:
        name = name.split(":", 1)[1].strip()
    return name


def load_vicon_xlsx(filepath: str, vicon_fps: float = 100.0) -> pd.DataFrame:
    """Parse a Vicon Trajectories export into the unified pose DataFrame.

    - axis row auto-detected (tolerant of Frame/Sub Frame columns);
    - marker names are colon-split to drop the subject prefix;
    - the duplicate '| ... |' block yields names like 'LKNE |' that simply
      become extra columns and are dropped at mapping time;
    - mm->m when median(|coord|) > 10;
    - adds zero-based ``timestamp`` = (frame - frame[0]) / vicon_fps and ``frame``.
    """
    rows = _read_rows(filepath)
    axis_i = _find_axis_row(rows)
    name_row = rows[axis_i - 1]
    axis_row = rows[axis_i]

    # forward-fill marker names across the 3 (X,Y,Z) columns they span
    names: list[str | None] = []
    last = None
    for c in name_row:
        nm = _marker_name(c)
        if nm is not None:
            last = nm
        names.append(last)

    # locate the Frame column in the axis row (first cell whose label is 'Frame')
    frame_col = None
    for j, a in enumerate(axis_row):
        if a is not None and str(a).strip().lower() == "frame":
            frame_col = j
            break

    columns: list[str | None] = []
    for name, axis in zip(names, axis_row):
        ax = str(axis).strip().upper() if axis is not None else ""
        if name is None or ax not in ("X", "Y", "Z"):
            columns.append(None)
            continue
        columns.append(f"{name}_{ax.lower()}")

    data_rows = rows[axis_i + 1:]
    records, frames = [], []
    for r in data_rows:
        if all(c is None for c in r):
            continue
        rec = {}
        for col, val in zip(columns, r):
            if col is None or val is None:
                continue
            try:
                rec[col] = float(val)
            except (TypeError, ValueError):
                continue
        if not rec:
            continue
        records.append(rec)
        fv = r[frame_col] if frame_col is not None and frame_col < len(r) else None
        try:
            frames.append(float(fv))
        except (TypeError, ValueError):
            frames.append(float(len(frames)))

    df = pd.DataFrame(records)
    # drop the units row (e.g. all 'mm') which produced an empty/NaN record set:
    df = df.dropna(how="all").reset_index(drop=True)

    coord = df.to_numpy(dtype=float)
    if np.nanmedian(np.abs(coord)) > 10.0:
        df = df / 1000.0

    frames = np.asarray(frames[: len(df)], dtype=float)
    df.insert(0, "timestamp", (frames - frames[0]) / vicon_fps)
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

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_vicon_reader.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Smoke-test against the real file**

Run:
```bash
.venv/bin/python -c "
from modules.data_loader.vicon_reader import load_vicon_xlsx, map_vicon_to_caliscope
import yaml
df = load_vicon_xlsx('data/Vicon_10_series/1.xlsx')
print('shape', df.shape, 'has ts', 'timestamp' in df.columns)
m = yaml.safe_load(open('settings.yaml'))['landmark_mapping']
out = map_vicon_to_caliscope(df, m)
print('left_knee_z present:', 'left_knee_z' in out.columns)
print('median |left_knee_z|:', float(out['left_knee_z'].abs().median()))
"
```
Expected: `(≈13610 rows, many marker columns — the `| … |` duplicate block adds extras that map-time ignores)`, `has ts True`, `left_knee_z present: True`, and median `|left_knee_z|` < 2 (meters, not mm).

- [ ] **Step 6: Commit**

```bash
git add modules/data_loader/vicon_reader.py tests/test_vicon_reader.py
git commit -m "fix(vicon): parse real Trajectories schema (subject prefix, Frame cols, mm, timestamp)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Configuration additions

**Files:**
- Modify: `gait_analysis/settings.yaml`

- [ ] **Step 1: Extend the `comparison:` block**

Replace the existing `comparison:` block at the end of `settings.yaml` with:
```yaml
comparison:
  icc_type: '3,1'
  good_rmse_threshold_deg: 5.0
  acceptable_rmse_threshold_deg: 10.0
  vicon_fps: 100.0                  # Vicon device sample rate
  sync_method: xcorr                # xcorr | jump | zmax (temporal alignment for position layer)
  reference_landmarks: null         # markers used for rigid alignment (null = all common)
  pairs: null                       # optional {caliscope_session: vicon_file} synchronous map
```

- [ ] **Step 2: Verify it loads**

Run: `.venv/bin/python -c "import yaml; c=yaml.safe_load(open('settings.yaml')); print(c['comparison'])"`
Expected: dict prints with the 7 keys.

- [ ] **Step 3: Commit**

```bash
git add settings.yaml
git commit -m "config(phase3): comparison vicon_fps/sync_method/reference_landmarks/pairs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `alignment.py` — Umeyama rigid+scale transform

**Files:**
- Create: `gait_analysis/modules/comparison/__init__.py`
- Create: `gait_analysis/modules/comparison/alignment.py`
- Create: `gait_analysis/tests/test_alignment.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_alignment.py`:
```python
import numpy as np

from modules.comparison.alignment import apply_transform, estimate_rigid_transform


def _rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0, 0, 1.0]])


def test_umeyama_recovers_known_transform():
    rng = np.random.default_rng(42)
    src = rng.normal(size=(50, 3))
    R_true, s_true, T_true = _rot_z(30.0), 2.5, np.array([1.0, -2.0, 0.5])
    dst = s_true * (src @ R_true.T) + T_true

    R, T, s = estimate_rigid_transform(src, dst)
    assert np.isclose(s, s_true, atol=1e-6)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(T, T_true, atol=1e-6)
    assert np.linalg.det(R) > 0          # reflection-free
    assert np.allclose(apply_transform(src, R, T, s), dst, atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: FAIL with `ModuleNotFoundError: modules.comparison`.

- [ ] **Step 3: Implement**

Create `modules/comparison/__init__.py` (empty file).

Create `modules/comparison/alignment.py`:
```python
"""Coordinate-system registration (Umeyama) and temporal sync for Vicon comparison.

Ported and cleaned from the VvsC.py baseline (compute_similarity_transform,
estimate_time_shift_by_xcorr, detect_event_index).
"""
import numpy as np


def estimate_rigid_transform(src: np.ndarray, dst: np.ndarray):
    """Umeyama (1991) similarity: find (R, T, s) with dst ≈ s * (src @ R.T) + T.

    Reflection-free (det(R) = +1), uniform scale. src/dst are (N, 3).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - mu_s, dst - mu_d
    H = s0.T @ d0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_s = (s0 ** 2).sum()
    s = S.sum() / var_s if var_s > 0 else 1.0
    T = mu_d - s * (R @ mu_s)
    return R, T, s


def apply_transform(points: np.ndarray, R: np.ndarray, T: np.ndarray, s: float) -> np.ndarray:
    """Apply (R, T, s): s * (points @ R.T) + T."""
    return s * (np.asarray(points, float) @ R.T) + T
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/comparison/__init__.py modules/comparison/alignment.py tests/test_alignment.py
git commit -m "feat(comparison): Umeyama rigid+scale transform (alignment.py)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `alignment.py` — temporal sync (xcorr + event detection)

**Files:**
- Modify: `gait_analysis/modules/comparison/alignment.py`
- Modify: `gait_analysis/tests/test_alignment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_alignment.py`:
```python
from modules.comparison.alignment import detect_sync_event, estimate_time_shift_xcorr


def test_xcorr_recovers_injected_time_shift():
    fs = 100.0
    t = np.arange(0, 4, 1 / fs)
    # a moving pelvis with a sharp speed burst (a "jump"): position ramp + bump
    pos = np.zeros((len(t), 3))
    pos[:, 0] = np.cumsum(np.exp(-((t - 1.0) ** 2) / 0.01)) / fs   # burst near t=1s
    shift = 0.20
    t_cal = t + shift                                              # caliscope lags by 0.2 s
    s = estimate_time_shift_xcorr(t, pos, t_cal, pos, fs_grid=fs, max_shift=0.5)
    assert abs(s - shift) < 0.03


def test_detect_sync_event_finds_speed_peak():
    fs = 100.0
    t = np.arange(0, 3, 1 / fs)
    pos = np.zeros((len(t), 3))
    pos[:, 2] = np.exp(-((t - 0.5) ** 2) / 0.005)                 # Z bump at 0.5 s
    idx = detect_sync_event(t, pos, mode="zmax")
    assert abs(t[idx] - 0.5) < 0.05
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: FAIL (`ImportError` for the two new names).

- [ ] **Step 3: Implement (append to `alignment.py`)**

```python
def detect_sync_event(time: np.ndarray, point: np.ndarray, mode: str = "speed") -> int:
    """Index of the sync event: 'speed' = max 3D speed, 'zmax' = max Z (jump apex)."""
    time = np.asarray(time, float)
    point = np.asarray(point, float)
    if len(time) < 2:
        return 0
    if mode == "zmax":
        return int(np.nanargmax(point[:, 2]))
    if mode == "speed":
        dt = np.diff(time)
        dt[dt == 0] = np.min(dt[dt > 0]) if np.any(dt > 0) else 1.0
        speed = np.linalg.norm(np.diff(point, axis=0) / dt[:, None], axis=1)
        return int(np.nanargmax(speed)) + 1
    raise ValueError(f"unknown sync mode {mode!r}")


def _speed(time: np.ndarray, point: np.ndarray):
    dt = np.diff(time)
    dt[dt == 0] = np.min(dt[dt > 0]) if np.any(dt > 0) else 1.0
    speed = np.linalg.norm(np.diff(point, axis=0) / dt[:, None], axis=1)
    t_mid = 0.5 * (time[:-1] + time[1:])
    return t_mid, speed


def estimate_time_shift_xcorr(t_a, pts_a, t_b, pts_b, fs_grid: float = 100.0,
                              max_shift: float = 0.5) -> float:
    """Lag (seconds) such that t_b - shift aligns stream B onto stream A.

    Cross-correlates the 3D speed of a shared point (e.g. pelvis). Ported from
    VvsC.estimate_time_shift_by_xcorr.
    """
    t_a, t_b = np.asarray(t_a, float), np.asarray(t_b, float)
    ta_s, sa = _speed(t_a, np.asarray(pts_a, float))
    tb_s, sb = _speed(t_b, np.asarray(pts_b, float))
    start, end = max(ta_s[0], tb_s[0]), min(ta_s[-1], tb_s[-1])
    if end - start <= 2.0 / fs_grid:
        raise ValueError("overlap too small to estimate time shift")
    grid = np.arange(start, end, 1.0 / fs_grid)
    ga = np.interp(grid, ta_s, sa)
    gb = np.interp(grid, tb_s, sb)
    ga = (ga - ga.mean()) / (ga.std() or 1.0)
    gb = (gb - gb.mean()) / (gb.std() or 1.0)
    corr = np.correlate(gb, ga, mode="full")
    lags = np.arange(-len(ga) + 1, len(gb))
    mlag = int(max_shift * fs_grid)
    mask = (lags >= -mlag) & (lags <= mlag)
    best = lags[mask][int(np.argmax(corr[mask]))]
    return best / fs_grid
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: PASS (4 tests total).

- [ ] **Step 5: Commit**

```bash
git add modules/comparison/alignment.py tests/test_alignment.py
git commit -m "feat(comparison): xcorr time-shift + sync-event detection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `metrics.py` — error metrics + ICC

**Files:**
- Create: `gait_analysis/modules/comparison/metrics.py`
- Create: `gait_analysis/tests/test_metrics_comparison.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics_comparison.py`:
```python
import numpy as np
import pytest

from modules.comparison.metrics import calc_icc, calc_mae, calc_pearson, calc_rmse


def test_identity_arrays_give_perfect_scores():
    a = np.linspace(0, 60, 101)
    assert calc_rmse(a, a) == pytest.approx(0.0, abs=1e-12)
    assert calc_mae(a, a) == pytest.approx(0.0, abs=1e-12)
    assert calc_pearson(a, a) == pytest.approx(1.0, abs=1e-9)
    assert calc_icc(a, a) == pytest.approx(1.0, abs=1e-6)


def test_rmse_mae_known_values():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert calc_rmse(a, b) == pytest.approx(np.sqrt((9 + 16 + 0) / 3))
    assert calc_mae(a, b) == pytest.approx((3 + 4 + 0) / 3)


def test_nan_aware():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, 5.0, 3.0])
    assert calc_rmse(a, b) == pytest.approx(0.0, abs=1e-12)


def test_icc_constant_input_is_nan():
    a = np.ones(50)
    b = np.linspace(0, 1, 50)
    assert np.isnan(calc_icc(a, b))


def test_icc_high_for_strongly_agreeing_raters():
    rng = np.random.default_rng(0)
    a = rng.normal(size=80)
    b = a + rng.normal(scale=0.05, size=80)     # near-identical raters
    assert calc_icc(a, b) > 0.9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_metrics_comparison.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

Create `modules/comparison/metrics.py`:
```python
"""Accuracy metrics for Vicon comparison: RMSE/MAE/Pearson (NaN-aware) + ICC."""
import numpy as np

_ICC_TYPE = {"1,1": "ICC1", "2,1": "ICC2", "3,1": "ICC3"}


def _paired(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    m = ~(np.isnan(a) | np.isnan(b))
    return a[m], b[m]


def calc_rmse(a, b) -> float:
    a, b = _paired(a, b)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def calc_mae(a, b) -> float:
    a, b = _paired(a, b)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def calc_pearson(a, b) -> float:
    a, b = _paired(a, b)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def calc_icc(a, b, icc_type: str = "3,1") -> float:
    """ICC between two raters a, b (1-D, paired). Uses pingouin; NaN if degenerate."""
    import pandas as pd
    import pingouin as pg

    a, b = _paired(a, b)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    n = a.size
    long = pd.DataFrame({
        "target": list(range(n)) * 2,
        "rater": ["a"] * n + ["b"] * n,
        "rating": np.concatenate([a, b]),
    })
    res = pg.intraclass_corr(data=long, targets="target", raters="rater",
                             ratings="rating").set_index("Type")
    return float(res.loc[_ICC_TYPE[icc_type], "ICC"])
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_metrics_comparison.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add modules/comparison/metrics.py tests/test_metrics_comparison.py
git commit -m "feat(comparison): RMSE/MAE/Pearson + pingouin ICC(3,1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `metrics.py` — angle + position report builders

**Files:**
- Modify: `gait_analysis/modules/comparison/metrics.py`
- Modify: `gait_analysis/tests/test_metrics_comparison.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_metrics_comparison.py`:
```python
from modules.comparison.metrics import (
    angle_comparison_report,
    position_comparison_report,
    verdict_for_rmse,
)


def test_verdict_thresholds():
    assert verdict_for_rmse(3.0, 5.0, 10.0) == "good"
    assert verdict_for_rmse(7.0, 5.0, 10.0) == "acceptable"
    assert verdict_for_rmse(12.0, 5.0, 10.0) == "poor"


def test_angle_report_uses_joint_intersection_and_verdicts():
    cal = {"left_knee": np.linspace(0, 60, 101), "left_ankle": np.linspace(0, 20, 101)}
    vic = {"left_knee": np.linspace(0, 60, 101)}          # ankle missing on one side
    rep = angle_comparison_report(cal, vic, good=5.0, acceptable=10.0)
    assert set(rep["joint"]) == {"left_knee"}              # intersection only
    row = rep.set_index("joint").loc["left_knee"]
    assert row["rmse_deg"] == 0.0 and row["verdict"] == "good"
    assert row["icc"] > 0.99


def test_position_report_zero_error_for_identical_points():
    pts = {"LKNE": np.tile([0.1, 0.2, 0.9], (10, 1))}
    rep = position_comparison_report(pts, pts)
    row = rep.set_index("joint").loc["LKNE"]
    assert row["rmse_m"] == 0.0 and row["mae_m"] == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_metrics_comparison.py -v`
Expected: FAIL (new names undefined).

- [ ] **Step 3: Implement**

First add `import pandas as pd` to the **top** of `metrics.py` (directly under `import numpy as np`). Then append the following functions:

```python
def verdict_for_rmse(rmse_deg: float, good: float, acceptable: float) -> str:
    if np.isnan(rmse_deg):
        return "n/a"
    if rmse_deg < good:
        return "good"
    if rmse_deg <= acceptable:
        return "acceptable"
    return "poor"


def angle_comparison_report(cal_curves: dict, vic_curves: dict,
                            good: float = 5.0, acceptable: float = 10.0,
                            icc_type: str = "3,1", joint_list=None) -> pd.DataFrame:
    """Per-joint angle agreement over matched 101-pt curves.

    cal_curves / vic_curves: {"<side>_<joint>": 101-array}. Only joints present in
    BOTH (and non-degenerate) are reported — this naturally drops hip on the Vicon
    side (no shoulder marker -> all-NaN curve).
    """
    joints = joint_list or sorted(set(cal_curves) & set(vic_curves))
    rows = []
    for j in joints:
        a, b = np.asarray(cal_curves[j], float), np.asarray(vic_curves[j], float)
        if np.isnan(a).all() or np.isnan(b).all():
            continue
        rmse = calc_rmse(a, b)
        rows.append({
            "joint": j,
            "rmse_deg": rmse,
            "mae_deg": calc_mae(a, b),
            "pearson": calc_pearson(a, b),
            "icc": calc_icc(a, b, icc_type=icc_type),
            "verdict": verdict_for_rmse(rmse, good, acceptable),
        })
    return pd.DataFrame(rows, columns=["joint", "rmse_deg", "mae_deg",
                                       "pearson", "icc", "verdict"])


def position_comparison_report(cal_pts: dict, vic_pts: dict, joint_list=None) -> pd.DataFrame:
    """Per-joint positional error (meters) between matched (N,3) arrays.

    cal_pts / vic_pts: {"JOINT": (N,3)} already on a common, aligned grid.
    """
    joints = joint_list or sorted(set(cal_pts) & set(vic_pts))
    rows = []
    for j in joints:
        v, c = np.asarray(vic_pts[j], float), np.asarray(cal_pts[j], float)
        diff = v - c
        mask = ~np.isnan(diff).any(axis=1)
        if not mask.any():
            continue
        d = diff[mask]
        dist = np.linalg.norm(d, axis=1)
        rows.append({
            "joint": j,
            "n_samples": int(dist.size),
            "rmse_m": float(np.sqrt(np.mean(dist ** 2))),
            "mae_m": float(np.mean(dist)),
            "max_m": float(np.max(dist)),
            "median_m": float(np.median(dist)),
            "rmse_x_m": float(np.sqrt(np.mean(d[:, 0] ** 2))),
            "rmse_y_m": float(np.sqrt(np.mean(d[:, 1] ** 2))),
            "rmse_z_m": float(np.sqrt(np.mean(d[:, 2] ** 2))),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_metrics_comparison.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add modules/comparison/metrics.py tests/test_metrics_comparison.py
git commit -m "feat(comparison): angle + position report builders with verdicts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: `report.py` — assemble `comparison_report.json`

**Files:**
- Create: `gait_analysis/modules/comparison/report.py`
- Create: `gait_analysis/tests/test_comparison_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_comparison_report.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_comparison_report.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

Create `modules/comparison/report.py`:
```python
"""Assemble the canonical comparison_report.json (angle + position layers)."""
import datetime as dt


def _df_to_keyed(df, key="joint"):
    out = {}
    for _, row in df.iterrows():
        d = row.to_dict()
        out[d.pop(key)] = d
    return out


def build_report(angle_df, position_df, overlay: dict, meta: dict) -> dict:
    """Combine the angle table, position table, and overlay curves into one dict.

    meta: pair_id, model, caliscope_fps, vicon_fps, time_shift_s, scale, low_confidence.
    """
    angle = _df_to_keyed(angle_df)
    knee = angle.get("left_knee", angle.get("right_knee", {}))
    worst = max((v["verdict"] for v in angle.values()),
                key=lambda x: {"good": 0, "acceptable": 1, "poor": 2, "n/a": -1}.get(x, -1),
                default="n/a")
    summary = (f"{len(angle)} joints compared; knee ICC="
               f"{knee.get('icc', float('nan')):.3f}; worst verdict: {worst}")
    return {
        "pair_id": meta.get("pair_id"),
        "model": meta.get("model"),
        "caliscope_fps": meta.get("caliscope_fps"),
        "vicon_fps": meta.get("vicon_fps"),
        "processed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "angle": angle,
        "angle_overlay": overlay,
        "position": {
            "joints": _df_to_keyed(position_df) if len(position_df) else {},
            "time_shift_s": meta.get("time_shift_s"),
            "scale": meta.get("scale"),
            "low_confidence": meta.get("low_confidence", False),
        },
        "verdict_summary": summary,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_comparison_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/comparison/report.py tests/test_comparison_report.py
git commit -m "feat(comparison): build_report assembles comparison_report.json

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: `compare_pipeline.py` — shared `run_comparison`

**Files:**
- Create: `gait_analysis/compare_pipeline.py`
- Create: `gait_analysis/tests/test_compare_pipeline.py`

- [ ] **Step 1: Write the failing test (synthetic transformed-copy stand-in)**

Create `tests/test_compare_pipeline.py`:
```python
import numpy as np
import pandas as pd
import yaml

from compare_pipeline import run_comparison


def _walking_df(n=240, fps=60.0):
    """A synthetic lower-body walker: oscillating knee/ankle so events + angles exist."""
    t = np.arange(n) / fps
    df = pd.DataFrame({"frame": range(n), "timestamp": t})
    phase = 2 * np.pi * 1.0 * t                      # ~1 Hz stride
    # vertical (z) bounce drives heel-strike detection; x advances forward
    for side, ph in (("left", 0.0), ("right", np.pi)):
        df[f"{side}_hip_x"] = 0.4 * t
        df[f"{side}_hip_y"] = 0.0
        df[f"{side}_hip_z"] = 0.9 + 0.02 * np.sin(phase + ph)
        df[f"{side}_knee_x"] = 0.4 * t + 0.05 * np.sin(phase + ph)
        df[f"{side}_knee_y"] = 0.0
        df[f"{side}_knee_z"] = 0.5 + 0.05 * np.cos(phase + ph)
        df[f"{side}_ankle_x"] = 0.4 * t
        df[f"{side}_ankle_y"] = 0.0
        df[f"{side}_ankle_z"] = 0.1 + 0.04 * np.sin(phase + ph)
        df[f"{side}_heel_z"] = 0.08 + 0.05 * (np.sin(phase + ph) ** 2)
        df[f"{side}_foot_index_z"] = 0.05 + 0.05 * (np.cos(phase + ph) ** 2)
        df[f"{side}_heel_x"] = 0.4 * t
        df[f"{side}_heel_y"] = 0.0
        df[f"{side}_foot_index_x"] = 0.4 * t
        df[f"{side}_foot_index_y"] = 0.0
    df.attrs["fps"] = fps
    return df


def test_run_comparison_identical_streams_perfect_angle_agreement():
    cfg = yaml.safe_load(open("settings.yaml"))
    cal = _walking_df()
    vic = cal.copy()
    vic.attrs["fps"] = cal.attrs["fps"]               # true copy (df.copy may drop attrs)
    report, _ = run_comparison(cal, vic, cfg, model="SIMPLE_HOLISTIC", pair_id="synthetic")
    assert report["angle"], "expected at least one comparable joint"
    # knee/ankle present in both; hip excluded (no shoulder markers here)
    assert all(j in ("left_knee", "right_knee", "left_ankle", "right_ankle")
               for j in report["angle"])
    knee = report["angle"].get("left_knee") or report["angle"].get("right_knee")
    assert knee["rmse_deg"] < 1.0                     # same motion -> near-zero angle error
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_compare_pipeline.py -v`
Expected: FAIL (`ModuleNotFoundError: compare_pipeline`).

- [ ] **Step 3: Implement**

Create `compare_pipeline.py` (sibling of `pipeline.py`):
```python
"""Shared Vicon-comparison pipeline: one implementation for cli.py and the GUI worker."""
import numpy as np

from modules.comparison.metrics import angle_comparison_report
from modules.comparison.report import build_report
from modules.kinematics.filters import fill_gaps
from modules.kinematics.gait_events import detect_gait_events
from modules.kinematics.joint_angles import calc_joint_angles_timeseries
from modules.kinematics.normalizer import get_mean_std_cycle, normalize_gait_cycle
from pipeline import _filter_coords                       # reuse the Phase-1 coord filter

_JOINTS = ("hip", "knee", "ankle")


def _joint_curves(df, cfg, fps):
    """Return {"<side>_<joint>": mean 101-pt curve} for joints with valid data.

    All-NaN angle columns (e.g. Vicon hip, which needs a shoulder marker) are skipped.
    """
    proc, g = cfg["processing"], cfg["gait_events"]
    df = fill_gaps(df, max_gap_frames=proc["max_gap_frames"])
    df = _filter_coords(df, proc["filter_cutoff_hz"], proc["filter_order"], fps)
    events = detect_gait_events(
        df, fps=fps, method=g["method"], heel=g["heel_landmark"],
        toe=g["toe_landmark"], vertical=g["vertical_axis"],
        min_stride_sec=proc["min_stride_duration_sec"], cutoff_hz=proc["filter_cutoff_hz"],
    )
    df = calc_joint_angles_timeseries(df)
    curves, cycles = {}, {}
    for side in ("left", "right"):
        if not events.get(f"{side}_HS"):
            continue
        for joint in _JOINTS:
            col = f"{side}_{joint}_angle"
            if col not in df.columns:
                continue
            arr = df[col].to_numpy()
            if np.isnan(arr).all():
                continue
            mat = normalize_gait_cycle(arr, events, side=side)
            if mat.shape[0] == 0:
                continue
            mean, _ = get_mean_std_cycle(mat)
            if np.isnan(mean).all():
                continue
            curves[f"{side}_{joint}"] = mean
            cycles[f"{side}_{joint}"] = mat
    return curves, cycles


def run_comparison(cal_df, vic_df, cfg, *, model, pair_id, progress_cb=None):
    """Compare one caliscope stream vs one Vicon stream → (report, artifacts).

    Angle layer (primary): same Module-2 kinematics on both → matched 101-pt curves.
    Position layer is added in a later task; here it is reported empty when unavailable.
    artifacts carries per-cycle matrices so callers (validate-vicon) can pool across clips.
    """
    import pandas as pd

    ccmp = cfg["comparison"]

    def report(frac, stage):
        if progress_cb is not None:
            progress_cb(frac, stage)

    cal_fps = cal_df.attrs.get("fps")
    vic_fps = vic_df.attrs.get("fps", ccmp.get("vicon_fps", 100.0))

    report(0.15, "Caliscope kinematics")
    cal_curves, cal_cycles = _joint_curves(cal_df, cfg, cal_fps)
    report(0.45, "Vicon kinematics")
    vic_curves, vic_cycles = _joint_curves(vic_df, cfg, vic_fps)

    report(0.70, "Angle metrics")
    angle_df = angle_comparison_report(
        cal_curves, vic_curves,
        good=ccmp["good_rmse_threshold_deg"],
        acceptable=ccmp["acceptable_rmse_threshold_deg"],
        icc_type=ccmp["icc_type"],
    )
    overlay = {
        j: {"caliscope": list(map(float, cal_curves[j])),
            "vicon": list(map(float, vic_curves[j]))}
        for j in set(cal_curves) & set(vic_curves)
    }

    report(0.85, "Position metrics")
    position_df = pd.DataFrame()        # position layer added in Task 9

    report(0.95, "Building report")
    rep = build_report(angle_df, position_df, overlay, meta={
        "pair_id": pair_id, "model": model,
        "caliscope_fps": None if cal_fps is None else round(float(cal_fps), 3),
        "vicon_fps": round(float(vic_fps), 3),
        "time_shift_s": None, "scale": None, "low_confidence": False,
    })
    report(1.0, "Done")
    artifacts = {"cal_cycles": cal_cycles, "vic_cycles": vic_cycles}
    return rep, artifacts
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_compare_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add compare_pipeline.py tests/test_compare_pipeline.py
git commit -m "feat(comparison): run_comparison angle layer (shared pipeline)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Position layer in `run_comparison` + VvsC parity

Add the positional comparison to `run_comparison` and prove it reproduces `VvsC.py` (criterion #1).

**Files:**
- Modify: `gait_analysis/compare_pipeline.py`
- Modify: `gait_analysis/modules/comparison/alignment.py` (add `align_streams` helper)
- Create: `gait_analysis/tests/test_vvsc_parity.py`

- [ ] **Step 1: Write the failing parity tests**

The load-bearing parity is the **core algorithm**: our `estimate_rigid_transform` and VvsC's `compute_similarity_transform` use identical math (`H = A0ᵀB0`, same SVD, same det-flip, same `s = ΣS/var_A`, same `T = μ_B − s·R·μ_A`) and the same convention `B ≈ s·(A@Rᵀ) + T`. That parity is exact and needs no real data. The end-to-end positional RMSE is a looser, real-data ballpark check (interp/window choices differ from VvsC).

Create `tests/test_vvsc_parity.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_vvsc_parity.py -v`
Expected: the exact-parity test FAILS only if `align_streams`/`estimate_rigid_transform` aren't importable yet (they are, from Task 3) — it should actually PASS already since `estimate_rigid_transform` exists. The ballpark test FAILS with `position layer produced no joints` until Steps 3–4 add the position layer. (If real data is absent it skips; confirm with `ls data/Vicon_10_series/1.xlsx`.)

- [ ] **Step 3: Add `align_streams` to `alignment.py`**

Append to `modules/comparison/alignment.py`:
```python
def _pelvis(df):
    import numpy as np
    cols = [("left_hip", "right_hip")]
    for lh, rh in cols:
        if all(f"{lh}_{a}" in df.columns for a in "xyz") and \
           all(f"{rh}_{a}" in df.columns for a in "xyz"):
            l = df[[f"{lh}_x", f"{lh}_y", f"{lh}_z"]].to_numpy(float)
            r = df[[f"{rh}_x", f"{rh}_y", f"{rh}_z"]].to_numpy(float)
            return 0.5 * (l + r)
    return None


def align_streams(cal_df, vic_df, joints, fs_grid=100.0, max_shift=0.5):
    """Time-align (xcorr on pelvis speed) + rigid+scale register Vicon onto caliscope.

    joints: {"JOINT": "canonical_landmark"} to compare, e.g. {"LKNE": "left_knee"}.
    Returns (cal_pts, vic_pts_aligned, info) where each *_pts is {"JOINT": (N,3)} on a
    shared grid, and info has time_shift_s + scale.
    """
    import numpy as np

    from modules.data_loader.synchronizer import synchronize

    cp, vp = _pelvis(cal_df), _pelvis(vic_df)
    shift = 0.0
    if cp is not None and vp is not None:
        shift = estimate_time_shift_xcorr(
            cal_df["timestamp"].to_numpy(), cp,
            vic_df["timestamp"].to_numpy(), vp, fs_grid=fs_grid, max_shift=max_shift)
    vic_shifted = vic_df.copy()
    vic_shifted["timestamp"] = vic_shifted["timestamp"] - shift
    cal_g, vic_g = synchronize(cal_df, vic_shifted, target_fps=fs_grid)

    def pts(df, landmark):
        cols = [f"{landmark}_{a}" for a in "xyz"]
        if not all(c in df.columns for c in cols):
            return None
        return df[cols].to_numpy(float)

    cal_pts, vic_pts = {}, {}
    for jname, landmark in joints.items():
        c, v = pts(cal_g, landmark), pts(vic_g, landmark)
        if c is None or v is None:
            continue
        cal_pts[jname], vic_pts[jname] = c, v

    # estimate one transform on all common, NaN-free samples (concatenated)
    A, B = [], []
    common = sorted(set(cal_pts) & set(vic_pts))
    for j in common:
        m = ~(np.isnan(cal_pts[j]).any(1) | np.isnan(vic_pts[j]).any(1))
        A.append(vic_pts[j][m])
        B.append(cal_pts[j][m])
    scale = None
    if A and sum(len(a) for a in A) >= 3:
        R, T, scale = estimate_rigid_transform(np.concatenate(A), np.concatenate(B))
        vic_pts = {j: apply_transform(vic_pts[j], R, T, scale) for j in vic_pts}
    return cal_pts, vic_pts, {"time_shift_s": float(shift),
                              "scale": None if scale is None else float(scale)}
```

- [ ] **Step 4: Wire the position layer into `run_comparison`**

First add this import to the top-of-file imports in `compare_pipeline.py` (next to the other `modules.comparison.metrics` import):
```python
from modules.comparison.metrics import position_comparison_report
```
Then in `compare_pipeline.py`, replace the line `position_df = pd.DataFrame()        # position layer added in Task 9` with:
```python
    from modules.comparison.alignment import align_streams

    pos_joints = {"LHIP": "left_hip", "RHIP": "right_hip", "LKNE": "left_knee",
                  "RKNE": "right_knee", "LANK": "left_ankle", "RANK": "right_ankle"}
    try:
        cal_pts, vic_pts, align_info = align_streams(
            cal_df, vic_df, pos_joints,
            fs_grid=ccmp.get("vicon_fps", 100.0), max_shift=0.5)
        position_df = position_comparison_report(cal_pts, vic_pts)
    except (ValueError, KeyError):
        position_df, align_info = pd.DataFrame(), {"time_shift_s": None, "scale": None}
```
And update the `meta=` dict in the `build_report(...)` call to use the alignment info:
```python
        "time_shift_s": align_info.get("time_shift_s"),
        "scale": align_info.get("scale"),
        "low_confidence": False,
```

- [ ] **Step 5: Run parity + existing pipeline tests**

Run:
```bash
.venv/bin/python -m pytest tests/test_vvsc_parity.py tests/test_compare_pipeline.py -v
```
Expected: exact-parity test PASS (our Umeyama == VvsC's transform); ballpark test PASS (global RMSE same order as VvsC, `rel=0.5`); compare_pipeline still PASS. If the ballpark test is off by >2×, check that mm→m happened in the reader (a ~1000× error means units), and that `align_streams` uses pelvis-speed xcorr like VvsC.

- [ ] **Step 6: Commit**

```bash
git add compare_pipeline.py modules/comparison/alignment.py tests/test_vvsc_parity.py
git commit -m "feat(comparison): position layer + VvsC parity (criterion #1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: `compare` CLI subcommand

**Files:**
- Modify: `gait_analysis/cli.py`
- Create: `gait_analysis/tests/test_compare_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compare_cli.py`:
```python
import json

import numpy as np
import pandas as pd

import cli


def _synthetic_pair(tmp_path):
    # write a tiny unified-format caliscope CSV session and a Vicon xlsx? Too heavy.
    # Instead exercise the `compare` function with monkeypatched loaders.
    pass


def test_compare_writes_report(tmp_path, monkeypatch):
    from tests.test_compare_pipeline import _walking_df

    cal = _walking_df()
    vic = cal.copy(); vic.attrs["fps"] = 100.0

    monkeypatch.setattr(cli, "load_caliscope_session", lambda *a, **k: cal)
    monkeypatch.setattr(cli, "_load_vicon_mapped", lambda *a, **k: vic)

    out = tmp_path / "comparison_report.json"
    res = cli.compare("ignored_session", "ignored.xlsx", "SIMPLE_HOLISTIC", str(out))
    assert out.exists()
    saved = json.loads(out.read_text())
    assert "angle" in saved and "position" in saved
    assert res["pair_id"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_compare_cli.py -v`
Expected: FAIL (`AttributeError: module 'cli' has no attribute 'compare'`).

- [ ] **Step 3: Implement (modify `cli.py`)**

Add imports near the top of `cli.py` (after the existing imports):
```python
from modules.data_loader.vicon_reader import load_vicon_xlsx, map_vicon_to_caliscope
from compare_pipeline import run_comparison
```

Add these functions after `reproducibility(...)`:
```python
def _load_vicon_mapped(vicon_path: str, cfg: dict):
    df = load_vicon_xlsx(vicon_path, vicon_fps=cfg["comparison"].get("vicon_fps", 100.0))
    return map_vicon_to_caliscope(df, cfg["landmark_mapping"])


def compare(session_dir: str, vicon_path: str, model: str, out_path: str) -> dict:
    cfg = _load_settings()
    cal = load_caliscope_session(session_dir, model=model)
    vic = _load_vicon_mapped(vicon_path, cfg)
    pair_id = f"{Path(session_dir).name}__{Path(vicon_path).stem}"
    report, _ = run_comparison(cal, vic, cfg, model=model, pair_id=pair_id)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))
    return report
```

In `main()`, add a subparser (after the `reproducibility` parser block):
```python
    c = sub.add_parser("compare", help="Compare one caliscope session vs one Vicon trial")
    c.add_argument("--session", required=True)
    c.add_argument("--vicon", required=True)
    c.add_argument("--model", default=default_model)
    c.add_argument("--out", required=True)
```

And add the dispatch branch (after the `reproducibility` branch):
```python
    elif args.command == "compare":
        rep = compare(args.session, args.vicon, args.model, args.out)
        knee = rep["angle"].get("left_knee") or rep["angle"].get("right_knee") or {}
        print(f"Wrote {args.out}: {rep['verdict_summary']}")
        if knee:
            print(f"  knee RMSE={knee['rmse_deg']:.2f}° ICC={knee['icc']:.3f}")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_compare_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke-test on real data**

Run:
```bash
.venv/bin/python cli.py compare --session data/caliscope_project/recordings/p1_1 \
  --vicon data/Vicon_10_series/1.xlsx --out results/compare_p1_1__1.json
```
Expected: prints a verdict summary + knee RMSE/ICC; writes `results/compare_p1_1__1.json`.

- [ ] **Step 6: Commit**

```bash
git add cli.py tests/test_compare_cli.py
git commit -m "feat(cli): compare subcommand -> comparison_report.json

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: `validate-vicon` CLI — Level A/C tables (pooled)

**Files:**
- Modify: `gait_analysis/cli.py`
- Create: `gait_analysis/tests/test_validate_vicon.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate_vicon.py`:
```python
import numpy as np

import cli


def test_pool_curves_concatenates_cycles_across_pairs():
    # two pairs, each contributing left_knee cycle matrices -> one ensemble mean
    a = {"cal_cycles": {"left_knee": np.full((2, 101), 10.0)},
         "vic_cycles": {"left_knee": np.full((3, 101), 10.0)}}
    b = {"cal_cycles": {"left_knee": np.full((1, 101), 20.0)},
         "vic_cycles": {"left_knee": np.full((1, 101), 20.0)}}
    cal_curves, vic_curves = cli._pool_curves([a, b])
    assert "left_knee" in cal_curves
    # cal ensemble mean = mean of 2x10 + 1x20 over 3 cycles = 13.333...
    assert cal_curves["left_knee"][0] == np.float64(np.mean([10, 10, 20]))
    assert vic_curves["left_knee"][0] == np.float64(np.mean([10, 10, 10, 20]))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_validate_vicon.py -v`
Expected: FAIL (`AttributeError: '_pool_curves'`).

- [ ] **Step 3: Implement (modify `cli.py`)**

Add after `compare(...)`:
```python
def _pool_curves(artifacts_list):
    """Concatenate per-cycle matrices across pairs → subject-level ensemble mean curve."""
    import numpy as np

    cal_stack, vic_stack = {}, {}
    for art in artifacts_list:
        for j, mat in art.get("cal_cycles", {}).items():
            cal_stack.setdefault(j, []).append(np.asarray(mat, float))
        for j, mat in art.get("vic_cycles", {}).items():
            vic_stack.setdefault(j, []).append(np.asarray(mat, float))
    cal_curves = {j: np.nanmean(np.vstack(v), axis=0) for j, v in cal_stack.items()}
    vic_curves = {j: np.nanmean(np.vstack(v), axis=0) for j, v in vic_stack.items()}
    return cal_curves, vic_curves


def validate_vicon(recordings_dir: str, vicon_dir: str, model: str, out_dir: str) -> dict:
    """Run all configured/auto pairs; emit pooled Level-A angle table + per-pair position rows."""
    import pandas as pd

    from modules.comparison.metrics import angle_comparison_report

    cfg = _load_settings()
    ccmp = cfg["comparison"]
    pairs = ccmp.get("pairs")
    if not pairs:                      # default: p1_1..p1_5 ↔ Vicon 1..5 by order
        pairs = {f"p1_{i}": f"{i}.xlsx" for i in range(1, 6)}

    artifacts, position_rows = [], []
    for sess, vfile in pairs.items():
        sess_dir = str(Path(recordings_dir) / sess)
        vpath = str(Path(vicon_dir) / vfile)
        if not (Path(sess_dir).exists() and Path(vpath).exists()):
            continue
        cal = load_caliscope_session(sess_dir, model=model)
        vic = _load_vicon_mapped(vpath, cfg)
        rep, art = run_comparison(cal, vic, cfg, model=model, pair_id=f"{sess}__{vfile}")
        artifacts.append(art)
        for j, m in rep["position"]["joints"].items():
            position_rows.append({"pair": f"{sess}__{vfile}", "joint": j, **m})

    cal_curves, vic_curves = _pool_curves(artifacts)
    level_a = angle_comparison_report(
        cal_curves, vic_curves,
        good=ccmp["good_rmse_threshold_deg"],
        acceptable=ccmp["acceptable_rmse_threshold_deg"],
        icc_type=ccmp["icc_type"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    level_a.to_csv(out / "level_a_angles.csv", index=False)
    pd.DataFrame(position_rows).to_csv(out / "level_c_positions.csv", index=False)
    summary = {"n_pairs": len(artifacts),
               "level_a": level_a.to_dict(orient="records")}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
```

Add the subparser in `main()`:
```python
    v = sub.add_parser("validate-vicon", help="Level A/C validation tables across the dataset")
    v.add_argument("--recordings", required=True)
    v.add_argument("--vicon-dir", required=True)
    v.add_argument("--model", default=default_model)
    v.add_argument("--out", required=True)
```

Add the dispatch branch:
```python
    elif args.command == "validate-vicon":
        res = validate_vicon(args.recordings, args.vicon_dir, args.model, args.out)
        print(f"Validated {res['n_pairs']} pairs. Level-A angle table:")
        for row in res["level_a"]:
            print(f"  {row['joint']}: RMSE={row['rmse_deg']:.2f}° "
                  f"ICC={row['icc']:.3f} ({row['verdict']})")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_validate_vicon.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke-test on real data**

Run:
```bash
.venv/bin/python cli.py validate-vicon --recordings data/caliscope_project/recordings \
  --vicon-dir data/Vicon_10_series --out results/vicon_validation
```
Expected: prints a per-joint Level-A table; writes `level_a_angles.csv`, `level_c_positions.csv`, `validation_summary.json` under `results/vicon_validation/`. (Default pairing p1_1..p1_5 ↔ 1..5.xlsx; refine via `comparison.pairs` if a true synchronous map is supplied.)

- [ ] **Step 6: Commit**

```bash
git add cli.py tests/test_validate_vicon.py
git commit -m "feat(cli): validate-vicon Level A/C tables (pooled ensemble)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: GUI `ComparisonWorker`

**Files:**
- Modify: `gait_analysis/gui/worker.py`
- Modify: `gait_analysis/tests/test_gui_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui_smoke.py`:
```python
def test_comparison_worker_error_path_emits_error():
    from gui.worker import ComparisonWorker
    worker = ComparisonWorker("/nonexistent/session", "/nonexistent/vicon.xlsx",
                              "SIMPLE_HOLISTIC", {})
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors          # bad paths -> error signal, no crash
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_comparison_worker_error_path_emits_error -v`
Expected: FAIL (`ImportError: cannot import name 'ComparisonWorker'`).

- [ ] **Step 3: Implement (append to `gui/worker.py`)**

```python
class ComparisonWorker(QObject):
    progress = pyqtSignal(float, str)        # (fraction 0..1, stage label)
    finished = pyqtSignal(dict, object)      # (report, artifacts)
    error = pyqtSignal(str)

    def __init__(self, session_dir, vicon_path, model, cfg):
        super().__init__()
        self._session = session_dir
        self._vicon = vicon_path
        self._model = model
        self._cfg = cfg

    def run(self):
        try:
            from compare_pipeline import run_comparison
            from modules.data_loader.vicon_reader import (
                load_vicon_xlsx,
                map_vicon_to_caliscope,
            )

            self.progress.emit(0.0, "Loading")
            cal = load_caliscope_session(self._session, model=self._model)
            vic = map_vicon_to_caliscope(
                load_vicon_xlsx(self._vicon,
                                vicon_fps=self._cfg["comparison"].get("vicon_fps", 100.0)),
                self._cfg["landmark_mapping"])
            pair_id = f"{Path(self._session).name}__{Path(self._vicon).stem}"
            report, artifacts = run_comparison(
                cal, vic, self._cfg, model=self._model, pair_id=pair_id,
                progress_cb=lambda f, s: self.progress.emit(f, s))
            self.finished.emit(report, artifacts)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, must never crash
            self.error.emit(str(exc))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_comparison_worker_error_path_emits_error -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gui/worker.py tests/test_gui_smoke.py
git commit -m "feat(gui): ComparisonWorker (compare off the GUI thread)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 13: GUI `ComparePanel` + overlay plot

**Files:**
- Create: `gait_analysis/gui/panels/compare_panel.py`
- Modify: `gait_analysis/tests/test_gui_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui_smoke.py`:
```python
def _fake_comparison_report():
    import numpy as np
    return {
        "pair_id": "p1_1__1", "model": "SIMPLE_HOLISTIC",
        "caliscope_fps": 19.0, "vicon_fps": 100.0,
        "angle": {
            "left_knee": {"rmse_deg": 4.2, "mae_deg": 3.1, "pearson": 0.97,
                          "icc": 0.88, "verdict": "good"},
        },
        "angle_overlay": {
            "left_knee": {"caliscope": list(np.linspace(0, 60, 101)),
                          "vicon": list(np.linspace(0, 58, 101))},
        },
        "position": {"joints": {}, "time_shift_s": 0.1, "scale": 1.0,
                     "low_confidence": False},
        "verdict_summary": "1 joints compared; knee ICC=0.880; worst verdict: good",
    }


def test_compare_panel_populates_table_and_plot_on_finished(qtbot):
    from gui.panels.compare_panel import ComparePanel
    panel = ComparePanel()
    qtbot.addWidget(panel)
    panel._on_finished(_fake_comparison_report(), {})
    assert panel.table.rowCount() == 1                  # one angle joint row
    assert "left_knee" in [panel.table.item(0, 0).text()]
    assert any(ax.lines for ax in panel.overlay.current_figure.axes)  # overlay drawn
    assert "ICC" in panel.verdict.text() or "knee" in panel.verdict.text()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_compare_panel_populates_table_and_plot_on_finished -v`
Expected: FAIL (`ModuleNotFoundError: gui.panels.compare_panel`).

- [ ] **Step 3: Implement**

Create `gui/panels/compare_panel.py`:
```python
"""Tab «Сравнение»: caliscope vs Vicon — run comparison, per-joint table, overlay, verdict."""
from pathlib import Path

import yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.worker import ComparisonWorker
from modules.data_loader.landmarks import MODELS

GAIT_DIR = Path(__file__).resolve().parents[2]


def _load_settings():
    with open(GAIT_DIR / "settings.yaml") as f:
        return yaml.safe_load(f)


class OverlayCanvas(FigureCanvasQTAgg):
    def __init__(self):
        super().__init__(Figure(figsize=(8, 3.2)))

    def render_overlay(self, overlay: dict):
        fig = self.figure
        fig.clear()
        joints = list(overlay) or ["(none)"]
        axes = fig.subplots(1, len(joints), squeeze=False)[0]
        import numpy as np
        x = np.linspace(0, 100, 101)
        for ax, j in zip(axes, overlay):
            cal = overlay[j]["caliscope"]
            vic = overlay[j]["vicon"]
            ax.plot(x, cal, color="tab:blue", label="caliscope")
            ax.plot(x, vic, color="tab:red", label="Vicon")
            ax.set_title(j)
            ax.set_xlabel("% gait cycle")
            ax.set_ylabel("angle (deg)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        self.draw()

    @property
    def current_figure(self):
        return self.figure


class ComparePanel(QWidget):
    comparison_done = pyqtSignal(dict, object)

    _COLS = ["joint", "rmse_deg", "mae_deg", "pearson", "icc", "verdict"]

    def __init__(self):
        super().__init__()
        self._cfg = _load_settings()
        self._session = None
        self._vicon = None
        self._report = None

        self.session_btn = QPushButton("Choose caliscope session…")
        self.session_label = QLabel("(none)")
        self.vicon_btn = QPushButton("Choose Vicon XLSX…")
        self.vicon_label = QLabel("(none)")
        self.model = QComboBox()
        self.model.addItems(MODELS)
        self.model.setCurrentText(self._cfg["processing"]["default_model"])
        self.run_btn = QPushButton("▶ Run comparison")
        self.run_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.table = QTableWidget(0, len(self._COLS))
        self.table.setHorizontalHeaderLabels(self._COLS)
        self.overlay = OverlayCanvas()
        self.verdict = QLabel("")

        form = QFormLayout()
        srow = QHBoxLayout(); srow.addWidget(self.session_btn); srow.addWidget(self.session_label)
        vrow = QHBoxLayout(); vrow.addWidget(self.vicon_btn); vrow.addWidget(self.vicon_label)
        form.addRow(srow)
        form.addRow(vrow)
        form.addRow("Model", self.model)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.table)
        layout.addWidget(self.overlay)
        layout.addWidget(self.verdict)

        self.session_btn.clicked.connect(self._choose_session)
        self.vicon_btn.clicked.connect(self._choose_vicon)
        self.run_btn.clicked.connect(self._run)

    def _maybe_enable(self):
        self.run_btn.setEnabled(bool(self._session and self._vicon))

    def _choose_session(self):
        path = QFileDialog.getExistingDirectory(self, "Choose caliscope session folder")
        if path:
            self._session = path
            self.session_label.setText(Path(path).name)
            self._maybe_enable()

    def _choose_vicon(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Vicon XLSX", "", "XLSX (*.xlsx)")
        if path:
            self._vicon = path
            self.vicon_label.setText(Path(path).name)
            self._maybe_enable()

    def _run(self):
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self._thread = QThread()
        self._worker = ComparisonWorker(self._session, self._vicon,
                                        self.model.currentText(), self._cfg)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda f, s: self.progress.setValue(int(f * 100)))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self.run_btn.setEnabled(True))
        self._thread.start()

    def _on_finished(self, report, artifacts):
        self._report = report
        angle = report.get("angle", {})
        self.table.setRowCount(len(angle))
        for row, (joint, m) in enumerate(angle.items()):
            values = [joint, f"{m['rmse_deg']:.2f}", f"{m['mae_deg']:.2f}",
                      f"{m['pearson']:.3f}", f"{m['icc']:.3f}", m["verdict"]]
            for col, val in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(val))
        self.overlay.render_overlay(report.get("angle_overlay", {}))
        self.verdict.setText(report.get("verdict_summary", ""))
        self.comparison_done.emit(report, artifacts)

    def _on_error(self, msg):
        QMessageBox.warning(self, "Comparison failed", msg)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_compare_panel_populates_table_and_plot_on_finished -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gui/panels/compare_panel.py tests/test_gui_smoke.py
git commit -m "feat(gui): ComparePanel with per-joint table + overlay curves

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 14: Wire the «Сравнение» tab into `MainWindow`

**Files:**
- Modify: `gait_analysis/gui/main_window.py`
- Modify: `gait_analysis/tests/test_gui_smoke.py` (update the two-tab test to three)

- [ ] **Step 1: Update the failing test**

In `tests/test_gui_smoke.py`, change `test_main_window_has_two_tabs_and_routes_results` so the tab count assertion is **3** and rename for clarity:
```python
def test_main_window_has_three_tabs_and_routes_results(qtbot):
    try:
        from gui.main_window import MainWindow
        win = MainWindow()
    except Exception as exc:                 # vispy/GL unavailable (VizPanel)
        pytest.skip(f"vispy widget unavailable: {exc}")
    qtbot.addWidget(win)
    assert win.centralWidget().count() == 3
    results, df = fixture_results_df()
    win.analyze._on_finished(results, df)            # emits analysis_done
    assert win.viz.slider.maximum() == len(df) - 1   # routed into the Viz tab
    assert hasattr(win, "compare")                   # compare tab present
```
(Delete the old `test_main_window_has_two_tabs_and_routes_results`.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py::test_main_window_has_three_tabs_and_routes_results -v`
Expected: FAIL (count is 2; no `win.compare`).

- [ ] **Step 3: Implement (modify `gui/main_window.py`)**

```python
"""Main window: three-tab shell — Анализ → Сравнение / Визуализация."""
from PyQt6.QtWidgets import QMainWindow, QTabWidget

from gui.panels.analyze_panel import AnalyzePanel
from gui.panels.compare_panel import ComparePanel
from gui.panels.viz_panel import VizPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Analysis")
        self.resize(1000, 750)
        self.analyze = AnalyzePanel()
        self.compare = ComparePanel()
        self.viz = VizPanel()
        tabs = QTabWidget()
        tabs.addTab(self.analyze, "Анализ")
        tabs.addTab(self.compare, "Сравнение")
        tabs.addTab(self.viz, "Визуализация")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready")
        self.analyze.analysis_done.connect(self.viz.set_data)
        self.analyze.analysis_done.connect(
            lambda *_: self.statusBar().showMessage("Analysis complete"))
        self.compare.comparison_done.connect(
            lambda *_: self.statusBar().showMessage("Comparison complete"))

    def closeEvent(self, event):
        for panel in (self.analyze, self.compare):
            thread = getattr(panel, "_thread", None)
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        super().closeEvent(event)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_gui_smoke.py -v`
Expected: PASS (all GUI smoke tests, including the new 3-tab test).

- [ ] **Step 5: Commit**

```bash
git add gui/main_window.py tests/test_gui_smoke.py
git commit -m "feat(gui): add Сравнение tab (3-tab app) + close-guard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 15: Full verification + manual-visual checklist

**Files:**
- Modify: `gait_analysis/docs/manual_visual_checklist.md`

- [ ] **Step 1: Run the full suite with coverage**

Run:
```bash
.venv/bin/python -m pytest --cov=. --cov-report=term-missing -q
```
Expected: all tests PASS; coverage > 70%. If `test_vvsc_parity` was skipped (data absent), note it in the run output — it must pass when real data is present (it is, per Task 0).

- [ ] **Step 2: Run ruff**

Run: `.venv/bin/python -m ruff check .`
Expected: clean. Fix any reported issues in the touched files (line length 100, import order I).

- [ ] **Step 3: Generate the real-data artifacts for visual review**

Run:
```bash
.venv/bin/python cli.py validate-vicon --recordings data/caliscope_project/recordings \
  --vicon-dir data/Vicon_10_series --out results/vicon_validation
```
Expected: Level-A table prints (knee target RMSE<5°/ICC>0.85; acceptance RMSE<10°, knee ICC>0.75); CSVs + summary written.

- [ ] **Step 4: Append a comparison section to the manual-visual checklist**

Add to `docs/manual_visual_checklist.md`:
```markdown
## Module 3 — Vicon comparison (Phase 3)

- [ ] `python app.py` → «Сравнение» tab: pick a caliscope session + a Vicon XLSX → Run.
- [ ] Per-joint table shows knee + ankle rows (hip absent — Vicon has no shoulder marker).
- [ ] Overlay plot: caliscope (blue) and Vicon (red) gait-cycle curves track each other.
- [ ] Verdict summary line reflects knee ICC + worst verdict.
- [ ] `cli.py validate-vicon …` Level-A knee RMSE/ICC are physically plausible (RMSE single-digit°).
- [ ] `test_vvsc_parity` passes (ported position metrics reproduce VvsC global RMSE).
```

- [ ] **Step 5: Commit**

```bash
git add docs/manual_visual_checklist.md
git commit -m "docs(phase3): manual-visual checklist for Vicon comparison

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes (addressed during authoring)

- **Spec coverage:** reader fix (Task 1) ↔ spec §5.1; alignment (3–4) ↔ §5.2; metrics+ICC (5–6) ↔ §5.3; report (7) ↔ §5.4; run_comparison (8–9) ↔ §5.5/§6; CLI compare + validate-vicon (10–11) ↔ §5.6/§8; GUI (12–14) ↔ §5.7; config (2) ↔ §9; pingouin (0) ↔ §11.4; parity (9) ↔ criterion #1/§7; tests throughout ↔ §7.
- **Hip exclusion:** `calc_joint_angles_timeseries` defines hip via shoulder–hip–knee; Vicon lacks a shoulder marker → all-NaN hip curve, dropped by `angle_comparison_report`'s intersection + all-NaN guard. Knee + ankle are the compared joints (knee ICC is the gating criterion #5). Documented in Tasks 6, 8, 13, 15.
- **Type/name consistency:** `estimate_rigid_transform → (R, T, s)` and `apply_transform(points, R, T, s)` used consistently in Tasks 3/9; `run_comparison(...) -> (report, artifacts)` consistent in Tasks 8/9/12; `_pool_curves(artifacts_list)` consumes the `{cal_cycles, vic_cycles}` artifact shape emitted by `run_comparison` (Tasks 8/11); report keys (`angle`, `angle_overlay`, `position.joints`, `verdict_summary`) consistent across Tasks 7/8/13.
- **Pairing default:** `validate-vicon` defaults to `p1_1..p1_5 ↔ 1..5.xlsx` and honors `comparison.pairs` when supplied — matches spec §6.3.
```

