# Design Spec — Gait Kinematics Analysis from Multi-Camera RGB Video

**Version:** 1.0
**Date:** 2026-05-29
**Source RFC:** `ТЗ_Анализ_кинематики_походки.docx` (v1.0, diploma thesis)
**Status:** Approved for planning

---

## 1. Overview & Goals

A software complex that estimates gait kinematics from synchronized multi-camera (3×) RGB
video. It consumes 3D landmark coordinates already produced by **caliscope**, computes
clinical gait parameters (joint angles, spatiotemporal metrics), and compares accuracy
against a Vicon reference system. A PyQt6 GUI makes it usable for clinical/research workflows.

The system delivers:

1. **Ingestion** of caliscope 3D output (POSE / SIMPLE_HOLISTIC / HOLISTIC) into a unified format.
2. **Automatic kinematics**: gait-event detection, joint angles, gait-cycle normalization,
   spatiotemporal parameters.
3. **Vicon comparison** (apparatus): coordinate-system alignment + accuracy metrics
   (RMSE / MAE / Pearson r / ICC). Built now against the documented Vicon schema, validated
   on real data when it arrives.
4. **Visualization & GUI**: animated 3D skeleton, gait-cycle angle plots, comparison overlays,
   export to CSV/XLSX/PNG.

### Design principles (from RFC §2.1)

- Each module is an importable Python package, usable without the GUI.
- Business logic is fully separated from the GUI; the GUI only calls module functions and renders results.
- All tunables live in `settings.yaml` / `config.toml` — nothing hardcoded.
- Every module is runnable via CLI (`cli.py`) for automated tests and apparatus runs.
- Intermediate results persist as JSON/CSV for reproducibility and debugging.
- Reproducibility: a fixed seed (`numpy.random.seed(42)`) for any stochastic step.

---

## 2. Scope & Phasing

**In scope:** all four modules + PyQt6 GUI + CLI + tests, built against the documented data
contracts.

**Decided deviations / clarifications** (resolved against the actual provided data — see §4):
canonical lowercase landmark naming, fps derived from data, vertical axis = `z`, timestamps
sourced from the model-level `frame_time_history.csv`.

**Validation reality:** No Vicon XLSX and no `VvsC.py` are present in the provided dataset.
Therefore:

- Module 3 is **built now** against the documented Vicon XLSX schema + `landmark_mapping`, and
  **tested on synthetic data**.
- **Test-retest reproducibility** (Level B, CV across `p1_1…p1_5`) is the **primary validation
  result**.
- Real-Vicon validation (apparatus Levels A & C, the `VvsC.py` baseline) and acceptance
  criteria #1/#4/#5 are **gated** on the user supplying the XLSX files + `VvsC.py`.

### Build phases

| Phase | Contents | Verifiable today? |
|-------|----------|-------------------|
| **1 — Core spine** | Module 1 (data_loader) + Module 2 (kinematics) + `cli.py` | ✅ end-to-end on `p1_*` data |
| **2 — Viz & GUI** | Module 4 (visualization) + PyQt6 GUI shell | ✅ on `p1_*` data |
| **3 — Vicon (gated)** | Module 3 alignment/metrics real-data validation + `VvsC.py` wrapper + Levels A/C | ⏸ needs Vicon XLSX + `VvsC.py` |

Module 3's *code* (alignment, metrics, report) is written in Phase 1/2 with synthetic tests;
Phase 3 is only the live-data validation gate.

**Out of scope:** running caliscope itself (its output is the input); marker-based Vicon capture;
any change to camera calibration.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  GUI Application (PyQt6)                  │   app.py
│         Load · Analysis · Compare · Visualize tabs        │
└──────┬──────────────┬──────────────┬─────────────────────┘
   ┌───▼────┐   ┌──────▼───┐   ┌──────▼──┐   ┌──────▼──────┐
   │ Mod 1  │   │  Mod 2   │   │  Mod 3  │   │   Mod 4     │
   │ Data   │   │ Kinemat. │   │ Vicon   │   │ Visualize   │
   │ Loader │   │ Analysis │   │ Compare │   │ 3D / Plots  │
   └───┬────┘   └──────┬───┘   └──────┬──┘   └──────┬──────┘
       └──────── unified pose DataFrame ────────────┘
                          │
              ┌───────────▼───────────┐
              │   State / Config       │  session.json + settings.yaml + config.toml
              └────────────────────────┘
```

`cli.py` is a parallel entry point exposing each module/pipeline step for headless runs.

### Project structure (per RFC §3)

```
gait_analysis/
├── app.py                 # GUI entry point (PyQt6)
├── cli.py                 # CLI entry point (tests / apparatus)
├── config.toml            # camera params (copied/linked from caliscope project)
├── settings.yaml          # paths, thresholds, algorithm params, landmark_mapping
├── requirements.txt
├── modules/
│   ├── data_loader/       # Module 1
│   │   ├── caliscope_reader.py
│   │   ├── vicon_reader.py
│   │   ├── config_reader.py
│   │   └── synchronizer.py
│   ├── kinematics/        # Module 2
│   │   ├── gait_events.py
│   │   ├── joint_angles.py
│   │   ├── spatiotemporal.py
│   │   ├── normalizer.py
│   │   └── filters.py
│   ├── comparison/        # Module 3
│   │   ├── alignment.py
│   │   ├── metrics.py
│   │   └── report.py
│   └── visualization/     # Module 4
│       ├── skeleton_3d.py
│       ├── angle_plots.py
│       └── export.py
├── gui/
│   ├── main_window.py
│   ├── panels/ (load_panel, analysis_panel, compare_panel, viz_panel)
│   └── widgets/ (skeleton_widget, plot_widget)
├── tests/
│   ├── test_gait_events.py / test_joint_angles.py / test_filters.py
│   ├── test_normalizer.py / test_synchronizer.py / test_metrics.py
│   └── fixtures/          # short synthetic/real fragments
└── data/                  # symlinks to real data
    ├── caliscope_project/ -> ../../caliscope_project_271025
    └── Vicon_10_series/   # (added when Vicon data arrives)
```

---

## 4. Data formats & conventions (resolved from real files)

These resolve the abstractions the RFC left open. They are **binding contracts** for all modules.

### 4.1 Unified pose DataFrame (the single inter-module format)

Wide format, one row per frame. This is exactly the shape caliscope already emits in
`xyz_<MODEL>_labelled.csv`, plus a real `timestamp`:

```
frame, timestamp, <landmark>_x, <landmark>_y, <landmark>_z, ...
```

- **Canonical landmark naming: lowercase, side-prefixed** — `left_hip`, `right_knee`,
  `left_heel`, `left_foot_index`, `left_ankle`, `left_shoulder`, etc. This matches caliscope's
  native output. The RFC's uppercase formula names (`LEFT_HIP`) are aliases of these.
- `timestamp` in **seconds** from start, monotonically increasing.
- Coordinates in **meters**.
- Missing landmarks are `NaN`.
- Column order is not significant; lookups are by name.

### 4.2 caliscope input (Module 1)

- Read `xyz_<MODEL>_labelled.csv` directly — it is already wide + landmark-named. Index key is
  `sync_index` (may start > 0 and contain gaps where triangulation failed; do **not** assume
  contiguity).
- **Timestamps**: join the **model-directory** `frame_time_history.csv`
  (`sync_index, port, frame_index, frame_time`). Its `sync_index` aligns with the labelled
  output. The **session-directory** `frame_time_history.csv` uses offset original capture
  indices and must **not** be used for this join. Collapse the per-port rows (3 ports) to one
  `frame_time` per `sync_index` (mean), then `timestamp = frame_time − frame_time[first]`.
- **FPS is derived**, never hardcoded: `fps = 1 / median(diff(timestamp))`. (Observed ≈ 19 fps
  for `walking_test`; varies per recording.)
- Models: `POSE` | `SIMPLE_HOLISTIC` | `HOLISTIC`. Default `SIMPLE_HOLISTIC`. Lower-body
  landmarks (hip/knee/ankle/heel/foot_index) are present and ≈97–100% populated across
  `p1_1…p1_5`; `foot_index` occasionally drops to ~85% (occlusion → handled by `fill_gaps`).

### 4.3 Vicon input (Module 1, documented contract)

- `load_vicon_xlsx(filepath)`: auto-detect the header row (Vicon XLSX header is not always row 1);
  parse `MarkerName:X|Y|Z` columns into the unified DataFrame.
- **Units**: Vicon is often in **mm** — auto-detect (coordinate magnitude / explicit unit cell)
  and convert to meters.
- `map_vicon_to_caliscope(df, model)`: rename Vicon markers to canonical landmark names via
  `settings.yaml → landmark_mapping` (e.g. `LASI→left_hip`, `LKNE→left_knee`, `LHEE→left_heel`,
  `LTOE→left_foot_index`).

### 4.4 Synchronization (Module 1)

`synchronize(df_a, df_b, target_fps=100.0)`: interpolate both onto a common grid
(`scipy.interpolate.interp1d`, cubic), clip to the overlapping time span. Precondition:
`timestamp` in seconds, monotonic.

### 4.5 Axes & units

- **Vertical axis = `z`** (default; confirmed from data: shoulder ≈ +0.6 → foot ≈ −0.79).
  Configurable via `settings.yaml → gait_events.vertical_axis`.
- Direction of travel is estimated from horizontal displacement (PCA on the horizontal plane)
  for step-length/step-width projections, rather than assuming an axis.
- **Mirror check**: with certain camera layouts caliscope's X may mirror Vicon's. The alignment
  step (Umeyama, allows reflection-free rotation) plus a visual check covers this; a sign flag
  is exposed in config if a hard flip is needed.

### 4.6 Intermediate / result artifacts

- `unified_pose.csv` — the §4.1 format (debug/repro).
- `gait_results.json` — per-session results (RFC §9.2): `gait_events`, `spatiotemporal`,
  `joint_angles_mean` / `joint_angles_std` (101 points, 0–100% cycle), plus `session_id`,
  `model`, `fps`, `processed_at`.

---

## 5. Module 1 — Data Loader & Synchronization

| File | Responsibility | Key functions |
|------|----------------|---------------|
| `caliscope_reader.py` | Read labelled CSV + attach real timestamps + derive fps | `load_caliscope_session(session_dir, model='SIMPLE_HOLISTIC') -> DataFrame`; `list_landmarks(model) -> list[str]` |
| `vicon_reader.py` | Read Vicon XLSX → unified format, map markers | `load_vicon_xlsx(path) -> DataFrame`; `map_vicon_to_caliscope(df, model) -> DataFrame` |
| `config_reader.py` | Parse camera intrinsics/extrinsics from `config.toml` | `load_camera_config(path) -> dict` |
| `synchronizer.py` | Resample two DataFrames onto a common time grid | `synchronize(df_a, df_b, target_fps=100.0) -> (DataFrame, DataFrame)` |

`config.toml` note: cameras are keyed `cam_1`/`cam_2`/`cam_3` with `matrix` (3×3 intrinsics),
`distortions` (5), `translation` (3), `rotation` (3, Rodrigues). `config_reader` normalizes these
into the RFC dict shape (intrinsics fx/fy/cx/cy + distortion; extrinsics R 3×3 / T 3).

---

## 6. Module 2 — Kinematics

Pipeline order is fixed: **`fill_gaps` → `butterworth_filter` → `detect_gait_events` →
`calc_joint_angles_timeseries` → `normalize_gait_cycle` → `calc_spatiotemporal`**.

### 6.1 `filters.py`
- `fill_gaps(df, max_gap_frames=10)`: cubic-interpolate NaN runs ≤ `max_gap_frames`; longer gaps
  stay NaN. **Called before filtering.**
- `butterworth_filter(signal, cutoff_hz=6.0, fs=<derived>, order=4, zero_phase=True)`: zero-phase
  `filtfilt`, `padtype='odd'`. `fs` is the derived fps, not a constant.

### 6.2 `gait_events.py`
- `detect_gait_events(df, side='both', method='velocity') -> dict` returning
  `{left_HS, left_TO, right_HS, right_TO}` as frame-index lists.
- `velocity` method: filter heel vertical (`<side>_heel_z`, Butterworth 4th-order fc=6 Hz),
  local minima → HS; toe vertical (`<side>_foot_index_z`) local maxima between consecutive HS
  → TO. Reject events closer than `min_stride_duration_sec` (0.3 s) apart.
- Also support `height` and `combined`.

### 6.3 `joint_angles.py`
- `calc_angle_3d(p1, vertex, p2)`: `arccos(dot(v1,v2)/(|v1||v2|))`.
- `calc_joint_angles_timeseries(df)`: appends hip/knee/ankle (sagittal) for both sides and pelvis
  (tilt/obliquity/rotation via plane fits / PCA). Sign convention: flexion > 0, extension < 0
  (Plug-In Gait / Vicon compatible).

### 6.4 `normalizer.py`
- `normalize_gait_cycle(signal, events, side, n_points=101)`: split into HS→HS cycles, interpolate
  each to 101 points → `[n_cycles × 101]`.
- `get_mean_std_cycle(cycles_matrix)`: mean[101], std[101] for ±1 STD corridor plots.

### 6.5 `spatiotemporal.py`
- `calc_spatiotemporal(df, events, fps) -> dict`: cadence, speed, stride/step length, step width,
  stance/swing/double-support %, per RFC §5.4. Direction of travel via horizontal-plane PCA.

---

## 7. Module 3 — Vicon Comparison (built now, validated when data arrives)

| File | Responsibility | Key functions |
|------|----------------|---------------|
| `alignment.py` | Register caliscope → Vicon frame | `estimate_rigid_transform(src, dst) -> (R, T, s)` (Umeyama 1991); `align_coordinate_systems(cal_df, vic_df, reference_landmarks=None) -> DataFrame` |
| `metrics.py` | Accuracy metrics | `calc_rmse`, `calc_mae`, `calc_pearson`, `calc_icc(type='3,1')` (pingouin); `full_comparison_report(cal_df, vic_df, joint_list=None) -> DataFrame` |
| `report.py` | Summary report generation | per-joint metric table + overlay-plot data + verdict thresholds |

Alignment strategy: compute the rigid+scale transform on a static-stance reference window
(first N frames) using mapped landmarks, then apply to the dynamic recording.

**`VvsC.py` wrapper** (`run_baseline_comparison`) runs the existing baseline via `subprocess`
and parses its output — **deferred to Phase 3** (script not yet provided). The function and its
interface are specified now; the body is implemented when `VvsC.py` lands.

**Testing without Vicon:** synthetic — apply a known `(R, T, s)` to a point cloud and assert it is
recovered; identical arrays → RMSE 0, MAE 0, Pearson 1, ICC ≈ 1.

---

## 8. Module 4 — Visualization & PyQt6 GUI

### 8.1 Visualization library
- `skeleton_3d.py`: 3D skeleton animation (vispy / `QOpenGLWidget`), frame slider, Play/Pause/Step,
  speed 0.25×/0.5×/1×.
- `angle_plots.py`: joint-angle curves over % gait cycle with mean ± STD corridor; optional
  normative corridor loaded from CSV.
- `export.py`: CSV / XLSX / PNG export.

### 8.2 GUI (PyQt6)
`MainWindow` = `QTabWidget` with four panels; heavy work runs in a `QThread` worker that emits
progress signals to the status bar (GUI must never block).

| Tab | Panel | Content |
|-----|-------|---------|
| Загрузка данных | `LoadPanel` | project folder picker, model dropdown, optional Vicon XLSX, preview table (first 10 rows), indicators: frame count / fps / duration / landmarks with >5% NaN |
| Анализ | `AnalysisPanel` | editable algorithm params (filter cutoff, event threshold), "Run analysis" → pipeline in QThread, progress bar, results table, export |
| Сравнение с Vicon | `ComparePanel` | run alignment (with quality viz), RMSE/MAE/Pearson/ICC table per joint, overlaid caliscope (blue) vs Vicon (red) cycle curves, verdict (RMSE <5° good, 5–10° acceptable, >10° poor) |
| Визуализация | `VizPanel` | animated 3D skeleton, knee/hip/ankle angle kinograms in % cycle, optional norm corridor, top-view foot trajectories |

---

## 9. Configuration (`settings.yaml`)

Mirrors RFC §8, with values defaulted from the real data. Notable keys (canonical lowercase names):

```yaml
paths:        { caliscope_root, vicon_root, output_dir, camera_config }
processing:   { default_model: SIMPLE_HOLISTIC, target_fps: 100,
                filter_cutoff_hz: 6.0, filter_order: 4, max_gap_frames: 10,
                min_stride_duration_sec: 0.3 }
gait_events:  { method: velocity, heel_landmark: heel, toe_landmark: foot_index,
                vertical_axis: z }
landmark_mapping:   # Vicon marker -> canonical landmark
  LASI: left_hip   ; RASI: right_hip
  LKNE: left_knee  ; RKNE: right_knee
  LANK: left_ankle ; RANK: right_ankle
  LHEE: left_heel  ; RHEE: right_heel
  LTOE: left_foot_index ; RTOE: right_foot_index
comparison:   { icc_type: '3,1', good_rmse_threshold_deg: 5.0,
                acceptable_rmse_threshold_deg: 10.0 }
```

Note `target_fps` is the synchronization grid for Vicon comparison; caliscope's own fps is always
derived from its `frame_time_history`.

---

## 10. Validation & apparatus plan

- **Level B — Reproducibility (PRIMARY).** Run the full pipeline on `p1_1…p1_5`, compute the
  coefficient of variation (CV) per parameter across the 5 sessions; boxplots per parameter.
  Target: CV < 15% (acceptance #6). **Caveat reported with results:** sessions are short
  (~64–102 frames, ~2–4 strides each), so per-session cycle counts are low; CV is presented with
  cycle-count annotations.
- **Level A — Vicon synchronous accuracy (GATED).** RMSE/MAE/ICC caliscope vs Vicon per joint;
  targets knee RMSE < 5°, ICC > 0.85. Runs when Vicon XLSX is provided.
- **Level C — Algorithm comparison table (GATED).** baseline (`VvsC.py`) vs HOLISTIC vs the
  developed algorithm. Fills the thesis "accuracy comparison" table. Runs when `VvsC.py` + Vicon
  provided.

---

## 11. Testing

pytest with synthetic fixtures (RFC §11.1), coverage > 70%, `ruff` clean.

| Test | Asserts |
|------|---------|
| `test_gait_events` | HS/TO detected on a synthetic sinusoidal foot signal with known events |
| `test_joint_angles` | `calc_angle_3d` returns 45° / 90° / 180° for constructed triples |
| `test_filters` | Butterworth passes < cutoff, attenuates > cutoff; `fill_gaps` respects `max_gap_frames` |
| `test_normalizer` | output length 101; correct cross-cycle averaging |
| `test_synchronizer` | equal length + matching timestamps after sync |
| `test_metrics` | RMSE 0 for identical arrays; Pearson 1 for linear; Umeyama recovers known transform |

First build step: create venv + install `requirements.txt` (pandas/scipy/etc. not currently
installed). Python 3.14 available; `tomllib` is stdlib.

---

## 12. Acceptance criteria (annotated)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `VvsC.py` runs, reproduces baseline | ⏸ **gated** (script not provided) |
| 2 | `fill_gaps` + `butterworth_filter` work on NaN data | ✅ buildable now (unit + visual) |
| 3 | `gait_events` finds HS/TO on all 5 `p1` sessions | ✅ buildable now (visual verification) |
| 4 | Joint angles vs Vicon RMSE < 10° (accept) / < 5° (good) | ⏸ **gated** (Level A) |
| 5 | Knee ICC > 0.75 | ⏸ **gated** (Level A) |
| 6 | CV across `p1_1…p1_5` < 15% | ✅ **primary** (Level B) |
| 7 | GUI loads project, shows skeleton + plots | ✅ Phase 2 |
| 8 | CSV + XLSX export works | ✅ Phase 2 |
| 9 | pytest passes, coverage > 70% | ✅ all phases |
| 10 | `ruff check .` clean | ✅ all phases |

---

## 13. Risks & open items

1. **Short recordings** → few gait cycles per session; weakens per-session statistics and CV.
   Mitigation: report cycle counts alongside every aggregate; aggregate cycles across sessions
   where clinically valid.
2. **Vicon data + `VvsC.py` not yet provided** → Levels A/C and criteria #1/#4/#5 gated. Module 3
   code + synthetic tests proceed regardless.
3. **Coordinate mirroring** between caliscope and Vicon → handled by Umeyama alignment + visual
   check + optional sign flag; only verifiable once Vicon data exists.
4. **Pelvis angles** (tilt/obliquity/rotation) need a robust pelvis plane from available hip
   (and where present, shoulder) landmarks; definition fixed in `joint_angles.py` and documented.
5. **3D viewer performance** in PyQt6 — vispy/OpenGL chosen for this reason; matplotlib fallback
   if a target machine lacks GL.

---

## 14. Summary

Build the verifiable spine first (Modules 1+2 + CLI), then visualization + GUI, then gate the
real-Vicon validation. Reproducibility across `p1_1…p1_5` is the headline result available with
today's data; the Vicon-comparison machinery is written and synthetic-tested now so that dropping
in the XLSX + `VvsC.py` later requires no rework — only running the gated apparatus.
