# Design Spec — Gait Analysis Phase 2: Visualization + Dead-Simple PyQt6 GUI

**Version:** 1.0
**Date:** 2026-05-30
**Builds on:** `2026-05-29-gait-kinematics-analysis-design.md` (Phase 1, merged to main)
**Status:** Approved for planning

---

## 1. Overview & Goals

Phase 1 delivered the verifiable spine — `data_loader` + `kinematics` + `cli.py` — done,
merged, 41 tests, `ruff` clean (headline Level-B result: cadence/stride/speed reproducible
< 11% CV after stride-rejection). Phase 2 adds **Module 4 (visualization)** and a **deliberately
minimal PyQt6 desktop GUI** on top of that spine.

The guiding constraint for this phase is **dead-simple GUI**: the application should do one
workflow well — *load a caliscope session, run the pipeline, look at the results* — with no
ceremony. The 3D skeleton is the one place where quality is prioritized over minimalism (see
§2 decisions); everything around it stays lean.

This phase satisfies acceptance criteria **#7** (GUI loads project, shows skeleton + plots) and
**#8** (CSV + XLSX export), and keeps **#9/#10** (tests/coverage, `ruff`) green.

---

## 2. Scope & Decisions

### In scope
- **Prerequisite refactor:** extract the analysis pipeline into a shared `run_pipeline` function
  used by both `cli.py` and the GUI worker.
- **Module 4 — visualization:** `skeleton_3d.py` (GL-free core), `angle_plots.py` (matplotlib),
  `export.py` (CSV/XLSX/PNG).
- **GUI:** a two-tab PyQt6 application (`Анализ`, `Визуализация`) with a non-blocking worker
  thread.

### Out of scope (Phase 2)
- **Vicon "Сравнение" tab** — gated to Phase 3 (no Vicon data exists; Module 3's *code* is
  already synthetic-tested). Building its UI now would be dead UI wired to nothing.
- **matplotlib-3D fallback renderer** — Phase 1 spec §13 risk #5 documents this as the GL
  mitigation. Decision below.
- Running caliscope itself; marker-based Vicon capture; camera-calibration changes.

### Decisions resolved during brainstorming (binding)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **GUI structure = two tabs** (`Анализ` / `Визуализация`), not the spec's four. | Vicon tab dropped (gated); a separate Load tab is ceremony for a one-folder-one-button flow, so Load merges into `Анализ`. |
| D2 | **3D skeleton rendered with vispy/OpenGL**, as pre-planned. | User explicitly chose visual quality here over minimalism. The one place "simple" yields to "good-looking". |
| D3 | **No matplotlib-3D fallback widget.** Skeleton *core* stays GL-free so a fallback remains trivially addable later. | vispy is verified working + constructs headlessly (see §3), so the GL risk is low. A second renderer is the "complication" to avoid. **Tradeoff accepted: the real GL render path has no automated pixel coverage — only the GL-free core (unit) + construction (smoke) + manual-visual tier.** |
| D4 | **matplotlib for angle plots; vispy only for the skeleton.** | matplotlib is already a dependency, pure-Python, and headless-testable. No pyqtgraph/plotly needed. |
| D5 | **Keep one compact data-quality line** in the `Анализ` tab (frames / fps / duration + landmarks with > 5% NaN). | Cheap (one label, no extra tab) and clinically meaningful — warns when foot tracking is unreliable before the numbers are trusted. |
| D6 | **No new dependencies beyond PyQt6 / vispy / PyOpenGL / pytest-qt.** pingouin/plotly/pyqtgraph excluded from Phase 2. | Keeps the phase lean; pingouin belongs to Module 3 (Phase 3). |

---

## 3. Dependency verification (done during brainstorming)

Installed into the project venv (`gait_analysis/.venv`, Python 3.14.4, glibc 2.42 ≥ 2.34) with
`pip install --only-binary=:all:`. Verified empirically:

- `PyQt6` **6.11.0** (cp310-abi3 wheel) imports. ✓
- `vispy` **0.16.2** (cp314 wheel) imports; `vispy.__version__` resolves — **no stale-PyPI-JSON
  anomaly** in practice. ✓
- A vispy `SceneCanvas` + `Markers` visual **constructs under `QT_QPA_PLATFORM=offscreen`**
  without crashing (warns `QOpenGLWidget is not supported on this platform`, but construction
  succeeds; native backend `CanvasBackendDesktop`). ✓
- `PyOpenGL` 3.1.10, `pytest-qt` 4.5.0 installed. ✓

**Testing consequence:** the skeleton widget can be *constructed* in a headless (`offscreen`)
smoke test — feed data, assert no crash and correct state — but **pixel-level rendering cannot
be asserted headlessly**. Real render verification stays in the manual-visual tier (§8).

To add to `requirements.txt`:
```
PyQt6>=6.11
vispy>=0.16.2
PyOpenGL>=3.1
pytest-qt>=4.5
```

---

## 4. Architecture

```
┌────────────────────────────────────────────────────────┐
│                 GUI Application (PyQt6)                  │   app.py
│        QTabWidget:  [ Анализ ]   [ Визуализация ]        │   gui/main_window.py
└───────┬───────────────────────────────┬─────────────────┘
        │ Run (folder, model, params)    │ (results, df)
   ┌────▼───────────────┐                │
   │ PipelineWorker      │  QThread       │
   │ (gui/worker.py)     │  off-UI-thread │
   └────┬───────────────┘                │
        │ load_caliscope_session          │
        │ run_pipeline(...)  ◀────────────┼──── shared with cli.py
   ┌────▼────────────────────────────────▼─────────────────┐
   │            gait_analysis/pipeline.py                    │
   │  run_pipeline(df, cfg, *, model, session_id,            │
   │               progress_cb) -> (results, df_processed)   │
   └────┬───────────────────────────────────────────────────┘
   ┌────▼────┐  ┌──────────┐  ┌───────────────────────────┐
   │ Module 1│  │ Module 2 │  │ Module 4 (visualization)  │
   │ loader  │  │ kinematics│ │ skeleton_3d / angle_plots │
   └─────────┘  └──────────┘  │ / export                  │
                              └───────────────────────────┘
```

`cli.py` and the GUI worker are **two entry points onto the same `run_pipeline`** — the canonical
`gait_results.json` schema is assembled in exactly one place.

### New / changed files (~22 incl. tests)

```
gait_analysis/
├── app.py                              # NEW — GUI entry point
├── pipeline.py                         # NEW — shared run_pipeline (extracted from cli.analyze)
├── cli.py                              # CHANGED — analyze() now calls run_pipeline
├── requirements.txt                    # CHANGED — add PyQt6/vispy/PyOpenGL/pytest-qt
├── modules/visualization/
│   ├── __init__.py                     # NEW
│   ├── skeleton_3d.py                  # NEW — GL-free skeleton core
│   ├── angle_plots.py                  # NEW — matplotlib angle-cycle figures
│   └── export.py                       # NEW — CSV / XLSX / PNG export
├── gui/
│   ├── __init__.py                     # NEW
│   ├── main_window.py                  # NEW — QTabWidget + status bar + result routing
│   ├── worker.py                       # NEW — PipelineWorker(QObject) on QThread
│   ├── panels/
│   │   ├── __init__.py                 # NEW
│   │   ├── analyze_panel.py            # NEW — load + params + run + table + quality line + export
│   │   └── viz_panel.py                # NEW — skeleton + plots + playback controls
│   └── widgets/
│       ├── __init__.py                 # NEW
│       ├── skeleton_widget.py          # NEW — vispy SceneCanvas widget
│       └── plot_widget.py              # NEW — FigureCanvasQTAgg wrapper
└── tests/
    ├── conftest.py                     # NEW/CHANGED — lock QT_QPA_PLATFORM=offscreen
    ├── test_pipeline.py                # NEW — run_pipeline unit test
    ├── test_skeleton_3d.py             # NEW — GL-free core
    ├── test_angle_plots.py             # NEW — Figure structure
    ├── test_export.py                  # NEW — write + read back
    └── test_gui_smoke.py               # NEW — pytest-qt offscreen smoke
```

---

## 5. Prerequisite refactor — `pipeline.py`

The analysis pipeline is currently inlined in `cli.analyze` (`cli.py:41-93`). Extract it so the
GUI worker and the CLI share one implementation.

```python
# gait_analysis/pipeline.py
def run_pipeline(df, cfg, *, model, session_id, progress_cb=None) -> tuple[dict, "DataFrame"]:
    """Run the full kinematics pipeline on a loaded pose DataFrame.

    df          : unified pose DataFrame from load_caliscope_session (carries fps in df.attrs).
    cfg         : settings dict (processing / gait_events / spatiotemporal sections).
    model       : model name, stamped into results.
    session_id  : session identifier (e.g. folder name), stamped into results.
    progress_cb : optional callable(fraction: float, stage: str); no-op if None.

    Returns (results_dict, df_processed). results_dict is the canonical gait_results.json schema.
    """
```

**Pipeline order (unchanged from Phase 1):** `fill_gaps → butterworth_filter → detect_gait_events
→ calc_joint_angles_timeseries → normalize_gait_cycle → calc_spatiotemporal → assemble results`.

`progress_cb` is invoked between stages with a monotonically increasing fraction and a
human-readable stage label (`"Filling gaps"`, `"Filtering"`, `"Detecting gait events"`,
`"Joint angles"`, `"Normalizing cycles"`, `"Spatiotemporal"`).

**`cli.analyze` becomes:**
```python
def analyze(session_dir, model, out_path) -> dict:
    cfg = _load_settings()
    df = load_caliscope_session(session_dir, model=model)
    results, _ = run_pipeline(df, cfg, model=model, session_id=Path(session_dir).name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    return results
```

**Regression guard:** the CLI's `gait_results.json` output must remain byte-identical. The
existing `test_cli_integration.py` (and the rest of the 41 tests) is the guard; a new
`test_pipeline.py` exercises `run_pipeline` directly (on a fixture df, assert schema keys + that
`progress_cb` is called with increasing fractions ending at 1.0).

---

## 6. Module 4 — Visualization

All three files are **pure logic / headless-testable** (strict TDD). No Qt, no GL in this layer.

### 6.1 `skeleton_3d.py` — GL-free skeleton core
- `SKELETON_EDGES: list[tuple[str, str]]` — connectivity using canonical lowercase landmark
  names (e.g. `("left_hip","left_knee")`, `("left_knee","left_ankle")`,
  `("left_ankle","left_heel")`, `("left_ankle","left_foot_index")`, `("left_hip","right_hip")`,
  and shoulder/hip links where present).
- `frame_points(df, frame_idx) -> dict[str, tuple[float,float,float]]` — landmark → (x,y,z) for
  one frame; missing/NaN landmarks omitted.
- `segment_lines(points, edges=SKELETON_EDGES) -> list[tuple[tuple,tuple]]` — drawable segments,
  skipping any edge with a missing endpoint.
- **Tests:** known frame → expected points; edges with a NaN endpoint are dropped; edge list
  references only valid landmark names.

### 6.2 `angle_plots.py` — matplotlib angle-cycle figures
- `plot_joint_angles(angles_mean, angles_std, joints=("hip","knee","ankle"),
  norm_corridor=None) -> matplotlib.figure.Figure` — mean line + ±1 STD corridor over 0–100%
  gait cycle, one subplot per joint, left/right overlaid; optional normative corridor.
- Pure: builds and returns a `Figure`; the widget owns the canvas.
- **Tests:** correct number of axes; line ydata matches input mean; band present when std given.

### 6.3 `export.py` — CSV / XLSX / PNG
- `export_results_csv(results, path)` — spatiotemporal table + per-joint angle curves to CSV.
- `export_results_xlsx(results, path)` — same, multi-sheet, via openpyxl.
- `export_figure_png(fig, path)` — save a matplotlib `Figure` to PNG.
- **Tests:** write to a tmp path, read back, assert key fields / sheet names / file exists & non-empty.

---

## 7. GUI (PyQt6) — two tabs

`MainWindow` is a `QTabWidget` with two panels and a status bar. **All heavy work runs in a worker
thread; the GUI never blocks.**

### 7.1 `worker.py` — `PipelineWorker(QObject)`
Lives on a `QThread`. One method `run()` does `load_caliscope_session(folder, model)` **then**
`run_pipeline(df, cfg, model=model, session_id=folder.name, progress_cb=self._emit_progress)`.

Signals:
- `progress(float, str)` — fraction + stage label → progress bar + status bar.
- `finished(dict, object)` — `(results, df_processed)`.
- `error(str)` — exception message.

The `run()` body is wrapped in `try/except`; any exception emits `error` (never crashes the UI).

### 7.2 `panels/analyze_panel.py` — tab «Анализ»
Layout (top → bottom):
- Folder picker (`QFileDialog`, directory mode) + model `QComboBox`
  (`POSE | SIMPLE_HOLISTIC | HOLISTIC`, default from `settings.yaml`).
- Param fields: filter cutoff (`QDoubleSpinBox`), min stride sec (`QDoubleSpinBox`) — defaulted
  from `settings.yaml`, overridable.
- **Run** `QPushButton` → starts the worker; disabled while running.
- `QProgressBar` driven by `progress`.
- **Data-quality line** (`QLabel`): `frames N · fps F · duration Ds · ⚠ landmarks >5% NaN: …`
  populated after load/finish.
- Results `QTableWidget`: spatiotemporal parameters (cadence, speed, stride/step length, step
  width, stance/swing %).
- **Export CSV** / **Export XLSX** buttons (enabled after a successful run).

On `finished`: fill table + quality line, re-enable Run, and hand `(results, df)` up to
`MainWindow`, which forwards to the Viz panel.

On `error`: `QMessageBox.warning` with the message; restore idle state.

### 7.3 `panels/viz_panel.py` — tab «Визуализация»
- `SkeletonWidget` (vispy) — top.
- Playback controls: frame `QSlider`, Play/Pause `QPushButton`, speed `QComboBox`
  (0.25× / 0.5× / 1×).
- `PlotWidget` (matplotlib angle plots) — bottom.
- **Export PNG** button (saves the current angle figure).
- `set_data(results, df)` wires frame range from `len(df)` and renders plots from
  `results["joint_angles_mean"]` / `["joint_angles_std"]`.

### 7.4 `widgets/skeleton_widget.py` — vispy
- Wraps a vispy `SceneCanvas` (`canvas.native` embedded as the Qt widget) with `Markers`
  (joints) + `Line` (segments) visuals and a turntable camera.
- `set_data(df)` stores the frame array; `set_frame(i)` recomputes points/segments via
  `skeleton_3d` and updates the visuals; NaN landmarks are skipped per frame.
- A `QTimer` advances frames during playback at the selected speed factor.

### 7.5 `widgets/plot_widget.py` — matplotlib
- `FigureCanvasQTAgg` hosting the `Figure` from `angle_plots.plot_joint_angles`; `render(mean,
  std)` rebuilds the figure; exposes the current `Figure` for PNG export.

---

## 8. Testing — three tiers

| Tier | Scope | How |
|------|-------|-----|
| **1. Headless-unit (strict TDD)** | `run_pipeline`, `skeleton_3d` core, `angle_plots`, `export` | Plain pytest on fixtures; assert data/structure. |
| **2. pytest-qt offscreen smoke** | `MainWindow`, panels, worker, `SkeletonWidget`, `PlotWidget` | `conftest.py` sets `QT_QPA_PLATFORM=offscreen`; `qtbot` constructs widgets, feeds a small fixture `(df, results)`, asserts **no crash + state** (table populated, slider range correct, worker emits `finished`). No pixel assertions. |
| **3. Manual-visual checklist** | Real GL render | Documented steps: run `app.py` on `p1_1`; confirm skeleton animates + scrubs, plots render with corridor, CSV/XLSX/PNG exports open. **Only tier covering real GL pixels** (D3 tradeoff). |

`ruff check .` clean; overall coverage maintained > 70% (Tier-1 cores carry the coverage; GUI
shell is smoke-covered).

**Fixtures:** reuse the existing mini caliscope fixtures; add a small `(df, results)` pair for GUI
smoke tests so they don't depend on real data.

---

## 9. Error handling

- **Worker thread** wraps load + pipeline in `try/except` → `error(str)` signal → `QMessageBox`;
  the UI thread is never blocked and never crashes on bad input.
- **Bad/empty folder, wrong model, missing required landmarks** → the loader/pipeline raises with
  a clear message, surfaced verbatim in the dialog.
- **Per-frame NaN landmarks** in the skeleton → that marker/segment is skipped, not an error.
- **Data-quality line** proactively flags landmarks with > 5% NaN so the user distrusts shaky
  sessions *before* reading the numbers.

---

## 10. Build sequence (detail comes in the implementation plan)

1. Add deps to `requirements.txt`; confirm `import PyQt6` / `import vispy` in the venv. *(deps
   already installed + verified during brainstorming — §3.)*
2. **Extract `run_pipeline`**; refactor `cli.analyze`; `test_pipeline.py` + existing 41 tests
   pass (regression guard).
3. **Module 4 cores** (parallelizable): `skeleton_3d`, `angle_plots`, `export` — TDD.
4. **GUI shell + worker**: `app.py`, `main_window.py`, `worker.py` — offscreen smoke.
5. **AnalyzePanel** — offscreen smoke.
6. **SkeletonWidget + PlotWidget** — core unit-tested; widget construction smoke + manual-visual.
7. **VizPanel** wiring.
8. **Integration**: end-to-end on `p1_1`; manual-visual checklist; `ruff`; coverage.

Per the prewarmed plan: parallel multi-agent fan-out for the pure-logic cores (step 3);
subagent-driven sequential review (Phase-1 style) for the GUI shell + integration tail.

---

## 11. Acceptance criteria touched by Phase 2

| # | Criterion | Phase 2 status |
|---|-----------|----------------|
| 7 | GUI loads project, shows skeleton + plots | ✅ delivered |
| 8 | CSV + XLSX export works | ✅ delivered |
| 9 | pytest passes, coverage > 70% | ✅ maintained |
| 10 | `ruff check .` clean | ✅ maintained |

(#1/#4/#5 remain gated to Phase 3 — Vicon; #2/#3/#6 delivered in Phase 1.)

---

## 12. Risks & open items

1. **GL render path has no automated pixel coverage** (D3). Mitigation: GL-free core is fully
   unit-tested; widget construction is smoke-tested headlessly; a manual-visual checklist covers
   real pixels; the GL-free core keeps a matplotlib fallback trivially addable if a target machine
   ever lacks GL.
2. **vispy under `offscreen`** warns `QOpenGLWidget is not supported` — expected; construction
   still succeeds, so smoke tests are valid. Do not assert rendered output in Tier 2.
3. **Short recordings** (~64–102 frames) carry over from Phase 1: few cycles per session. The
   angle-plot corridor may be wide; this is a data property, surfaced (not hidden) by the plots.
4. **`run_pipeline` extraction must not change CLI output** — guarded by the existing integration
   test; verify byte-identical `gait_results.json` before building on it.

---

## 13. Summary

Extract the pipeline into one shared `run_pipeline`, add a headless-testable visualization module
(matplotlib plots + GL-free skeleton core), and wrap it in a deliberately minimal two-tab PyQt6
app with a non-blocking worker. The skeleton renders with verified vispy/OpenGL for quality; every
other surface stays simple. The Vicon comparison tab and a second skeleton renderer are explicitly
deferred. Result: criteria #7 and #8 met, the test suite stays green, and the GUI does one
workflow — load, analyze, look — without ceremony.
