# Design Spec — Phase 3: Vicon Comparison (Module 3)

**Version:** 1.0
**Date:** 2026-06-06
**Status:** Approved for planning
**Predecessors:** Phase 1 (`2026-05-29-gait-kinematics-analysis-design.md`), Phase 2
(`2026-05-30-gait-phase2-gui-design.md`), bridge (`2026-05-31-phase3-vicon-bridge.md`).
**Source of truth for algorithms:** Phase 1 spec §7 (Module 3). This spec supersedes the bridge
doc where they conflict, because the gated inputs have now arrived and contradict the bridge's
"drop-in, no rework" premise (see §2).

---

## 1. Goal

Build Module 3 (Vicon comparison) and unlock the gated validation Levels A and C. The deliverable
has **two layers**, both emitted in one `comparison_report.json`:

1. **Angle layer (primary, degrees).** Compare caliscope vs Vicon **joint-angle gait-cycle
   curves** — per-joint RMSE°, MAE°, Pearson r, ICC(3,1). Satisfies ТЗ acceptance criteria
   **#4** (joint RMSE < 10° acceptable / < 5° good) and **#5** (knee ICC > 0.75), and Level A.
2. **Position layer (baseline, meters).** Reproduce the existing `VvsC.py` baseline —
   similarity-align the two marker clouds and compute per-joint positional RMSE/MAE in meters
   (+ pelvis-relative). Satisfies criterion **#1** ("VvsC.py runs, reproduces baseline") and
   feeds the Level C comparison table.

Both layers also surface in a new GUI "Сравнение" tab, completing the 4-tab app of Phase-1 §8.2.

---

## 2. What the real data revealed (reconciliation)

The bridge doc assumed Module 3 was "written, synthetic-tested, drop in data and run." The real
artifacts (`Vicon_10_series/*.xlsx`, `VvsC.py`) contradict that on four fronts. This spec resolves
each:

| Finding | Reality | Resolution |
|---------|---------|------------|
| **`vicon_reader` is broken on real files** | `load_vicon_xlsx` raises `ValueError` — the real axis row is `Frame, Sub Frame, X, Y, Z, …` (not X/Y/Z only); marker names are colon-prefixed with the subject (`Derevesnikova Darya Alexandrovna:LASI`); there is a duplicate `\| … \|` marker block and a `Trajectory Count` column; units row `mm` sits below the axis row; frame numbers start at 225. | **Fix `vicon_reader`** by porting VvsC's proven parsing (§5.1). |
| **Domain mismatch** | `VvsC.py` measures **positional** error (meters); ТЗ #4/#5 require **angular** error (degrees) + knee ICC. | Build **both layers** (§1). They are complementary, not competing. |
| **Sync is event-based, not steady-state** | Vicon ≈ 136 s @ 100 fps (≈13 615 frames); caliscope clips are 64–207 frames (~3–10 s @ ~19–20 fps). VvsC time-aligns via a jump velocity/Z peak + pelvis-speed cross-correlation. Our `synchronize()` does neither. | Port VvsC's xcorr/jump alignment into `alignment.py`; the angle layer mostly avoids it via gait-cycle normalization (§6.1). |
| **No 1:1 trial pairing** | 10 Vicon trials, all the **same subject** (Derevesnikova Darya Alexandrovna); on the caliscope side only `p1_1..p1_5` (+ `walking_test`, `walking_test_2`) have SIMPLE_HOLISTIC. | Angle layer pools the subject's cycles (pairing-free); position layer uses explicit pairs if configured, else auto-matches by xcorr quality (§6.3). |

**Key enabling insight:** VvsC's `compute_similarity_transform` *is* Umeyama (= the spec's
`alignment.py`) and its `estimate_time_shift_by_xcorr` *is* the temporal sync the real data needs.
Porting VvsC and building the spec's alignment are the **same work**. Only VvsC's `main()`
(hardcoded paths, `plt.show()`, CSV side-effects) is unusable; its functions are import-clean.

---

## 3. Architecture

```
                       compare_pipeline.run_comparison(cal_df, vic_df, cfg)
                                          │
              ┌───────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
   data_loader.vicon_reader      comparison.alignment        comparison.metrics
   (FIXED: real schema)          (Umeyama + xcorr/jump)      (angle ° + position m + ICC)
              │                            │                            │
              └──────────── reuse Module 2 kinematics on BOTH streams ───┘
                                          │
                              comparison.report  →  comparison_report.json
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                             ▼
              cli.py: compare / validate-vicon          gui ComparePanel + ComparisonWorker
```

`run_comparison` is the single source of truth for the report JSON — mirrors the `run_pipeline`
pattern (one function, two entry points). VvsC.py remains in the repo unmodified as the reference
baseline for the parity test.

---

## 4. Module 3 file plan

```
modules/comparison/
├── __init__.py
├── alignment.py     # rigid+scale registration, temporal sync
├── metrics.py       # error metrics (° and m), ICC
└── report.py        # assemble comparison_report.json + overlay-plot data
compare_pipeline.py  # run_comparison(...) — shared by CLI + GUI (sibling of pipeline.py)
```

Touched existing files: `modules/data_loader/vicon_reader.py` (fix), `cli.py` (subcommands),
`gui/worker.py`, `gui/main_window.py`, `gui/panels/compare_panel.py` (new),
`settings.yaml` (config), `requirements.txt` / venv (`pingouin`).

---

## 5. Components — interfaces

### 5.1 `vicon_reader.py` (FIX)

Port VvsC's `load_vicon_points_with_pelvis` parsing into the existing functions:

- `load_vicon_xlsx(filepath, vicon_fps=100.0) -> DataFrame`
  - Locate the axis row as the row containing `X`/`Y`/`Z` labels (now tolerant of `Frame`,
    `Sub Frame`, and other non-axis cells in the same row); marker-name row is directly above.
  - **Colon-split** each marker cell on `:` and take the trailing token (`…:LASI` → `LASI`),
    strip whitespace.
  - **Ignore** the duplicate `\| … \|` block (its split yields names like `LASI |`, which are not
    canonical and get dropped at mapping time) and the `Trajectory Count` column.
  - Build `MARKER_{x,y,z}` columns; auto-detect mm→m (existing `median(|coord|) > 10` rule).
  - **Add `timestamp`** = `(frame − frame[0]) / vicon_fps` (seconds) and keep `frame`, so
    `synchronize()` / xcorr can consume it. `vicon_fps` defaults to 100 (the value also present in
    sheet row 1; reading it from the sheet is optional polish).
- `map_vicon_to_caliscope(vicon_df, mapping) -> DataFrame` — unchanged (rename via
  `settings.yaml → landmark_mapping`). Markers absent from the mapping (LPSI/LTHI/LTIB/…) are left
  untouched and ignored downstream.

### 5.2 `alignment.py` (new)

- `estimate_rigid_transform(src, dst) -> (R, T, s)` — Umeyama (1991), reflection-free
  (`det(R)` fix), uniform scale. Cleaned from VvsC `compute_similarity_transform`.
- `apply_transform(points, R, T, s) -> ndarray`.
- `align_coordinate_systems(cal_df, vic_df, reference_landmarks=None) -> DataFrame` — estimate the
  transform on common landmarks (default: all mapped markers present in both, NaN-masked) and apply
  it to the Vicon stream. (Used by the position layer; **not** needed by the angle layer — angles
  are invariant to rigid+scale.)
- `estimate_time_shift_xcorr(t_a, pts_a, t_b, pts_b, fs_grid=100.0, max_shift=0.5) -> float` —
  pelvis-speed cross-correlation lag. Ported from VvsC.
- `detect_sync_event(t, pts, mode='speed'|'zmax') -> int` — jump apex / max-speed index, for the
  jump-sync alternative. Ported from VvsC `detect_event_index`.

### 5.3 `metrics.py` (new)

- `calc_rmse(a, b)`, `calc_mae(a, b)`, `calc_pearson(a, b)` — NaN-aware, 1-D.
- `calc_icc(values_by_rater, icc_type='3,1') -> float` — **pingouin** `intraclass_corr`; guard
  degenerate (constant) input → `NaN`.
- `angle_comparison_report(cal_curves, vic_curves, joint_list=None) -> DataFrame` — per joint:
  `rmse_deg, mae_deg, pearson, icc, verdict` where verdict = good (<5°) / acceptable (5–10°) /
  poor (>10°) from `settings.comparison`.
- `position_comparison_report(cal_pts, vic_pts_aligned, joint_list=None) -> DataFrame` — per joint:
  `rmse_m, mae_m, max_m, median_m, rmse_{x,y,z}_m`, plus pelvis-relative variants. Mirrors VvsC
  `compute_errors_for_trial` outputs (for parity).

### 5.4 `report.py` (new)

- `build_report(angle_df, position_df, meta) -> dict` — canonical `comparison_report.json`:
  `{session/pair ids, model, fps, processed_at, angle: {...}, position: {...},
  verdict_summary, overlay: {per-joint 101-pt cal/vic curves}}`.

### 5.5 `compare_pipeline.py` (new)

- `run_comparison(cal_df, vic_df, cfg, *, model, pair_id, progress_cb=None) -> (report, artifacts)`
  — operates on the **single** (cal_df, vic_df) it is handed (one clip vs one Vicon trial).
  - Angle layer: run Module 2 kinematics (`calc_joint_angles_timeseries`, `detect_gait_events`,
    `normalize_gait_cycle`, `get_mean_std_cycle`) on **both** DataFrames → mean 101-pt curve per
    joint per system → `angle_comparison_report`. Also returns the per-cycle normalized matrices so
    the apparatus can pool across clips.
  - Position layer: `estimate_time_shift_xcorr` → crop overlap → `align_coordinate_systems` →
    `position_comparison_report`.
  - Assemble via `report.build_report`. `progress_cb(frac, stage)` like `run_pipeline`.

  **Pooling lives one level up:** the subject-level Level-A table (§8) is assembled by
  `validate-vicon`, which concatenates the per-cycle normalized matrices across the subject's clips
  into one caliscope ensemble curve and one Vicon ensemble curve per joint, then computes one set of
  RMSE°/ICC. `run_comparison` itself never concatenates raw DataFrames across clips (their
  timestamps are not continuous).

### 5.6 `cli.py` (extend)

- `compare --caliscope <session_dir> --vicon <xlsx> --out <json>` — one pair → report.
- `validate-vicon --recordings <dir> --vicon-dir <dir> --out <dir>` — run the dataset, emit the
  **Level A** table (per-joint caliscope-vs-Vicon RMSE°/ICC) and **Level C** table (VvsC positional
  baseline vs developed-angle algorithm) + overlay PNGs. Honors `comparison.pairs` or auto-matches.

### 5.7 GUI (extend)

- `gui/panels/compare_panel.py` — `ComparePanel` mirroring `AnalyzePanel`: caliscope-session +
  Vicon-XLSX pickers, "Run comparison" → `ComparisonWorker` in a `QThread`, progress bar, per-joint
  RMSE°/MAE/Pearson/ICC table, **matplotlib** overlaid caliscope(blue)/Vicon(red) 101-pt cycle
  curves (no GL — avoids the Wayland/NVIDIA `#version 120` issue), verdict label, export.
- `gui/worker.py` — `ComparisonWorker(QObject)` with `progress`/`finished`/`error` signals; calls
  `run_comparison`. Same close-guard pattern as the pipeline worker.
- `gui/main_window.py` — add a 3rd tab `"Сравнение"` between Анализ and Визуализация.

---

## 6. Data flow & methods

### 6.1 Angle layer (primary, pairing-free)

Joint flexion angles are `arccos` of normalized vectors → **invariant to rotation, translation,
and uniform scale**. So alignment is *not* on the critical path here. Method:

1. caliscope: load session → Module 2 → per-joint normalized 101-pt mean curve (pooled across the
   subject's cycles / clips).
2. Vicon: load XLSX → map markers → run the **same** Module 2 code (the real Vicon files carry
   hip/knee/ankle/heel/toe markers) → per-joint normalized 101-pt mean curve (pooled across the
   10 trials' walking cycles).
3. Compare the two mean curves per joint: RMSE°, MAE°, Pearson r, ICC(3,1) treating the 101 cycle
   points as targets and {caliscope, Vicon} as the two raters.

Gait-cycle normalization *is* the temporal alignment for ensemble comparison; pooling the subject's
cycles makes this robust to the missing 1:1 trial map. Pelvis tilt/obliquity/rotation are defined
vs lab axes and are **not** rotation-invariant — they are reported but excluded from the gating
verdict (they are not in #4/#5).

### 6.2 Position layer (VvsC baseline, m)

For a designated synchronous (caliscope, Vicon) pair: `estimate_time_shift_xcorr` → crop to overlap
→ `estimate_rigid_transform` on common markers → `position_comparison_report`. Reproduces VvsC's
`results_summary.csv`. **Criterion #1** is met by the parity test (§7).

### 6.3 Pairing strategy

- **Angle layer:** none required — pools the subject's cycles.
- **Position layer:** requires *synchronous* pairs (the jump gesture is the manual sync event).
  Resolution order: (1) explicit `comparison.pairs:` in `settings.yaml`; (2) fallback —
  auto-match each caliscope clip to the Vicon trial with the best xcorr peak / lowest post-align
  RMSE. Auto-matched pairs are flagged low-confidence in the report.

**Assumption (confirmed at design approval; refine if a pairing is later supplied):** subject =
participant 1 → `p1_1..p1_5` are the caliscope partners of 5 of the 10 Vicon trials. The headline
angle validation does not depend on this.

---

## 7. Testing (tiers as Phase 2; coverage > 70%, `ruff` clean)

| Test | Asserts |
|------|---------|
| `test_vicon_reader` (extend) | Real-schema fixture: subject-prefixed colon names, `Frame`/`Sub Frame` cols, `\| … \|` duplicate block, `Trajectory Count`, mm→m, `timestamp` derived. `load_vicon_xlsx` succeeds and yields canonical columns after mapping. |
| `test_alignment` | Umeyama recovers a known `(R, T, s)` applied to a point cloud; reflection-free. xcorr recovers a known injected time shift. |
| `test_metrics_comparison` | Identical arrays → RMSE 0, MAE 0, Pearson 1, ICC ≈ 1; ICC(3,1) matches a hand-computed value; verdict thresholds correct. |
| `test_vvsc_parity` | Ported `position_comparison_report` == VvsC `compute_errors_for_trial` (within tol) on one real pair → **criterion #1**. May be marked `@pytest.mark.realdata` if it needs the XLSX. |
| `test_compare_pipeline` | `run_comparison` returns the full report schema (angle + position sections) on a synthetic transformed-copy stand-in. |
| `test_compare_gui_smoke` | pytest-qt offscreen: ComparePanel constructs, run wires to worker, results populate (no GL). |
| manual-visual | overlay curves + Level A/C tables look right (`docs/manual_visual_checklist.md`). |

Synthetic stand-in for unit tests: feed a known-transformed, time-shifted copy of caliscope data as
the "Vicon" stream so alignment/sync/metrics are exercised without real files.

---

## 8. Validation apparatus (Levels A & C — now unlockable)

`validate-vicon` produces the thesis "accuracy comparison" deliverables:

- **Level A** — per-joint caliscope-vs-Vicon angle accuracy table (RMSE°, MAE°, Pearson, ICC);
  targets: knee RMSE < 5°, ICC > 0.85 (acceptance: RMSE < 10° / < 5°; knee ICC > 0.75).
- **Level C** — comparison table: VvsC positional baseline vs developed-angle algorithm
  (+ optional HOLISTIC vs SIMPLE_HOLISTIC), with overlay PNGs.

Level B (reproducibility CV across `p1_1..p1_5`) remains the Phase-1 primary result and is
unchanged.

---

## 9. Configuration additions (`settings.yaml`)

```yaml
comparison:
  icc_type: '3,1'                  # existing
  good_rmse_threshold_deg: 5.0     # existing — angle-path verdict
  acceptable_rmse_threshold_deg: 10.0
  vicon_fps: 100.0                 # new — Vicon device rate
  sync_method: xcorr               # new — xcorr | jump | zmax
  reference_landmarks: null        # new — markers for alignment (null = all common)
  pairs: null                      # new — optional {caliscope_session: vicon_file} map
```

caliscope fps stays **derived** from `frame_time_history` (≈19–20), never the hardcoded 20 VvsC
used.

---

## 10. Acceptance criteria addressed

| # | Criterion | This phase |
|---|-----------|-----------|
| 1 | `VvsC.py` runs, reproduces baseline | ✅ parity test (§7) |
| 4 | Joint angles vs Vicon RMSE < 10° / < 5° | ✅ Level A angle table |
| 5 | Knee ICC > 0.75 | ✅ Level A (pingouin ICC(3,1)) |
| 7/8 | GUI tabs, export | ✅ Сравнение tab + export |
| 9/10 | pytest > 70%, ruff clean | ✅ all phases |

---

## 11. Risks & open items

1. **Pairing for the position layer** — wrong (non-synchronous) pairs give meaningless positional
   RMSE. Mitigated by explicit `pairs:` or xcorr auto-match with low-confidence flagging; the angle
   layer is unaffected.
2. **Coordinate mirroring** — handled by reflection-free Umeyama + a visual overlay check; sign flag
   available if a hard flip is ever needed.
3. **Long Vicon recordings** (jump + walk + idle) — gait-event detection + `min_stride` filtering
   isolate the walking cycles; xcorr windowing gives a cleaner crop when a pair is known.
4. **pingouin install** pulls statsmodels + scikit-learn — accepted (user decision) for a
   battle-tested ICC with CIs.

---

## 12. Repo conventions (carry forward)

Python 3.14 venv at `gait_analysis/.venv`; run from `gait_analysis/` with `.venv/bin/python -m
pytest`; `ruff` (line 100, E/F/I/W); tests import `from modules…` / `from compare_pipeline import…`;
commit with explicit pathspecs (never `git add -A`). `main` is ahead of `origin/main` and unpushed —
pushing stays a separate user decision.
