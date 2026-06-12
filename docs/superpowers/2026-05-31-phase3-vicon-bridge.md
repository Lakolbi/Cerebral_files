# Phase 3 Bridge — Vicon Comparison (planning handoff)

**Date:** 2026-05-31
**Status:** Phase 1 + Phase 2 DONE & merged to `main`. Phase 3 not started.
**Purpose:** Carry forward everything needed to plan and build Phase 3 (Module 3 — Vicon
comparison) in a fresh session, and make explicit what is buildable now vs. gated on user data.

**Read these first:**
- Phase 1 design spec — `docs/superpowers/specs/2026-05-29-gait-kinematics-analysis-design.md`
  (Module 3 is fully specified there: §3 structure, §7 Module 3, §9 config, §10 validation
  Levels A/C, §12 acceptance criteria). **This bridge does not restate the algorithms — §7 is the
  source of truth.**
- Phase 2 design + plan — `docs/superpowers/specs/2026-05-30-gait-phase2-gui-design.md`,
  `docs/superpowers/plans/2026-05-30-gait-phase2-gui.md`.

---

## 1. Where the project stands

| Module | State |
|--------|-------|
| **Module 1 — data_loader** | DONE. Incl. `vicon_reader.py` (`load_vicon_xlsx`, `map_vicon_to_caliscope`) + `test_vicon_reader.py`, and `synchronizer.py` (`synchronize(df_a, df_b, target_fps)`). **Vicon ingestion + resampling already exist and are tested.** |
| **Module 2 — kinematics** | DONE. Joint angles, gait events, normalizer, spatiotemporal, filters. |
| **`pipeline.py` / `cli.py`** | DONE. Shared `run_pipeline(df, cfg, *, model, session_id, progress_cb) -> (results, df)`. CLI has `analyze` + `reproducibility` subcommands only. |
| **Module 4 — visualization + GUI** | DONE (Phase 2). Two-tab PyQt6 app (`Анализ`/`Визуализация`), vispy skeleton, matplotlib plots, CSV/XLSX/PNG export. 64 tests / 95% cov / ruff clean. |
| **Module 3 — comparison** | **NOT BUILT.** `modules/comparison/` does not exist. The original Phase-1 spec assumed this code would be written in Phase 1/2 with synthetic tests — it was not. Building it is the core of Phase 3. |
| **GUI "Сравнение с Vicon" tab** | NOT BUILT (deliberately dropped from Phase 2, decision D1). |
| **VvsC.py baseline wrapper** | NOT BUILT (script not provided). |

Config already in place (`settings.yaml`):
- `landmark_mapping:` Vicon marker → canonical lowercase landmark (LASI→left_hip, LKNE→left_knee,
  LANK→left_ankle, LHEE→left_heel, LTOE→left_foot_index, and right-side; 10 markers).
- `comparison:` `icc_type: '3,1'`, `good_rmse_threshold_deg: 5.0`, `acceptable_rmse_threshold_deg: 10.0`.

---

## 2. Phase 3 work — split by gate

### 2a. Buildable NOW (synthetic-testable, no Vicon data needed)
Build `modules/comparison/` per Phase-1 spec §7, TDD with synthetic fixtures:
- **`alignment.py`** — `estimate_rigid_transform(src, dst) -> (R, T, s)` (Umeyama 1991, allows
  scale, reflection-free); `align_coordinate_systems(cal_df, vic_df, reference_landmarks=None)`.
  Test: apply a known (R,T,s) to a point cloud, assert it is recovered.
- **`metrics.py`** — `calc_rmse`, `calc_mae`, `calc_pearson`, `calc_icc(type='3,1')`;
  `full_comparison_report(cal_df, vic_df, joint_list=None) -> DataFrame`. Test: identical arrays →
  RMSE 0 / MAE 0 / Pearson 1 / ICC ≈ 1.
- **`report.py`** — per-joint metric table + overlay-plot data + verdict thresholds (good <5°,
  acceptable 5–10°, poor >10°).
- **`compare` CLI subcommand** in `cli.py` — load caliscope + Vicon, synchronize (Module 1's
  `synchronize`), align, compute metrics, write `comparison_report.json`. Synthetic-testable by
  feeding a transformed copy of caliscope data as a stand-in "Vicon" stream.
- **GUI "Сравнение" tab** (optional for Phase 3, or defer) — run alignment, per-joint
  RMSE/MAE/Pearson/ICC table, overlaid caliscope(blue)/Vicon(red) cycle curves, verdict. Buildable
  against synthetic data; mirrors the Phase-2 panel/worker pattern.

### 2b. GATED on the user supplying data
- **`VvsC.py` wrapper** (`run_baseline_comparison`, runs the baseline via `subprocess`, parses
  output) — interface specified now, **body implemented when `VvsC.py` lands**.
- **Level A** — Vicon synchronous accuracy: RMSE/MAE/ICC caliscope vs real Vicon per joint
  (targets: knee RMSE < 5°, ICC > 0.85).
- **Level C** — algorithm comparison table: baseline (VvsC) vs HOLISTIC vs developed algorithm.
- **Acceptance criteria #1** (VvsC reproduces baseline), **#4** (joint RMSE <10°/<5°), **#5**
  (knee ICC > 0.75) — all unlock only with real Vicon + VvsC.py.

---

## 3. What the user must provide to unlock 2b
1. **Vicon XLSX file(s)** for the same/comparable walking trials. `vicon_reader.load_vicon_xlsx`
   already: auto-detects the header row, parses `MarkerName:X|Y|Z` columns, auto-detects mm vs m
   and converts to meters. If the real files deviate from that schema (different header layout,
   unit cell, marker naming), `vicon_reader` is where to adjust — re-validate it against the real
   file first.
2. **`VvsC.py`** (the existing baseline script) — needed for criterion #1 and Level C only.

Until then, build everything in 2a and leave 2b interfaces stubbed with synthetic tests, so
dropping in the data later requires only running the gated apparatus (no rework).

---

## 4. Patterns + gotchas to carry forward from Phases 1–2

- **Reuse the shared-function pattern:** factor any compare pipeline into a plain function (like
  `run_pipeline`) callable from both `cli.py` and a GUI worker; assemble the canonical report JSON
  in exactly one place.
- **Test tiers (same as Phase 2):** headless-unit (strict TDD for alignment/metrics — these are
  pure numpy, fully testable) / pytest-qt offscreen smoke (if a compare tab is built) /
  manual-visual (overlay plots). `tests/conftest.py` already forces `QT_QPA_PLATFORM=offscreen`.
- **ICC dependency decision:** `pingouin` is listed in `requirements.txt` but **NOT installed**
  (Phase 2 only installed matplotlib/PyQt6/vispy/PyOpenGL/pytest-qt). `pingouin` is pure-Python
  but pulls scipy/statsmodels/scikit-learn. **Recommended: implement `calc_icc('3,1')` with a
  small numpy formula** (ICC(3,1) is a closed-form from a one-way/two-way ANOVA table) and avoid
  the heavy dependency — keeps the install lean. Decide this in brainstorming.
- **GL gotcha (already fixed, but relevant if the compare tab adds vispy/GL overlays):** vispy
  emits GLSL `#version 120`; on Wayland/NVIDIA the embedded `QOpenGLWidget` gets a core-profile
  context that rejects it. `app.py` already reroutes Wayland→`xcb` (`_platform_for`) so any new GL
  widget inherits the working context. Matplotlib overlay plots avoid GL entirely (preferred for
  the compare curves).
- **"Dead-simple" mandate still holds:** if a compare tab is added, keep it lean (one tab, one
  results table, one overlay-plot canvas, one verdict label) — mirror the Phase-2 `analyze_panel`
  structure.
- **Coordinate mirroring** (Phase-1 spec §4.5, §13.3): caliscope X may mirror Vicon's; Umeyama
  alignment (reflection-free rotation) + a visual check covers it; a sign flag is exposed in
  config if a hard flip is needed. Only verifiable once real Vicon data exists.
- **Repo conventions:** Python 3.14 venv at `gait_analysis/.venv`; run from `gait_analysis/` with
  `.venv/bin/python -m pytest`; `ruff` config in `pyproject.toml` (line 100, select E/F/I/W); tests
  import `from modules...`/`from pipeline import...`; commit with explicit pathspecs (never
  `git add -A`) — an unrelated `.docx` and `.idea/` sit untracked in the tree.

---

## 5. Recommended kickoff for the Phase 3 session
1. If real Vicon XLSX + `VvsC.py` are now available, re-validate `vicon_reader` against the real
   file before anything else.
2. Run `/superpowers:brainstorming` for Phase 3 — key decisions to settle: (a) ICC via numpy vs
   pingouin; (b) build the GUI "Сравнение" tab now or keep it CLI/report-only; (c) scope of the
   synthetic test harness for the gated pieces.
3. Then `/superpowers:writing-plans` → subagent-driven execution (same flow that worked for
   Phase 2: viz/logic cores are independent and parallelizable; GUI/CLI tails sequential).

**`main` is N commits ahead of `origin/main` and not pushed** — pushing is a separate user
decision.
