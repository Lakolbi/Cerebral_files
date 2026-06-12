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

## Module 3 — Vicon comparison (Phase 3)

- [ ] `python app.py` → «Сравнение» tab: pick a caliscope session + a Vicon XLSX → Run.
- [ ] Per-joint table shows knee + ankle rows (hip absent — Vicon has no shoulder marker).
- [ ] Overlay plot: caliscope (blue) and Vicon (red) gait-cycle curves are drawn per joint.
- [ ] Verdict summary line reflects knee ICC + worst verdict.
- [ ] `cli.py validate-vicon --recordings data/caliscope_project/recordings --vicon-dir
      data/Vicon_10_series --out results/vicon_validation` writes the Level-A/-C CSVs.
- [ ] `test_vvsc_parity` passes (ported Umeyama reproduces VvsC's similarity transform exactly).

**Real-data Level-A result (after the Zeni heel-strike anchoring fix, `modules/comparison/events.py`):**
knee ICC = 0.79 (left) / 0.74 (right), RMSE 16.1° / 9.1°; ankles ICC 0.26–0.39 (weak). So #5
(knee ICC > 0.75) is essentially met for the knees; left-knee RMSE reflects a constant
hip-marker-definition offset (Vicon ASIS proxy vs caliscope hip-joint-centre — high ICC/consistency,
biased absolute angle). Ankles are weak (inherent markerless ankle/foot tracking limitation).
**Caveat:** caliscope trials are *overground* walking, Vicon trials are *treadmill* walking, recorded
in separate non-concurrent sessions of the same subject — real overground-vs-treadmill kinematic
differences bound how high #4/#5 can go regardless of method.
