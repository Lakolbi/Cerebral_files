from pathlib import Path

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


def test_plot_widget_renders_angles(qtbot):
    from gui.widgets.plot_widget import PlotWidget
    w = PlotWidget()
    qtbot.addWidget(w)
    results, _ = fixture_results_df()
    w.render_angles(results["joint_angles_mean"], results["joint_angles_std"])
    assert w.current_figure.axes                     # axes exist after render
    assert any(ax.lines for ax in w.current_figure.axes)   # at least one curve drawn


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


def test_analyze_panel_quality_line_only_flags_gait_landmarks(qtbot):
    from gui.panels.analyze_panel import AnalyzePanel
    panel = AnalyzePanel()
    qtbot.addWidget(panel)
    results, df = fixture_results_df()
    df = df.copy()
    # a non-gait landmark that is entirely untracked must NOT appear in the warning
    df["nose_tip_x"] = np.nan
    df["nose_tip_y"] = np.nan
    df["nose_tip_z"] = np.nan
    # a gait landmark that is untracked MUST appear in the warning
    df["left_foot_index_x"] = np.nan
    df["left_foot_index_y"] = np.nan
    df["left_foot_index_z"] = np.nan
    panel._on_finished(results, df)
    text = panel.quality.text()
    assert "nose_tip" not in text
    assert "left_foot_index" in text


_P1_3 = Path(__file__).resolve().parents[1] / "data" / "caliscope_project" / "recordings" / "p1_3"


@pytest.mark.skipif(not _P1_3.exists(), reason="p1_3 caliscope data not present")
def test_real_thread_run_completes_and_rearms(qtbot):
    try:
        from gui.main_window import MainWindow
        win = MainWindow()
    except Exception as exc:                 # vispy/GL unavailable
        pytest.skip(f"vispy widget unavailable: {exc}")
    qtbot.addWidget(win)
    panel = win.analyze
    panel._folder = str(_P1_3)
    with qtbot.waitSignal(panel.analysis_done, timeout=30000):
        panel._run()                          # real QThread + worker + pipeline
    qtbot.waitUntil(lambda: not panel._thread.isRunning(), timeout=5000)
    assert panel.run_btn.isEnabled()          # re-armed only after thread fully stopped
    assert panel.table.rowCount() > 0         # results populated
    assert win.viz.slider.maximum() > 0       # data routed into the viz tab


def test_platform_reroutes_wayland_to_xcb():
    # vispy's GLSL 1.20 shaders fail on the core-profile context Wayland/EGL gives
    # NVIDIA; xcb (XWayland/GLX) provides a compatible context. offscreen is preserved.
    from app import _platform_for
    assert _platform_for("") == "xcb"
    assert _platform_for("wayland") == "xcb"
    assert _platform_for("wayland;xcb") == "xcb"
    assert _platform_for("offscreen") == "offscreen"
    assert _platform_for("xcb") == "xcb"


def test_configure_opengl_requests_compatibility_profile():
    from PyQt6.QtGui import QSurfaceFormat

    import app
    prev = QSurfaceFormat.defaultFormat()
    try:
        app.configure_opengl()
        assert (QSurfaceFormat.defaultFormat().profile()
                == QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    finally:
        QSurfaceFormat.setDefaultFormat(prev)


def test_viz_panel_playback_interval_uses_capture_fps(qtbot):
    try:
        from gui.panels.viz_panel import VizPanel
        panel = VizPanel()
    except Exception as exc:                 # vispy/GL unavailable
        pytest.skip(f"vispy widget unavailable: {exc}")
    qtbot.addWidget(panel)
    results, df = fixture_results_df()       # fps 19.0
    panel.set_data(results, df)
    # 1x playback must track the recording's fps, not a hardcoded 30.
    assert panel._frame_interval_ms(1.0) == max(1, int(1000 / 19.0))
    assert panel._frame_interval_ms(0.5) == max(1, int(1000 / (19.0 * 0.5)))


def test_overlay_worker_error_path_emits_error():
    from gui.overlay_worker import OverlayWorker
    worker = OverlayWorker("/nonexistent/session/folder", "SIMPLE_HOLISTIC", "/tmp/none")
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors          # missing videos/xy -> error signal, no crash


def test_analyze_panel_has_produce_button_disabled_initially(qtbot):
    from gui.panels.analyze_panel import AnalyzePanel
    panel = AnalyzePanel()
    qtbot.addWidget(panel)
    assert hasattr(panel, "produce_btn")
    assert not panel.produce_btn.isEnabled()       # disabled until a folder is chosen


def test_comparison_worker_error_path_emits_error():
    from gui.worker import ComparisonWorker
    worker = ComparisonWorker("/nonexistent/session", "/nonexistent/vicon.xlsx",
                              "SIMPLE_HOLISTIC", {})
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors          # bad paths -> error signal, no crash


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


def test_compare_panel_emits_comparison_done(qtbot):
    from gui.panels.compare_panel import ComparePanel
    panel = ComparePanel()
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.comparison_done, timeout=1000):
        panel._on_finished(_fake_comparison_report(), {})


def test_compare_panel_overlay_handles_empty(qtbot):
    from gui.panels.compare_panel import ComparePanel
    panel = ComparePanel()
    qtbot.addWidget(panel)
    rep = _fake_comparison_report()
    rep["angle"], rep["angle_overlay"] = {}, {}        # no comparable joints
    panel._on_finished(rep, {})                        # must not raise
    assert panel.table.rowCount() == 0
