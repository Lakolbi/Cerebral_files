"""Tab «Визуализация»: 3D skeleton + playback controls + angle plots."""
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

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
        self._fps = None    # capture fps; playback matches the real recording duration

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
        self._fps = results.get("fps")
        self.skeleton.set_data(df)
        self.slider.setRange(0, max(0, len(df) - 1))
        self.slider.setValue(0)
        self.plots.render_angles(results.get("joint_angles_mean", {}),
                                 results.get("joint_angles_std", {}))

    def _frame_interval_ms(self, factor):
        """Per-frame timer interval so 1x playback matches the recording's real duration."""
        fps = self._fps if self._fps and self._fps > 0 else 30.0
        return max(1, int(1000 / (fps * factor)))

    def _toggle_play(self):
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("▶")
        else:
            factor = _SPEEDS[self.speed.currentText()]
            self._timer.start(self._frame_interval_ms(factor))
            self.play_btn.setText("⏸")

    def _advance(self):
        nxt = self.slider.value() + 1
        self.slider.setValue(0 if nxt > self.slider.maximum() else nxt)

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "angles.png", "PNG (*.png)")
        if path:
            export_figure_png(self.plots.current_figure, path)
