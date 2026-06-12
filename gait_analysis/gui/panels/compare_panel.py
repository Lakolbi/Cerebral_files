"""Tab «Сравнение»: caliscope vs Vicon — run comparison, per-joint table, overlay, verdict."""
from pathlib import Path

import numpy as np
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
        if not overlay:
            ax = fig.subplots(1, 1)
            ax.set_title("no comparable joints")
            ax.set_axis_off()
            self.draw()
            return
        axes = fig.subplots(1, len(overlay), squeeze=False)[0]
        for ax, j in zip(axes, overlay):
            cal = overlay[j]["caliscope"]
            vic = overlay[j]["vicon"]
            x = np.linspace(0, 100, len(cal))
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
        srow = QHBoxLayout()
        srow.addWidget(self.session_btn)
        srow.addWidget(self.session_label)
        vrow = QHBoxLayout()
        vrow.addWidget(self.vicon_btn)
        vrow.addWidget(self.vicon_label)
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
        self._worker = ComparisonWorker(
            self._session, self._vicon, self.model.currentText(), self._cfg
        )
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
            values = [
                joint,
                f"{m['rmse_deg']:.2f}",
                f"{m['mae_deg']:.2f}",
                f"{m['pearson']:.3f}",
                f"{m['icc']:.3f}",
                m["verdict"],
            ]
            for col, val in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(val))
        self.overlay.render_overlay(report.get("angle_overlay", {}))
        self.verdict.setText(report.get("verdict_summary", ""))
        self.comparison_done.emit(report, artifacts)

    def _on_error(self, msg):
        QMessageBox.warning(self, "Comparison failed", msg)
