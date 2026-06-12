"""Tab «Анализ»: load + params + run + results table + data-quality line + export."""
from pathlib import Path

import numpy as np
import yaml
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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

from gui.overlay_worker import OverlayWorker
from gui.worker import PipelineWorker
from modules.data_loader.landmarks import GAIT_LANDMARKS, MODELS

GAIT_DIR = Path(__file__).resolve().parents[2]      # gait_analysis/


def _load_settings():
    with open(GAIT_DIR / "settings.yaml") as f:
        return yaml.safe_load(f)


class AnalyzePanel(QWidget):
    analysis_done = pyqtSignal(dict, object)        # (results, df) -> MainWindow -> VizPanel

    def __init__(self):
        super().__init__()
        self._cfg = _load_settings()
        self._folder = None
        self._results = None

        proc = self._cfg["processing"]
        self.folder_btn = QPushButton("Choose session folder…")
        self.folder_label = QLabel("(none)")
        self.model = QComboBox()
        self.model.addItems(MODELS)
        self.model.setCurrentText(proc["default_model"])
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0.5, 20.0)
        self.cutoff.setValue(proc["filter_cutoff_hz"])
        self.min_stride = QDoubleSpinBox()
        self.min_stride.setRange(0.1, 3.0)
        self.min_stride.setSingleStep(0.1)
        self.min_stride.setValue(proc["min_stride_duration_sec"])
        self.run_btn = QPushButton("▶ Run analysis")
        self.run_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.quality = QLabel("")
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["parameter", "value"])
        self.csv_btn = QPushButton("Export CSV")
        self.csv_btn.setEnabled(False)
        self.xlsx_btn = QPushButton("Export XLSX")
        self.xlsx_btn.setEnabled(False)
        self.produce_btn = QPushButton("Produce marked videos")
        self.produce_btn.setEnabled(False)

        form = QFormLayout()
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_btn)
        folder_row.addWidget(self.folder_label)
        form.addRow(folder_row)
        form.addRow("Model", self.model)
        form.addRow("Filter cutoff (Hz)", self.cutoff)
        form.addRow("Min stride (s)", self.min_stride)
        export_row = QHBoxLayout()
        export_row.addWidget(self.csv_btn)
        export_row.addWidget(self.xlsx_btn)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.quality)
        layout.addWidget(self.table)
        layout.addLayout(export_row)
        layout.addWidget(self.produce_btn)

        self.folder_btn.clicked.connect(self._choose_folder)
        self.run_btn.clicked.connect(self._run)
        self.csv_btn.clicked.connect(self._export_csv)
        self.xlsx_btn.clicked.connect(self._export_xlsx)
        self.produce_btn.clicked.connect(self._produce_videos)

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose caliscope session folder")
        if path:
            self._folder = path
            self.folder_label.setText(Path(path).name)
            self.run_btn.setEnabled(True)
            self.produce_btn.setEnabled(True)

    def _build_cfg(self):
        cfg = dict(self._cfg)
        cfg["processing"] = dict(self._cfg["processing"])
        cfg["processing"]["filter_cutoff_hz"] = self.cutoff.value()
        cfg["processing"]["min_stride_duration_sec"] = self.min_stride.value()
        return cfg

    def _run(self):
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self._thread = QThread()
        self._worker = PipelineWorker(self._folder, self.model.currentText(),
                                      self._build_cfg())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self.run_btn.setEnabled(True))
        self._thread.start()

    def _on_progress(self, frac, stage):
        self.progress.setValue(int(frac * 100))

    def _on_finished(self, results, df):
        self._results = results
        st = results.get("spatiotemporal", {})
        self.table.setRowCount(len(st))
        for row, (k, v) in enumerate(st.items()):
            self.table.setItem(row, 0, QTableWidgetItem(k))
            self.table.setItem(row, 1, QTableWidgetItem("" if v is None else str(v)))
        self._fill_quality(results, df)
        self.csv_btn.setEnabled(True)
        self.xlsx_btn.setEnabled(True)
        self.analysis_done.emit(results, df)

    def _on_error(self, msg):
        QMessageBox.warning(self, "Analysis failed", msg)

    def _fill_quality(self, results, df):
        fps = results.get("fps")
        n = results.get("n_frames", len(df))
        duration = round(n / fps, 1) if fps else 0
        bad = []
        for name in GAIT_LANDMARKS:
            col = f"{name}_x"
            if col in df.columns:
                frac = float(np.isnan(df[col].to_numpy()).mean())
                if frac > 0.05:
                    bad.append(f"{name} ({frac * 100:.0f}%)")
        warn = ("  ⚠ >5% NaN: " + ", ".join(bad)) if bad else ""
        self.quality.setText(f"frames {n} · fps {fps} · {duration}s{warn}")

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "gait.csv", "CSV (*.csv)")
        if path:
            from modules.visualization.export import export_results_csv
            export_results_csv(self._results, path)

    def _export_xlsx(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export XLSX", "gait.xlsx", "XLSX (*.xlsx)")
        if path:
            from modules.visualization.export import export_results_xlsx
            export_results_xlsx(self._results, path)

    def _produce_videos(self):
        self.produce_btn.setEnabled(False)
        self.progress.setValue(0)
        out_dir = (GAIT_DIR / self._cfg["paths"]["output_dir"]
                   / f"{Path(self._folder).name}_marked")
        self._overlay_thread = QThread()
        self._overlay_worker = OverlayWorker(self._folder, self.model.currentText(),
                                             str(out_dir))
        self._overlay_worker.moveToThread(self._overlay_thread)
        self._overlay_thread.started.connect(self._overlay_worker.run)
        self._overlay_worker.progress.connect(self._on_progress)
        self._overlay_worker.finished.connect(self._on_videos_done)
        self._overlay_worker.error.connect(self._on_error)
        self._overlay_worker.finished.connect(self._overlay_thread.quit)
        self._overlay_worker.error.connect(self._overlay_thread.quit)
        self._overlay_thread.finished.connect(lambda: self.produce_btn.setEnabled(True))
        self._overlay_thread.start()

    def _on_videos_done(self, paths):
        self.progress.setValue(100)
        where = paths[0].parent if paths else "(none)"
        QMessageBox.information(self, "Marked videos",
                               f"Saved {len(paths)} videos to:\n{where}")
