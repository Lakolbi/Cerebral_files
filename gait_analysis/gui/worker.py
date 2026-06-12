"""Background worker: load a session + run the pipeline off the GUI thread."""
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from modules.data_loader.caliscope_reader import load_caliscope_session
from pipeline import run_pipeline


class PipelineWorker(QObject):
    progress = pyqtSignal(float, str)        # (fraction 0..1, stage label)
    finished = pyqtSignal(dict, object)      # (results, df_processed)
    error = pyqtSignal(str)

    def __init__(self, folder, model, cfg):
        super().__init__()
        self._folder = folder
        self._model = model
        self._cfg = cfg

    def run(self):
        try:
            self.progress.emit(0.0, "Loading session")
            df = load_caliscope_session(self._folder, model=self._model)
            results, df_out = run_pipeline(
                df, self._cfg, model=self._model,
                session_id=Path(self._folder).name,
                progress_cb=lambda frac, stage: self.progress.emit(frac, stage),
            )
            self.finished.emit(results, df_out)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, must never crash
            self.error.emit(str(exc))


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
