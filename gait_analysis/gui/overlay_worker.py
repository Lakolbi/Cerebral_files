"""Background worker: produce marked videos off the GUI thread."""
from PyQt6.QtCore import QObject, pyqtSignal

from modules.visualization.video_overlay import produce_marked_videos


class OverlayWorker(QObject):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(list)          # list[Path]
    error = pyqtSignal(str)

    def __init__(self, session_dir, model, out_dir):
        super().__init__()
        self._session_dir = session_dir
        self._model = model
        self._out_dir = out_dir

    def run(self):
        try:
            outs = produce_marked_videos(
                self._session_dir, self._model, self._out_dir,
                progress_cb=lambda frac, stage: self.progress.emit(frac, stage),
            )
            self.finished.emit(outs)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, must never crash
            self.error.emit(str(exc))
