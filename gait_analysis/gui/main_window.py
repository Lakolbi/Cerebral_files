"""Main window: three-tab shell — Анализ → Сравнение / Визуализация."""
from PyQt6.QtWidgets import QMainWindow, QTabWidget

from gui.panels.analyze_panel import AnalyzePanel
from gui.panels.compare_panel import ComparePanel
from gui.panels.viz_panel import VizPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Analysis")
        self.resize(1000, 750)
        self.analyze = AnalyzePanel()
        self.compare = ComparePanel()
        self.viz = VizPanel()
        tabs = QTabWidget()
        tabs.addTab(self.analyze, "Анализ")
        tabs.addTab(self.compare, "Сравнение")
        tabs.addTab(self.viz, "Визуализация")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready")
        self.analyze.analysis_done.connect(self.viz.set_data)
        self.analyze.analysis_done.connect(
            lambda *_: self.statusBar().showMessage("Analysis complete"))
        self.compare.comparison_done.connect(
            lambda *_: self.statusBar().showMessage("Comparison complete"))

    def closeEvent(self, event):
        # AnalyzePanel runs both a pipeline thread and a video-overlay thread;
        # ComparePanel runs a comparison thread. Quit any that are still running
        # so we never destroy a live QThread on close.
        threads = {self.analyze: ("_thread", "_overlay_thread"),
                   self.compare: ("_thread",)}
        for panel, attrs in threads.items():
            for attr in attrs:
                thread = getattr(panel, attr, None)
                if thread is not None and thread.isRunning():
                    thread.quit()
                    thread.wait()
        super().closeEvent(event)
