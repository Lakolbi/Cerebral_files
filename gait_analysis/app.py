"""GUI entry point for the gait analysis application."""
import os
import sys


def _platform_for(current):
    """Choose the Qt platform plugin.

    vispy's GL backend emits GLSL 1.20 shaders. On a Wayland session NVIDIA gives the
    embedded QOpenGLWidget a core-profile EGL context that rejects them ("unsupported
    version 120"), and Qt ignores compatibility-profile requests under EGL. Routing
    through XWayland (xcb) uses GLX, which provides a compatibility context that compiles
    them. 'offscreen' (tests/headless) and any explicit non-wayland choice are preserved.
    """
    if current == "offscreen":
        return "offscreen"
    if current == "" or "wayland" in current:
        return "xcb"
    return current


os.environ["QT_QPA_PLATFORM"] = _platform_for(os.environ.get("QT_QPA_PLATFORM", ""))

import vispy  # noqa: E402

vispy.use(app="pyqt6")
from PyQt6.QtGui import QSurfaceFormat  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402


def configure_opengl():
    """Request a compatibility-profile GL context so vispy's GLSL 1.20 shaders compile.

    Belt-and-suspenders alongside the xcb platform routing; must run before the
    QApplication / first GL widget is created.
    """
    fmt = QSurfaceFormat()
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)


def main():
    configure_opengl()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
