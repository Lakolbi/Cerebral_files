"""Force a headless Qt platform before pytest-qt creates the QApplication."""
import os

# Force (not setdefault): a desktop shell may export QT_QPA_PLATFORM=wayland/xcb,
# which would make GUI tests target a real display non-deterministically.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
