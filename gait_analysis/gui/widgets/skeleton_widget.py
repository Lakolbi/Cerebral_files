"""vispy 3D skeleton widget embedded in Qt (geometry from the GL-free core)."""
import numpy as np
import vispy

vispy.use(app="pyqt6")
from vispy import scene  # noqa: E402

from modules.visualization.skeleton_3d import (  # noqa: E402
    SKELETON_LANDMARKS,
    bounds,
    frame_points,
    segment_lines,
)


class SkeletonWidget:
    """Wraps a vispy SceneCanvas. Embed `self.native` (a QWidget) into a layout."""

    def __init__(self):
        self.canvas = scene.SceneCanvas(keys="interactive", show=False, bgcolor="white")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45)
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.lines = scene.visuals.Line(parent=self.view.scene, color="black",
                                        width=2, connect="segments")
        self._df = None

    @property
    def native(self):
        return self.canvas.native

    @property
    def n_frames(self):
        return 0 if self._df is None else len(self._df)

    def set_data(self, df):
        self._df = df
        self.set_frame(0)
        # Frame the whole walk once, at a stable scale, so the skeleton does not
        # translate out of view as the subject moves across the capture volume.
        bb = bounds(df)
        if bb is not None:
            (x0, y0, z0), (x1, y1, z1) = bb
            self.view.camera.set_range(x=(x0, x1), y=(y0, y1), z=(z0, z1), margin=0.15)
        else:
            self.view.camera.set_range()

    def set_frame(self, i):
        if self._df is None or not (0 <= i < len(self._df)):
            return
        pts = frame_points(self._df, i, names=SKELETON_LANDMARKS)
        if pts:
            coords = np.array(list(pts.values()), dtype=np.float32)
            self.markers.set_data(coords, size=8, face_color="red")
        segs = segment_lines(pts)
        if segs:
            flat = np.array([p for seg in segs for p in seg], dtype=np.float32)
            self.lines.set_data(pos=flat)
