"""Qt canvas hosting the joint-angle matplotlib figure."""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from modules.visualization.angle_plots import plot_joint_angles


class PlotWidget(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(figsize=(8, 3.2))
        super().__init__(fig)
        plot_joint_angles({}, {}, fig=self.figure)

    def render_angles(self, angles_mean, angles_std):
        plot_joint_angles(angles_mean, angles_std, fig=self.figure)
        self.draw()

    @property
    def current_figure(self):
        return self.figure
