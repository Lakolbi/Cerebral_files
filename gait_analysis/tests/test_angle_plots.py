import numpy as np

from modules.visualization.angle_plots import plot_joint_angles


def test_returns_one_axis_per_joint():
    fig = plot_joint_angles({}, {}, joints=("hip", "knee", "ankle"))
    assert len(fig.axes) == 3


def test_line_ydata_matches_mean():
    vals = list(np.linspace(0, 60, 101))
    fig = plot_joint_angles({"left_knee": vals}, {"left_knee": [0.0] * 101}, joints=("knee",))
    line = fig.axes[0].lines[0]
    assert line.get_ydata()[-1] == vals[-1]


def test_reuses_provided_figure():
    from matplotlib.figure import Figure
    fig = Figure()
    out = plot_joint_angles({"left_knee": [0.0] * 101}, {}, joints=("knee",), fig=fig)
    assert out is fig                       # draws onto the given figure, returns it
    assert len(fig.axes) == 1
