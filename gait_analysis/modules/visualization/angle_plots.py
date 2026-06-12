"""Matplotlib figures for joint-angle curves over the gait cycle (OO API, no pyplot)."""
import numpy as np
from matplotlib.figure import Figure


def plot_joint_angles(angles_mean, angles_std, joints=("hip", "knee", "ankle"),
                      norm_corridor=None, fig=None):
    """Build/refresh a Figure: one subplot per joint, left/right mean +- STD over 0-100% cycle.

    angles_mean / angles_std : {"<side>_<joint>": [101 values]} (e.g. "left_knee").
    norm_corridor : optional {"<joint>": (low[101], high[101])} normative band.
    fig : optional existing Figure to draw onto (cleared first); a new one is created if None.
    Returns the Figure.
    """
    if fig is None:
        fig = Figure(figsize=(4 * len(joints), 3.2))
    else:
        fig.clear()
    axes = fig.subplots(1, len(joints), squeeze=False)[0]
    x = np.linspace(0, 100, 101)
    for ax, joint in zip(axes, joints):
        for side, color in (("left", "tab:blue"), ("right", "tab:red")):
            mean = angles_mean.get(f"{side}_{joint}")
            if mean is None:
                continue
            mean = np.asarray(mean, float)
            ax.plot(x, mean, color=color, label=side)
            std = angles_std.get(f"{side}_{joint}")
            if std is not None:
                std = np.asarray(std, float)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        if norm_corridor and joint in norm_corridor:
            low, high = norm_corridor[joint]
            ax.fill_between(x, low, high, color="tab:green", alpha=0.15, label="norm")
        ax.set_title(joint)
        ax.set_xlabel("% gait cycle")
        ax.set_ylabel("angle (deg)")
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
