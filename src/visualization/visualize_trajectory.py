import numpy as np
import matplotlib.pyplot as plt

from vehicle.trajectory import Trajectory


def plot_trajectory_xy(trajectory: Trajectory) -> plt.Figure:
    fig, ax = plt.subplots()

    ax.plot(trajectory.position_world[:, 0], trajectory.position_world[:, 1])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    return fig