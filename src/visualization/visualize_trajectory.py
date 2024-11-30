import numpy as np
import matplotlib.pyplot as plt

from vehicle.trajectory import Trajectory


def plot_trajectory_xy(trajectory: Trajectory) -> plt.Figure:
    fig, ax = plt.subplots()

    ax.plot(trajectory.position_world[:, 0], trajectory.position_world[:, 1])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    return fig


def plot_trajectory_traces(trajectory: Trajectory) -> plt.Figure:
    fig, axs = plt.subplots(3)

    axs[0].plot(trajectory.time, trajectory.position_world[:, 0], label="X")
    axs[0].plot(trajectory.time, trajectory.position_world[:, 1], label="Y")
    axs[0].plot(trajectory.time, trajectory.position_world[:, 2], label="Z")
    axs[1].set_title("Position")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(trajectory.time, trajectory.velocity_world[:, 0], label="X")
    axs[1].plot(trajectory.time, trajectory.velocity_world[:, 1], label="Y")
    axs[1].plot(trajectory.time, trajectory.velocity_world[:, 2], label="Z")
    axs[1].set_title("Velocity")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(trajectory.time, trajectory.acceleration_world[:, 0], label="X")
    axs[2].plot(trajectory.time, trajectory.acceleration_world[:, 1], label="Y")
    axs[2].plot(trajectory.time, trajectory.acceleration_world[:, 2], label="Z")
    axs[2].set_title("Acceleration")
    axs[2].legend()
    axs[2].grid()

    return fig