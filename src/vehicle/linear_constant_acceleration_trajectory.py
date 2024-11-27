import numpy as np
from spatialmath import SE3
from typing import List

from vehicle.trajectory import Trajectory


class LinearConstantAccelerationTrajectory(Trajectory):

    def __init__(self, keyposes: List[SE3], max_velocity: float, acceleration: float, dt: float):
        self._keyposes = keyposes

        times = []
        poses = []
        positions = []
        velocities = []
        accelerations = []
        orientations_rpy = []
        angular_velocities = []

        a = acceleration
        vmax = max_velocity

        for segment_i in range(len(keyposes) - 1):
            pose_start = self._keyposes[segment_i]
            pose_end = self._keyposes[segment_i + 1]

            distance = np.linalg.norm(pose_start.t - pose_end.t)

            vmax_reached = distance > (vmax ** 2) / a

            t_acc = vmax / a

        self._poses = poses
        self._positions = np.array(positions)
        self._velocities = np.array(velocities)
        self._accelerations = np.array(accelerations)
        self._orientations_rpy = np.array(orientations_rpy)
        self._angular_velocities = np.array(angular_velocities)

    @property
    def keyposes(self) -> List[SE3]:
        return self._keyposes

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def time(self) -> np.array:
        return self._time

    @property
    def position_world(self) -> np.array:
        return self._positions

    @property
    def velocity_world(self) -> np.array:
        return self._velocities

    @property
    def acceleration_world(self) -> np.array:
        return self._accelerations

    @property
    def orientation_rpy_world(self) -> np.array:
        return self._orientations_rpy

    @property
    def angular_velocity_world(self) -> np.array:
        return self._angular_velocities


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from visualization.visualize_trajectory import plot_trajectory_xy

    trajectory = LinearConstantAccelerationTrajectory(
        keyposes=[
            SE3.Trans(0.0, 0.0, 0.0),
            SE3.Trans(1.0, 0.0, 0.0),
            SE3.Trans(1.0, 1.0, 0.0),
        ],
        max_velocity=0.1,
        acceleration=0.1,
        dt=0.05,
    )

    fig = plot_trajectory_xy(trajectory)

    fig.show()
    plt.show()
