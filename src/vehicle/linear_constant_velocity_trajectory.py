import numpy as np
from spatialmath import SE3
from typing import List

from vehicle.trajectory import Trajectory


class LinearConstantVelocityTrajectory(Trajectory):

    def __init__(self, keyposes: List[SE3], velocity: float, dt: float):
        self._keyposes = keyposes

        segment_distances: np.array = np.array([
            np.linalg.norm(self._keyposes[i + 1].t - self._keyposes[i].t)
            for i in range(len(self._keyposes) - 1)
        ])

        segment_durations: np.array = segment_distances / velocity
        segment_start_times: np.array = np.concatenate((
            np.array([0]),
            np.cumsum(segment_durations[:-1]),
        ))

        self._duration: float = np.sum(segment_durations)

        self._time: np.array = dt * np.arange(np.ceil(self._duration / dt) + 1)

        segment_i = 0

        poses = []
        positions = []
        velocities = []
        accelerations = []
        orientations_rpy = []
        angular_velocities = []

        for time in self._time:
            if segment_i + 1 < len(segment_start_times) and time >= segment_start_times[segment_i + 1]:
                segment_i += 1

            pose_start = self._keyposes[segment_i]
            pose_end = self._keyposes[segment_i + 1]

            delta = (time - segment_start_times[segment_i]) / segment_durations[segment_i]

            pose = pose_start.interp(pose_end, delta)

            poses.append(pose)

            positions.append(pose.t)
            orientations_rpy.append(pose.rpy())
            velocities.append((pose_end.t - pose_start.t) / segment_durations[segment_i])
            accelerations.append(np.zeros((3,)))

            relative_pose = pose_end @ pose_start.inv()

            theta, vec = relative_pose.angvec()

            angular_velocities.append((theta * vec) / segment_durations[segment_i])

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

    trajectory = LinearConstantVelocityTrajectory(
        keyposes=[
            SE3.Trans(0.0, 0.0, 0.0),
            SE3.Trans(1.0, 0.0, 0.0),
            SE3.Trans(1.0, 1.0, 0.0),
        ],
        velocity=0.1,
        dt=0.05,
    )

    fig = plot_trajectory_xy(trajectory)

    fig.show()
    plt.show()
