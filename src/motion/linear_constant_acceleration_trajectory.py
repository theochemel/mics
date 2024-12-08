import numpy as np
from spatialmath import SE3
from typing import List, Tuple

from motion.trajectory import Trajectory


class LinearConstantAccelerationTrajectory(Trajectory):

    def __init__(self, keyposes: List[SE3], max_velocity: float, acceleration: float, dt: float):
        self._keyposes = keyposes

        a = acceleration
        vmax = max_velocity

        segment_durations = np.empty((len(keyposes) - 1,))

        for segment_i in range(len(keyposes) - 1):
            pose_start = self._keyposes[segment_i]
            pose_end = self._keyposes[segment_i + 1]

            distance = np.linalg.norm(pose_start.t - pose_end.t)

            vmax_reached = distance > (vmax ** 2) / a
            t_acc = vmax / a
            d_acc = 0.5 * a * t_acc ** 2
            t_const = (distance - 2 * d_acc) / vmax if vmax_reached else 0.0

            t_total = t_const + 2 * t_acc

            segment_durations[segment_i] = t_total

        segment_start_times: np.array = np.concatenate((
            np.array([0]),
            np.cumsum(segment_durations[:-1]),
        ))

        self._duration: float = np.sum(segment_durations)

        self._time: np.array = dt * np.arange(np.ceil(self._duration / dt) + 1)

        poses = []
        positions = []
        velocities = []
        accelerations = []
        orientations_rpy = []
        angular_velocities = []

        segment_i = 0

        for time in self._time:
            if segment_i + 1 < len(segment_start_times) and time >= segment_start_times[segment_i + 1]:
                segment_i += 1

            pose_start = self._keyposes[segment_i]
            pose_end = self._keyposes[segment_i + 1]

            relative_pose = pose_end @ pose_start.inv()

            t = time - segment_start_times[segment_i]

            distance = np.linalg.norm(pose_end.t - pose_start.t)
            direction = (pose_end.t - pose_start.t) / distance

            vmax_reached = distance > (vmax ** 2) / a
            t_acc = vmax / a
            d_acc = 0.5 * a * t_acc ** 2
            t_const = (distance - 2 * d_acc) / vmax if vmax_reached else 0.0

            if t < t_acc:
                d = 0.5 * a * t ** 2
                v = a * t
                acc = a
            elif t < t_const + t_acc:
                d = 0.5 * a * t_acc ** 2 + vmax * (t - t_acc)
                v = vmax
                acc = 0
            else:
                d = 0.5 * a * t_acc ** 2 + vmax * t_const + vmax * (t - t_const - t_acc) - 0.5 * a * (t - t_const - t_acc) ** 2
                v = vmax - a * (t - t_const - t_acc)
                acc = -a

            delta = d / distance

            pose = pose_start.interp(pose_end, delta)
            poses.append(pose)

            positions.append(pose.t)
            orientations_rpy.append(pose.rpy())
            velocities.append(v * direction)
            accelerations.append(acc * direction)

            rot_theta, rot_vec = relative_pose.angvec()

            angular_velocities.append(
                (v / distance) * rot_theta * rot_vec
            )

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
    def poses(self) -> List[SE3]:
        return self._poses

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

    def __getitem__(self, value) -> Tuple[float, SE3]:
        return self._time[value], self._poses[value]

    def __len__(self) -> int:
        return len(self._poses)

    def __iter__(self):
        return self._poses.__iter__()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from visualization.visualize_trajectory import plot_trajectory_traces

    trajectory = LinearConstantAccelerationTrajectory(
        keyposes=[
            SE3.Trans(0.0, 0.0, 0.0),
            SE3.Trans(1.0, 0.0, 0.0),
            SE3.Trans(0.0, 1.0, 0.0),
        ],
        max_velocity=0.1,
        acceleration=0.1,
        dt=0.05,
    )

    fig = plot_trajectory_traces(trajectory)

    fig.show()
    plt.show()
