from dataclasses import dataclass
import numpy as np

from motion.trajectory import Trajectory


@dataclass
class IMUMeasurement:
    time: np.array
    acceleration_body: np.array
    orientation_rpy: np.array


class IMU:

    def __init__(self,
                 acceleration_white_sigma: float,
                 acceleration_walk_sigma: float,
                 orientation_white_sigma: float,
                 orientation_walk_sigma: float,
                 ):
        self._acceleration_white_sigma = acceleration_white_sigma
        self._acceleration_walk_sigma = acceleration_walk_sigma
        self._orientation_white_sigma = orientation_white_sigma
        self._orientation_walk_sigma = orientation_walk_sigma

    def measure(self, trajectory: Trajectory) -> IMUMeasurement:
        time = trajectory.time
        poses = trajectory.poses
        acceleration_world = trajectory.acceleration_world
        orientation_rpy = trajectory.orientation_rpy_world

        acceleration_body = []

        for i in range(len(poses)):
            acceleration_body.append(
                poses[i].inv().R @ acceleration_world[i]
            )

        acceleration_body = np.array(acceleration_body)

        acceleration_walk_noise = np.random.normal(loc=0, scale=self._acceleration_walk_sigma, size=acceleration_body.shape)
        acceleration_white_noise = np.random.normal(loc=0, scale=self._acceleration_white_sigma, size=acceleration_body.shape)

        orientation_walk_noise = np.random.normal(loc=0, scale=self._orientation_walk_sigma, size=acceleration_body.shape)
        orientation_white_noise = np.random.normal(loc=0, scale=self._orientation_white_sigma, size=acceleration_body.shape)

        acceleration_body += np.cumsum(acceleration_walk_noise, axis=0) + acceleration_white_noise
        orientation_rpy += np.cumsum(orientation_walk_noise, axis=0) + orientation_white_noise

        return IMUMeasurement(time, acceleration_body, orientation_rpy)


if __name__ == "__main__":
    from spatialmath import SE3
    from motion.linear_constant_acceleration_trajectory import LinearConstantAccelerationTrajectory
    from visualization.visualize_imu import plot_imu_traces
    import matplotlib.pyplot as plt

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

    imu = IMU(
        acceleration_white_sigma=1e-3,
        acceleration_walk_sigma=1e-3,
        orientation_white_sigma=1e-3,
        orientation_walk_sigma=1e-6,
    )

    measurement = imu.measure(trajectory)

    fig = plot_imu_traces(measurement)
    plt.show()
