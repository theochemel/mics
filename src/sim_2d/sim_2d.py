from typing import Callable, Tuple, List

import numpy as np

from motion.imu import IMU, IMUMeasurement
from motion.trajectory import Trajectory
from util.config import Config
from util.signals import modulated_chirp


class World:

    def __init__(self, config: Config):
        self._targets: np.ndarray = np.zeros((0,2), dtype=np.float64)
        self._c = config

    def add_targets(self, targets: np.ndarray) -> None:
        self._targets = np.concatenate((self._targets, targets.astype(np.float64)), axis=0)

    @property
    def targets(self):
        return self._targets

    def get_signal(self,
                   pose: np.ndarray,
                   signal_fn: Callable[[np.ndarray], np.ndarray],
                   signal_t: np.ndarray) -> np.ndarray:
        return_signal = np.zeros_like(signal_t, dtype=np.complex128)

        for target in self._targets:
            target_range = np.linalg.norm(target - pose)

            if target_range > self._c.max_range:
                continue

            target_rt_t = (2 * target_range) / self._c.C

            return_signal += signal_fn(signal_t - target_rt_t)

        return return_signal # * np.exp(-2.0j * np.pi * self._c.chirp_fc * signal_t)


def make_forest_targets(extent: Tuple[float, float, float, float], nx: int, ny: int, sigma: float = 0):
    minx, maxx, miny, maxy = extent
    target_points_x = np.linspace(minx, maxx, nx, endpoint=True)
    target_points_y = np.linspace(miny, maxy, ny, endpoint=True)
    target_points_y, target_points_x = np.meshgrid(target_points_y, target_points_x, indexing="ij")
    target_points_y = target_points_y.flatten()
    target_points_x = target_points_x.flatten()
    target_points = np.stack((target_points_x, target_points_y), axis=-1)
    target_points += np.random.normal(loc=0, scale=sigma, size=target_points.shape)
    return target_points


def make_sine_targets():
    t = np.linspace(0, 1, 50)
    target_points_x = 1 + t * 8
    target_points_y = 5 + np.sin(2 * np.pi * t)
    target_points = np.stack((target_points_x, target_points_y), axis=-1)
    return target_points


def make_corner_target():
    target_points_x = np.concatenate((np.linspace(5, 9, 50), 9 * np.ones((50,))))
    target_points_y = np.concatenate((9 * np.ones((50,)), np.linspace(5, 9, 50)))
    target_points = np.stack((target_points_x, target_points_y), axis=-1)
    return target_points


class Sim2d:

    def __init__(self, world: World,
                 trajectory: Trajectory,
                 config: Config):
        self._world = world
        self._traj = trajectory
        self._imu = IMU.from_config(config)
        self._c = config

        self._signal_t = self._c.Ts * np.arange(int(self._c.max_rt_t / self._c.Ts))

    def get_imu_measurements(self) -> IMUMeasurement:
        return self._imu.measure(self._traj)

    def get_signal_at_pose(self, pose_idx):
        pose2d = self._traj.poses[pose_idx].t[:2]

        signal = self._world.get_signal(pose2d,
                                       lambda t: modulated_chirp(t, self._c),
                                       self._signal_t)

        return signal
