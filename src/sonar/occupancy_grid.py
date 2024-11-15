import functools
from typing import Tuple

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from spatialmath import SE3

from sonar.phased_array import RectangularArray
from sonar.utils import BarkerCode, FMBarker


class OccupancyGridMap:

    def __init__(self, x: int, y: int, z: int, size: float,
                 world_t_map: SE3,
                 code: BarkerCode,
                 array: RectangularArray):
        self._map = np.zeros((x, y, z), dtype=np.float64)
        self._size = size
        self._world_t_map = world_t_map
        self._C = 1500
        self._code = code
        self._array = array

    @functools.cached_property
    def _grid_coordinates(self):
        nx, ny, nz = self._map.shape
        x = np.linspace(0, nx)*self._size
        y = np.linspace(0, ny)*self._size
        z = np.linspace(0, nz)*self._size
        grid_local = np.transpose(np.meshgrid(x, y, z))
        world_t_grid = self._world_t_map.inv() * grid_local
        return world_t_grid

    def add_measurement(self,
                        rx_pattern: np.array,
                        T: float,
                        steering_dir: np.array,
                        world_t_vehicle: SE3,
                        barker_code: np.array,
                        barker_frequencies: Tuple[float, float],
                        T_bit: float,
                        receive):

        # compute point locations relative to array and corresponding delays
        world_t_grid = self._grid_coordinates
        vehicle_t_grid = world_t_vehicle.inv() * world_t_grid
        grid_distances = np.linalg.norm(vehicle_t_grid, -1)
        grid_delays = 2 * grid_distances / self._C
        grid_delays_samples = np.round(grid_delays / T).astype(np.int64)

        # compute signals for each element
        f_low, f_high = barker_frequencies
        k_low = 2*pi * f_low / self._C
        k_high = 2*pi * f_high / self._C
        signals = self._array.beamform_receive(np.array([k_low, k_high]), steering_dir, rx_pattern, T)

        # correlate with barker code
        signals = np.transpose(signals, (1, 0, 2))  # (n_steering, n_k, n_samples)
        correlation = []
        n_steering = signals.shape[0]
        for i_steering in range(n_steering):
            correlation.append(self._code.correlate(signals[i_steering]))
        correlation = np.array(correlation)

        # TODO: threshold intensity

        # compute array gain for every point for every steering angle
        # TODO: Using only f_low. Better to compute correlations and gains for f_high and f_low separately
        looking_dir = vehicle_t_grid / grid_distances
        gains = self._array.get_gain(steering_dir, looking_dir.reshape(-1, 3), np.array([f_low]))  # TODO: check reshape
        nx, ny, nz, _ = world_t_grid.shape
        gains = np.reshape((n_steering, nx, ny, nz))

        grid_delays_samples = np.clip(grid_delays_samples, 0, correlation.shape[1])

        for i_steering in range(correlation.shape[0]):
            self._map += 2*np.log(grid_distances)
            self._map += gains[i_steering] / 20
            self._map += correlation[i_steering][grid_delays_samples]

