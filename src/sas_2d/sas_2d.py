import numpy as np
from matplotlib import pyplot as plt

from util.util import wrap2pi
from util.config import Config
from util.signals import *


class SAS:

    def __init__(self, config: Config):
        self._c = config

        # Initialize map
        grid_x = self._c.grid_resolution_xy * np.arange(self._c.grid_size_xy) # -
                 # (self._c.grid_resolution_xy * self._c.grid_size_xy / 2))

        grid_y, grid_x = np.meshgrid(np.flip(grid_x), grid_x, indexing="ij")
        self._grid_pos = np.stack((grid_x, grid_y), axis=-1)
        self._map = np.zeros(self._grid_pos.shape[0:2], dtype=np.complex128)

    @property
    def map(self) -> np.ndarray:
        return self._map

    @property
    def grid_pos(self):
        return self._grid_pos

    def update_map(self, pose: np.ndarray, pulse: np.ndarray, vis=False) -> None:
        grid_range = np.linalg.norm(self._grid_pos - pose[np.newaxis, np.newaxis], axis=-1)
        grid_rt_t = (2.0 * grid_range) / self._c.C

        k = grid_rt_t / self._c.Ts
        k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
        k_a = k - k_i  # Fractional parts

        # Ensure we don't go out of bounds for the upper index
        k_i_plus_1 = np.clip(k_i + 1, 0, len(pulse) - 1)  # Upper bounds (clipped)

        # Perform linear interpolation
        interp_pulse = (1 - k_a) * pulse[k_i] + k_a * pulse[k_i_plus_1]

        update = interp_pulse * np.exp((2.0j * np.pi * self._c.chirp_fc / self._c.C) * (2.0 * grid_range))

        if vis:
            width, height = 4, 4
            fig, ax = plt.subplots(figsize=(width, height))
            ax.imshow(np.abs(update), extent=self._c.grid_extent_xy)
            fig.savefig("2d_map_update.png", dpi=600, bbox_inches='tight')

        self._map += update
