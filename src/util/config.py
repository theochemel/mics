from dataclasses import dataclass
from typing import Tuple

import numpy as np

@dataclass
class Config:
    C: float = 1500

    chirp_fc: float = 75e3
    chirp_bw: float = 50e3
    chirp_duration: float = 1e-3

    fs: float = 1e6

    max_visibility_range = 10
    max_range: float = 15

    grid_size_xy = 1000
    grid_size_z = 2

    grid_resolution_xy = 1e-2

    acceleration_white_sigma = 1e-3
    acceleration_walk_sigma = 1e-5
    orientation_white_sigma = 1e-3
    orientation_walk_sigma = 1e-8

    @property
    def chirp_K(self) -> float:
        return self.chirp_bw / self.chirp_duration

    @property
    def K(self) -> float:
        return 2 * np.pi * self.chirp_fc / self.C

    @property
    def w(self) -> float:
        return 2 * np.pi * self.chirp_fc

    @property
    def Ts(self) -> float:
        return 1 / self.fs

    @property
    def max_rt_t(self) -> float:
        return (2 * self.max_range) / self.C

    @property
    def grid_extent_xy(self) -> Tuple[float, float, float, float]:
        side = self.grid_size_xy * self.grid_resolution_xy
        return (0, side, 0, side)
