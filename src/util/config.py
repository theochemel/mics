from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    c: float = 1500

    chirp_fc: float = 50e3
    chirp_bw: float = 50e3
    chirp_duration: float = 2e-4

    @property
    def chirp_K(self) -> float:
        return self.chirp_bw / self.chirp_duration

    @property
    def K(self) -> float:
        return 2 * np.pi * self.chirp_fc / self.c

    @property
    def w(self) -> float:
        return 2 * np.pi * self.chirp_fc

    fs: float = 1e6

    @property
    def Ts(self) -> float:
        return 1 / self.fs

    max_range: float = 15

    @property
    def max_rt_t(self) -> float:
        return (2 * self.max_range) / self.c

    max_visibility_range = 10

    grid_size_xy = 300
    grid_size_z = 2

    grid_resolution_xy = 5e-3

    grid_min_z = -2

    acceleration_white_sigma = 1e-3
    acceleration_walk_sigma = 1e-5
    orientation_white_sigma = 1e-3
    orientation_walk_sigma = 1e-8
