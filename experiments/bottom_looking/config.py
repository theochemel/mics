from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    c: float = 1500

    chirp_fc: float = 50e3
    chirp_bw: float = 50e3
    chirp_duration: float = 1e-3

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

    grid_size_xy = 512
    grid_size_z = 1

    grid_resolution_xy = 1e-2
    grid_resolution_z = 1e-2

    grid_min_z = 0

    @property
    def spatial_f(self) -> float:
        return self.c / (2 * self.grid_resolution_xy)

    fov: float = np.deg2rad(60)