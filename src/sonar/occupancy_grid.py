import functools
from typing import Tuple

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from spatialmath import SE3
import torch
import torch.nn.functional as F

class OccupancyGridMap:

    def __init__(self, x: int, y: int, z: int, size: float,
                 world_t_map: SE3,
                 device: torch.device):
        self._map = torch.zeros([x, y, z], dtype=torch.complex64, device=device)
        self._size = size
        self._world_t_map = world_t_map
        self._C = 1500
        self._device = device

    @functools.cached_property
    def _world_t_grid(self):
        nx, ny, nz = self._map.shape
        x = torch.linspace(0, nx-1, nx, device=self._device, dtype=torch.double) * self._size
        y = torch.linspace(0, ny-1, ny, device=self._device, dtype=torch.double) * self._size
        z = torch.linspace(0, nz-1, nz, device=self._device, dtype=torch.double) * self._size
        yy, xx, zz = torch.meshgrid(y, x, z, indexing='ij')
        grid_local = torch.stack([xx, yy, zz, torch.ones_like(xx)], dim=-1)  # Shape: (ny, nx, nz, 4)
        grid_local_flat = grid_local.reshape(-1, 4)
        world_t_map_t = torch.tensor(np.array(self._world_t_map), device=self._device)
        world_t_grid_flat = (world_t_map_t @ grid_local_flat.T).T
        world_t_grid = world_t_grid_flat.reshape(ny, nx, nz, 4)
        return world_t_grid

    def add_measurement(self,
                        phi,
                        k: float,
                        directivity: torch.Tensor,
                        world_t_array: SE3):

        array_t_world = world_t_array.inv()
        array_t_world_t = torch.tensor(np.array(array_t_world), device=self._device)

        array_t_grid = (array_t_world_t @ self._world_t_grid.reshape((-1, 4)).T).T
        array_t_grid = array_t_grid.reshape(self._world_t_grid.shape)
        del array_t_world_t

        array_t_grid_norm = array_t_grid[..., :3].norm(dim=-1)
        array_t_grid_unit = array_t_grid / array_t_grid_norm
        del array_t_grid

        az = torch.atan2(array_t_grid_unit[..., 1], array_t_grid_unit[..., 0])
        el = torch.arcsin(array_t_grid_unit[..., 2])
        del array_t_grid_unit

        directivity_lookup = torch.cat((az, el), dim=-1).flatten(0, 2).reshape((1, 1, -1, 2))
        directivity = directivity.reshape((1, 1) + directivity.shape)
        gain = F.grid_sample(directivity, directivity_lookup)[0, 0, 0, :]
        gain = gain.unflatten(0, self._world_t_grid.shape[:3])

        psi = phi * torch.exp(-2j * np.pi * k * array_t_grid_norm) * gain * array_t_grid_norm**2

        self._map += psi

    def get_map(self):
        return self._map


if __name__ == "__main__":
    dev = torch.device('cuda')
    ogm = OccupancyGridMap(100, 100, 100, 0.1, SE3(), dev)
    gc = ogm._world_t_grid
    print(gc.shape)
