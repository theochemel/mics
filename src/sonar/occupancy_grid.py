import functools
from typing import Tuple

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from spatialmath import SE3
import torch
import torch.nn.functional as F

from visualization.visualize_map import plot_slices_with_colormap

torch.set_grad_enabled(False)

class OccupancyGridMap:

    def __init__(self, x: int, y: int, z: int, size: float,
                 world_t_map: SE3,
                 device: torch.device):
        self._map = torch.zeros([x, y, z], dtype=torch.complex128, device=device)
        self._size = size
        self._world_t_map = world_t_map
        self._C = 1500
        self._device = device

    @functools.cached_property
    def world_t_grid(self):
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
                        intensity: torch.Tensor,
                        k: float,
                        T_rx: float,
                        C: float,
                        world_t_source: SE3,
                        world_t_sink: SE3,
                        visualization_geometry=None):

        # source_t_world = world_t_source.inv()
        # source_t_world_t = torch.tensor(np.array(source_t_world), device=self._device)
        #
        # sink_t_world = world_t_sink.inv()
        # sink_t_world_t = torch.tensor(np.array(sink_t_world), device=self._device)
        #
        # source_t_grid = (source_t_world_t @ self.world_t_grid.reshape((-1, 4)).T).T
        # source_t_grid = source_t_grid.reshape(self.world_t_grid.shape)
        #
        # sink_t_grid = (sink_t_world_t @ self.world_t_grid.reshape((-1, 4)).T).T
        # sink_t_grid = sink_t_grid.reshape(self.world_t_grid.shape)
        #
        # source_t_grid_norm = source_t_grid[..., :3].norm(dim=-1)
        # sink_t_grid_norm = sink_t_grid[..., :3].norm(dim=-1)

        grid_pos = self.world_t_grid[..., :3]
        source_pos = torch.tensor(world_t_source.t).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        sink_pos = torch.tensor(world_t_sink.t).unsqueeze(0).unsqueeze(1).unsqueeze(2)

        grid_round_trip_range = (grid_pos - source_pos).norm(dim=-1) + (grid_pos - sink_pos).norm(dim=-1)

        # array_t_grid_norm = array_t_grid[..., :3].norm(dim=-1) array_t_grid_unit = array_t_grid / array_t_grid_norm.unsqueeze(-1)

        grid_range_index = (grid_round_trip_range / (T_rx * C)).to(torch.int)

        grid_update_valid = grid_range_index < len(intensity)

        # plt.plot(np.abs(intensity))
        # plt.show()

        # plt.imshow(grid_range_index[:, :, 0])
        # plt.show()

        grid_update = torch.zeros_like(self._map)
        grid_update[grid_update_valid] = \
            intensity[grid_range_index[grid_update_valid]] \
            * torch.exp(-1.0j * k * grid_round_trip_range[grid_update_valid])

        # plt.imshow(np.abs(grid_update[:, :, 0]))
        # plt.show()
        #
        # plt.imshow(np.angle(grid_update[:, :, 0]))
        # plt.show()

        self._map = self._map + grid_update

    def get_map(self):
        return self._map


if __name__ == "__main__":
    dev = torch.device('cuda')
    ogm = OccupancyGridMap(100, 100, 100, 0.1, SE3(), dev)
    gc = ogm.world_t_grid
    print(gc.shape)
