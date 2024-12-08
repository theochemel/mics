import numpy as np

from config import Config


def get_grid_points(config: Config):
    grid_x = np.flip((
            config.grid_resolution_xy
            * (np.arange(config.grid_size_xy) - (config.grid_size_xy - 1) / 2)
    ))
    grid_y = np.flip(grid_x)
    grid_z = config.grid_resolution_z * np.arange(config.grid_size_z) + config.grid_min_z

    grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")

    return grid_x, grid_y, grid_z

def get_grid_xy_extent(config: Config):
    pixel = config.grid_resolution_xy

    return [
        -config.grid_resolution_xy * (config.grid_size_xy / 2) - pixel / 2,
        config.grid_resolution_xy * (config.grid_size_xy / 2) + pixel / 2,
        -config.grid_resolution_xy * (config.grid_size_xy / 2) - pixel / 2,
        config.grid_resolution_xy * (config.grid_size_xy / 2) + pixel / 2,
    ]

