import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

from signals import get_demod_signals, pulse_compress_signals
from grid import get_grid_points, get_grid_xy_extent
from sas import get_sas_updates
from config import Config

import pickle
import os


def main():
    config = Config()

    with open("terrain.pkl", "rb") as f:
        terrain = pickle.load(f)

    terrain_x = terrain["terrain_x"]
    terrain_y = terrain["terrain_y"]
    terrain_z = terrain["terrain_z"]
    targets = np.stack((terrain_x.flatten(), terrain_y.flatten(), terrain_z.flatten()), axis=-1)

    # targets = np.array([
    #     [0.0, 0.0, 0.0],
    #     [1.0, 0.0, 0.1],
    #     [2.0, 2.0, 0.1],
    # ])

    # if os.path.exists("cache.pkl"):
    #     with open("cache.pkl", "rb") as f:
    #         cache = pickle.load(f)
    #
    #     sinks = cache["sinks"]
    #     signal_t = cache["signal_t"]
    #     signals = cache["signals"]
    # else:
    array_x = 0.01 * np.arange(10)
    array_y = 0.01 * np.arange(10)
    array_x, array_y = np.meshgrid(array_x, array_y, indexing="xy")
    array_x = array_x.flatten()
    array_y = array_y.flatten()
    array_z = np.zeros_like(array_x)

    array = np.stack((array_x, array_y, array_z), axis=-1)

    grid_x, grid_y, grid_z = get_grid_points(config)
    grid_xy_extent = get_grid_xy_extent(config)
    grid_points = np.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)

    map = np.zeros_like(grid_x, dtype=np.complex128)

    for i in range(10):
        position = np.array([0.1 * i, 0.0, 5.0])

        sinks = array + position

        signal_t, raw_signals = get_demod_signals(sinks, targets, config)

        plt.plot(np.real(raw_signals[0]))
        plt.plot(np.imag(raw_signals[0]))
        plt.show()

        signals = pulse_compress_signals(raw_signals, config)

        # with open("cache.pkl", "wb") as f:
        #     pickle.dump({"sinks": sinks, "signal_t": signal_t, "signals": signals}, f)


        updates = get_sas_updates(grid_points, sinks, signal_t, signals, config)

        updates = updates.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

        sum_update = np.sum(updates, axis=0)

        map += sum_update



if __name__ == "__main__":
    main()
