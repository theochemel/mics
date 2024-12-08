import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from tqdm import tqdm

from experiments.bottom_looking.signals import pulse_compress_signals
from grid import get_grid_points, get_grid_xy_extent
from sas import get_sas_updates
from signals import demod_signal
from config import Config
from visualize import plot_map_slices_animated



def main():
    config = Config()
    config.grid_min_z = -4

    with open("no-cubes.pkl", "rb") as f:
        exp = pickle.load(f)

    traj = exp["trajectory"]
    rx_pattern = exp["rx_pattern"]
    array = exp["array"]

    grid_x, grid_y, grid_z = get_grid_points(config)
    grid_xy_extent = get_grid_xy_extent(config)
    grid_points = np.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)

    map = np.zeros_like(grid_x, dtype=np.complex128)

    for i in tqdm(range(len(rx_pattern))):
        pose_i = i + 1

        pose = traj.poses[pose_i]

        raw_signals = rx_pattern[i]
        signal_t = config.Ts * np.arange(raw_signals.shape[-1])

        signals = demod_signal(signal_t, raw_signals, config)

        pulses = pulse_compress_signals(signals, config)

        updates = get_sas_updates(grid_points, array.positions + pose.t, signal_t, pulses, config)
        updates = updates.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

        sum_updates = np.sum(updates, axis=0)

        map += sum_updates


    plot_map_slices_animated(map, grid_xy_extent)


if __name__ == "__main__":
    main()

