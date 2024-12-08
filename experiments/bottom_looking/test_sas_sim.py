import pickle

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation
from tqdm import tqdm

from experiments.bottom_looking.signals import pulse_compress_signals
from grid import get_grid_points, get_grid_xy_extent
from sas import get_sas_updates
from signals import demod_signal, chirp
from config import Config
from visualize import plot_map_slices_animated



def main():
    config = Config()

    with open("cubes-only.pkl", "rb") as f:
        exp = pickle.load(f)

    traj = exp["trajectory"]
    tx_pattern = exp["tx_pattern"]
    rx_pattern = exp["rx_pattern"]
    array = exp["array"]

    source = np.array([0,0,0])

    grid_x, grid_y, grid_z = get_grid_points(config)
    grid_xy_extent = get_grid_xy_extent(config)
    grid_points = np.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)

    map = np.zeros_like(grid_x, dtype=np.complex128)

    for i in tqdm(range(len(rx_pattern))):
        pose_i = i + 1

        pose = traj.poses[pose_i]

        current_array_positions = (array.positions + pose.t)#[4:5]
        current_source_position = source + pose.t

        raw_signals = rx_pattern[i]#[4:5]
        signal_t = config.Ts * np.arange(raw_signals.shape[-1]) - config.chirp_duration / 2

        signals = demod_signal(signal_t, raw_signals, config)

        # tx_pattern_t = config.Ts * np.arange(len(tx_pattern.baseband))

        # plt.plot(tx_pattern_t, tx_pattern.baseband, label="real")
        # plt.plot(tx_pattern_t, np.real(sp.signal.hilbert(tx_pattern.baseband)), label="hilbert real")
        # plt.plot(tx_pattern_t, np.imag(sp.signal.hilbert(tx_pattern.baseband)), label="hilbert imag")
        # plt.plot(tx_pattern_t, np.real(chirp(tx_pattern_t - config.chirp_duration / 2, config)), label="analytic real")
        # plt.plot(tx_pattern_t, np.imag(chirp(tx_pattern_t - config.chirp_duration / 2, config)), label="analytic imag")
        # plt.legend()
        # plt.show()

        # demod_tx = demod_signal(config.Ts * np.arange(len(tx_pattern.baseband)), tx_pattern.baseband, config)
        # pulse_tx = pulse_compress_signals(demod_tx, config)
        #
        # plt.plot(np.real(pulse_tx[0]))
        # plt.plot(np.imag(pulse_tx[0]))
        # plt.show()

        # plt.plot(config.c * signal_t / 2, np.real(signals[0]))
        # plt.plot(config.c * signal_t / 2, np.imag(signals[0]))
        # plt.show()

        pulses = pulse_compress_signals(signals, config)

        updates = get_sas_updates(grid_points, current_array_positions, source, signal_t, pulses, config)

        updates = updates.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

        sum_updates = np.sum(updates, axis=0)

        map += sum_updates


    plt.imshow(np.abs(map[:, :, 0]), extent=grid_xy_extent)
    plt.imshow(np.log(np.abs(map[:, :, 0])), cmap='viridis', aspect='equal')
    # plt.scatter(current_array_positions[:, 0], current_array_positions[:, 1])
    plt.show()

    plot_map_slices_animated(map, grid_xy_extent)


if __name__ == "__main__":
    main()

