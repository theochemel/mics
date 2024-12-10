import pickle

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation
from tqdm import tqdm

from signals import pulse_compress_signals
from grid import get_grid_points, get_grid_xy_extent
from sas import get_sas_updates, get_sas_weights
from signals import demod_signal, chirp
from config import Config
from visualize import plot_map_slices_animated, plot_map_slices



def main():
    config = Config()

    with open("single-cube.pkl", "rb") as f:
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
    map_weights = np.zeros_like(grid_x, dtype=np.float64)

    positions = np.array([pose.t for pose in traj.poses])

    for i in tqdm(range(len(rx_pattern) )):
        pose_i = i + 1

        pose = traj.poses[pose_i]

        current_array_positions = (array.positions + pose.t)[1:2]
        current_source_position = source + pose.t

        raw_signals = rx_pattern[i][1:2]
        # unshifted_t = config.Ts * np.arange(raw_signals.shape[-1])
        # plt.plot(unshifted_t, raw_signals[0])
        # plt.show()

        signal_t = config.Ts * np.arange(raw_signals.shape[-1])
        signals = demod_signal(signal_t, raw_signals, config)

        signal_t_shifted = signal_t + config.chirp_duration / 2

        for i in range(signals.shape[0]):
            signals[i] = np.interp(signal_t_shifted, signal_t, signals[i])


        # plt.plot(signal_t, np.real(signals[0]))
        # plt.plot(signal_t, np.imag(signals[0]))
        # plt.show()
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
        # plt.plot(signal_t, np.real(pulses[0]))
        # plt.plot(signal_t, np.imag(pulses[0]))
        # plt.show()
        # # pulses *= signal_t ** 4
        # pulses *= 1e9


        # for signal in signals:
        #     axs[0].plot(np.real(signal))
        #
        # fig, axs = plt.subplots(3, sharex=True)

        # for pulse in pulses:
        #     plt.plot(1500 * (signal_t / 2), np.abs(pulse))
        #
        # plt.show()

        updates = get_sas_updates(grid_points, current_array_positions, current_source_position, signal_t, pulses, config)

        updates = updates.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))
        sum_updates = np.sum(updates, axis=0)
        map += sum_updates

        weights = get_sas_weights(grid_points, current_array_positions, current_source_position, signal_t, pulses, config)
        weights = weights.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))
        sum_weights = np.sum(weights, axis=0)
        map_weights += sum_weights

        if not i % 10:
            extent = [np.min(grid_x), np.max(grid_x), np.min(grid_y), np.max(grid_y)]
            plot_map_slices(map, grid_z, extent, positions)

        # plt.imshow(np.abs(sum_updates[:, :, 0]), extent=grid_xy_extent)
        # plt.show()
        # plt.imshow(np.abs(map[:, :, 0]), extent=grid_xy_extent)
        # plt.show()

        # if i % 30 == 0:
        #     plot_map_slices(map, grid_z, grid_xy_extent)

    # plt.imshow(np.abs(map[:, :, 0]), extent=grid_xy_extent)
    # plt.imshow(np.log(np.abs(map[:, :, 0])), cmap='viridis', aspect='equal')
    # plt.scatter(current_array_positions[:, 0], current_array_positions[:, 1])
    # plt.show()

    with open("lines-0_1ms-zigzag-map.pkl", "wb") as f:
        pickle.dump(
            { "grid_x": grid_x, "grid_y": grid_y, "grid_z": grid_z, "map": map, "map_weights": map_weights },
            f
        )


if __name__ == "__main__":
    main()

