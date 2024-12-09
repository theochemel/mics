import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from visualize import plot_map_slices_animated
from sas import get_sas_updates
from signals import demod_signal, pulse_compress_signals
from config import Config


with open("lines-0_1ms-zigzag-map.pkl", "rb") as fp:
    map_res = pickle.load(fp)

with open("lines-0_1ms-10x10.pkl", "rb") as fp:
    exp_res = pickle.load(fp)

base_map = map_res["map"]
base_map_weights = map_res["map_weights"]
grid_x = map_res["grid_x"]
grid_y = map_res["grid_y"]
grid_z = map_res["grid_z"]
grid_points = np.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)

traj = exp_res["trajectory"]
rx_pattern = exp_res["rx_pattern"]
array = exp_res["array"]

grid_xy_extent = [
    np.min(grid_x) - 5e-3,
    np.max(grid_x) + 5e-3,
    np.min(grid_y) - 5e-3,
    np.max(grid_y) + 5e-3,
]

plot_map_slices_animated(base_map_weights, grid_xy_extent, traj)
plot_map_slices_animated(base_map / base_map_weights, grid_xy_extent, traj)

rx_pattern_i = len(rx_pattern) - 1
pose_i = rx_pattern_i + 1

config = Config()

x_displacements = np.linspace(-1e-1, 1e-1, 100)
contrasts = np.empty_like(x_displacements)
raw_signals = rx_pattern[rx_pattern_i]

signal_t = config.Ts * np.arange(raw_signals.shape[-1]) - config.chirp_duration / 2

signals = demod_signal(signal_t, raw_signals, config)

pulses = pulse_compress_signals(signals, config)

gt_array_position = traj.poses[pose_i].t

for i in tqdm(range(len(x_displacements))):
    array_position = gt_array_position + np.array([0, x_displacements[i], 0])

    updates = get_sas_updates(grid_points, array.positions + array_position, array_position, signal_t, pulses, config)

    updates = updates.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

    sum_updates = np.sum(updates, axis=0)

    new_map = base_map + sum_updates
    contrast = np.mean(np.sum(np.abs(new_map) ** 2))

    contrasts[i] = contrast

plt.plot(x_displacements, contrasts)
plt.axvline(x=0, c="r")
plt.show()