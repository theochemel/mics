import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from visualize import plot_map_slices_animated
from sas import get_sas_updates
from signals import demod_signal, pulse_compress_signals
from config import Config


with open("lines-strip-map.pkl", "rb") as fp:
    map_res = pickle.load(fp)

with open("lines-strip.pkl", "rb") as fp:
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

# plot_map_slices_animated(base_map_weights, grid_xy_extent, traj)
# plot_map_slices_animated(base_map, grid_xy_extent, traj)

# plt.imshow(
#     np.transpose(np.abs(base_map)[:, base_map.shape[1] // 2, :])
# )
# plt.show()

weights = (np.abs(base_map) ** 4)[:, base_map.shape[1] // 2, :]
z_coords = grid_z[:, base_map.shape[1] // 2, :]

avg_height = np.sum(weights * z_coords, axis=-1) / np.sum(weights, axis=-1)

plt.imshow(
    np.flip(np.transpose(np.abs(base_map)[:, base_map.shape[1] // 2, :]), 0),
    extent=[-0.6, 0.6, -1, -0.75],
)
# plt.plot(np.linspace(-0.6, 0.6, len(avg_height)), avg_height, c="r")
plt.show()

config = Config()

start_i = len(rx_pattern) // 2
end_i = len(rx_pattern) // 2 + 30

dilations = np.linspace(-1e-2, 1e-2, 10)
contrasts = np.empty_like(dilations)

maps = []


def wrap2pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

for dilation_i, dilation in tqdm(enumerate(dilations)):
    gt_start_array_position = traj.poses[start_i].t

    update_map = np.zeros_like(base_map)

    for pose_i in range(start_i, end_i):
        gt_array_position = traj.poses[pose_i].t
        array_position = gt_array_position + np.array([0, dilation, 0])

        rx_pattern_i = pose_i - 1
        raw_signals = rx_pattern[rx_pattern_i]

        signal_t = config.Ts * np.arange(raw_signals.shape[-1]) - config.chirp_duration / 2

        signals = demod_signal(signal_t, raw_signals, config)

        pulses = pulse_compress_signals(signals, config)

        updates = get_sas_updates(grid_points, array.positions + array_position, array_position, signal_t, pulses, config)

        updates = updates.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

        sum_updates = np.sum(updates, axis=0)

        update_map += sum_updates

    new_map = base_map + update_map
    contrast = np.mean(
        np.sum(
            np.abs(
                new_map[
                    :
                    new_map.shape[1] // 2 - 1 : new_map.shape[1] // 2,
                    :,
                ]
            ) ** 2
        )
    )

    maps.append(new_map)

    # plt.imshow(np.abs(new_map[:, :, 0]))
    # plt.show()

    contrasts[dilation_i] = contrast

    # phase_error = np.mean(wrap2pi(np.angle(base_map) - np.angle(sum_updates)))
    # weighted_phase_error = np.mean(np.abs(base_map) * wrap2pi(np.angle(base_map) - np.angle(sum_updates)))

    # phase_errors[i] = phase_error
    # weighted_phase_errors[i] = weighted_phase_error

plt.plot(dilations, contrasts)
plt.axvline(x=0, c="r")
plt.show()

maps = np.array(maps)
maps = np.abs(maps)

vmax = np.max(maps)
vmin = np.min(maps)

for i in range(maps.shape[0]):
    plt.imshow(
        np.flip(np.transpose(maps[i, :, base_map.shape[1] // 2, :]), 0),
        extent=[-0.6, 0.6, -1, -0.75],
    )
    plt.show()
    # img_display = ax.imshow(maps[0, :, :, 0], vmin=vmin, vmax=vmax)

# def animate(frame_i):
#     img_display.set_data(maps[frame_i, :, :, 0])
#     ax.set_title(f"{frame_i}")
#     return [img_display]
#
#
# import matplotlib
# anim = matplotlib.animation.FuncAnimation(fig, animate, frames=maps.shape[0], interval=500)

plt.show()

