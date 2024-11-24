import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize


C = 1500

# l_m = 1e-1
# f_m = C / l_m
# w_m = 2 * np.pi * f_m
# k_m = w_m / C

chirp_fc = 75e3
chirp_bw = 50e3
chirp_duration = 1e-3
chirp_K = chirp_bw / chirp_duration

chirp_fhi = chirp_fc + chirp_bw / 2

fs = 4 * chirp_fhi
Ts = 1 / fs

l_m = C / chirp_fhi

max_range = 10
max_rt_t = (2 * max_range) / C

target_points = np.array([
    [3, 3],
    [2, 3],
    [3, 2],
    [8, 8],
])

gt_traj_x = (l_m / 2) * np.arange(8 / (l_m / 2)) + 1
gt_traj_y = np.full_like(gt_traj_x, fill_value=1)
gt_traj = np.stack((gt_traj_x, gt_traj_y), axis=-1)

noisy_traj = gt_traj + np.random.normal(loc=0, scale=l_m / 4, size=gt_traj.shape)

grid_width = 10
grid_height = 10
grid_size = 1e-1

grid_x = grid_size * np.arange(int(grid_width / grid_size)) + grid_size / 2
grid_y = grid_size * np.arange(int(grid_height / grid_size)) + grid_size / 2

grid_y, grid_x = np.meshgrid(np.flip(grid_y), grid_x, indexing="ij")
grid_pos = np.stack((grid_x, grid_y), axis=-1)
grid_extent = [0, grid_width, 0, grid_height]

def chirp_envelope(t):
    return np.where(
        (t >= -chirp_duration / 2) & (t <= chirp_duration / 2),
        (1 / chirp_duration) * np.cos(np.pi * t / chirp_duration) ** 2,
        0
    )

def chirp(t):
    return chirp_envelope(t) * np.exp(2.0j * np.pi * chirp_fc * t + 1.0j * np.pi * chirp_K * t ** 2)

def get_signal(position):
    signal_t = Ts * np.arange(int(max_rt_t / Ts))

    signal = np.zeros_like(signal_t, dtype=np.complex128)

    for target_point in target_points:
        target_rt_t = (2 * np.linalg.norm(target_point - position)) / C

        signal += chirp(signal_t - target_rt_t)

    return signal_t, signal

# signal_t, signal = get_signal(gt_traj[0])
# plt.plot(signal_t, np.real(signal))
# plt.plot(signal_t, np.imag(signal))
# plt.show()

def build_map(traj):
    map = np.zeros(grid_pos.shape[0:2], dtype=np.complex128)

    for position in traj:
        signal_t, signal = get_signal(position)

        grid_range = np.linalg.norm(grid_pos - position[np.newaxis, np.newaxis], axis=-1)
        grid_rt_t = (2.0 * grid_range) / C

        signal_demod = signal * np.exp(-2.0j * np.pi * chirp_fc * signal_t)

        grid_signal_t = signal_t[np.newaxis, np.newaxis, :] - grid_rt_t[:, :, np.newaxis]

        reference_signal = chirp(grid_signal_t)
        reference_signal_demod = reference_signal * np.exp(-2.0j * np.pi * chirp_fc * signal_t[np.newaxis, np.newaxis, :])

        update = np.sum(signal_demod * reference_signal_demod, axis=-1)

        # plt.plot(np.real(signal_demod))
        # plt.plot(np.real(reference_signal_demod[66, 33]))
        # plt.show()

        map += update

        # plt.subplot(1, 2, 1)
        # plt.imshow(np.abs(map), extent=grid_extent)
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.angle(map), extent=grid_extent)
        # plt.show()

    return map

known_traj = gt_traj[:100]

base_map = build_map(known_traj)

plt.imshow(np.abs(base_map))
plt.show()

plt.imshow(np.angle(base_map))
plt.show()