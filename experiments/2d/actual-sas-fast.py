import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize


C = 1500

chirp_fc = 7.5e3
chirp_bw = 5e3
chirp_duration = 1e-3
chirp_K = chirp_bw / chirp_duration

chirp_fhi = chirp_fc + chirp_bw / 2

fs = 1e6
Ts = 1 / fs

l_m = C / chirp_fhi

max_range = 20
max_rt_t = (2 * max_range) / C

target_points = np.array([
    [5, 5],
])

# gt_traj_x = (l_m / 2) * np.arange(8 / (l_m / 2)) + 1
# gt_traj_y = np.full_like(gt_traj_x, fill_value=1)
gt_traj_x = 1e-2 * np.arange(100)
gt_traj_y = np.zeros_like(gt_traj_x)
gt_traj = np.stack((gt_traj_x, gt_traj_y), axis=-1)

noisy_traj = gt_traj + np.random.normal(loc=0, scale=l_m / 4, size=gt_traj.shape)

grid_width = 10
grid_height = 10
grid_size = 1e-2

grid_x = grid_size * np.arange(int(grid_width / grid_size)) + grid_size / 2
grid_y = grid_size * np.arange(int(grid_height / grid_size)) + grid_size / 2

grid_y, grid_x = np.meshgrid(np.flip(grid_y), grid_x, indexing="ij")
grid_pos = np.stack((grid_x, grid_y), axis=-1)
grid_extent = [0, grid_width, 0, grid_height]

def chirp_envelope(t):
    return np.where(
        (t >= -chirp_duration / 2) & (t <= chirp_duration / 2),
        (1 / chirp_duration) * np.cos(np.pi * t / chirp_duration) ** 2,
        # 1,
        0
    )

def chirp(t):
    return chirp_envelope(t) * np.exp(2.0j * np.pi * chirp_fc * t + 1.0j * np.pi * chirp_K * t ** 2)

def chirp_reference(t):
    return chirp_envelope(t) * np.exp(1.0j * np.pi * chirp_K * t ** 2)

def get_signal(position, signal_t):
    signal = np.zeros_like(signal_t, dtype=np.complex128)

    for target_point in target_points:
        target_rt_t = (2 * np.linalg.norm(target_point - position)) / C

        signal += chirp(signal_t - target_rt_t)

    return signal

signal_t = Ts * np.arange(int(max_rt_t / Ts))

signal = get_signal(gt_traj[0], signal_t)

def pulse_compress(signal, signal_t):
    reference_signal_t = Ts * np.arange(int(chirp_duration / Ts)) - (chirp_duration / 2)
    reference_signal = chirp_reference(reference_signal_t)

    correlation = sp.signal.correlate(signal, reference_signal, mode="same")

    return correlation # * np.exp(-2.0j * np.pi * chirp_fc * signal_t)

# reference_signal_t = Ts * np.arange(int(chirp_duration / Ts)) - (chirp_duration / 2)
# reference_signal = chirp(reference_signal_t)
#
# correlation = sp.signal.correlate(signal, reference_signal, mode="full")
# lags = sp.signal.correlation_lags(len(signal), len(reference_signal), mode="full")
# print(lags)
# # correlation = np.concatenate((
# #     # np.zeros(-lags[0]),
# # ))
# correlation = correlation[-lags[0]:]
#
# plt.subplot(3, 1, 1)
# plt.plot(np.real(signal))
# plt.plot(np.imag(signal))
# plt.subplot(3, 1, 2)
# plt.plot(reference_signal_t, np.real(reference_signal))
# plt.plot(reference_signal_t, np.imag(reference_signal))
# plt.subplot(3, 1, 3)
# plt.plot(np.real(correlation))
# plt.plot(np.imag(correlation))
# plt.show()

def build_map(traj):
    # traj is [n_poses, 2]
    # signal_t is [n_samples]
    # signals is [n_poses, n_samples]

    map = np.zeros(grid_pos.shape[0:2], dtype=np.complex128)

    for i, position in enumerate(traj):
        signal = get_signal(position, signal_t)
        pulse = pulse_compress(signal, signal_t)

        grid_range = np.linalg.norm(grid_pos - position[np.newaxis, np.newaxis], axis=-1)
        grid_rt_t = (2.0 * grid_range) / C

        plt.imshow(grid_range)
        plt.show()

        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(np.real(pulse))
        axs[0].plot(np.imag(pulse))
        axs[0].axvline(x=((2 * np.linalg.norm(position - target_points[0])) / C) / Ts, c="r")
        axs[1].plot(np.real(signal))
        axs[1].plot(np.imag(signal))
        axs[1].axvline(x=((2 * np.linalg.norm(position - target_points[0])) / C) / Ts, c="r")
        plt.show()

        k = grid_rt_t / Ts
        k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
        k_a = k - k_i  # Fractional parts

        # Ensure we don't go out of bounds for the upper index
        k_i_plus_1 = np.clip(k_i + 1, 0, len(pulse) - 1)  # Upper bounds (clipped)

        # Perform linear interpolation
        # interp_pulse = (1 - k_a) * pulse[k_i] + k_a * pulse[k_i_plus_1]
        interp_pulse = pulse[k_i]

        update = interp_pulse * np.exp((2.0j * np.pi * chirp_fc / C) * (2.0 * grid_range))

        plt.subplot(1, 2, 1)
        plt.imshow(np.abs(update), extent=grid_extent)
        plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
        plt.subplot(1, 2, 2)
        plt.imshow(np.angle(update), extent=grid_extent)
        plt.show()

        map += update

        plt.subplot(1, 2, 1)
        plt.imshow(np.abs(map), extent=grid_extent)
        plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
        plt.subplot(1, 2, 2)
        plt.imshow(np.angle(map), extent=grid_extent)
        plt.show()

    return map


known_traj = gt_traj[:100]

base_map = build_map(known_traj)

plt.imshow(np.abs(base_map))
plt.show()

plt.imshow(np.angle(base_map))
plt.show()