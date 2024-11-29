import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


C = 1500

chirp_fc = 75e3
chirp_bw = 50e3
chirp_duration = 1e-3
chirp_K = chirp_bw / chirp_duration
K = 2 * np.pi * chirp_fc / C

chirp_fhi = chirp_fc + chirp_bw / 2

fs = 1e6
Ts = 1 / fs

l_m = C / chirp_fhi

max_range = 20
max_rt_t = (2 * max_range) / C

signal_t = Ts * np.arange(int(max_rt_t / Ts))

grid_width = 10
grid_height = 10
grid_size = 1e-2

"""
SIMULATION & SIGNAL PROCESSING
"""

def make_targets():
    target_points_x = np.linspace(1, 9, 9)
    target_points_y = np.linspace(1, 9, 9)
    target_points_y, target_points_x = np.meshgrid(target_points_y, target_points_x, indexing="ij")
    target_points_y = target_points_y.flatten()
    target_points_x = target_points_x.flatten()
    target_points = np.stack((target_points_x, target_points_y), axis=-1)
    target_points += np.random.normal(loc=0, scale=2e-2, size=target_points.shape)
    return target_points


def wrap2pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def chirp_envelope(t):
    return np.where(
        (t >= -chirp_duration / 2) & (t <= chirp_duration / 2),
        (1 / chirp_duration) * np.cos(np.pi * t / chirp_duration) ** 2,
        0
    )


def chirp(t):
    return chirp_envelope(t) * np.exp(2.0j * np.pi * chirp_fc * t + 1.0j * np.pi * chirp_K * t ** 2)


def reference_chirp(t):
    return chirp_envelope(t) * np.exp(1.0j * np.pi * chirp_K * t ** 2)


def get_signal(position, signal_t, target_points):
    signal = np.zeros_like(signal_t, dtype=np.complex128)

    for target_point in target_points:
        target_rt_t = (2 * np.linalg.norm(target_point - position)) / C

        signal += chirp(signal_t - target_rt_t)

    return signal * np.exp(-2.0j * np.pi * chirp_fc * signal_t)


def pulse_compress(signal, signal_t):
    reference_signal_t = Ts * np.arange(int(chirp_duration / Ts)) - (chirp_duration / 2)
    reference_signal = reference_chirp(reference_signal_t)

    correlation = sp.signal.correlate(signal, reference_signal, mode="same")

    return correlation

"""
SYNTHETIC-APERTURE PROCESSING
"""

def initialize_map():
    grid_x = grid_size * np.arange(int(grid_width / grid_size)) + grid_size / 2
    grid_y = grid_size * np.arange(int(grid_height / grid_size)) + grid_size / 2

    grid_y, grid_x = np.meshgrid(np.flip(grid_y), grid_x, indexing="ij")
    grid_pos = np.stack((grid_x, grid_y), axis=-1)

    map = np.zeros(grid_pos.shape[0:2], dtype=np.complex128)

    return grid_pos, map

def update_map(map, grid_pos, gt_position, position,
               visualize=False, target_points=None):
    signal = get_signal(gt_position, signal_t)
    pulse = pulse_compress(signal, signal_t)

    if visualize:
        pulse_range = (signal_t * C) / 2.0

        pulse_update = pulse * np.exp((2.0j * np.pi * chirp_fc / C) * (2.0 * pulse_range))

        plt.plot(pulse_range, np.real(pulse_update))
        plt.plot(pulse_range, np.imag(pulse_update))
        plt.show()

        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(pulse_range, np.abs(pulse_update))
        axs[1].plot(pulse_range, np.where(np.abs(pulse_update) > 1e-9, np.angle(pulse_update), 0))
        plt.show()

    grid_range = np.linalg.norm(grid_pos - position[np.newaxis, np.newaxis], axis=-1)
    grid_rt_t = (2.0 * grid_range) / C

    k = grid_rt_t / Ts
    k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
    k_a = k - k_i  # Fractional parts

    # Ensure we don't go out of bounds for the upper index
    k_i_plus_1 = np.clip(k_i + 1, 0, len(pulse) - 1)  # Upper bounds (clipped)

    # Perform linear interpolation
    interp_pulse = (1 - k_a) * pulse[k_i] + k_a * pulse[k_i_plus_1]

    update = interp_pulse * np.exp((2.0j * np.pi * chirp_fc / C) * (2.0 * grid_range))

    map += update

    if visualize:
        grid_extent = [0, grid_width, 0, grid_height]
        plt.subplot(1, 2, 1)
        plt.imshow(np.abs(map), extent=grid_extent)
        plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
        plt.subplot(1, 2, 2)
        plt.imshow(np.angle(map), extent=grid_extent)
        plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
        plt.suptitle("Map")
        plt.show()


def build_map_from_gt_traj(map, grid_pos, gt_traj):
    for i, gt_position in enumerate(gt_traj):
        update_map(map, grid_pos, gt_position, gt_position)

"""
SLAM
"""

def importance_sample(img, n_samples):
    weights = img ** 4
    weights = weights / np.sum(weights)
    flat_weights = weights.flatten()
    indices = np.arange(len(flat_weights))

    samples = np.random.choice(indices, size=n_samples, replace=False, p=flat_weights)

    sample_y, sample_x = np.unravel_index(samples, img.shape)

    return np.stack((sample_x, sample_y), axis=-1)


def build_amplitude_linear_system(pos,
                                  pulses, map_sample_coords, map_sample_weigths,
                                  odom, odom_cov,
                                  pos_prior, pos_prior_cov):
    # Dimensions
    N_poses = pos.shape[0]
    N_odoms = odom.shape[0]
    assert N_poses == N_odoms + 1
    N_samples = map_sample_coords.shape[0]

    # Samples
    sample_vec = map_sample_coords[np.newaxis, :] - pos[:, np.newaxis]
    sample_range = np.linalg.norm(sample_vec, axis=-1)
    sample_dir = sample_vec / sample_range  # N_poses x N_samples x 2
    sample_rt_t = 2 * sample_range / C

    # Pulse derivative interpolation
    k = sample_rt_t / Ts
    k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
    k_a = k - k_i  # Fractional parts
    k_i_plus_1 = np.clip(k_i + 1, 0, pulses.shape[1] - 1)  # Upper bounds (clipped)

    row_indices = np.arange(N_poses)[:, np.newaxis]
    row_indices = np.repeat(row_indices, N_samples, axis=1)

    pulses = np.abs(pulses).astype(np.float64)
    dpulses_dt = np.gradient(pulses, Ts, axis=-1)
    dpulses_dt = (1 - k_a) * dpulses_dt[row_indices, k_i] + k_a * dpulses_dt[row_indices, k_i_plus_1]  # N_poses x N_samples

    # Measurement Jacobian
    A = np.zeros((2 + N_odoms * 2 + N_samples * N_poses, N_poses * 2))

    H_odom = np.array([
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ])
    H_prior = np.eye(2)

    inv_sqrt_odom = np.linalg.inv(sp.linalg.sqrtm(odom_cov))
    inv_sqrt_pos_prior = np.linalg.inv(sp.linalg.sqrtm(pos_prior_cov))

    A_odom = inv_sqrt_odom @ H_odom
    A_pos_prior = inv_sqrt_pos_prior @ H_prior

    A[:2, :2] = A_odom
    for i in range(2, 2 + N_odoms * 2, 2):
        A[i:i+2, i:i+4] = A_odom

    # r is roundtrip distance to the target, t is return time
    dt_dr = 1 / C
    dr_dpos = -2 * sample_dir

    dpulses_dpos = dt_dr * dpulses_dt[:, :, np.newaxis] * dr_dpos  # N_poses x N_samples x 2

    for pose_i in range(0, N_poses):
        l = 2 * pose_i
        t = 2 * N_odoms + N_samples * pose_i
        H[t:t+N_samples, l:l+2] = dpulses_dpos[pose_i]

    # Residuals

