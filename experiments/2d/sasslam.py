import matplotlib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from tracer.scene import Scene

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
ACOUSTIC SIM
"""

def make_forest_targets():
    target_points_x = np.linspace(1, 9, 9)
    target_points_y = np.linspace(1, 9, 9)
    target_points_y, target_points_x = np.meshgrid(target_points_y, target_points_x, indexing="ij")
    target_points_y = target_points_y.flatten()
    target_points_x = target_points_x.flatten()
    target_points = np.stack((target_points_x, target_points_y), axis=-1)
    # target_points += np.random.normal(loc=0, scale=3e-1, size=target_points.shape)
    return target_points


def make_sine_targets():
    t = np.linspace(0, 1, 50)
    target_points_x = 1 + t * 8
    target_points_y = 5 + np.sin(2*np.pi*t)
    target_points = np.stack((target_points_x, target_points_y), axis=-1)
    return target_points

def wrap2pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def chirp_envelope(t):
    return np.where(
        (t >= -chirp_duration / 2) & (t <= chirp_duration / 2),
        # (1 / chirp_duration) * np.cos(np.pi * t / chirp_duration) ** 2,
        np.cos(np.pi * t / chirp_duration) ** 2,
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
MOTION SIM
"""

def odom_from_traj(traj, cov):
    odom = np.zeros((traj.shape[0] - 1, 2))
    for i in range(1, traj.shape[0]):
        mean = traj[i] - traj[i-1]
        sample = np.random.multivariate_normal(mean, cov)
        odom[i-1] = sample

    return odom

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

def update_map(map, grid_pos, position, signal,
               visualize=False, target_points=None):
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


def build_map_from_gt_traj(map, grid_pos, gt_traj, target_points):
    for i, gt_position in enumerate(gt_traj):
        signal = get_signal(gt_position, signal_t, target_points)
        update_map(map, grid_pos, gt_position, signal)

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
                                  pulses, map_sample_coords, map_sample_weights, pulse_scale_factor,
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
    sample_dir = sample_vec / sample_range[:, :, np.newaxis]  # N_poses x N_samples x 2
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
    pulses = (1 - k_a) * pulses[row_indices, k_i] + k_a * pulses[row_indices, k_i_plus_1]
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

    A[:2, :2] = A_pos_prior
    for j in range(0, N_odoms * 2, 2):
        A[j+2:j+4, j:j+4] = A_odom

    # r is roundtrip distance to the target, t is return time
    dt_dr = 1 / C
    dr_dpos = -2 * sample_dir

    dpulses_dpos = dt_dr * dpulses_dt[:, :, np.newaxis] * dr_dpos  # N_poses x N_samples x 2

    map_sample_weights *= pulse_scale_factor

    for pose_i in range(0, N_poses):
        l = 2 * pose_i
        t = 2 + 2 * N_odoms + N_samples * pose_i
        A[t:t+N_samples, l:l+2] = map_sample_weights[:, np.newaxis] * dpulses_dpos[pose_i]

    # Residuals
    b = np.zeros(2 + N_odoms * 2 + N_samples * N_poses)

    b[:2] = inv_sqrt_pos_prior @ (pos[0] - pos_prior)

    h_odom = np.diff(pos, axis=0)
    odom_error = odom - h_odom
    b[2 : 2+N_odoms*2] = (inv_sqrt_odom @ odom_error.T).T.reshape(-1)

    b[2+N_odoms*2:] = (map_sample_weights * pulses).reshape(-1)

    return A, b


def plot_amplitude_error(sample_coords, sample_weights,
                         pos, pulse, pos_history, gt_pos):

    sample_range = np.linalg.norm(sample_coords[:, np.newaxis, np.newaxis] - pos[np.newaxis], axis=-1)
    sample_rt_t = (2.0 * sample_range) / C
    k = sample_rt_t / Ts
    k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
    k_a = k - k_i  # Fractional parts
    k_i_plus_1 = np.clip(k_i + 1, 0, len(pulse) - 1)  # Upper bounds (clipped)
    interp_pulse = (1 - k_a) * pulse[k_i] + k_a * pulse[k_i_plus_1]
    update = interp_pulse * np.exp((2.0j * np.pi * chirp_fc / C) * (2.0 * sample_range))
    weighted_magnitudes = np.sum(sample_weights[:, np.newaxis, np.newaxis] * np.abs(update), axis=0)


    fig, ax1 = plt.subplots()
    # surf = ax1.plot_surface(pos[..., 1], pos[..., 0], weighted_magnitudes, cmap=matplotlib.cm.coolwarm)
    plt.contourf(pos[..., 1], pos[..., 0], weighted_magnitudes, cmap='viridis', levels=100)
    plt.plot(pos_history[:, 0], pos_history[:, 1], color='red', linewidth=2)
    plt.scatter(gt_pos[0], gt_pos[1], color='green', linewidths=2)
    plt.show()


if __name__ == "__main__":
    # Generate trajectory and odometry
    gt_traj_x = 1e-2 * np.arange(100)
    gt_traj_y = np.zeros_like(gt_traj_x)
    gt_traj = np.stack((gt_traj_x, gt_traj_y), axis=-1)

    odom_cov = np.array([
        [0.001, 0],
        [0, 0.001]
    ])
    odom = odom_from_traj(gt_traj, odom_cov)

    target_points = make_forest_targets()
    grid_pos, map = initialize_map()

    n_init_poses = 50
    build_map_from_gt_traj(map, grid_pos, gt_traj[:n_init_poses], target_points)

    grid_extent = [0, grid_width, 0, grid_height]
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(map), extent=grid_extent)
    plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(map), extent=grid_extent)
    plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
    plt.suptitle("Update")
    plt.show()

    # Start with a shitty prior
    pos_prior_cov = odom_cov
    pos_prior = np.random.multivariate_normal(gt_traj[n_init_poses], pos_prior_cov)
    print(pos_prior - gt_traj[n_init_poses])

    # Dead reckon initial poses
    lag = 1
    pos = np.zeros((lag, 2))
    pos[0] = pos_prior
    for i in range(1, lag):
        pos[i] = pos[i-1] + odom[n_init_poses + i - 1]


    # SLAMMING
    for odom_i in range(n_init_poses-1, odom.shape[0]):
        # Simulate
        gt_pos = gt_traj[odom_i+1]
        pulses = np.empty((lag, signal_t.shape[0]), dtype=np.complex128)
        for i in range(lag):
            signal = get_signal(gt_pos, signal_t, target_points)
            pulses[i] = pulse_compress(signal, signal_t)

        # Normalize map
        map /= np.max(np.abs(map))

        # Sample map
        sample_idx = importance_sample(np.abs(map), 128)
        sample_coords = grid_pos[sample_idx[:, 1], sample_idx[:, 0]]
        sample_weights = np.abs(map)[sample_idx[:, 1], sample_idx[:, 0]]


        err_x = np.linspace(0, 0.5, 50) - 0.25
        err_y = np.linspace(0, 1, 50) - 0.5
        err_x += gt_pos[0]
        err_y += gt_pos[1]
        err_x, err_y = np.meshgrid(np.flip(err_y), err_x, indexing='ij')
        err_pos = np.stack((err_x, err_y), axis=-1)


        print(f'Error before opt: {np.linalg.norm(gt_pos - pos, axis=-1)}')

        n_iterations = 10
        pos_history = np.empty((n_iterations+1, 2))
        pos_history[0] = pos[0]
        for i in range(n_iterations):
            A, b = build_amplitude_linear_system(pos,
                                                 pulses, sample_coords, sample_weights, 100,
                                                 odom[odom_i : odom_i+lag-1], odom_cov,
                                                 pos_prior, pos_prior_cov)

            x, residuals, rank, s = np.linalg.lstsq(A, b)

            dpos = x.reshape((lag, 2))
            pos +=  dpos
            pos_history[i+1] = pos[0]

        pos_history = np.array(pos_history)
        plot_amplitude_error(sample_coords, sample_weights, err_pos, pulses[0], pos_history, gt_pos)

            # print(dpos, pos)

        gt_pos = gt_traj[odom_i+1]
        print(f'Error after opt: {np.linalg.norm(gt_pos - pos, axis=-1)}')

        break
