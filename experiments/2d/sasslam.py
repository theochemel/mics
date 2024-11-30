import matplotlib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

C = 1500

chirp_fc = 75e3
chirp_bw = 50e3
chirp_duration = 1e-3
chirp_K = chirp_bw / chirp_duration
K = 2 * np.pi * chirp_fc / C
w = 2 * np.pi * chirp_fc

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
    target_points += np.random.normal(loc=0, scale=1e-1, size=target_points.shape)
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

def odom_from_traj(traj, cov, clip_val=None, bias=None):
    odom = np.zeros((traj.shape[0] - 1, 2))
    mean = np.zeros((2,))
    for i in range(1, traj.shape[0]):
        sample = np.random.multivariate_normal(mean, cov)
        if bias is not None:
            sample += bias
        if clip_val:
            sample = np.clip(sample, -clip_val, clip_val)
        odom[i-1] = traj[i] - traj[i-1] + sample

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

def update_map(map, grid_pos, position, pulse,
               visualize=False, target_points=None):

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

    update /= np.max(update)

    map += update

    if visualize:
        grid_extent = [0, grid_width, 0, grid_height]
        plt.subplot(1, 2, 1)
        plt.imshow(np.abs(map), extent=grid_extent)
        if target_points is not None:
            plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
        plt.subplot(1, 2, 2)
        plt.imshow(np.angle(map), extent=grid_extent)
        if target_points is not None:
            plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
        plt.suptitle("Map")
        plt.show()


def build_map_from_traj(map, grid_pos, gt_traj, target_points, est_traj=None):
    update_traj = est_traj if est_traj is not None else gt_traj
    for i, (gt_position, update_pos) in enumerate(zip(gt_traj, update_traj)):
        signal = get_signal(gt_position, signal_t, target_points)
        pulse = pulse_compress(signal, signal_t)
        update_map(map, grid_pos, update_pos, pulse)

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


def compute_sample_roundtrip_t(sample_coords, poses):
    sample_vec = sample_coords[np.newaxis, :] - poses[:, np.newaxis]
    sample_range = np.linalg.norm(sample_vec, axis=-1)
    sample_dir = sample_vec / sample_range[:, :, np.newaxis]  # N_poses x N_samples x 2
    sample_rt_t = 2 * sample_range / C

    return sample_rt_t, sample_dir


def build_amplitude_linear_system(poses,
                                  pulses, map_sample_coords, map_sample_weights, pulse_scale_factor,
                                  odom, odom_cov,
                                  pos_prior, pos_prior_cov):
    # Dimensions
    N_poses = poses.shape[0]
    N_odoms = odom.shape[0]
    assert N_poses == N_odoms + 1
    N_samples = map_sample_coords.shape[0]

    # Samples
    sample_vec = map_sample_coords[np.newaxis, :] - poses[:, np.newaxis]
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

    b[:2] = inv_sqrt_pos_prior @ (poses[0] - pos_prior)

    h_odom = np.diff(poses, axis=0)
    odom_error = odom - h_odom
    b[2 : 2+N_odoms*2] = (inv_sqrt_odom @ odom_error.T).T.reshape(-1)

    b[2+N_odoms*2:] = (map_sample_weights * pulses).reshape(-1)

    return A, b


def get_pose_prior_whitened_jacobian(pose, cov):
    sqrt_inv_cov = np.linalg.inv(sp.linalg.sqrtm(cov))
    A = sqrt_inv_cov  # @ np.eye(2)
    b = sqrt_inv_cov @ pose
    return A, b


def get_odometry_whitened_jacobian(odom, cov):

    N_odoms = odom.shape[0]

    H_odom = np.array([
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ])
    sqrt_inv_cov = np.linalg.inv(sp.linalg.sqrtm(cov))
    A_odom = sqrt_inv_cov @ H_odom

    A = np.zeros((N_odoms * 2, N_odoms * 2 + 2))
    for j in range(0, N_odoms * 2, 2):
        A[j:j+2, j:j+4] = A_odom

    b = (sqrt_inv_cov @ odom.T).T.reshape(-1)

    return A, b


def get_phase_whitened_jacobian(phase_error, sample_dir, sample_weights, phase_var):

    N_poses, N_samples = phase_error.shape

    phase_grad = 2 * K * sample_dir # N_poses x N_samples x 2

    sqrt_inv_var = 1 / np.sqrt(phase_var)

    A = np.zeros((N_poses * N_samples, N_poses * 2))
    for pose_i in range(N_poses):
        l = 2 * pose_i
        t = N_samples * pose_i
        A[t:t+N_samples, l:l+2] = sqrt_inv_var * sample_weights[:, np.newaxis] * phase_grad[pose_i]

    b = sqrt_inv_var * (sample_weights * phase_error).reshape(-1)

    return A, b


def build_phase_linear_system(poses,
                              pulses, map_sample_coords, map_samples, phase_var,
                              odom, odom_cov,
                              pose_prior, pose_prior_cov):
    # Dimensions
    N_poses = poses.shape[0]
    N_odoms = odom.shape[0]
    assert N_poses == N_odoms + 1
    N_samples = map_sample_coords.shape[0]

    sample_rt_t, sample_dir = compute_sample_roundtrip_t(map_sample_coords, poses)  # N_poses x N_samples

    # Pulse interpolation
    k = sample_rt_t / Ts
    k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
    k_a = k - k_i  # Fractional parts
    k_i_plus_1 = np.clip(k_i + 1, 0, pulses.shape[1] - 1)  # Upper bounds (clipped)
    row_indices = np.arange(N_poses)[:, np.newaxis]
    row_indices = np.repeat(row_indices, N_samples, axis=1)

    pulses = (1 - k_a) * pulses[row_indices, k_i] + k_a * pulses[row_indices, k_i_plus_1]

    # update = pulses * np.exp(w * sample_rt_t)  # N_poses x N_samples
    update = pulses * np.exp(1j * w * sample_rt_t)

    sample_phases = np.angle(map_samples)
    sample_weights = np.abs(map_samples)

    est_phase = np.angle(update)
    phase_error = wrap2pi(est_phase - sample_phases)

    # Measurement Jacobian
    A = np.zeros((2 + N_odoms * 2 + N_samples * N_poses, N_poses * 2))

    b = np.empty((2 + N_odoms * 2 + N_samples * N_poses))

    A[:2, :2], b[:2] = get_pose_prior_whitened_jacobian(pose_prior, pose_prior_cov)

    A[2:N_odoms*2+2, :], b[2:N_odoms*2+2] = get_odometry_whitened_jacobian(odom, odom_cov)

    A[-N_poses*N_samples:, :], b[-N_poses*N_samples:] = get_phase_whitened_jacobian(phase_error,
                                                                                    sample_dir,
                                                                                    sample_weights,
                                                                                    phase_var)

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
    update = interp_pulse * np.exp(2.0j * w * sample_rt_t)
    weighted_magnitudes = np.sum(sample_weights[:, np.newaxis, np.newaxis] * np.abs(update), axis=0)

    fig, ax1 = plt.subplots()
    # surf = ax1.plot_surface(pos[..., 1], pos[..., 0], weighted_magnitudes, cmap=matplotlib.cm.coolwarm)
    plt.contourf(pos[..., 1], pos[..., 0], weighted_magnitudes, cmap='viridis', levels=100)
    plt.plot(pos_history[:, 0], pos_history[:, 1], color='red', linewidth=2)
    plt.scatter(gt_pos[0], gt_pos[1], color='green', linewidths=2)
    plt.show()


def plot_phase_error(sample_coords, samples,
                     poses, pulse, pose_history=None, gt_pose=None):

    sample_weights = np.abs(samples)
    sample_phases = np.angle(samples)

    sample_range = np.linalg.norm(sample_coords[:, np.newaxis, np.newaxis] - poses[np.newaxis], axis=-1)
    sample_rt_t = (2.0 * sample_range) / C
    k = sample_rt_t / Ts
    k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
    k_a = k - k_i  # Fractional parts
    k_i_plus_1 = np.clip(k_i + 1, 0, len(pulse) - 1)  # Upper bounds (clipped)
    interp_pulse = (1 - k_a) * pulse[k_i] + k_a * pulse[k_i_plus_1]
    update = interp_pulse * np.exp(1.0j * w * sample_rt_t)
    est_phase = np.angle(update)
    avg_phase_error = np.sum(
        sample_weights[:, np.newaxis, np.newaxis] * (wrap2pi(sample_phases[:, np.newaxis, np.newaxis] - est_phase))**2, axis=0
    ) / np.sum(sample_weights)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection=None))
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.contourf(poses[..., 0], poses[..., 1], avg_phase_error, cmap='viridis', levels=100)
    ax2.plot_surface(poses[..., 0], poses[..., 1], avg_phase_error, cmap=matplotlib.cm.coolwarm)
    if pose_history is not None:
        ax1.plot(pose_history[:, 0], pose_history[:, 1], color='red', linewidth=2)
    if gt_pose is not None:
        ax1.scatter(gt_pose[0], gt_pose[1], color='green', linewidths=2)
    plt.show()


def visualize_map(map, traj=None, targets=None, ax=None):
    grid_extent = [0, grid_width, 0, grid_height]
    if ax is None:
        fig, ax = plt.subplot()
    ax.imshow(np.abs(map), extent=grid_extent)
    if targets is not None:
        ax.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
    if traj is not None:
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2)
    if ax is None:
        plt.suptitle("Map")
        plt.show()


if __name__ == "__main__":
    # Generate trajectory and odometry
    gt_traj_x = 3e-2 * np.arange(200)
    gt_traj_y = np.zeros_like(gt_traj_x)
    gt_traj = np.stack((gt_traj_x, gt_traj_y), axis=-1)

    odom_sigma = (C / chirp_fhi) / 4
    print(f"odom std dev: {odom_sigma}")
    odom_cov = np.array([
        [odom_sigma**2, 0],
        [0, odom_sigma**2]
    ])
    odom_clip_val = 0.0035
    odom_bias = np.array([-0.001, 0.001])
    odom = odom_from_traj(gt_traj, odom_cov, clip_val=odom_clip_val)

    target_points = make_forest_targets()
    grid_pos, map = initialize_map()

    slam_start_pose = 20
    lag = 8

    build_map_from_traj(map, grid_pos, gt_traj[:slam_start_pose - lag + 1], target_points)

    grid_extent = [0, grid_width, 0, grid_height]
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(map), extent=grid_extent)
    plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(map), extent=grid_extent)
    plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
    plt.suptitle("Update")
    plt.show()

    # Offset grid for error plotting
    l = C / chirp_fc
    offset_x = np.linspace(0, 2 * l, 20) - l
    offset_y = np.linspace(0, 2 * l, 20) - l
    offset_x, offset_y = np.meshgrid(np.flip(offset_y), offset_x, indexing='ij')
    err_pos = np.stack((offset_x, offset_y), axis=-1)

    # Start with a shitty prior
    pose_prior_cov = odom_cov
    noise = np.random.multivariate_normal(np.zeros((2,)), pose_prior_cov)
    pose_prior = gt_traj[slam_start_pose - lag] + np.clip(noise, -odom_clip_val, odom_clip_val)

    # Initialize poses from ground truth
    poses = np.zeros((lag, 2))
    for i in range(lag):
        noise = np.random.multivariate_normal(np.zeros((2,)), pose_prior_cov)
        poses[i] = gt_traj[slam_start_pose - lag + 1 + i] + np.clip(noise, -odom_clip_val, odom_clip_val)

    # Prepare pulses
    pulses = np.empty((lag, signal_t.shape[0]), dtype=np.complex128)
    for i in range(lag):
        signal = get_signal(gt_traj[slam_start_pose - lag + 1 + i], signal_t, target_points)
        pulses[i] = pulse_compress(signal, signal_t)

    computed_trajectory = np.zeros_like(gt_traj)
    computed_trajectory[:slam_start_pose] = gt_traj[:slam_start_pose]

    # SLAMMING
    for last_pose_i in range(slam_start_pose, gt_traj.shape[0]-1):
        first_pose_i = last_pose_i - lag + 1

        gt_poses = gt_traj[first_pose_i:last_pose_i + 1]  # for evaluation & sim only

        # Normalize map
        # map /= np.max(np.abs(map))

        # Sample map
        sample_idx = importance_sample(np.abs(map), 128)
        sample_coords = grid_pos[sample_idx[:, 1], sample_idx[:, 0]]
        samples = map[sample_idx[:, 1], sample_idx[:, 0]]

        print(f'Errors before opt: {np.linalg.norm(gt_poses - poses, axis=-1)}')

        n_iterations = 10
        pose_history = np.empty((n_iterations + 1,) + poses.shape)
        pose_history[0] = poses
        for i in range(n_iterations):
            A, b = build_phase_linear_system(poses,
                                             pulses,
                                             sample_coords, samples,
                                             1e-7,
                                             odom[last_pose_i-lag+1:last_pose_i], odom_cov,
                                             pose_prior, pose_prior_cov)

            x = np.linalg.solve(A.T @ A, A.T @ b)

            dpos = x.reshape((lag, 2))
            poses += dpos
            pose_history[i + 1] = poses

        print(f'Error after opt: {np.linalg.norm(gt_poses - poses, axis=-1)}')

        signal = get_signal(gt_poses[0], signal_t, target_points)
        pulse = pulse_compress(signal, signal_t)
        # plot_phase_error(sample_coords, samples, err_pos + gt_poses[0], pulse, pose_history[:, 0], gt_pose=gt_poses[0])

        # Update map
        update_map(map, grid_pos, poses[0], pulse, visualize=False)
        computed_trajectory[last_pose_i-lag+1] = poses[0]


        # Marginalize out first pose
        # Todo: handle covariance properly
        pose_prior = poses[0] + odom[last_pose_i-lag+1]

        # Add initial value for next pose
        next_pose = poses[-1] + odom[last_pose_i]
        poses[:-1] = poses[1:]
        poses[-1] = next_pose

        # Compute next pulse
        # SIMULATION START
        next_gt_pose = gt_traj[last_pose_i + 1]
        signal = get_signal(next_gt_pose, signal_t, target_points)
        pulses[:-1] = pulses[1:]
        pulses[-1] = pulse_compress(signal, signal_t)
        # SIMULATION END


    dead_reckon_traj = np.empty_like(gt_traj)
    dead_reckon_traj[:slam_start_pose] = gt_traj[:slam_start_pose]
    for i in range(slam_start_pose, dead_reckon_traj.shape[0]):
        dead_reckon_traj[i] = dead_reckon_traj[i-1] + odom[i-1]

    _, dead_reckon_map = initialize_map()
    _, gt_map = initialize_map()
    build_map_from_traj(dead_reckon_map, grid_pos, gt_traj, target_points, est_traj=dead_reckon_traj)
    build_map_from_traj(gt_map, grid_pos, gt_traj, target_points)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    visualize_map(map, computed_trajectory, target_points, ax=ax1)
    visualize_map(gt_map, gt_traj, target_points, ax=ax2)
    visualize_map(dead_reckon_map, dead_reckon_traj, target_points, ax=ax3)
    plt.show()
