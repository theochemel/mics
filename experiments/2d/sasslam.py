import matplotlib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from spatialmath.pose3d import SE3

from motion.imu import IMU, IMUMeasurement
from motion.linear_constant_acceleration_trajectory import LinearConstantAccelerationTrajectory

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

def make_corner_target():
    target_points_x = np.concatenate((np.linspace(5,9,50), 9 * np.ones((50,))))
    target_points_y = np.concatenate((9 * np.ones((50,)), np.linspace(5,9,50)))
    target_points = np.stack((target_points_x, target_points_y), axis=-1)
    return target_points

def wrap2pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def cosine_envelope(t):
    return np.where(
        (t >= -chirp_duration / 2) & (t <= chirp_duration / 2),
        # (1 / chirp_duration) * np.cos(np.pi * t / chirp_duration) ** 2,
        np.cos(np.pi * t / chirp_duration) ** 2,
        0
    )


def chirp(t):
    return cosine_envelope(t) * np.exp(2.0j * np.pi * chirp_fc * t + 1.0j * np.pi * chirp_K * t ** 2)




def reference_chirp(t):
    return cosine_envelope(t) * np.exp(1.0j * np.pi * chirp_K * t ** 2)


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


def build_prior_system(state, prior, cov):
    sqrt_inv_cov = np.linalg.inv(sp.linalg.sqrtm(cov))
    A = sqrt_inv_cov
    b = -sqrt_inv_cov @ (state - prior)
    return A, b


def build_motion_system(accel, accel_cov, dt):

    N_poses = accel.shape[0]
    # State: x, y, v_x, v_y
    H = np.array([
        [0, 0, 1/dt, 0],
        [0, 0, 0, 1/dt]
    ])
    sqrt_inv_cov = np.linalg.inv(sp.linalg.sqrtm(accel_cov))
    A_motion = sqrt_inv_cov @ H

    A = np.zeros((N_poses * 2, N_poses * 4))
    for j in range(0, N_poses):
        t = j * 2
        l = j * 4
        A[t:t+2, l:l+4] = A_motion

    b = (sqrt_inv_cov @ accel.T).T.reshape(-1)

    return A, b


def build_phase_system(phase_error, sample_dir, sample_weights, phase_var):

    N_poses, N_samples = phase_error.shape

    phase_grad = 2 * K * sample_dir # N_poses x N_samples x 2

    sqrt_inv_var = 1 / np.sqrt(phase_var)

    A = np.zeros((N_poses * N_samples, N_poses * 4))
    for pose_i in range(N_poses):
        l = 4 * pose_i
        t = N_samples * pose_i
        A[t:t+N_samples, l:l+2] = sqrt_inv_var * sample_weights[:, np.newaxis] * phase_grad[pose_i]

    b = sqrt_inv_var * (sample_weights * phase_error).reshape(-1)

    return A, b


def build_linear_system(state,
                        pulses, map_sample_coords, map_samples, phase_var,
                        accel, accel_cov, dt,
                        prior, prior_cov):
    # Dimensions
    N_poses = state.shape[0]
    N_samples = map_sample_coords.shape[0]
    assert state.shape[0] == accel.shape[0]

    sample_rt_t, sample_dir = compute_sample_roundtrip_t(map_sample_coords, state[:, :2])  # N_poses x N_samples

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
    A = np.zeros((4 + N_poses * 2 + N_samples * N_poses, N_poses * 4))
    # A = np.zeros((N_poses * 2 + N_samples * N_poses, N_poses * 4))

    b = np.empty((4 + N_poses * 2 + N_samples * N_poses))
    # b = np.empty((N_poses * 2 + N_samples * N_poses))

    A[:4, :4], b[:4] = build_prior_system(state[0], prior, prior_cov)

    A[4:N_poses*2+4, :], b[4:N_poses*2+4] = build_motion_system(accel, accel_cov, dt)
    # A[:N_poses*2, :], b[:N_poses*2] = get_motion_whitened_jacobian(accel, accel_cov, dt)

    A[-N_poses*N_samples:, :], b[-N_poses*N_samples:] = build_phase_system(phase_error,
                                                                           sample_dir,
                                                                           sample_weights,
                                                                           phase_var)

    return A, b


def plot_phase_error(sample_coords, samples,
                     offset_grid, pulses, gt_poses, pose_history=None):

    sample_weights = np.abs(samples)
    sample_phases = np.angle(samples)

    N_poses = len(gt_poses)

    fig, axes = plt.subplots(2, int(np.ceil(N_poses / 2)))
    axes = axes.reshape(-1)
    for i in range(N_poses):
        grid = offset_grid + gt_poses[i]
        sample_range = np.linalg.norm(sample_coords[:, np.newaxis, np.newaxis] - grid[np.newaxis], axis=-1)
        sample_rt_t = (2.0 * sample_range) / C
        k = sample_rt_t / Ts
        k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
        k_a = k - k_i  # Fractional parts
        k_i_plus_1 = np.clip(k_i + 1, 0, pulses.shape[1] - 1)  # Upper bounds (clipped)
        interp_pulse = (1 - k_a) * pulses[i, k_i] + k_a * pulses[i, k_i_plus_1]
        update = interp_pulse * np.exp(1.0j * w * sample_rt_t)
        est_phase = np.angle(update)
        avg_phase_error = np.sum(
            sample_weights[:, np.newaxis, np.newaxis] * (wrap2pi(sample_phases[:, np.newaxis, np.newaxis] - est_phase))**2, axis=0
        ) / np.sum(sample_weights)

        axes[i].contourf(grid[..., 0], grid[..., 1], avg_phase_error, cmap='viridis', levels=100)
        axes[i].scatter(gt_poses[i, 0], gt_poses[i, 1], color='green', linewidths=2)
        if pose_history is not None:
            axes[i].plot(pose_history[:, i, 0], pose_history[:, i, 1], color='red', linewidth=2)
        axes[i].set_title(f'State {i}')
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


def propagate_state(state, accel, dt):
    new_state = np.empty_like(state)
    new_state[:2] = state[:2] + dt * state[2:] + 0.5 * accel * dt ** 2
    new_state[2:] = state[2:] + accel * dt
    return new_state


if __name__ == "__main__":
    dt = 0.05

    trajectory = LinearConstantAccelerationTrajectory(
        keyposes=[
            SE3.Trans(0.0, 0.0, 0.0),
            SE3.Trans(1.0, 0.0, 0.0),
            SE3.Trans(0.0, 1.0, 0.0),
        ],
        max_velocity=0.1,
        acceleration=0.1,
        dt=dt,
    )

    print(f"Trajectory length: {len(trajectory.poses)}")

    imu_accel_white_sigma = 1e-4
    imu_accel_walk_sigma = 1e-4
    imu = IMU(
        acceleration_white_sigma=imu_accel_white_sigma,
        acceleration_walk_sigma=imu_accel_walk_sigma,
        orientation_white_sigma=1e-3,
        orientation_walk_sigma=1e-6,
    )

    imu_measurement = imu.measure(trajectory)

    odom_clip_val = 0.0035

    target_points = make_forest_targets()
    grid_pos, map = initialize_map()

    slam_start_pose_idx = 80
    lag = 8

    gt_pose = np.array(trajectory.poses)[:, :2, 3]
    gt_vel = trajectory.velocity_world[:, :2]
    imu_accel_body = imu_measurement.acceleration_body
    yaw = imu_measurement.orientation_rpy[:, 2]
    imu_accel_world = np.empty((imu_accel_body.shape[0], 2))
    imu_accel_world[:, 0] = imu_accel_body[:, 0] * np.cos(yaw) - imu_accel_body[:, 1] * np.sin(yaw)
    imu_accel_world[:, 1] = imu_accel_body[:, 0] * np.sin(yaw) + imu_accel_body[:, 1] * np.cos(yaw)
    # IMU acceleration covariance
    imu_accel_cov = np.eye(2) * imu_accel_white_sigma ** 2

    build_map_from_traj(map, grid_pos, gt_pose[:slam_start_pose_idx], target_points)

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
    err_x = np.linspace(0, 2 * l, 20) - l
    err_y = np.linspace(0, 2 * l, 20) - l
    err_x, err_y = np.meshgrid(np.flip(err_y), err_x, indexing='ij')
    err_offset = np.stack((err_x, err_y), axis=-1)

    prior_cov = np.eye(4) * 0.002 ** 2
    # noise = np.random.multivariate_normal(np.zeros((2,)), prior_cov)
    prior = np.empty(4)
    prior[:2] = gt_pose[slam_start_pose_idx - lag] # + np.clip(noise, -odom_clip_val, odom_clip_val)
    prior[2:] = gt_vel[slam_start_pose_idx - lag]

    # Initialize poses from ground truth
    state = np.zeros((lag, 4))
    for i in range(lag):
        state[i, :2] = gt_pose[slam_start_pose_idx - lag + 1 + i]
        state[i, 2:] = gt_vel[slam_start_pose_idx - lag + 1 + i]

    # Prepare pulses
    pulses = np.empty((lag, signal_t.shape[0]), dtype=np.complex128)
    for i in range(lag):
        signal = get_signal(gt_pose[slam_start_pose_idx - lag + 1 + i], signal_t, target_points)
        pulses[i] = pulse_compress(signal, signal_t)

    computed_trajectory = np.zeros((gt_pose.shape[0], 4))
    computed_trajectory[:slam_start_pose_idx] = (
        np.concatenate((gt_pose[:slam_start_pose_idx], gt_vel[:slam_start_pose_idx]), axis=1))

    # Generate dead-reckoned trajectory for comparison
    dead_reckon_traj = np.empty_like(computed_trajectory)
    dead_reckon_traj[:slam_start_pose_idx] = computed_trajectory[:slam_start_pose_idx]
    for i in range(slam_start_pose_idx, dead_reckon_traj.shape[0]):
        dead_reckon_traj[i] = propagate_state(dead_reckon_traj[i-1], imu_accel_world[i-1], dt)
    dr_pose = dead_reckon_traj[:, :2]

    # SLAMMING
    for last_pose_i in range(slam_start_pose_idx, gt_pose.shape[0] - 1):
        first_pose_i = last_pose_i - lag + 1

        current_gt_poses = gt_pose[first_pose_i:last_pose_i + 1]  # for evaluation & sim only
        current_dr_poses = dr_pose[first_pose_i:last_pose_i + 1]  # for evaluation & sim only

        # Sample map
        sample_idx = importance_sample(np.abs(map), 128)
        sample_coords = grid_pos[sample_idx[:, 1], sample_idx[:, 0]]
        samples = map[sample_idx[:, 1], sample_idx[:, 0]]

        print()
        print(f"--- POSES {first_pose_i} - {last_pose_i} ---")
        print("Prior covariance:")
        print(prior_cov)
        print(f'Position error before opt: {np.linalg.norm(current_gt_poses - state[:, :2], axis=-1)}')

        n_iterations = 10
        state_optimization_history = np.empty((n_iterations + 1,) + state.shape)
        state_optimization_history[0] = state
        for i in range(n_iterations):
            A, b = build_linear_system(state,
                                       pulses,
                                       sample_coords, samples,
                                       1e-12,
                                             imu_accel_world[first_pose_i:last_pose_i+1], imu_accel_cov, dt,
                                       prior, prior_cov)

            delta = np.linalg.solve(A.T @ A, A.T @ b)

            delta = delta.reshape((lag, 4))
            state += delta
            state_optimization_history[i + 1] = state

        print(f'Position error after opt: {np.linalg.norm(current_gt_poses - state[:, :2], axis=-1)}')
        print(f'Dead-reckoned position error: {np.linalg.norm(current_gt_poses - current_dr_poses, axis=-1)}')

        # plot_phase_error(sample_coords, samples, err_offset, pulses, current_gt_poses,
        #                  pose_history=state_optimization_history[..., :2])

        # Update map
        update_map(map, grid_pos, state[0, :2], pulses[0], visualize=False)
        computed_trajectory[last_pose_i-lag+1] = state[0]

        # Marginalize out first pose
        prior = state[1]  # todo: this is hacky but should work well enough
        prior_cov = np.linalg.inv(A.T @ A)[:4, :4]

        # Add initial value for next pose
        next_state = propagate_state(state[-1], imu_accel_world[last_pose_i], dt)
        state[:-1] = state[1:]
        state[-1] = next_state

        # Compute next pulse
        # SIMULATION START
        next_gt_pose = gt_pose[last_pose_i + 1]
        signal = get_signal(next_gt_pose, signal_t, target_points)
        pulses[:-1] = pulses[1:]
        pulses[-1] = pulse_compress(signal, signal_t)
        # SIMULATION END



    _, dead_reckon_map = initialize_map()
    _, gt_map = initialize_map()
    build_map_from_traj(dead_reckon_map, grid_pos, gt_pose, target_points, est_traj=dead_reckon_traj[:, :2])
    build_map_from_traj(gt_map, grid_pos, gt_pose, target_points)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    visualize_map(map, computed_trajectory[:, :2], target_points, ax=ax1)
    visualize_map(gt_map, gt_pose, target_points, ax=ax2)
    visualize_map(dead_reckon_map, dead_reckon_traj[:, :2], target_points, ax=ax3)
    plt.show()
