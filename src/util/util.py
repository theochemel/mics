import numpy as np
from matplotlib import pyplot as plt

from util.config import Config


def wrap2pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def visualize_map(map, config: Config, traj=None, vel=None, targets=None, ax=None, save=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.abs(map), extent=config.grid_extent_xy)
    if targets is not None:
        ax.scatter(targets[:, 0], targets[:, 1], c="r", marker="x")
    if traj is not None:
        if vel is not None:
            scatter = ax.scatter(traj[:,0], traj[:,1], c=vel, s=1, cmap='plasma')
            colorbar = plt.colorbar(scatter)
            colorbar.set_label('Vel')
        else:
            ax.plot(traj[:, 0], traj[:, 1], linewidth=2)

    if ax is None and save is not None:
        fig.savefig(save, dpi=600, bbox_inches='tight')
    if ax is None:
        plt.suptitle("Map")
        plt.show()

def plot_phase_error(config: Config, sample_coords, samples,
                     offset_grid, pulses, gt_poses, pose_history=None, prior=None):

    sample_weights = np.abs(samples)
    sample_phases = np.angle(samples)

    N_poses = len(gt_poses)

    fig, axes = plt.subplots(2, int(np.ceil(N_poses / 2)))
    axes = axes.reshape(-1)
    for i in range(N_poses):
        grid = offset_grid + gt_poses[i]
        sample_range = np.linalg.norm(sample_coords[:, np.newaxis, np.newaxis] - grid[np.newaxis], axis=-1)
        sample_rt_t = (2.0 * sample_range) / config.C
        k = sample_rt_t / config.Ts
        k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
        k_a = k - k_i  # Fractional parts
        k_i_plus_1 = np.clip(k_i + 1, 0, pulses.shape[1] - 1)  # Upper bounds (clipped)
        interp_pulse = (1 - k_a) * pulses[i, k_i] + k_a * pulses[i, k_i_plus_1]
        update = interp_pulse * np.exp(1.0j * config.w * sample_rt_t)
        est_phase = np.angle(update)
        avg_phase_error = np.sum(
            sample_weights[:, np.newaxis, np.newaxis] * (wrap2pi(sample_phases[:, np.newaxis, np.newaxis] - est_phase))**2, axis=0
        ) / np.sum(sample_weights)

        axes[i].contourf(grid[..., 0], grid[..., 1], avg_phase_error, cmap='viridis', levels=100)
        axes[i].scatter(gt_poses[i, 0], gt_poses[i, 1], color='green', linewidths=2)
        if pose_history is not None:
            axes[i].plot(pose_history[:, i, 0], pose_history[:, i, 1], color='red', linewidth=2)
        axes[i].set_title(f'State {i}')
    if prior is not None:
        axes[0].scatter(prior[0], prior[1], color='red', linewidths=2)
    plt.show()

def plot_phase_error_3d(config: Config, sample_coords, samples,
                        offset_grid, pulse, gt_pose):

    sample_weights = np.abs(samples)
    sample_phases = np.angle(samples)

    grid = offset_grid + gt_pose
    sample_range = np.linalg.norm(sample_coords[:, np.newaxis, np.newaxis] - grid[np.newaxis], axis=-1)
    sample_rt_t = (2.0 * sample_range) / config.C
    k = sample_rt_t / config.Ts
    k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
    k_a = k - k_i  # Fractional parts
    k_i_plus_1 = np.clip(k_i + 1, 0, pulse.shape[-1] - 1)  # Upper bounds (clipped)
    interp_pulse = (1 - k_a) * pulse[k_i] + k_a * pulse[k_i_plus_1]
    update = interp_pulse * np.exp(1.0j * config.w * sample_rt_t)
    est_phase = np.angle(update)
    avg_phase_error = np.sum(
        sample_weights[:, np.newaxis, np.newaxis] * (wrap2pi(sample_phases[:, np.newaxis, np.newaxis] - est_phase))**2, axis=0
    ) / np.sum(sample_weights)

    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid[..., 0], grid[..., 1], avg_phase_error, cmap='cividis')
    plt.show()
    fig.savefig('2d_phase_err_landscape.png', dpi=600, bbox_inches='tight')
