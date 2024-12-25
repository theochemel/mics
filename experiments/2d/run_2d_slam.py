import argparse
import copy
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle as pkl

import numpy as np

from motion.imu import IMUMeasurement, get_world_accel_2d
from motion.trajectory import Trajectory
from util.config import Config
from sas_2d.sas_2d import SAS
from sas_2d.slam import FixedLagSmoother
from util.signals import pulse_compress
from util.util import visualize_map, plot_phase_error, plot_phase_error_3d

if __name__ == '__main__':

    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    with open(Path.cwd() / Path(args.filename), 'rb') as f:
        sim_result = pkl.load(f)

    waves: np.ndarray = sim_result["waves"]
    gt_traj: Trajectory = sim_result["trajectory"]
    imu_measurements: IMUMeasurement = sim_result["imu_measurements"]

    gt_poses = np.array([p.t[:2] for p in gt_traj.poses])

    config = Config()
    dt = 0.05  # todo do we need this?

    sas = SAS(config)

    # Initialize map with ground-truth trajectory
    n_init_poses = 2
    init_pulses = []
    signal_t = config.Ts * np.arange(waves[0].shape[0])
    for pose_idx in range(n_init_poses):
        pose_2d = gt_poses[pose_idx]
        signal = waves[pose_idx]
        # plt.plot(signal_t, np.real(signal))
        # plt.plot(signal_t, np.imag(signal))
        # plt.show()
        pulse = pulse_compress(signal, config)
        # plt.plot(signal_t, np.real(pulse))
        # plt.plot(signal_t, np.imag(pulse))
        # plt.show()
        init_pulses.append(pulse)
        sas.update_map(pose_2d, pulse, vis=(pose_idx==n_init_poses-1))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.abs(sas.map), extent=config.grid_extent_xy)
    fig.savefig('2d_sas_init_map.png', dpi=600, bbox_inches='tight')
    # Initialize fixed-lag smoother
    lag = 2

    fls_init_poses = np.array([gt_traj.poses[i].t[:2] for i in range(n_init_poses - lag, n_init_poses)])
    fls_init_vel = gt_traj.velocity_world[n_init_poses-lag:n_init_poses, :2]
    fls_init_state = np.concatenate((fls_init_poses, fls_init_vel), axis=1)

    fls_init_pulses = init_pulses[-lag:]
    fls_init_accel = [
        get_world_accel_2d(imu_measurements.acceleration_body[i], imu_measurements.orientation_rpy[i, 2])
        for i in range(n_init_poses-lag, n_init_poses)
    ]

    fls_init_prior = fls_init_state[0]
    fls_init_prior_cov = np.eye(4) * 1e-6 ** 2

    fls_p = FixedLagSmoother(config,
                             sas,
                             np.copy(fls_init_state),
                             np.array(fls_init_pulses),
                             np.array(fls_init_accel),
                             fls_init_prior,
                             fls_init_prior_cov,
                             lag,
                             use_prior=True,
                             use_motion=True
                             )
    #
    # fls_np = FixedLagSmoother(config,
    #                          sas,
    #                          fls_init_state,
    #                          np.array(fls_init_pulses),
    #                          np.array(fls_init_accel),
    #                          fls_init_prior,
    #                          fls_init_prior_cov,
    #                          lag,
    #                          use_prior=False,
    #                          use_motion=False)


    # Offset grid for error plotting
    l = config.C / config.chirp_fc
    err_x = np.linspace(0, 2 * l, 20) - l
    err_y = np.linspace(0, 2 * l, 20) - l
    err_x, err_y = np.meshgrid(np.flip(err_y), err_x, indexing='ij')
    err_offset = np.stack((err_x, err_y), axis=-1)

    trajectory = np.zeros((len(waves), 4))
    trajectory[:n_init_poses] = np.concatenate((gt_poses[:n_init_poses], gt_traj.velocity_world[:n_init_poses, :2]),
                                               axis=-1)

    gt_vel = gt_traj.velocity_world[:, :2]
    visualize_map(sas.map, config,
                  gt_poses[:n_init_poses], vel=np.linalg.norm(gt_vel[:n_init_poses], axis=1))
    plt.show()

    for pose_idx in (range(n_init_poses, len(waves))):
        current_gt_poses = gt_poses[pose_idx-lag+1:pose_idx+1]
        print()
        print(f"--- POSES {pose_idx - lag + 1} - {pose_idx} ---")
        # print("Prior covariance:")
        # print(prior_cov)
        signal = waves[pose_idx]
        pulse = pulse_compress(signal, config)

        accel = get_world_accel_2d(imu_measurements.acceleration_body[pose_idx],
                                   imu_measurements.orientation_rpy[pose_idx, 2])

        sample_coords, samples, state_opt_traj = fls_p.update(accel, pulse, dt)
        # sample_coords_np, samples_np, state_opt_traj_np = fls_np.update(accel, pulse, dt)

        print(f"delta: {state_opt_traj[-1] - state_opt_traj[0]}")
        print(f'Position error after opt: {np.linalg.norm(current_gt_poses - fls_p.state[:, :2], axis=-1)}')
        # print(f'Position error after opt (np): {np.linalg.norm(current_gt_poses - fls_np.state[:, :2], axis=-1)}')
        # plot_phase_error(config, sample_coords, samples, err_offset, fls_p.pulses,
        #                  gt_poses[pose_idx-lag+1:pose_idx+1],
        #                  pose_history=state_opt_traj[..., :2],
        #                  prior=fls_p.prior)
        print(f"GT POSE: {gt_poses[pose_idx-lag+1]}")
        # plot_phase_error_3d(config, sample_coords, samples, err_offset, fls_p.pulses[0],
        #                     gt_poses[pose_idx-lag+1])
        trajectory[pose_idx-lag+1] = fls_p.state[0]

        # if pose_idx % 20 == 0:
        #     visualize_map(sas.map, config,
        #                   trajectory[:pose_idx-lag+2, :2])
        #     plt.show()


    visualize_map(sas.map, config,
                  trajectory[:-5, :2], vel=np.linalg.norm(trajectory[:-5, 2:], axis=1))
    plt.show()

    with open(f'results_lag_{lag}_init_{n_init_poses}.pkl', 'wb') as f:
        res = {
            'computed_traj': trajectory,
            'gt_traj': gt_traj,
            'map': sas.map,
            'imu_measurements': imu_measurements,
            'config': config
        }
        pkl.dump(res, f)

