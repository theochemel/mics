import argparse
from pathlib import Path

from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle as pkl

import numpy as np
from spatialmath import SE3

from motion.imu import IMU
from sim_2d.sim_2d import Sim2d, World, make_forest_targets
from motion.linear_constant_acceleration_trajectory import LinearConstantAccelerationTrajectory
from util.config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o')
    args = parser.parse_args()

    np.random.seed(1)

    config = Config()
    dt = 0.05

    world = World(config)
    targets = make_forest_targets((1, 9, 1, 9), 9, 9, 0.1)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.scatter(targets[..., 0], targets[..., 1], marker='x', c='r', linewidths=2)
    fig.savefig('target_forest.svg', dpi=600, bbox_inches='tight')
    plt.show()
    world.add_targets(targets)

    trajectory = LinearConstantAccelerationTrajectory(
        keyposes = [
            SE3.Trans(4.5, 4.5, 0),
            SE3.Trans(5.5, 4.5, 0),
            SE3.Trans(5.5, 5.5, 0),
            SE3.Trans(4.5, 5.5, 0),
            SE3.Trans(4.5, 4.5, 0),
        ],
        max_velocity=0.1,
        acceleration=0.1,
        dt=dt
    )

    sim = Sim2d(world, trajectory, config)

    waves = []

    for pose_idx in tqdm(range(len(trajectory.poses))):
        pose_wave = sim.get_signal_at_pose(pose_idx)
        waves.append(pose_wave)

    waves = np.array(waves)

    imu_measurements = sim.get_imu_measurements()

    result_dict = {
        "waves": waves,
        "trajectory": trajectory,
        "imu_measurements": imu_measurements
    }

    with open(Path.cwd() / Path(args.o), 'wb') as f:
        pkl.dump(result_dict, f)
