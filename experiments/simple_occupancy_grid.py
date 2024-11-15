import pickle
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from spatialmath import SE3, SO3
from tqdm import tqdm

from sonar.occupancy_grid import OccupancyGridMap
from sonar.phased_array import RectangularArray
from sonar.utils import BarkerCode, FMBarker
from tracer.motion_random_tracer import Trajectory
from tracer.scene import UniformContinuousAngularDistribution

with open('exp_res.pkl', "rb") as f:
    simulation_results = pickle.load(f)

n_sinks: int = simulation_results['n_sinks']
rx_pattern: List[np.array] = simulation_results['rx_pattern']
T_rx: float = simulation_results['T_rx']
trajectory: Trajectory = simulation_results['trajectory']

arr = RectangularArray(10, 10, 0.0075, UniformContinuousAngularDistribution(
    min_az=-pi, max_az=pi, min_el=0, max_el=pi/2
))

f_low = 100_000
f_high = 150_000
T_bit = 1e-3
code = FMBarker(BarkerCode.Sequence.BARKER_7, f_low, f_high, T_rx, T_bit)


map = OccupancyGridMap(50, 50, 50, 0.08,
                       SE3.Rt(SO3(), [-2, -2, -2]),
                       code,
                       arr)

azimuth = np.linspace(0, 2 * np.pi, 18)
elevation = np.linspace(0, np.pi / 2, 4)
azimuth, elevation = np.meshgrid(azimuth, elevation)
azimuth = azimuth.reshape(-1)
elevation = elevation.reshape(-1)

x = np.cos(elevation) * np.cos(azimuth)
y = np.cos(elevation) * np.sin(azimuth)
z = np.sin(elevation)

steering_dirs = np.transpose([x, y, z])


for pose_i in range(len(rx_pattern)):
    pose_pattern = rx_pattern[pose_i]
    pose_pattern = pose_pattern.reshape((arr.nx, arr.ny, -1))
    _, world_t_vehicle = trajectory[pose_i]
    map.add_measurement(pose_pattern,
                        T_rx,
                        steering_dirs,
                        world_t_vehicle,
                        code,
                        (f_low, f_high),
                        T_bit,
                        arr.beamform_receive)

    with open('map.pkl', "wb") as f:
        pickle.dump(map, f)

    exit()

with open('map.pkl', "wb") as f:
    pickle.dump(map, f)
