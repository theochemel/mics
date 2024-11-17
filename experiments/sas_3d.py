import pickle

import numpy as np
from numpy import pi
import torch
from scipy.signal import correlate, cheby1, sosfilt
from spatialmath import SE3, SO3

from sonar.occupancy_grid import OccupancyGridMap
from sonar.phased_array import RectangularArray
from sonar.utils import BarkerCode, az_el_to_direction_grid
from tracer.motion_random_tracer import Trajectory

filename = 'test.pkl'

with open(filename, "rb") as f:
    simulation_results = pickle.load(f)

n_sinks = simulation_results['n_sinks']
rx_pattern = simulation_results['rx_pattern']
T_rx = simulation_results['T_rx']
T_tx = simulation_results['T_tx']
trajectory: Trajectory = simulation_results['trajectory']
array: RectangularArray = simulation_results['array']
code: BarkerCode = simulation_results['code']
C = simulation_results['C']

device = torch.device('cuda')

map = OccupancyGridMap(100, 100, 100, 0.03,
                       SE3.Rt(SO3(), (-1, -1, -1)),
                       device)

# Configure steering angles
steering_az = np.linspace(-pi, pi, 18)
steering_el = np.linspace(0, pi / 2, 5, endpoint=True)  # elevation from x-y plane toward +z
steering_dir = az_el_to_direction_grid(steering_az, steering_el)

# Gain LUT
looking_res_deg = 1
looking_az = np.linspace(-pi, pi, 360 // looking_res_deg)
looking_el = np.linspace(0, pi / 2, 90 // looking_res_deg)  # elevation from x-y plane toward +z
looking_dir = az_el_to_direction_grid(looking_az, looking_el)

f = np.array([code.carrier])
k = 2*pi*f / C

gain_db = array.get_gain(steering_dir, looking_dir, np.array([k]))
gain = 10 ** (gain_db[..., 0] / 20)

filter = cheby1(4, 3, 0.5*f, btype='low', fs=1/T_rx)

for pose_i in range(len(rx_pattern)):
    pose_pattern = rx_pattern[pose_i]

    beamformed_signal = array.beamform_receive(np.array([k]), steering_dir, pose_pattern, T_rx)[0]

    for steering_i in range(len(steering_dir)):
        correlation = correlate(beamformed_signal[steering_i], code.baseband, mode='valid')
        correlation = sosfilt(filter, correlation)

