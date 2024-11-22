import numpy as np
import matplotlib.pyplot as plt
import pickle
from math import pi
import torch
from scipy.signal import correlate

from tracer.motion_random_tracer import Trajectory
from sonar.phased_array import RectangularArray, DASBeamformer
from tracer.geometry import db_to_amplitude, az_el_to_direction_grid


filename = 'exp_res.pkl'

with open(filename, "rb") as f:
    simulation_results = pickle.load(f)

n_sinks = simulation_results['n_sinks']
rx_signals = simulation_results['rx_pattern']
code = simulation_results['tx_pattern']
T_rx = simulation_results['T_rx']
T_tx = simulation_results['T_tx']
trajectory: Trajectory = simulation_results['trajectory']
array: RectangularArray = simulation_results['array']
C = simulation_results['C']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

beamformer = DASBeamformer(array, C)

# Configure steering angles
steering_az = np.array([0])
steering_el = np.array([pi / 2])
steering_dir = az_el_to_direction_grid(steering_az, steering_el)
steering_dir = steering_dir.reshape(-1, 3)

# Gain LUT
looking_res_deg = 1
looking_az = np.linspace(-pi, pi, 360 // looking_res_deg)
looking_el = np.linspace(0, pi, 180 // looking_res_deg)  # elevation from x-y plane toward +z
looking_dir = az_el_to_direction_grid(looking_az, looking_el)

k = 2 * pi * code._f_hi / C

gain_db = beamformer.get_gain(steering_dir, looking_dir.reshape(-1, 3), k)
gain_db = gain_db.reshape((steering_dir.shape[0], looking_dir.shape[0], looking_dir.shape[1]))
gain = db_to_amplitude(gain_db)

poses = []
correlations = []

for pose_i in range(1, 20):
    pattern_i = pose_i - 1

    beamformed_signal = beamformer.beamform_receive(steering_dir, rx_signals[pattern_i], T_rx, k)

    _, pose = trajectory[pose_i]

    correlation = np.array([correlate(s, code.baseband, mode="valid") for s in beamformed_signal])

    poses.append(pose)
    correlations.append(correlation)

with open("autofocus.pkl", "wb") as fp:
    pickle.dump({ "poses": poses, "correlations": correlations }, fp)
