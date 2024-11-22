import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
import torch
from scipy.signal import correlate, correlation_lags, cheby1, sosfiltfilt
from spatialmath import SE3, SO3
from tqdm import tqdm
import matplotlib

from sonar.occupancy_grid import OccupancyGridMap
from sonar.phased_array import RectangularArray, DASBeamformer
from sonar.utils import BarkerCode
from tracer.geometry import db_to_amplitude, az_el_to_direction_grid, direction_to_az_el
from tracer.motion_random_tracer import Trajectory
from tracer.scene import Surface, SimpleMaterial, Scene

from visualization.visualize_map import np_to_voxels, plot_slices_with_colormap, SliceViewer
import open3d as o3d


filename = "exp_res.pkl"

with open(filename, "rb") as f:
    simulation_results = pickle.load(f)

n_sinks = simulation_results['n_sinks']
rx_signals = simulation_results['rx_pattern']
code: BarkerCode = simulation_results['tx_pattern']
T_rx = simulation_results['T_rx']
T_tx = simulation_results['T_tx']
trajectory: Trajectory = simulation_results['trajectory']
array: RectangularArray = simulation_results['array']
C = simulation_results['C']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

beamformer = DASBeamformer(array, C)

k = 2 * pi * code.carrier / C

img_h = 4
img_w = 4

steering_y = np.linspace(-0.5, 0.5, img_h)
steering_x = np.linspace(-0.5, 0.5, img_w)

steering_y, steering_x = np.meshgrid(steering_y, steering_x, indexing="ij")
steering_dir = np.stack((np.ones_like(steering_x), steering_x, steering_y), axis=-1)
steering_dir /= np.linalg.norm(steering_dir, axis=-1)[:, :, np.newaxis]
steering_dir_flat = steering_dir.reshape((-1, 3))

filter = cheby1(4, 3, 10e3, btype='low', fs=1 / T_rx, output='sos')

# for rx_signal in rx_signals:
#     rx_signal += np.random.normal(loc=0, scale=np.sqrt(np.var(rx_signal)), size=rx_signal.shape)

for i in range(0, len(rx_signals)):
    beamformed_signal = beamformer.beamform_receive(steering_dir_flat, rx_signals[i], T_rx, k)

    distance_map = np.zeros((steering_dir.shape[0], steering_dir.shape[1]))

    for beam_i in range(beamformed_signal.shape[0]):
        correlation = correlate(beamformed_signal[beam_i], code.baseband, mode="valid")
        lags = correlation_lags(len(beamformed_signal[beam_i]), len(code.baseband), mode="valid")

        correlation_distance = (lags * T_rx * C) / 2
        signal_distance = (np.arange(len(beamformed_signal[beam_i])) * T_rx * C) / 2

        intensity = correlation * (correlation_distance ** 4)

        # correlation *= correlation_distance ** 2

        plt.subplot(2, 1, 1)
        plt.plot(signal_distance, beamformed_signal[beam_i])
        plt.subplot(2, 1, 2)
        plt.plot(correlation_distance, correlation)
        plt.show()

        max_i = np.argmax(correlation)
        max_d = correlation_distance[max_i]

        distance_map[beam_i // steering_dir.shape[1], beam_i % steering_dir.shape[1]] = max_d

    # plt.imshow(distance_map)
    # plt.show()

    depth_map = (distance_map[:, :, np.newaxis] * steering_dir.reshape((img_h, img_w, 3)))[:, :, 0]

    plt.imshow(depth_map)
    plt.show()