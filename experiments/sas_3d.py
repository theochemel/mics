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
from tracer.geometry import db_to_amplitude, az_el_to_direction_grid
from tracer.motion_random_tracer import Trajectory
from tracer.scene import Surface, SimpleMaterial, Scene

from visualization.visualize_map import plot_slices_with_colormap
import open3d as o3d

filename = 'exp_res.pkl'

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

sand_material = SimpleMaterial(
    absorption=0.9,
)
geometry = [
    Surface(
        id=f"cube",
        pose=SE3.Rt(SO3(), np.array([0.0, 0.0, 0.0])),
        material=sand_material,
        mesh=o3d.io.read_triangle_mesh("assets/cube.ply"),
    ),
]
scene = Scene([], [], geometry)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

beamformer = DASBeamformer(array, C)

# Configure steering angles
steering_az = np.linspace(-pi / 3, pi / 3, 3)
steering_el = np.array([pi / 2])
steering_dir = az_el_to_direction_grid(steering_az, steering_el)
steering_dir = steering_dir.reshape(-1, 3)

# Gain LUT
looking_res_deg = 1
looking_az = np.linspace(-pi, pi, 360 // looking_res_deg)
looking_el = np.linspace(0, pi, 180 // looking_res_deg)  # elevation from x-y plane toward +z
looking_dir = az_el_to_direction_grid(looking_az, looking_el)

k = 2 * pi * code.carrier / C

gain_db = beamformer.get_gain(steering_dir, looking_dir.reshape(-1, 3), k)
gain_db = gain_db.reshape((steering_dir.shape[0], looking_dir.shape[0], looking_dir.shape[1]))
gain = db_to_amplitude(gain_db)

# interference wave

# TODO: Figure this out for fm chirp, sinc interference
# T_m = len(code.baseband) * T_rx
f_m = 10e3
w_m = 2 * pi * f_m # rad / s
k_m = w_m / C # rad / m
l_m = C / f_m

map_size = l_m
map = OccupancyGridMap(100, 100, 1, map_size,
                       SE3.Rt(SO3(), (-50 * map_size, -50 * map_size, 0.0)),
                       device)

vehicle_poses = [
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=2).transform(p)
    for _, p in trajectory
]

# geometry = scene.visualization_geometry() + vehicle_poses
geometry = vehicle_poses

filter = cheby1(4, 0.1, 1e3, btype='low', fs=1 / T_rx, output='sos')

for pose_i in tqdm(range(1, len(rx_signals))):
    pattern_i = pose_i - 1

    beamformed_signal = beamformer.beamform_receive(steering_dir, rx_signals[pattern_i], T_rx, k)

    beamformed_signal += np.random.normal(loc=0, scale=1e-2, size=beamformed_signal.shape)

    _, pose = trajectory[pose_i]

    for beam_i in range(beamformed_signal.shape[0]):
        correlation = correlate(beamformed_signal[beam_i], code.baseband, mode="valid")
        lags = correlation_lags(beamformed_signal.shape[1], len(code.baseband), mode="valid")
        correlation = sosfiltfilt(filter, np.abs(correlation))

        correlation_t = lags * T_rx

        complex_baseband = correlation * (correlation_t ** 4)

        map.add_measurement(
            torch.tensor(complex_baseband, device=device),
            torch.tensor(gain[beam_i], device=device),
            k_m,
            T_rx,
            C,
            world_t_source=pose,
            world_t_sink=pose,
            visualization_geometry=geometry
        )

    map_abs = np.abs(map.get_map().cpu().numpy())

    # ts = np.array([(map._world_t_map.inv() @ pose[1]).t for pose in trajectory._poses])

    # plt.subplot(1, 3, 1)
    # plt.imshow(map_abs[map_abs.shape[0] // 2, :, :])
    # plt.title("X = 0")
    # plt.subplot(1, 3, 2)
    # plt.imshow(map_abs[:, map_abs.shape[1] // 2, :])
    # plt.title("Y = 0")
    # plt.subplot(1, 3, 3)
plt.imshow(map_abs[:, :, map_abs.shape[2] // 2])
plt.title("Z = 0")
plt.show()

plt.imshow(np.angle(map.get_map().cpu().numpy()))
plt.show()

# cmap = matplotlib.cm.viridis
# norm = matplotlib.colors.Normalize(vmin=0, vmax=len(intensities) - 1)
# colors = cmap(norm(range(len(intensities))))
#
# fig, axs = plt.subplots(2)
#
# for i, (intensity, signal) in enumerate(zip(intensities, signals)):
#     axs[0].plot(intensity, c=colors[i])
#     axs[1].plot(signal, c=colors[i])
#
# plt.show()
#
plot_slices_with_colormap(map_abs / map_abs.max(), map.world_t_grid,
                          geometry=[],
                          n_slices=5,
                          axis=1,
                          vehicle_pose=pose)

with open('map.pkl', 'wb') as f:
    pickle.dump(map.get_map().cpu().numpy(), f)

print('done')
