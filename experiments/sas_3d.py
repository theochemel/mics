import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
import torch
from scipy.signal import correlate, cheby1, sosfilt
from spatialmath import SE3, SO3
from tqdm import tqdm

from sonar.occupancy_grid import OccupancyGridMap
from sonar.phased_array import RectangularArray
from sonar.utils import BarkerCode, az_el_to_direction_grid
from tracer.motion_random_tracer import Trajectory
from tracer.scene import Surface, SimpleMaterial, Scene

from visualization.visualize_map import np_to_voxels, plot_slices_with_colormap, SliceViewer
import open3d as o3d

filename = 'exp_res.pkl'

with open(filename, "rb") as f:
    simulation_results = pickle.load(f)

n_sinks = simulation_results['n_sinks']
rx_pattern = simulation_results['rx_pattern']
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

map = OccupancyGridMap(100, 100, 100, 0.2,
                       SE3.Rt(SO3(), (-10, -10, -10)),
                       device)

# Configure steering angles
# steering_az = np.linspace(-pi, pi, 18)
# steering_el = np.linspace(0, pi / 2, 5, endpoint=True)  # elevation from x-y plane toward +z
steering_az = np.array([0])
steering_el = np.array([0])
steering_dir = az_el_to_direction_grid(steering_az, steering_el)
steering_dir = steering_dir.reshape(-1, 3)

# Gain LUT
looking_res_deg = 1
looking_az = np.linspace(-pi, pi, 360 // looking_res_deg)
looking_el = np.linspace(0, pi / 2, 90 // looking_res_deg)  # elevation from x-y plane toward +z
looking_dir = az_el_to_direction_grid(looking_az, looking_el)


fc = code.carrier
f = np.array([code.carrier,])
k = 2*pi*f / C

gain_db = array.get_gain(steering_dir, looking_dir.reshape(-1, 3), k)
gain = 10 ** (gain_db / 20)
gain = gain[0].reshape((len(steering_dir),) + looking_dir.shape[:2])

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# c = ax.pcolormesh(looking_az, looking_el, gain[0].T, shading='auto', cmap='viridis')
#
# plt.show()

filter = cheby1(4, 3, 0.1*fc, btype='low', fs=1/T_rx, output='sos')

# interference wave
T_m = len(code.baseband) * T_rx
f_m = 1 / T_m
w_m = 2*pi*f_m # rad / s
w_m_sample = w_m * T_rx  # rad / sample
k_m = w_m / C

print(f'T_m = {T_m}, f_m = {f_m}, lambda_m = {C / f_m}')

vehicle_poses = [
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(p)
    for _, p in trajectory
]

geometry = scene.visualization_geometry() + vehicle_poses


for pose_i in tqdm(range(1, len(rx_pattern))):
    pattern_i = pose_i - 1
    pose_pattern = rx_pattern[pattern_i].reshape((array.nx, array.ny, -1))

    beamformed_signal = array.beamform_receive(k, steering_dir, pose_pattern, T_rx)[0]
    # beamformed_signal = pose_pattern[0]

    _, pose = trajectory[pose_i]

    for steering_i in range(len(steering_dir)):
        raw_correlation = correlate(beamformed_signal[steering_i], code.baseband, mode='full')

        correlation = sosfilt(filter, np.abs(raw_correlation)) # ensure shift here is correct

        # plt.subplot(2, 1, 1)
        # plt.plot(raw_correlation)
        # plt.subplot(2, 1, 2)
        # plt.plot(correlation)
        # plt.show()

        # correlation_tt = np.arange(len(correlation)) * T_rx
        # plt.plot(correlation_tt, correlation)
        # plt.show()

        # range = (np.arange(len(correlation)) * T_rx * C) / 2.0
        range_spacing = (T_rx * C) / 2.0
        intensity = correlation * np.exp(2j * w_m_sample * np.arange(len(correlation)))
        intensity /= intensity.max()

        steering_gain = gain[steering_i]

        map.add_measurement(range_spacing,
                            torch.tensor(intensity, device=device),
                            k_m,
                            torch.tensor(gain[steering_i], device=device),
                            pose,
                            visualization_geometry=geometry)

    map_abs = np.abs(map.get_map().cpu().numpy())
    map_abs = (map_abs - map_abs.min()) / (map_abs.max() - map_abs.min())

    plt.subplot(1, 3, 1)
    plt.imshow(map_abs[map_abs.shape[0] // 2, :, :])
    plt.title("X = 0")
    plt.subplot(1, 3, 2)
    plt.imshow(map_abs[:, map_abs.shape[1] // 2, :])
    plt.title("Y = 0")
    plt.subplot(1, 3, 3)
    plt.imshow(map_abs[:, :, map_abs.shape[2] // 2])
    plt.title("Z = 0")
    plt.show()

    plot_slices_with_colormap(map_abs, map.world_t_grid,
                              geometry=geometry,
                              n_slices=10,
                              axis=1,
                              vehicle_pose=pose)

with open('map.pkl', 'wb') as f:
    pickle.dump(map.get_map().cpu().numpy(), f)

print('done')
