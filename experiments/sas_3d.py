import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
import torch
from scipy.signal import correlate, cheby1, sosfilt
from spatialmath import SE3, SO3
from tqdm import tqdm
import matplotlib

from sonar.occupancy_grid import OccupancyGridMap
from sonar.phased_array import RectangularArray, DASBeamformer
from sonar.utils import BarkerCode
from tracer.geometry import db_to_amplitude, az_el_to_direction_grid
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

beamformer = DASBeamformer(array, C)

# Configure steering angles
# steering_az = np.array([-pi / 2, pi / 2])
# steering_el = np.linspace(0, pi / 2, 4, endpoint=False)
steering_az = np.array([0])
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

filter = cheby1(4, 0.1, code.carrier, btype='low', fs=1/T_rx, output='sos')

# interference wave
T_m = len(code.baseband) * T_rx
f_m = 1 / T_m
w_m = 2 * pi * f_m # rad / s
k_m = w_m / C
l_m = C / f_m

map_size = l_m / 2
map = OccupancyGridMap(100, 100, 100, map_size,
                       SE3.Rt(SO3(), (-50 * map_size, -50 * map_size, -50 * map_size)),
                       device)

vehicle_poses = [
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=2).transform(p)
    for _, p in trajectory
]

print(vehicle_poses)

# geometry = scene.visualization_geometry() + vehicle_poses
geometry = vehicle_poses

intensities = []
signals = []

for pose_i in tqdm(range(1, len(rx_pattern))):
    pattern_i = pose_i - 1

    beamformed_signal = beamformer.beamform_receive(steering_dir, rx_pattern[pattern_i], T_rx, k)

    _, pose = trajectory[pose_i]

    for steering_i in range(len(steering_dir)):
        reference_signal = np.cos(2 * np.pi * code.carrier * T_rx * np.arange(len(beamformed_signal[0])))
        demod_signal = reference_signal * beamformed_signal[steering_i]

        filt_demod_signal = sosfilt(filter, demod_signal)

        correlation = correlate(filt_demod_signal, code._digital, mode="valid")

        range_spacing = (T_rx * C) / 2.0
        intensity = correlation

        intensities.append(intensity)
        signals.append(rx_pattern[pattern_i][0])

        steering_gain = db_to_amplitude(gain_db[steering_i])

        map.add_measurement(range_spacing,
                            torch.tensor(intensity, device=device),
                            torch.tensor(steering_gain, device=device),
                            k_m,
                            pose,
                            visualization_geometry=geometry)

map_abs = np.abs(map.get_map().cpu().numpy())
map_abs = (map_abs - map_abs.min()) / (map_abs.max() - map_abs.min())

ts = np.array([(map._world_t_map.inv() @ pose[1]).t for pose in trajectory._poses])

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

cmap = matplotlib.cm.viridis
norm = matplotlib.colors.Normalize(vmin=0, vmax=len(intensities) - 1)
colors = cmap(norm(range(len(intensities))))

fig, axs = plt.subplots(2)

for i, (intensity, signal) in enumerate(zip(intensities, signals)):
    axs[0].plot(intensity, c=colors[i])
    axs[1].plot(signal, c=colors[i])

plt.show()

plot_slices_with_colormap(map_abs, map.world_t_grid,
                          geometry=geometry,
                          n_slices=30,
                          axis=1,
                          vehicle_pose=pose)

with open('map.pkl', 'wb') as f:
    pickle.dump(map.get_map().cpu().numpy(), f)

print('done')
