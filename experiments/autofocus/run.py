import numpy as np
from math import pi
import pickle
from spatialmath import SE3
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt, cheby1
from scipy.optimize import minimize
import torch

with open("autofocus.pkl", "rb") as fp:
    res = pickle.load(fp)
    poses = res["poses"]
    correlations = res["correlations"]

correlations = [correlation[0] for correlation in correlations]

T_rx = 1e-6

C = 1500
f_m = 7.5e3
w_m = 2 * pi * f_m # rad / s
k_m = w_m / C # rad / m
l_m = C / f_m

filter = cheby1(4, 0.1, 0.5 * f_m, btype="low", fs=1 / T_rx, output="sos")

correlations = [
    1e-6 * sosfiltfilt(filter, correlation, axis=-1) for correlation in correlations
]

old_poses = poses[:10]
old_correlations = correlations[:10]
new_poses = poses[10:]
new_correlations = correlations[10:]

def evaluate_map(poses, correlations, grid_x, grid_y):
    map = np.zeros(grid_x.shape, dtype=np.complex128)

    for pose, correlation in zip(poses, correlations):
        d = np.sqrt((grid_x - pose.t[0]) ** 2 + (grid_y - pose.t[1]) ** 2)
        rt_d = 2 * d

        index = (rt_d / (T_rx * C)).astype(int)
        valid = (0 <= index) & (index < len(correlation))

        phasor = correlation * np.exp(1j * 2 * np.pi * f_m * T_rx * np.arange(len(correlation)))

        map[valid] += phasor[index[valid]] * np.exp(-1.0j * k_m * rt_d[valid])

    return map

grid_spacing = l_m / 8
grid_size_x = 8
grid_size_y = 8
grid_x = grid_spacing * np.arange(grid_size_x / grid_spacing) - (grid_size_x / 2) + (grid_spacing / 2)
grid_y = grid_spacing * np.arange(grid_size_y / grid_spacing) - (grid_size_y / 2) + (grid_spacing / 2)
grid_x, grid_y = np.meshgrid(grid_x, grid_y, indexing="xy")
grid_h, grid_w = grid_x.shape

map = evaluate_map(poses, correlations, grid_x, grid_y)

ts = np.array([pose.t for pose in poses])

noisy_poses = [pose @ SE3.Trans(np.random.uniform(low=-1e-2, high=1e-2, size=3)) for pose in new_poses]
noisy_ts = np.array([pose.t for pose in noisy_poses])

noisy_map = evaluate_map(noisy_poses, new_correlations, grid_x, grid_y) + evaluate_map(old_poses, old_correlations, grid_x, grid_y)

plt.subplot(3, 1, 1)
plt.imshow(np.abs(map))
plt.subplot(3, 1, 2)
plt.imshow(np.abs(noisy_map))
plt.subplot(3, 1, 3)
plt.plot(ts[:, 0], ts[:, 1])
plt.plot(noisy_ts[:, 0], noisy_ts[:, 1])
plt.show()

# def wrap2pi(x):
#     return ((x + pi) % (2 * pi)) - pi
#
# def grad(positions, target_positions, target_phases):
#
#     grads = np.zeros_like(positions)
#
#     for i, correlation in enumerate(correlations):
#         correlation = correlation[0]
#         position = positions[i]
#
#         for target_i, target_position in enumerate(target_positions):
#             d = np.linalg.norm(target_position - position)
#             rt_d = 2 * d
#
#             index = (rt_d / (T_rx * C)).astype(int)
#             valid = (0 <= index) & (index < len(correlation))
#
#             if not valid:
#                 continue
#
#             phasor = correlation[index] * np.exp(1j * 2 * np.pi * f_m * T_rx * index)
#
#             print(index)
#
#             # Maximize amount of phasor that matches with target_phase
#             # == minimize angle between phasor and target_phase
#
#             angle_diff = wrap2pi(target_phases[target_i] - np.angle(phasor))
#
#             grads[2 * i: 2 * i + 2] += angle_diff * (target_position - position)
#
#     return grads


# opt_poses = noisy_poses.copy()
#
# target_positions = np.array([[1.0, 0.0]])
# target_phases = np.array([0.0])
#
# alpha = 1e-3
#
# opt_ts = np.array([pose.t for pose in opt_poses])
# opt_positions = opt_ts[:, 0:2]
#
# for i in range(10):
#     print(opt_positions)
#
#     opt_positions -= alpha * grad(opt_positions, target_positions, target_phases)
#
#     plt.subplot(3, 1, 1)
#     plt.imshow(np.abs(map))
#     plt.subplot(3, 1, 3)
#     plt.plot(ts[:, 0], ts[:, 1])
#     plt.plot(opt_positions[:, 0], opt_positions[:, 1])
#     plt.show()

# BEGIN SECTION - NAIVE CONTRAST MAXIMIZATION

# def f(x):
#     translated_poses = [None] * len(poses)
#
#     for i in range(len(x) // 2):
#         translated_poses[i] = poses[i] @ SE3.Trans(x[2 * i], x[2 * i + 1], 0)
#
#     # target point
#
#     grid_x = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
#     grid_y = np.array([[-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]])
#
#     map = evaluate_map(translated_poses, correlations, grid_x, grid_y)
#
#     contrast = np.mean(np.abs(map) ** 2)
#
#     return -contrast
#
# x0 = np.zeros((2 * len(poses)))
# # res = minimize(f, x0, bounds=[(-l_m / 2, l_m / 2)] * (2 * len(poses)), options={"disp": True})
# res = minimize(f, x0, options={"disp": True}, method="Powell")
#
# print(res)
#
# adjusted_poses = [pose @ SE3.Trans(res.x[2 * i], res.x[2 * i + 1], 0) for (i, pose) in enumerate(noisy_poses)]
# adjusted_ts = np.array([pose.t for pose in adjusted_poses])
#
# eval_grid_x = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
# eval_grid_y = np.array([[-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]])
# print(f"before: {evaluate_map(noisy_poses, correlations, eval_grid_x, eval_grid_y)}")
# print(f"after: {evaluate_map(adjusted_poses, correlations, eval_grid_x, eval_grid_y)}")
#
# adjusted_map = evaluate_map(adjusted_poses, correlations, grid_x, grid_y)
#
# plt.subplot(3, 1, 1)
# plt.imshow(np.abs(map))
# plt.subplot(3, 1, 2)
# plt.imshow(np.abs(adjusted_map))
# plt.subplot(3, 1, 3)
# plt.plot(ts[:, 0], ts[:, 1])
# plt.plot(noisy_ts[:, 0], noisy_ts[:, 1])
# plt.plot(adjusted_ts[:, 0], adjusted_ts[:, 1])
# plt.show()

# END SECTION - NAIVE CONTRAST MAXIMIZATION

# BEGIN SECTION - TORCH CONTRAST MAXIMIZATION

def evaluate_map_pt(positions, correlations, sample_positions):
    map = torch.zeros(sample_positions.shape[:-1], dtype=torch.complex128)

    for position, correlation in zip(positions, correlations):
        d = torch.linalg.norm(sample_positions - position, axis=-1)
        rt_d = 2 * d

        index = (rt_d / (T_rx * C)).to(int)
        valid = (0 <= index) & (index < len(correlation))

        phasor = correlation * torch.exp(1j * 2 * np.pi * f_m * T_rx * torch.arange(len(correlation)))

        map[valid] += phasor[index[valid]] * torch.exp(-1.0j * k_m * rt_d[valid])

    return map

positions_pt = torch.tensor(noisy_ts[:, :2], requires_grad=True)
correlations_pt = [torch.tensor(correlation, requires_grad=False) for correlation in new_correlations]
sample_positions_pt = torch.tensor(np.stack((grid_x, grid_y), axis=-1), requires_grad=False)

old_positions_pt = torch.tensor(ts[:10, :2], requires_grad=False)
old_correlations_pt = [torch.tensor(correlation, requires_grad=False) for correlation in old_correlations]

base_map_pt = evaluate_map_pt(old_positions_pt, old_correlations_pt, sample_positions_pt)

optimizer = torch.optim.Adam([positions_pt], lr=1e-3)

while True:
    map_pt = evaluate_map_pt(positions_pt, correlations_pt, sample_positions_pt) + base_map_pt

    plt.imshow(np.abs(map_pt.detach()))
    plt.show()

    plt.plot(ts[:, 0], ts[:, 1], label="true")
    plt.plot(noisy_ts[:, 0], noisy_ts[:, 1], label="noisy")
    plt.plot(positions_pt.detach()[:, 0], positions_pt.detach()[:, 1], label="opt")
    plt.legend()
    plt.show()

    # map_norm = torch.abs(map_pt)
    # map_norm = (map_norm - map_norm.min()) / (map_norm.max() - map_norm.min())

    # bins = 16
    #
    # bin_edges = torch.linspace(0, 1, bins + 1)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins
    #
    # # Compute the "soft" histogram using a Gaussian kernel
    # # Reshape for broadcasting: flattened_image -> (N, 1), bin_centers -> (1, bins)
    # diff = map_norm.reshape(-1).unsqueeze(1) - bin_centers.unsqueeze(0)
    # bin_width = bin_edges[1] - bin_edges[0]
    # hist = torch.exp(-0.5 * (diff / (bin_width / 2)) ** 2)  # Gaussian kernel
    # hist = hist.sum(dim=0)  # Sum across all pixels
    #
    # # Normalize the histogram to get probabilities
    # probs = hist / hist.sum()
    #
    # # Compute entropy
    # entropy = -torch.sum(probs * torch.log2(probs + 1e-8))

    loss_pt = -(torch.abs(map_pt) ** 2).sum()

    optimizer.zero_grad()

    loss_pt.backward()

    optimizer.step()

# END SECTION - TORCH CONTRAST MAXIMIZATION