import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize


C = 1500

# l_m = 1e-1
# f_m = C / l_m
# w_m = 2 * np.pi * f_m
# k_m = w_m / C

chirp_fc = 75e3
chirp_bw = 50e3
chirp_duration = 1e-3
chirp_K = chirp_bw / chirp_duration

chirp_fhi = chirp_fc + chirp_bw / 2

fs = 4 * chirp_fhi
Ts = 1 / fs

l_m = C / chirp_fhi

max_range = 10
max_rt_t = (2 * max_range) / C

target_points = np.array([
    [3, 3],
    [2, 3],
    [3, 2],
    [8, 8],
])

gt_traj_x = (l_m / 2) * np.arange(8 / (l_m / 2)) + 1
gt_traj_y = np.full_like(gt_traj_x, fill_value=1)
gt_traj = np.stack((gt_traj_x, gt_traj_y), axis=-1)

noisy_traj = gt_traj + np.random.normal(loc=0, scale=l_m / 4, size=gt_traj.shape)

grid_width = 10
grid_height = 10
grid_size = 1e-1

grid_x = grid_size * np.arange(int(grid_width / grid_size)) + grid_size / 2
grid_y = grid_size * np.arange(int(grid_height / grid_size)) + grid_size / 2

grid_y, grid_x = np.meshgrid(np.flip(grid_y), grid_x, indexing="ij")
grid_pos = np.stack((grid_x, grid_y), axis=-1)
grid_extent = [0, grid_width, 0, grid_height]

def chirp_envelope(t):
    return np.where(
        (t >= -chirp_duration / 2) & (t <= chirp_duration / 2),
        (1 / chirp_duration) * np.cos(np.pi * t / chirp_duration) ** 2,
        0
    )

def chirp(t):
    return chirp_envelope(t) * np.exp(2.0j * np.pi * chirp_fc * t + 1.0j * np.pi * chirp_K * t ** 2)

def get_signal(position):
    signal_t = Ts * np.arange(int(max_rt_t / Ts))

    signal = np.zeros_like(signal_t, dtype=np.complex128)

    for target_point in target_points:
        target_rt_t = (2 * np.linalg.norm(target_point - position)) / C

        signal += chirp(signal_t - target_rt_t)

    return signal_t, signal

signal_t, signal = get_signal(gt_traj[0])
plt.plot(signal_t, np.real(signal))
plt.plot(signal_t, np.imag(signal))
plt.show()

def importance_sample(img, n_samples):
    weights = img ** 4
    weights = weights / np.sum(weights)
    flat_weights = weights.flatten()
    indices = np.arange(len(flat_weights))

    samples = np.random.choice(indices, size=n_samples, replace=False, p=flat_weights)

    sample_y, sample_x = np.unravel_index(samples, img.shape)

    return np.stack((sample_x, sample_y), axis=-1)


# def opt(x, start_pos, odom, base_samples, sample_pos, signal, signal_range):
#     traj = x.reshape((-1, 2))
#
#     phasor = signal * np.exp(1.0j * 2 * np.pi * f_m * (2 * signal_range / C))
#
#     sample_range = np.linalg.norm(sample_pos[np.newaxis, :] - traj[:, np.newaxis], axis=-1)
#
#     sample_range_index = (sample_range / range_res).astype(int)
#
#     update_samples = base_samples + np.sum(
#         phasor[:, sample_range_index] \
#         * np.exp(-1.0j * k_m * (2 * sample_range)),
#         axis=0,
#     )
#
#     contrast = np.mean(np.abs(update_samples) ** 2) / 1e6
#
#     est_start_pos, est_odom = build_odom(traj)
#
#     start_pos_err = np.linalg.norm(start_pos - est_start_pos) ** 2
#     odom_err = np.sum(np.linalg.norm(est_odom - odom, axis=-1) ** 2)
#
#     return -contrast + 1e4 * start_pos_err + 1e-1 * odom_err

def build_odom(traj):
    start_pos = traj[0]

    odom = np.empty((traj.shape[0] - 1, 2))

    for i in range(1, len(traj)):
        odom[i - 1] = traj[i] - traj[i - 1]

    return start_pos, odom

def build_map(traj):
    map = np.zeros(grid_pos.shape[0:2], dtype=np.complex128)

    for position in traj:
        signal_t, signal = get_signal(position)

        grid_range = np.linalg.norm(grid_pos - position[np.newaxis, np.newaxis], axis=-1)
        grid_rt_t = (2.0 * grid_range) / C

        signal_demod = signal * np.exp(-2.0j * np.pi * chirp_fc * signal_t)

        grid_signal_t = signal_t[np.newaxis, np.newaxis, :] - grid_rt_t[:, :, np.newaxis]

        reference_signal = chirp(grid_signal_t)
        reference_signal_demod = reference_signal * np.exp(-2.0j * np.pi * chirp_fc * signal_t[np.newaxis, np.newaxis, :])

        update = np.sum(signal_demod * reference_signal_demod, axis=-1)

        plt.plot(np.real(signal_demod))
        plt.plot(np.real(reference_signal_demod[66, 33]))
        plt.show()

        map += update

        plt.subplot(1, 2, 1)
        plt.imshow(np.abs(map), extent=grid_extent)
        plt.subplot(1, 2, 2)
        plt.imshow(np.angle(map), extent=grid_extent)
        plt.show()

    return map


def estimate_odom(x, i):
    return x[2 * i + 2 : 2 * i + 4] - x[2 * i : 2 * i + 2]


def build_system(x, start_pos, odom, signal, sample_pos, base_sample, sigma_odom):
    sqrt_inv_odom = np.linalg.inv(sp.linalg.sqrtm(sigma_odom))

    n_odom = len(odom)
    n_sample = len(sample_pos)

    M = (n_odom + 1) * 2 + n_sample
    N = len(x)

    A = np.zeros((M, N))
    b = np.zeros((M,))

    # Anchor first pose
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    b[0] = start_pos[0]
    b[1] = start_pos[1]

    H_odom_pose1 = np.array([
        [-1, 0],
        [0, -1],
    ])

    H_odom_pose2 = np.array([
        [1, 0],
        [0, 1],
    ])

    # Odom measurements
    for i in range(n_odom):
        m_i = 2 + 2 * i

        A[m_i : m_i + 2, 2 * i : 2 * i + 2] = sqrt_inv_odom @ H_odom_pose1
        A[m_i : m_i + 2, 2 * i + 2 : 2 * i + 4] = sqrt_inv_odom @ H_odom_pose2

        error = odom[i] - estimate_odom(x, i)
        b[m_i : m_i + 2] = sqrt_inv_odom @ error

    for i in range(n_sample):
        pass

    return A, b

known_traj = gt_traj[:100]
unknown_traj = np.concatenate((gt_traj[100][np.newaxis], noisy_traj[101:105]), axis=0)

start_pos, unknown_odom = build_odom(unknown_traj)

base_map = build_map(known_traj)

plt.imshow(np.abs(base_map))
plt.show()

plt.imshow(np.angle(base_map))
plt.show()

sample_index = importance_sample(np.abs(base_map), n_samples=16)
sample_pos = grid_pos[sample_index[:, 1], sample_index[:, 0]]
base_sample = base_map[sample_index[:, 1], sample_index[:, 0]]

signals = []
signal_ranges = []

for pos in gt_traj[100:105]:
    signal, signal_range = get_signal(pos)

    signals.append(signal)
    signal_ranges.append(signal_range)

signals = np.array(signals)
signal_ranges = np.array(signal_ranges)

sigma_odom = 1e-3 * np.eye(2)

x0 = unknown_traj.flatten()

res = minimize(opt, x0, args=(start_pos, unknown_odom, base_sample, sample_pos, signals, signal_ranges))

opt_x = res.x
opt_traj = opt_x.reshape((-1, 2))

pass
