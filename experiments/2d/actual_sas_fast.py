import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize

C = 1500

chirp_fc = 75e3
chirp_bw = 50e3
chirp_duration = 1e-3
chirp_K = chirp_bw / chirp_duration

chirp_fhi = chirp_fc + chirp_bw / 2

fs = 1e6
Ts = 1 / fs

l_m = C / chirp_fhi

max_range = 20
max_rt_t = (2 * max_range) / C

signal_t = Ts * np.arange(int(max_rt_t / Ts))

target_points_x = np.linspace(1, 9, 9)
target_points_y = np.linspace(1, 9, 9)
target_points_y, target_points_x = np.meshgrid(target_points_y, target_points_x, indexing="ij")
target_points_y = target_points_y.flatten()
target_points_x = target_points_x.flatten()
target_points = np.stack((target_points_x, target_points_y), axis=-1)
target_points += np.random.normal(loc=0, scale=5e-3, size=target_points.shape)

gt_traj_x = 1e-2 * np.arange(100)
gt_traj_y = np.zeros_like(gt_traj_x)
gt_traj = np.stack((gt_traj_x, gt_traj_y), axis=-1)

noisy_traj = gt_traj + np.random.normal(loc=0, scale=1e-5, size=gt_traj.shape)

grid_width = 10
grid_height = 10
grid_size = 1e-2

grid_x = grid_size * np.arange(int(grid_width / grid_size)) + grid_size / 2
grid_y = grid_size * np.arange(int(grid_height / grid_size)) + grid_size / 2

grid_y, grid_x = np.meshgrid(np.flip(grid_y), grid_x, indexing="ij")
grid_pos = np.stack((grid_x, grid_y), axis=-1)
grid_extent = [0, grid_width, 0, grid_height]


def wrap2pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def chirp_envelope(t):
    return np.where(
        (t >= -chirp_duration / 2) & (t <= chirp_duration / 2),
        (1 / chirp_duration) * np.cos(np.pi * t / chirp_duration) ** 2,
        0
    )

def chirp(t):
    return chirp_envelope(t) * np.exp(2.0j * np.pi * chirp_fc * t + 1.0j * np.pi * chirp_K * t ** 2)

def reference_chirp(t):
    return chirp_envelope(t) * np.exp(1.0j * np.pi * chirp_K * t ** 2)

def get_signal(position, signal_t):
    signal = np.zeros_like(signal_t, dtype=np.complex128)

    for target_point in target_points:
        target_rt_t = (2 * np.linalg.norm(target_point - position)) / C

        signal += chirp(signal_t - target_rt_t)

    return signal * np.exp(-2.0j * np.pi * chirp_fc * signal_t)


def pulse_compress(signal, signal_t):
    reference_signal_t = Ts * np.arange(int(chirp_duration / Ts)) - (chirp_duration / 2)
    reference_signal = reference_chirp(reference_signal_t)

    correlation = sp.signal.correlate(signal, reference_signal, mode="same")

    return correlation # * np.exp(-2.0j * np.pi * chirp_fc * signal_t)

def build_map(gt_traj, noisy_traj, visualize: bool = False):
    # traj is [n_poses, 2]
    # signal_t is [n_samples]
    # signals is [n_poses, n_samples]

    map = np.zeros(grid_pos.shape[0:2], dtype=np.complex128)

    for i, (gt_position, position) in enumerate(zip(gt_traj, noisy_traj)):
        signal = get_signal(gt_position, signal_t)
        pulse = pulse_compress(signal, signal_t)

        if visualize:
            pulse_range = (signal_t * C) / 2.0

            pulse_update = pulse * np.exp((2.0j * np.pi * chirp_fc / C) * (2.0 * pulse_range))

            plt.plot(pulse_range, np.real(pulse_update))
            plt.plot(pulse_range, np.imag(pulse_update))
            plt.show()

            fig, axs = plt.subplots(2, sharex=True)
            axs[0].plot(pulse_range, np.abs(pulse_update))
            axs[1].plot(pulse_range, np.where(np.abs(pulse_update) > 1e-9, np.angle(pulse_update), 0))
            plt.show()

        grid_range = np.linalg.norm(grid_pos - position[np.newaxis, np.newaxis], axis=-1)
        grid_rt_t = (2.0 * grid_range) / C

        k = grid_rt_t / Ts
        k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
        k_a = k - k_i  # Fractional parts

        # Ensure we don't go out of bounds for the upper index
        k_i_plus_1 = np.clip(k_i + 1, 0, len(pulse) - 1)  # Upper bounds (clipped)

        # Perform linear interpolation
        interp_pulse = (1 - k_a) * pulse[k_i] + k_a * pulse[k_i_plus_1]

        update = interp_pulse * np.exp((2.0j * np.pi * chirp_fc / C) * (2.0 * grid_range))

        if visualize:
            plt.subplot(1, 2, 1)
            plt.imshow(np.abs(update), extent=grid_extent)
            plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
            plt.subplot(1, 2, 2)
            plt.imshow(np.angle(update), extent=grid_extent)
            plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
            plt.suptitle("Update")
            plt.show()

        map += update

        if visualize:
            plt.subplot(1, 2, 1)
            plt.imshow(np.abs(map), extent=grid_extent)
            plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
            plt.subplot(1, 2, 2)
            plt.imshow(np.angle(map), extent=grid_extent)
            plt.scatter(target_points[:, 0], target_points[:, 1], c="r", marker="x")
            plt.suptitle("Map")
            plt.show()

    return map

def importance_sample(img, n_samples):
    weights = img ** 4
    weights = weights / np.sum(weights)
    flat_weights = weights.flatten()
    indices = np.arange(len(flat_weights))

    samples = np.random.choice(indices, size=n_samples, replace=False, p=flat_weights)

    sample_y, sample_x = np.unravel_index(samples, img.shape)

    return np.stack((sample_x, sample_y), axis=-1)

base_map = build_map(gt_traj[:50], gt_traj[:50])

# Now do the optimization to adjust a noisy pose 50 relative to
# map built from GT poses 0-49

sample_px = importance_sample(np.abs(base_map), 128)
sample_pos = grid_pos[sample_px[:, 1], sample_px[:, 0]]
sample_weight = np.abs(base_map)[sample_px[:, 1], sample_px[:, 0]]

plt.imshow(np.abs(base_map), extent=grid_extent)
plt.scatter(sample_pos[:, 0], sample_pos[:, 1], c="b")
plt.plot(gt_traj[:, 0], gt_traj[:, 1], c="r")
plt.show()

signal = get_signal(gt_traj[50], signal_t)
pulse = pulse_compress(signal, signal_t)

offset_x = np.linspace(-5e-2, 5e-2, 50)
offset_y = np.linspace(-5e-2, 5e-2, 50)

offset_x, offset_y = np.meshgrid(offset_x, offset_y)

errors = np.zeros(offset_x.shape)

for i in range(offset_x.shape[0]):
    for j in range(offset_x.shape[1]):
        print(f"{i}, {j}")

        est_pos = gt_traj[50] + np.array([
            offset_x[i, j],
            offset_y[i, j],
        ])

        base_map_phase = np.angle(base_map)[sample_px[:, 1], sample_px[:, 0]]

        sample_range = np.linalg.norm(sample_pos - est_pos[np.newaxis], axis=-1)
        sample_rt_t = (2.0 * sample_range) / C

        k = sample_rt_t / Ts
        k_i = np.floor(k).astype(int)  # Lower bounds (integer indices)
        k_a = k - k_i  # Fractional parts

        # Ensure we don't go out of bounds for the upper index
        k_i_plus_1 = np.clip(k_i + 1, 0, len(pulse) - 1)  # Upper bounds (clipped)

        # Perform linear interpolation
        interp_pulse = (1 - k_a) * pulse[k_i] + k_a * pulse[k_i_plus_1]

        update = interp_pulse * np.exp((2.0j * np.pi * chirp_fc / C) * (2.0 * sample_range))

        est_phase = np.angle(update)

        avg_phase_error = np.sum(
            sample_weight * (wrap2pi(est_phase - base_map_phase) ** 2)
        ) / np.sum(sample_weight)

        errors[i, j] = avg_phase_error

# plt.pcolormesh(offset_x, offset_y, errors)
# plt.gca().set_aspect("equal")
# plt.show()

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
surf = ax.plot_surface(offset_x, offset_y, errors, cmap=matplotlib.cm.coolwarm)
plt.show()

pass