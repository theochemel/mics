import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import skimage.measure

from config import Config
from sas import get_sas_updates, get_sas_weights
from signals import demod_signal, pulse_compress_signals
from motion.imu import IMU

np.random.seed(0)

window_size = 40

config = Config()

with open("GOOD-KINDA.pkl", "rb") as fp:
    exp_res = pickle.load(fp)

traj = exp_res["trajectory"]
rx_pattern = exp_res["rx_pattern"]
array = exp_res["array"]

pulse_rx_pattern = []

for i in range(len(rx_pattern)):
    raw_signals = rx_pattern[i]

    signal_t = config.Ts * np.arange(raw_signals.shape[-1])
    signals = demod_signal(signal_t, raw_signals, config)

    signal_t_shifted = signal_t + config.chirp_duration / 2

    for i in range(signals.shape[0]):
        signals[i] = np.interp(signal_t_shifted, signal_t, signals[i])

    pulses = pulse_compress_signals(signals, config)

    pulse_rx_pattern.append(pulses)

# imu = IMU(
#     acceleration_white_sigma=1e-4,
#     acceleration_walk_sigma=1e-3,
#     orientation_white_sigma=1e-9,
#     orientation_walk_sigma=1e-9,
# )

# imu_measurement = imu.measure(traj)

t = traj.time
dt = t[1] - t[0]

gt_p_y = traj.position_world[:, 1]
gt_v_y = traj.velocity_world[:, 1]

start_p_y = gt_p_y[0]

# imu_accel_y = imu_measurement.acceleration_body[:, 1]
imu_accel_y = np.diff(gt_v_y) / dt
imu_walk_noise_y = np.random.normal(loc=0, scale=1e-3, size=imu_accel_y.shape)
imu_accel_y += np.cumsum(imu_walk_noise_y)

grid_x = np.array([0.0])
grid_y = 1e-2 * np.arange(int(1.0 / 1e-2)) - 0.5
grid_z = 5e-3 * np.arange(int(0.2 / 5e-3)) - 1.0

grid_y, grid_x, grid_z = np.meshgrid(grid_y, grid_x, grid_z, indexing="ij")
grid_points = np.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)


def measure_sharpness(img):
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)

    # plt.imshow(np.abs(f_shift))
    # plt.show()

    y = np.arange(img.shape[0]) / (img.shape[0] - 1)
    x = np.arange(img.shape[1]) / (img.shape[1] - 1)

    y, x = np.meshgrid(y, x, indexing="ij")

    spread_x = 10
    spread_y = 1

    highpass = 1 - np.exp(-(spread_x * (x - 0.5) ** 2 + spread_y * (y - 0.5) ** 2))

    # plt.imshow(highpass)
    # plt.show()

    sharpness = np.mean(np.abs(f_shift) * highpass)

    return sharpness

    # magnitude_spectrum = np.abs(f_shift)
    # return np.sum(magnitude_spectrum[magnitude_spectrum > np.mean(magnitude_spectrum)])

def sharpness_opt_direct_v(p, v, window_i):
    window_v = np.concatenate((np.array([0]), np.cumsum(imu_accel_y[window_i:window_i + window_size - 1]))) * dt + v

    # plt.plot(window_v)
    # plt.plot(gt_v_y[window_i:window_i + window_size])
    # plt.show()

    v_errors = np.linspace(-1e-2, 1e-2, 10)
    sharpnesses = []

    imgs = []
    w_imgs = []

    for v_error in v_errors:
        map = np.zeros_like(grid_x, dtype=np.complex128)
        weights = np.zeros_like(grid_x)

        est_window_v = window_v + v_error
        est_window_pos = np.concatenate((np.array([0]), np.cumsum(est_window_v) * dt)) + p

        # plt.plot(est_window_pos)
        # plt.plot(gt_p_y[window_i:window_i + window_size] - gt_p_y[window_i])
        # plt.show()

        for i in range(window_i, window_i + window_size):
            pos = np.array([0.0, est_window_pos[i - window_i], 0.0])

            array_positions = array.positions + pos

            updates = get_sas_updates(grid_points, array_positions, pos, signal_t, pulse_rx_pattern[i - 1], config)
            updates = updates.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

            weight_updates = get_sas_weights(grid_points, array_positions, pos, signal_t, pulse_rx_pattern[i - 1], config)
            weight_updates = weight_updates.reshape((weight_updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

            map += np.sum(updates, axis=0)
            weights += np.sum(weight_updates, axis=0)

        map *= 1e-9

        # map /= weights
        img = np.transpose(np.abs(map[:, 0, :]))
        imgs.append(img)

        w_img = np.transpose(weights[:, 0, :])
        w_imgs.append(w_img)

        sharpness = measure_sharpness(img)

        # plt.imshow(img)
        # plt.title(f"{v_error} - {sharpness}")
        # plt.show()

        sharpnesses.append(sharpness)

    plt.plot(v_errors, sharpnesses)
    plt.show()


def sharpness_opt_direct_p(p, v, window_i, map):
    window_v = np.concatenate((np.array([0]), np.cumsum(imu_accel_y[window_i:window_i + window_size - 1]))) * dt + v
    window_p = np.cumsum(window_v) * dt + p

    p_errors = np.linspace(-1e-2, 1e-2, 5)

    sharpnesses = []

    imgs = []

    update_maps = []

    for p_error in p_errors:
        update_map = np.zeros_like(grid_x, dtype=np.complex128)

        est_window_pos = window_p + p_error

        for i in range(window_i, window_i + window_size):
            pos = np.array([0.0, est_window_pos[i - window_i], 0.0])

            array_positions = array.positions + pos

            updates = get_sas_updates(grid_points, array_positions, pos, signal_t, pulse_rx_pattern[i - 1], config)
            updates = updates.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

            update_map += np.sum(updates, axis=0)

        update_maps.append(update_map)

        new_map = map + update_map

        new_map *= 1e-9

        img = np.transpose(np.abs(new_map[:, 0, :]))
        imgs.append(img)

        sharpness = measure_sharpness(img)

        sharpnesses.append(sharpness)

    return update_maps[2]


def sharpness_d_v(p, v, est_bias, window_i, eps):
    map_va = np.zeros_like(grid_x, dtype=np.complex128)
    map_vb = np.zeros_like(grid_x, dtype=np.complex128)

    window_v = np.concatenate((np.array([0]), np.cumsum(imu_accel_y[window_i:window_i + window_size - 1] - est_bias))) * dt + v
    window_va = window_v - eps
    window_vb = window_v + eps

    est_pos_a = np.concatenate((np.array([0]), np.cumsum(window_va) * dt)) + p
    est_pos_b = np.concatenate((np.array([0]), np.cumsum(window_vb) * dt)) + p

    for i in range(window_i, window_i + window_size):
        position_va = np.array([0.0, est_pos_a[i - window_i], 0.0])
        position_vb = np.array([0.0, est_pos_b[i - window_i], 0.0])

        array_positions_va = array.positions + position_va
        array_positions_vb = array.positions + position_vb

        updates_va = get_sas_updates(grid_points, array_positions_va, position_va, signal_t, pulse_rx_pattern[i - 1], config)
        updates_vb = get_sas_updates(grid_points, array_positions_vb, position_vb, signal_t, pulse_rx_pattern[i - 1], config)

        updates_va = updates_va.reshape((updates_va.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))
        updates_vb = updates_vb.reshape((updates_vb.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

        map_va += np.sum(updates_va, axis=0)
        map_vb += np.sum(updates_vb, axis=0)

    map_va *= 1e-9
    map_vb *= 1e-9

    img_va = np.transpose(np.abs(map_va[:, 0, :]))
    img_vb = np.transpose(np.abs(map_vb[:, 0, :]))

    sharpness_va = measure_sharpness(img_va)
    sharpness_vb = measure_sharpness(img_vb)

    return (sharpness_vb - sharpness_va) / (2 * eps), (sharpness_va + sharpness_vb) / 2


def sharpness_d_b(p, v, est_bias, window_i, eps):
    map_a = np.zeros_like(grid_x, dtype=np.complex128)
    map_b = np.zeros_like(grid_x, dtype=np.complex128)

    window_a = np.concatenate((np.array([0]), np.cumsum(imu_accel_y[window_i:window_i + window_size - 1] - (est_bias - eps)))) * dt + v
    window_b = np.concatenate((np.array([0]), np.cumsum(imu_accel_y[window_i:window_i + window_size - 1] - (est_bias + eps)))) * dt + v

    est_pos_a = np.concatenate((np.array([0]), np.cumsum(window_a) * dt)) + p
    est_pos_b = np.concatenate((np.array([0]), np.cumsum(window_b) * dt)) + p

    for i in range(window_i, window_i + window_size):
        position_a = np.array([0.0, est_pos_a[i - window_i], 0.0])
        position_b = np.array([0.0, est_pos_b[i - window_i], 0.0])

        array_positions_a = array.positions + position_a
        array_positions_b = array.positions + position_b

        updates_a = get_sas_updates(grid_points, array_positions_a, position_a, signal_t, pulse_rx_pattern[i - 1], config)
        updates_b = get_sas_updates(grid_points, array_positions_b, position_b, signal_t, pulse_rx_pattern[i - 1], config)

        updates_a = updates_a.reshape((updates_a.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))
        updates_b = updates_b.reshape((updates_b.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

        map_a += np.sum(updates_a, axis=0)
        map_b += np.sum(updates_b, axis=0)

    map_a *= 1e-9
    map_b *= 1e-9

    img_a = np.transpose(np.abs(map_a[:, 0, :]))
    img_b = np.transpose(np.abs(map_b[:, 0, :]))

    sharpness_a = measure_sharpness(img_a)
    sharpness_b = measure_sharpness(img_b)

    return (sharpness_b - sharpness_a) / (2 * eps), (sharpness_a + sharpness_b) / 2


def sharpness_d_p(p, v, window_i, map, eps):
    window_v = np.concatenate((np.array([0]), np.cumsum(imu_accel_y[window_i:window_i + window_size - 1]))) * dt + v

    window_p = np.cumsum(window_v) * dt + p

    window_pa = window_p - eps
    window_pb = window_p + eps

    update_map_pa = np.zeros_like(grid_x, dtype=np.complex128)
    update_map_pb = np.zeros_like(grid_x, dtype=np.complex128)

    for i in range(window_i, window_i + window_size):
        position_pa = np.array([0.0, window_pa[i - window_i], 0.0])
        position_pb = np.array([0.0, window_pb[i - window_i], 0.0])

        array_positions_pa = array.positions + position_pa
        array_positions_pb = array.positions + position_pb

        updates_pa = get_sas_updates(grid_points, array_positions_pa, position_pa, signal_t, pulse_rx_pattern[i - 1], config)
        updates_pb = get_sas_updates(grid_points, array_positions_pb, position_pb, signal_t, pulse_rx_pattern[i - 1], config)

        updates_pa = updates_pa.reshape((updates_pa.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))
        updates_pb = updates_pb.reshape((updates_pb.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

        update_map_pa += np.sum(updates_pa, axis=0)
        update_map_pb += np.sum(updates_pb, axis=0)

    new_map_pa = map + update_map_pa
    new_map_pb = map + update_map_pb

    new_map_pa *= 1e-9
    new_map_pb *= 1e-9

    img_pa = np.transpose(np.abs(new_map_pa[:, 0, :]))
    img_pb = np.transpose(np.abs(new_map_pb[:, 0, :]))

    sharpness_pa = measure_sharpness(img_pa)
    sharpness_pb = measure_sharpness(img_pb)

    return (sharpness_pb - sharpness_pa) / (2 * eps), (sharpness_pa + sharpness_pb) / 2


est_v_y = 0
est_p_y = start_p_y
est_v_y_history = []
est_p_y_history = []

naive_v_y = 0
naive_p_y = start_p_y
naive_v_y_history = []
naive_p_y_history = []

for window_i in tqdm(range(len(rx_pattern) - window_size)):
    naive_v_y_history.append(naive_v_y)
    naive_p_y_history.append(naive_p_y)

    naive_p_y += naive_v_y * dt
    naive_v_y += imu_accel_y[window_i] * dt

plt.plot(naive_v_y_history, label="naive")
plt.plot(gt_v_y, label="gt")
plt.legend()
plt.show()

plt.plot(naive_p_y_history, label="naive")
plt.plot(gt_p_y, label="gt")
plt.legend()
plt.show()

plt.show()

est_v_y = naive_v_y_history[0]
est_p_y = naive_p_y_history[0]
est_b_y = 0

n_v_iters = 50
n_p_iters = 20

alpha_v = 2e-2
beta_v = 1e-1

alpha_b = 1e-2
beta_b = 1e-5
gamma_b = 1e-2

mu_v = 0.1
mu_b = 0.1

alpha_p = 1e-3
beta_p = 1e-4

mu_p = 0.1

base_map = np.zeros_like(grid_x, dtype=np.complex128)

for window_i in tqdm(range(0, len(rx_pattern) - window_size)):
    print()
    print(f"gt p: {gt_p_y[window_i]}, est p: {est_p_y}, naive p: {naive_p_y_history[window_i]}")
    print(f"gt v: {gt_v_y[window_i]}, est v: {est_v_y}, naive v: {naive_v_y_history[window_i]}")

    est_v_y_history.append(est_v_y)
    est_p_y_history.append(est_p_y)

    # Update estimated velocity by assuming constant velocity and optimizing over window

    sharpness_opt_direct_v(est_p_y, est_v_y, window_i)

    prior_est_p_y = est_p_y
    prior_est_v_y = est_v_y
    prior_est_b_y = est_b_y

    avg_grad_v = 0
    avg_grad_b = 0

    # for i in range(n_v_iters):
    i = 0
    while i < n_v_iters and (avg_grad_v == 0 or np.abs(avg_grad_v) + np.abs(avg_grad_b) > 1e-5):
        i += 1

        sharpness_grad_v, sharpness = sharpness_d_v(est_p_y, est_v_y, est_b_y, window_i, eps=1e-4)
        sharpness_grad_b, _ = sharpness_d_b(est_p_y, est_v_y, est_b_y, window_i, eps=1e-4)

        grad_v = alpha_v * sharpness_grad_v + beta_v * (prior_est_v_y - est_v_y)

        grad_b = alpha_b * sharpness_grad_b + beta_b * (prior_est_b_y - est_b_y) - gamma_b * est_b_y

        if avg_grad_v == 0:
            avg_grad_v = grad_v
            avg_grad_b = grad_b
        else:
            avg_grad_v = mu_v * avg_grad_v + (1 - mu_v) * grad_v
            avg_grad_b = mu_b * avg_grad_b + (1 - mu_b) * grad_b

        print(f"i: {i}, est_v_y: {est_v_y}, grad_v: {avg_grad_v}, est_b_y: {est_b_y}, grad_b: {avg_grad_b}")

        est_v_y += avg_grad_v
        est_b_y += avg_grad_b

    # est_bias_y = alpha_bias * est_bias_y + (1 - alpha_bias) * (prior_est_v_y - est_v_y) / dt

        # if len(est_p_y_history) >= 2:
        #     est_v_y += gamma_v * ((est_p_y_history[-1] - est_p_y_history[-2]) / dt - est_v_y)

    # TODO: Sharpness optimize est_p_y
    # sharpness_opt_direct_p(est_p_y, est_v_y, window_i, map)


    # avg_grad_p = 0
    #
    # # for i in range(n_p_iters):
    # while window_i > 0 and avg_grad_p == 0 or np.abs(avg_grad_p) > 1e-8:
    #     sharpness_grad, sharpness = sharpness_d_p(est_p_y, est_v_y, window_i, base_map, eps=1e-4)
    #     grad = alpha_p * sharpness_grad + beta_p * (prior_est_p_y - est_p_y)
    #     if avg_grad_p == 0:
    #         avg_grad_p = grad
    #     else:
    #         avg_grad_p = mu_p * avg_grad_p + (1 - mu_p) * grad
    #     print(f"i: {i}, est_p_y: {est_p_y}, grad: {avg_grad_p}, sharpness: {sharpness}")
    #     est_p_y += grad

    if window_i == 0:
        # Add whole map

        window_v = np.concatenate((np.array([0]), np.cumsum(imu_accel_y[window_i:window_i + window_size - 1]))) * dt + est_v_y
        window_p = np.cumsum(window_v) * dt + est_p_y

        update_map = np.zeros_like(base_map)

        for i in range(window_i, window_i + window_size):
            pos = np.array([0.0, window_p[i - window_i], 0.0])

            array_positions = array.positions + pos

            updates = get_sas_updates(grid_points, array_positions, pos, signal_t, pulse_rx_pattern[i - 1], config)
            updates = updates.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

            update_map += np.sum(updates, axis=0)

        base_map += update_map
    else:
        # Only add new measurement
        pos = np.array([0.0, est_p_y, 0.0])

        array_positions = array.positions + pos

        updates = get_sas_updates(grid_points, array_positions, pos, signal_t, pulse_rx_pattern[window_i - 1], config)
        updates = updates.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

        update_map = np.sum(updates, axis=0)

        base_map += update_map

    est_p_y += est_v_y * dt
    est_v_y += (imu_accel_y[window_i] - est_b_y) * dt

    img = np.transpose(np.abs(base_map[:, 0, :]))
    plt.imshow(img)
    plt.show()

    plt.plot(est_v_y_history, label="est")
    plt.plot(naive_v_y_history[:window_i], label="naive")
    plt.plot(gt_v_y[:window_i], label="gt")
    plt.legend()
    plt.title("v")
    plt.show()

    plt.plot(est_p_y_history, label="est")
    plt.plot(naive_p_y_history[:window_i], label="naive")
    plt.plot(gt_p_y[:window_i], label="gt")
    plt.legend()
    plt.title("p")
    plt.show()
