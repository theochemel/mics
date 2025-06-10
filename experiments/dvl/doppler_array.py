import os.path
from scipy.special import rel_entr

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from tqdm import tqdm
import pickle

c = 1500
f = 75e3
f_array = 75e3
pulse_duration = 1e0

fs = 300e3
ts = 1 / fs

lam = c / f_array

v = np.array([
    1e0, 0.0, 0.0
])

array_x = (lam / 2) * np.arange(10)
array_y = (lam / 2) * np.arange(10)

array_x = array_x - np.mean(array_x)
array_y = array_y - np.mean(array_y)

array_y, array_x = np.meshgrid(array_y, array_x, indexing="ij")
array_y = array_y.flatten()
array_x = array_x.flatten()
array_z = np.zeros_like(array_x)

ps = np.array([
    [x, y, 0.0] for (x, y) in zip(array_x, array_y)
])

surface_z = 10

n_rays = int(1e3)

ray_elevation = np.arccos(np.random.uniform(np.cos(0), np.cos(np.pi / 4), n_rays))
ray_azimuth = np.random.uniform(-np.pi, np.pi, n_rays)

ray_x = np.cos(ray_azimuth) * np.sin(ray_elevation)
ray_y = np.sin(ray_azimuth) * np.sin(ray_elevation)
ray_z = np.cos(ray_elevation)

ray_dir = np.stack((ray_x, ray_y, ray_z), axis=-1)

ray_f = f * (1 + 2 * np.sum(ray_dir * v[np.newaxis, :], axis=-1) / c)

n_rays = len(ray_dir)

rx_t = (1 / fs) * np.arange(int(pulse_duration * fs))

def get_samples(p):
    rx_sig = 0

    hit_point = ray_dir * (surface_z / ray_dir[:, 2][:, np.newaxis])
    ray_d = np.linalg.norm(hit_point, axis=-1) + np.linalg.norm(hit_point - p, axis=-1)
    ray_t = ray_d / c

    for i in range(n_rays):
        rx_sig += (1 / (ray_d[i] ** 2)) * np.sin((2 * np.pi * ray_f[i] * (rx_t - ray_t[i])))

    rx_sig /= n_rays

    rx_sig += np.random.normal(loc=0, scale=1e-3, size=rx_sig.shape)

    return rx_sig

if os.path.exists("sigs.pkl"):
    with open("sigs.pkl", "rb") as fp:
        rx_sigs = pickle.load(fp)
else:
    rx_sigs = np.zeros((len(rx_t), len(ps)))

    for i, p in tqdm(enumerate(ps)):
        rx_sigs[:, i] = get_samples(p)

    with open("sigs.pkl", "wb") as fp:
        pickle.dump(rx_sigs, fp)

def beamform_delay_and_sum(data, dir):
    M, N = data.shape

    dir = dir / np.linalg.norm(dir)

    # Calculate delays for each receiver
    delays = np.sum(dir * ps, axis=-1) / c
    delay_samples = delays * fs

    # Create beamformed signal (zero-initialized)
    beamformed_signal = np.zeros(M)

    for i in range(N):
        # Shift each signal using interpolation for fractional delay
        shifted = fractional_delay(data[:, i], delay_samples[i])
        beamformed_signal += shifted

    # Normalize
    beamformed_signal /= N

    return beamformed_signal


def fractional_delay(signal, delay):
    n = np.arange(-32, 33)  # filter length = 101
    h = np.sinc(n - delay) * np.hamming(len(n))
    h /= np.sum(h)  # normalize filter gain

    # Apply filter
    delayed_signal = np.convolve(signal, h, mode='same')
    return delayed_signal


def estimate_doppler_phase_method(signal, fs, f0, frame_size, hop_size):
    N = frame_size
    hop = hop_size
    omega0 = 2 * np.pi * f0

    t = (1 / fs) * np.arange(len(signal))

    phases = []

    for i in range(0, len(signal) - N + 1, hop_size):
        frame = signal[i : i + N]
        frame_t = t[i : i + N]
        window = gaussian(len(frame), std=20)
        mix = np.sum(window * frame * np.exp(-1j * omega0 * frame_t))
        # Sum to get phase
        ph = np.angle(mix)
        phases.append(ph)

    # Unwrap phase
    phases = np.unwrap(phases)

    plt.plot(phases)

    T = hop / fs  # time between frames

    slope, _ = np.polyfit(T * np.arange(len(phases)), phases, 1)

    fd = slope / (2 * np.pi)

    return fd


def evaluate_likelihood(candidate_v):
    beamform_azimuth = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    beamform_elevation = np.array([0.5])

    beamform_azimuth, beamform_elevation = np.meshgrid(beamform_azimuth, beamform_elevation, indexing="xy")
    beamform_azimuth = beamform_azimuth.flatten()
    beamform_elevation = beamform_elevation.flatten()

    beamform_dirs = np.stack((
        np.cos(beamform_azimuth) * np.sin(beamform_elevation),
        np.sin(beamform_azimuth) * np.sin(beamform_elevation),
        np.cos(beamform_elevation)
    ), axis=-1)

    likelihoods = []

    for beamform_dir in beamform_dirs:
        signal = beamform_delay_and_sum(rx_sigs, beamform_dir)

        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
        freq_shifts = freqs - f

        fft_valid = (-100 < freq_shifts) & (freq_shifts < 100)
        freq_shifts = freq_shifts[fft_valid]
        freqs = freqs[fft_valid]
        fft = fft[fft_valid]

        freq_bin_width = freq_shifts[1] - freq_shifts[0]

        freq_bin_edges = np.concatenate((
            freq_shifts - freq_bin_width / 2,
            np.array([freq_shifts[-1] + freq_bin_width / 2])
        ))

        power_spectrum = np.abs(fft) ** 2
        power_spectrum /= np.sum(power_spectrum * (freqs[1] - freqs[0]))

        sample_doppler_shifts, sample_weights = get_doppler_spectrum(beamform_dir, candidate_v)

        hist, _ = np.histogram(sample_doppler_shifts, bins=freq_bin_edges, weights=sample_weights)
        hist /= np.sum(hist) * freq_bin_width

        plt.plot(freq_shifts, hist, c="orange")
        plt.plot(freq_shifts, power_spectrum)
        plt.show()

        # divergence = np.sum(rel_entr(power_spectrum, hist + 1e-9))
        likelihood = np.sum(power_spectrum * hist)

        likelihoods.append(likelihood)

    likelihoods = np.array(likelihoods)
    likelihood = np.sum(likelihoods)

    return likelihood

def get_doppler_spectrum(beamform_dir: np.array, candidate_v: np.array):
    N = int(1e6)

    sample_azimuth = np.random.uniform(0, 2 * np.pi, N)
    sample_elevation = np.arccos(np.random.uniform(np.cos(0), np.cos(np.pi / 4), N))

    sample_dirs = np.stack((
        np.cos(sample_azimuth) * np.sin(sample_elevation),
        np.sin(sample_azimuth) * np.sin(sample_elevation),
        np.cos(sample_elevation)
    ), axis=-1)

    beamform_delays = -np.sum(beamform_dir * ps, axis=-1) * (2 * np.pi * f / c)
    sample_delays = -np.sum(sample_dirs[:, np.newaxis, :] * ps[np.newaxis, :, :], axis=-1) * (2 * np.pi * f / c)

    delta_delays = beamform_delays - sample_delays

    gain = np.sqrt(
        np.sum(np.sin(delta_delays), axis=-1) ** 2 + np.sum(np.cos(delta_delays), axis=-1) ** 2
    ) / delta_delays.shape[-1]

    power_gain = gain ** 2

    # fig = plt.figure()
    # ax1 = fig.add_subplot(131, projection='polar')
    # ax1.scatter(sample_azimuth, sample_elevation, c=power_gain)
    # plt.show()

    sample_doppler_shifts = f * (2 * np.sum(sample_dirs * candidate_v, axis=-1) / c)

    weights = power_gain / np.sum(power_gain)

    return sample_doppler_shifts, weights

# beamform_azimuth = np.linspace(0, 2 * np.pi, 10, endpoint=False)
# beamform_elevation = np.arccos(np.linspace(np.cos(0), np.cos(np.pi / 4), 11)[1:])

nx = 3
ny = 3

candidate_vx = v[0] + np.linspace(-1e-2, 1e-2, nx)
candidate_vy = v[1] + np.linspace(-1e-2, 1e-2, ny)
candidate_vy, candidate_vx = np.meshgrid(candidate_vy, candidate_vx, indexing="ij")

candidate_vx = candidate_vx.flatten()
candidate_vy = candidate_vy.flatten()
candidate_vz = np.zeros_like(candidate_vx)

candidate_vs = np.stack((candidate_vx, candidate_vy, candidate_vz), axis=-1)

likelihoods = []

for candidate_v in tqdm(candidate_vs):
    likelihood = evaluate_likelihood(candidate_v)
    likelihoods.append(likelihood)

likelihoods = np.array(likelihoods).reshape((ny, nx))
plt.imshow(likelihoods)
plt.show()

