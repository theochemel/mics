import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

v = 1500

target_distance = 10

target_x = np.linspace(-5, 5, 3)
target_y = np.linspace(-5, 5, 3)
target_x, target_y = np.meshgrid(target_x, target_y, indexing="xy")
target_x = target_x.flatten()
target_y = target_y.flatten()
target_z = np.where((target_x >= -0.5) & (target_x <= 0.5) & (target_y >= -0.5) & (target_y <= 0.5), target_distance, target_distance + 1.0)

target_points = np.stack((target_x, target_y, target_z), axis=1)

# target_points = np.array([
#     [0, 0, target_distance],
#     [5, 0, target_distance + 1],
# ])

# plt.scatter(target_points[:, 0], target_points[:, 1], c=target_points[:, 2])
# plt.show()

carrier_freq = 1000e3
modulation_freq = 50e3
sample_freq = 10 * carrier_freq

downsample_freq = 4 * modulation_freq

sample_period = 1 / sample_freq
downsample_period = 1 / downsample_freq
n_samples = 2000

forward_return_time = 2 * (target_distance / v)

modulation_wavelength = v / modulation_freq

spacing = modulation_wavelength / 2
n_elems = 15

tx_point = np.array([0, 0, 0])
rx_x = spacing * (np.arange(n_elems) - float(n_elems - 1) / 2)
rx_y = spacing * (np.arange(n_elems) - float(n_elems - 1) / 2)
rx_x, rx_y = np.meshgrid(rx_x, rx_y, indexing="xy")
rx_x = rx_x.flatten()
rx_y = rx_y.flatten()
rx_z = np.zeros_like(rx_x)

rx_points = np.stack((rx_x, rx_y, rx_z), axis=1)

target_indices = np.arange(target_points.shape[0])
rx_indices = np.arange(rx_points.shape[0])
target_indices, rx_indices = np.meshgrid(target_indices, rx_indices, indexing="ij")

path_distances = np.linalg.norm(tx_point - target_points[target_indices], axis=-1) + np.linalg.norm(target_points[target_indices] - rx_points[rx_indices], axis=-1)

transit_times = path_distances / v

start_time = forward_return_time
sample_times = sample_period * np.arange(int(n_samples * sample_freq / downsample_freq)) + start_time

tx_times = sample_times - transit_times[:, :, np.newaxis]

ramp_time_constant = 1e-2

chirp_period = 5e-3
c = 10000000

samples = np.where(
    tx_times < 0,
    0,
    (1 - np.exp(-tx_times / ramp_time_constant)) * np.sin(2 * np.pi * ((c / 2) * (tx_times % chirp_period) ** 2 + (carrier_freq + modulation_freq / 2) * tx_times))
)

samples = np.sum(samples, axis=0)

sos = signal.bessel(4, [carrier_freq, carrier_freq + modulation_freq], "bandpass", fs=sample_freq, output="sos")
# sos = signal.bessel(4, carrier_freq + modulation_freq, "low", fs=sample_freq, output="sos")
filter_samples = signal.sosfilt(sos, samples, axis=-1)

downsample_times = downsample_period * np.arange(n_samples) + start_time

downsample_samples = np.array([np.interp(downsample_times, sample_times, filter_samples[i]) for i in range(filter_samples.shape[0])])

# Now downsample samples

# plt.plot(sample_times, samples[samples.shape[0] // 2], label=f"rx = {rx_points[samples.shape[0] // 2, 0]}")
# plt.plot(sample_times, samples[1], label=f"rx = {rx_points[1, 0]}")
# plt.plot(sample_times, samples[2], label=f"rx = {rx_points[2, 0]}")
# plt.axvline(x=forward_return_time, c="r")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.show()
#
def plot_fft(signal, fs):
    N = signal.shape[1]
    fft_result = np.fft.fft(signal, axis=1)
    fft_freq = np.fft.fftfreq(N, 1 / fs)

    positive_freqs = fft_freq[:N // 2]
    positive_magnitude = np.abs(fft_result)[:, :N // 2]
    positive_phase = np.angle(fft_result)[:, :N // 2]

    _, axs = plt.subplots(nrows=2, sharex=True)

    for i in range(positive_magnitude.shape[0]):
        axs[0].plot(positive_freqs, positive_magnitude[i])
    axs[0].set_title('Frequency Domain (FFT)')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude')
    axs[0].set_yscale("log")

    for i in range(positive_phase.shape[0]):
        axs[1].plot(positive_freqs, positive_phase[i])
    axs[1].set_title('Frequency Domain (FFT)')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Phase (rad)')

    plt.tight_layout()
    plt.show()

# plot_fft(samples[0:3], sample_freq)
# plot_fft(filter_samples[0:3], sample_freq)
# plot_fft(downsample_samples[0:3], downsample_freq)
#
# delay_and_sum_broadside_samples = np.mean(downsample_samples, axis=0)
# plt.plot(downsample_times, delay_and_sum_broadside_samples, label="beamformed")
# plt.plot(downsample_times, downsample_samples[0], label="raw")
# plt.legend()
# plt.show()

# plt.plot(sample_times, samples[0], label="samples")
# plt.plot(sample_times, filter_samples[0], label="filter")

beamform_samples = np.mean(downsample_samples, axis=0)
# plt.plot(downsample_times, downsample_samples[0], label="raw")
# plt.plot(downsample_times, beamform_samples, label="beamformed")
# plt.legend()
# plt.show()
#
# window = np.ones(50) / 50
#
# raw_power = downsample_samples[0] ** 2
# beamform_power = beamform_samples ** 2
#
# raw_power_filt = np.convolve(raw_power, window, mode="same")
# beamform_power_filt = np.convolve(beamform_power, window, mode="same")

# total_raw_power = np.sqrt(np.sum(raw_power))
# total_beamform_power = np.sqrt(np.sum(beamform_power))
#
# raw_power = raw_power / total_raw_power
# beamform_power = beamform_power / total_beamform_power

# plt.plot(downsample_times, raw_power, label="raw")
# plt.plot(downsample_times, beamform_power, label="beamformed")
# plt.plot(downsample_times, raw_power_filt, label="raw filt")
# plt.plot(downsample_times, beamform_power_filt, label="beamformed filt")
# plt.legend()
# plt.show()

# plot_fft(np.array([downsample_samples[0]]), downsample_freq)

frequencies, times, Sxx_raw = signal.spectrogram(downsample_samples[0], downsample_freq, nperseg=64)
_frequencies, _times, Sxx_beamformed = signal.spectrogram(beamform_samples, downsample_freq, nperseg=64)
times += start_time

fig, axs = plt.subplots(nrows=2)

axs[0].set_title("Raw")
axs[0].pcolormesh(times, frequencies, 10 * np.log10(Sxx_raw), shading='gouraud')
axs[0].set_ylabel('Frequency [Hz]')
axs[0].set_xlabel('Time [sec]')

axs[1].set_title("Beamformed")
axs[1].pcolormesh(times, frequencies, 10 * np.log10(Sxx_beamformed), shading='gouraud')
axs[1].set_ylabel('Frequency [Hz]')
axs[1].set_xlabel('Time [sec]')

fig.tight_layout()
plt.show()

