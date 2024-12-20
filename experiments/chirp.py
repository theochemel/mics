import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import correlate, cheby1, sosfiltfilt, correlation_lags

fs = 1e6
fc = 75e3
bw = 10e3

duration = 1e-3
K = (fc - bw) / duration

t = np.linspace(-duration / 2, duration / 2, int(fs * duration), endpoint=False)

signal_if = np.exp(2 * np.pi * 1.0j * fc * t + np.pi * 1.0j * K * t ** 2)

signal = np.real(signal_if) * np.cos(t) + np.imag(signal_if) * np.sin(t)

rx_signal = np.zeros((10000))
rx_t = np.arange(len(rx_signal)) / fs

i = 0
rx_signal[1000 + i:1000 + i + len(signal)] += signal

rx_signal += 1e-3 * np.random.uniform(low=-1, high=1, size=rx_signal.shape)

plt.plot(rx_signal)
plt.show()

correlation = correlate(rx_signal, signal, mode="valid")
plt.plot(correlation)
plt.show()

lags = correlation_lags(len(rx_signal), len(signal), mode="valid")

complex_carrier = np.exp(1j * 2 * np.pi * fc * rx_t)

complex_baseband = correlation * complex_carrier

filter = cheby1(4, 0.1, 0.5 * fc, btype='low', fs=fs, output='sos')

filt_complex_baseband = sosfiltfilt(filter, complex_baseband)

plt.plot(np.abs(complex_baseband))
plt.plot(np.abs(filt_complex_baseband))
plt.show()

plt.plot(np.angle(complex_baseband))
plt.plot(np.angle(filt_complex_baseband))
plt.show()

