import numpy as np
import matplotlib.pyplot as plt


x0 = 1
x1 = 2

f = 85e3
fs = 300e3
c = 1500

T = 1e-1

N = int(1e1)

t = (1 / fs) * np.arange(int(T * fs))
signal = np.zeros_like(t)

x = np.random.uniform(x0, x1, N)
rtt = (2 * x) / c

for sample_rtt in rtt:
    signal += np.cos(2 * np.pi * f * (t - sample_rtt))

signal += np.random.normal(loc=0, scale=1e-3, size=signal.shape)

# Compute and plot spectrum
freqs = np.fft.fftfreq(len(signal), 1/fs)
spectrum = np.fft.fft(signal)
spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12)  # dB scale

# Plot time domain
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t[:1000], signal[:1000])
plt.title(f"Time Domain Signal (Sum of {N} Random-Phase Sinusoids)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot frequency domain
plt.subplot(1, 2, 2)
plt.plot(freqs[:len(freqs)//2], spectrum_db[:len(freqs)//2])
plt.title("Frequency Spectrum (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()

pass