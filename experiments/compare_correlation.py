import numpy as np
import torch
import scipy as sp
import matplotlib.pyplot as plt

chirp_fc = 75e3
chirp_bw = 50e3
chirp_duration = 1e-3
chirp_K = chirp_bw / chirp_duration

def chirp(t):
    return np.exp(2.0j * np.pi * chirp_fc * t + 1.0j * np.pi * chirp_K * t ** 2)


signal_duration = 5e-3
T = 1e-6
signal_N = int(signal_duration // T)
signal_t = T * np.arange(signal_N)
signal = chirp(signal_t - signal_duration / 2)

reference_duration = chirp_duration
reference_N = int(reference_duration / T)
reference_t = T * np.arange(reference_N) - reference_duration / 2
reference = chirp(reference_t)

corr = sp.signal.correlate(signal, reference, mode='same')

fig, ax = plt.subplots()
ax.plot(signal_t, np.real(corr), label='Re')
ax.plot(signal_t, np.imag(corr), label='Im')
plt.show()
