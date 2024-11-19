from sonar.utils import PMBarker, BarkerCode
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, cheby1, sosfilt

T_tx = T_rx = 1e-6 # 1 MHz
code = PMBarker(BarkerCode.Sequence.BARKER_2, 100_000, T_tx, 100e-6)

rx = np.zeros((10000,))
rx[2000:2000 + len(code.baseband)] += code.baseband
rx[2050:2050 + len(code.baseband)] += code.baseband
rx[2100:2100 + len(code.baseband)] += code.baseband
rx[2200:2200 + len(code.baseband)] += code.baseband
rx[2300:2300 + len(code.baseband)] += code.baseband
# rx[600:600 + len(code.baseband)] += code.baseband

filter = cheby1(4, 0.1, code.carrier, btype='low', fs=1/T_rx, output='sos')

reference_signal = np.cos(2 * np.pi * code.carrier * T_rx * np.arange(len(rx)))
demod_signal = reference_signal * rx

filt_demod_signal = sosfilt(filter, demod_signal)

correlation = correlate(filt_demod_signal, code._digital, mode="valid")

plt.subplot(4, 1, 1)
plt.plot(rx)
plt.subplot(4, 1, 2)
plt.plot(demod_signal)
plt.subplot(4, 1, 3)
plt.plot(filt_demod_signal)
plt.subplot(4, 1, 4)
plt.plot(correlation)
plt.show()
