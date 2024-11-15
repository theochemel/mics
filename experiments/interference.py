import numpy as np
import matplotlib.pyplot as plt


target_x = 1.0
rx_1_x = 9.0
rx_2_x = 9.5

c = 1
f = 1
l = c / f
k = np.pi / l
ts = 0.1

x_eval = np.linspace(0, 10, 200)

sigma_range = 0.1

rx1_rx = np.maximum(np.exp(-(x_eval - rx_1_x + target_x) ** 2 / sigma_range), 1e-3) * np.exp(1.0j * 2 * (x_eval - rx_1_x + target_x) * k)
rx2_rx = np.maximum(np.exp(-(x_eval - rx_2_x + target_x) ** 2 / sigma_range), 1e-3) * np.exp(1.0j * 2 * (x_eval - rx_2_x + target_x) * k)

_, axs = plt.subplots(nrows=2, sharex=True)
axs[0].plot(x_eval, np.abs(rx1_rx))
axs[0].plot(x_eval, np.abs(rx2_rx))
axs[1].plot(x_eval, np.angle(rx1_rx))
axs[1].plot(x_eval, np.angle(rx2_rx))
axs[1].set_xlabel("range")
axs[0].set_ylabel("magnitude")
axs[1].set_ylabel("phase")
plt.show()

buffer = np.zeros_like(x_eval, dtype=np.complex128)

td1 = (rx_1_x - x_eval[0]) / c
td2 = (rx_2_x - x_eval[0]) / c

sd1 = int(td1 / ts)
sd2 = int(td2 / ts)

phasor_1 = np.exp(-1.0j * 2 * (rx_1_x - x_eval) * k) * rx1_rx
phasor_1_x = rx_1_x - x_eval
phasor_2 = np.exp(-1.0j * 2 * (rx_2_x - x_eval) * k) * rx2_rx
phasor_2_x = rx_2_x - x_eval

_, axs = plt.subplots(nrows=2, sharex=True)
axs[0].plot(phasor_1_x, np.abs(phasor_1))
axs[0].plot(phasor_2_x, np.abs(phasor_2))
axs[1].plot(phasor_1_x, np.angle(phasor_1))
axs[1].plot(phasor_2_x, np.angle(phasor_2))
axs[0].set_ylabel("magnitude")
axs[1].set_ylabel("phase")
plt.show()

buffer += np.interp(x_eval, np.flip(phasor_1_x), np.flip(phasor_1))
buffer += np.interp(x_eval, np.flip(phasor_2_x), np.flip(phasor_2))

plt.plot(np.abs(buffer))
plt.show()