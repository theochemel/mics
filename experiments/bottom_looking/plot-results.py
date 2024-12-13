import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("results.pkl", "rb") as fp:
    res = pickle.load(fp)


time = 0.05 * np.arange(1000)

N = len(res["est_v_y_history"])

fig, axs = plt.subplots(2, figsize=(6, 4))

axs[0].set_title("Velocities")
axs[0].plot(time[:N], res["gt_v_y"][:N], label="gt")
axs[0].plot(time[:N], res["est_v_y_history"], label="est")
axs[0].plot(time[:N], res["naive_v_y_history"], label="naive")
axs[0].legend()
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Velocity (m/s)")

axs[1].set_title("Errors")
axs[1].plot(time[:N], np.array(res["est_v_y_history"]) - res["gt_v_y"][:N], label="est")
axs[1].plot(time[:N], np.array(res["naive_v_y_history"]) - res["gt_v_y"][:N], label="naive")
axs[1].legend()
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Error (m/s)")

fig.tight_layout()

plt.savefig("vel.svg")
plt.show()

fig, axs = plt.subplots(2, figsize=(6, 4))

axs[0].set_title("Velocities")
axs[0].plot(time[:20], res["gt_v_y"][:20], label="gt")
axs[0].plot(time[:20], res["est_v_y_history"][:20], label="est")
axs[0].plot(time[:20], res["naive_v_y_history"][:20], label="naive")
axs[0].legend()
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Velocity (m/s)")

axs[1].set_title("Errors")
axs[1].plot(time[:50], np.array(res["est_v_y_history"][:50]) - res["gt_v_y"][:50], label="est")
axs[1].plot(time[:50], np.array(res["naive_v_y_history"][:50]) - res["gt_v_y"][:50], label="naive")
axs[1].legend()
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Error (m/s)")

fig.tight_layout()

plt.savefig("vel-zoom.svg")
plt.show()

fig, ax = plt.subplots(figsize=(4, 2))
ax.set_title("Final Map")
ax.imshow(res["img"])
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()

plt.savefig("map.svg")
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("Acceleration Biases")
ax.plot(time[:N], res["est_b_y_history"], label="est")
ax.plot(time[:N], res["imu_walk_noise_y"][:N], label="gt")

from scipy.signal import butter, sosfiltfilt

sos = butter(4, 0.1, btype='low', output='sos')

ax.plot(time[:N], sosfiltfilt(sos, res["imu_walk_noise_y"][:N]), label="gt lowpass")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Bias (m/s^2)")
plt.legend(loc="upper right")
fig.tight_layout()

plt.savefig("bias.svg")
plt.show()
