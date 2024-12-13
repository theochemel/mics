import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("sharpness-opt.pkl", "rb") as fp:
    res = pickle.load(fp)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(4, 3))

axs[0].imshow(res["imgs"][0])
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_title(f"velocity error = -0.1")

axs[1].imshow(res["imgs"][25])
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].set_title(f"velocity error = 0")

axs[2].imshow(res["imgs"][49])
axs[2].set_xticks([])
axs[2].set_yticks([])
axs[2].set_title(f"velocity error = 0.1")

fig.tight_layout()
plt.savefig("sharpness-imgs.svg")

plt.show()

fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(res["v_errors"], res["sharpnesses"])
ax.set_yticks([])
ax.set_xticks([-0.1, 0.0, 0.1])
ax.set_xlabel("Velocity Error")
ax.set_title("Map Quality")

fig.tight_layout()

plt.savefig("sharpness-plot.svg")

plt.show()

fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(res["v_errors"][90:110], res["sharpnesses"][90:110])
ax.set_yticks([])
ax.set_xticks([-0.01, 0.0, 0.01])
ax.set_xlim(-0.01, 0.01)
ax.autoscale(axis="y")
ax.set_xlabel("Velocity Error")
ax.set_title("Map Quality")

fig.tight_layout()

plt.savefig("sharpness-plot-zoom.svg")

plt.show()
