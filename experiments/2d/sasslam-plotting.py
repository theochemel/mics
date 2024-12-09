import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("res.pkl", "rb") as fp:
    res = pickle.load(fp)

plt.figure(figsize=(8, 4))
plt.subplot(2, 1, 1)
plt.plot(res["trajectory"][:-2, 0], label="SLAM")
plt.plot(res["gt_pose"][:, 0], label="Ground Truth")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(res["trajectory"][:-2, 1], label="SLAM")
plt.plot(res["gt_pose"][:, 1], label="Ground Truth")
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(res["trajectory"][:-2, 0], res["trajectory"][:-2, 1])
plt.title("SLAM")
plt.xlim(4, 6)
plt.ylim(4, 6)
plt.gca().set_aspect("equal")
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(res["gt_pose"][:, 0], res["gt_pose"][:, 1])
plt.title("Ground Truth")
plt.xlim(4, 6)
plt.ylim(4, 6)
plt.gca().set_aspect("equal")
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(res["dead_reckon_traj"][:, 0], res["dead_reckon_traj"][:, 1])
plt.title("Dead Reckon")
plt.xlim(4, 6)
plt.ylim(4, 6)
plt.gca().set_aspect("equal")
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(np.abs(res["initial_map"]), extent=[0, 10, 0, 10])
plt.title("Initial Map")
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(np.abs(res["map"]), extent=[0, 10, 0, 10])
plt.title("SLAM Map")
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(np.abs(res["map"]), extent=[0, 10, 0, 10])
plt.plot(res["trajectory"][:-2, 0], res["trajectory"][:-2, 1])
plt.title("SLAM Map")
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(np.abs(res["gt_map"]), extent=[0, 10, 0, 10])
plt.plot(res["gt_pose"][:, 0], res["gt_pose"][:, 1])
plt.title("Ground Truth Map")
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(np.abs(res["dead_reckon_map"]), extent=[0, 10, 0, 10])
plt.plot(res["dead_reckon_traj"][:, 0], res["dead_reckon_traj"][:, 1])
plt.title("Dead Reckon Map")
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(res["target_points"][:, 0], res["target_points"][:, 1], c="r", marker="x")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title("Target Locations")
plt.show()

pass
