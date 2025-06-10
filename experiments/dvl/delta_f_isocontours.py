import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

v = np.array([1, 0, 0])
h = 1

v_norm = v / np.linalg.norm(v)
vx = v_norm[0]
vy = v_norm[1]
vz = v_norm[2]

fig_1, ax_1 = plt.subplots(subplot_kw={"projection": "polar"})
fig_2, ax_2 = plt.subplots()
fig_3, ax_3 = plt.subplots()

for delta in np.linspace(-1, 1,  10):

    theta = np.linspace(0, 2 * np.pi, 500)

    A = vz
    B = vx * np.cos(theta) + vy * np.sin(theta)

    arccos_arg = delta / np.sqrt(A ** 2 + B ** 2)
    arccos_arg = np.where((arccos_arg > 1) | (arccos_arg < -1), np.nan, arccos_arg)

    phi = np.arccos(arccos_arg) + np.arctan2(B, A)

    # Take results only for 0 <= phi <= pi / 2

    phi = np.where(phi >= np.pi / 2, np.nan, phi)

    theta_adj = np.where(phi < 0, theta + np.pi, theta)
    phi_adj = np.abs(phi)

    ax_1.plot(theta_adj, phi_adj)
    ax_1.set_rlim([0, np.pi / 4])

    x = h * np.cos(theta) * np.tan(phi)
    y = h * np.sin(theta) * np.tan(phi)

    ax_2.plot(x, y)

    grid_y = np.linspace(-10, 10, 100)
    grid_x = np.linspace(-10, 10, 100)
    grid_y, grid_x = np.meshgrid(grid_y, grid_x, indexing="ij")

    grid_y = grid_y.flatten()
    grid_x = grid_x.flatten()
    grid_z = np.full_like(grid_x, fill_value=h)

    grid = np.stack((grid_x, grid_y, grid_z), axis=-1)
    grid_r = np.linalg.norm(grid, axis=-1)
    grid_d_norm = grid / np.linalg.norm(grid, axis=-1, keepdims=True)

    grid_delta_f = np.sum(grid_d_norm * v_norm, axis=-1)

    grid_r = grid_r.reshape((100, 100))
    grid_delta_f = grid_delta_f.reshape((100, 100))

    ax_2.imshow(grid_delta_f, extent=[-10, 10, -10, 10], zorder=0)
    ax_2.set_xlim([-1, 1])
    ax_2.set_ylim([-1, 1])
    ax_2.add_patch(
        patches.Circle((0, 0), radius=1, edgecolor="black", facecolor="none")
    )

    ax_3.imshow(grid_r, extent=[-10, 10, -10, 10], zorder=0)

plt.show()