import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt


def plot_map_slices(map, grid_z, extent, poses=None):
    map_abs = np.abs(map)

    vmin = np.min(map_abs)
    vmax = np.max(map_abs)

    rows, cols = 1, map_abs.shape[2]
    fig, axes = plt.subplots(rows, cols, figsize=(3 * map_abs.shape[2], 4))

    # Loop through each subplot and add the images
    for i, ax in enumerate(axes.flat):  # Flatten the 2D axes array for easy indexing
        if i < map_abs.shape[2]:  # Ensure we don't exceed the number of images
            ax.imshow(map_abs[:, :, i], cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)  # Display the image
            if poses is not None:
                ax.scatter(poses[:, 0], poses[:, 1])
            ax.set_title(f"z = {grid_z[0, 0, i]}")  # Set a title for each subplot
        else:
            ax.axis("off")  # Hide empty subplots if any

    # Adjust spacing and display the grid
    plt.tight_layout()
    plt.show()


def plot_map_slices_animated(map, extent, traj):
    map_abs = np.abs(map)

    vmax = np.max(map_abs)
    vmin = np.min(map_abs)

    fig, ax = plt.subplots()

    img_display = ax.imshow(map_abs[:, :, 0], extent=extent, vmin=vmin, vmax=vmax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    positions = np.array([pose.t for pose in traj.poses])

    plt.plot(positions[:, 0], positions[:, 1])

    def animate(frame_i):
        img_display.set_data(map_abs[:, :, frame_i])
        # plt.plot(positions[:, 0], positions[:, 1])
        return [img_display]


    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=map_abs.shape[-1], interval=500)

    plt.show()


if __name__ == '__main__':
    import pickle as pkl

    with open('lines-0_1ms-zigzag-map.pkl', 'rb') as f:
        map = pkl.load(f)

    grid_x, grid_y, grid_z = map['grid_x'], map['grid_y'], map['grid_z']
    map_weights = map['map_weights']
    map = map['map']

    extent = [np.min(grid_x), np.max(grid_x), np.min(grid_y), np.max(grid_y)]

    plot_map_slices(map, grid_z, extent)
