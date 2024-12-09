import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt


def plot_map_slices(map, grid_z, extent):
    map_abs = np.abs(map)

    vmin = np.min(map_abs)
    vmax = np.max(map_abs)

    rows, cols = 1, map_abs.shape[2]
    fig, axes = plt.subplots(rows, cols, figsize=(3 * map_abs.shape[2], 4))

    # Loop through each subplot and add the images
    for i, ax in enumerate(axes.flat):  # Flatten the 2D axes array for easy indexing
        if i < map_abs.shape[2]:  # Ensure we don't exceed the number of images
            ax.imshow(map_abs[:, :, i], cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)  # Display the image
            ax.set_title(f"z = {grid_z[0, 0, i]}")  # Set a title for each subplot
        else:
            ax.axis("off")  # Hide empty subplots if any

    # Adjust spacing and display the grid
    plt.tight_layout()
    plt.show()


def plot_map_slices_animated(map, extent):
    map_abs = np.abs(map)

    vmax = np.max(map_abs)
    vmin = np.min(map_abs)

    fig, ax = plt.subplots()

    img_display = ax.imshow(map_abs[:, :, 0], extent=extent, vmin=vmin, vmax=vmax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")


    def animate(frame_i):
        img_display.set_data(map_abs[:, :, frame_i])
        return [img_display]


    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=map_abs.shape[-1], interval=1000, blit=True)

    plt.show()
