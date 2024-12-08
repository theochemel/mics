import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt


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
