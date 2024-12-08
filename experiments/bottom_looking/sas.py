import numpy as np

from config import Config


def get_sas_updates(points: np.array, sinks: np.array, signal_t: np.array, signals: np.array, config: Config) -> np.array:
    # points is [n, 3]
    # sinks is [s, 3]
    # signal_t is [d]
    # signals is [s, d]

    n = points.shape[0]
    s = sinks.shape[0]
    d = len(signal_t)

    # [n, s]
    range = np.linalg.norm(points[:, np.newaxis, :] - sinks[np.newaxis, :, :], axis=-1)
    rtt = (2 * range) / config.c

    # [n, s]
    d_interp = (rtt - signal_t[0]) / config.Ts
    d_i = np.floor(d_interp).astype(int)
    d_i_plus_1 = d_i + 1
    d_a = d_interp - d_i

    # [n, s]
    s_i = np.repeat(np.arange(s)[np.newaxis, :], n, axis=0)

    # [n, s]
    valid = (0 <= d_i) & (d_i_plus_1 < d)

    # [v]
    interp_pulse = (1 - d_a[valid]) * signals[s_i[valid], d_i[valid]] + d_a[valid] * signals[s_i[valid], d_i_plus_1[valid]]

    # [n, s]
    updates = np.zeros((n, s), dtype=np.complex128)

    # [n, s]
    updates[valid] = interp_pulse * np.exp(2.0j * np.pi * config.spatial_f * rtt[valid])

    # [s, n]
    updates = np.transpose(updates)

    return updates


def main():
    import matplotlib.pyplot as plt
    import matplotlib.animation

    from signals import get_demod_signals, pulse_compress_signals
    from grid import get_grid_points, get_grid_xy_extent

    import pickle
    import os

    config = Config()

    with open("terrain.pkl", "rb") as f:
        terrain = pickle.load(f)

    terrain_x = terrain["terrain_x"]
    terrain_y = terrain["terrain_y"]
    terrain_z = terrain["terrain_z"]
    targets = np.stack((terrain_x.flatten(), terrain_y.flatten(), terrain_z.flatten()), axis=-1)

    # targets = np.array([
    #     [0.0, 0.0, 0.0],
    #     [1.0, 0.0, 0.1],
    #     [2.0, 2.0, 0.1],
    # ])

    # if os.path.exists("cache.pkl"):
    #     with open("cache.pkl", "rb") as f:
    #         cache = pickle.load(f)
    #
    #     sinks = cache["sinks"]
    #     signal_t = cache["signal_t"]
    #     signals = cache["signals"]
    # else:
    array_x = 0.01 * np.arange(10)
    array_y = 0.01 * np.arange(10)
    array_x, array_y = np.meshgrid(array_x, array_y, indexing="xy")
    array_x = array_x.flatten()
    array_y = array_y.flatten()
    array_z = np.zeros_like(array_x)

    array = np.stack((array_x, array_y, array_z), axis=-1)

    grid_x, grid_y, grid_z = get_grid_points(config)
    grid_xy_extent = get_grid_xy_extent(config)

    grid_points = np.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)

    map = np.zeros_like(grid_x, dtype=np.complex128)

    for i in range(10):
        position = np.array([0.1 * i, 0.0, 5.0])

        sinks = array + position

        signal_t, raw_signals = get_demod_signals(sinks, targets, config)

        signals = pulse_compress_signals(raw_signals, config)

        # with open("cache.pkl", "wb") as f:
        #     pickle.dump({"sinks": sinks, "signal_t": signal_t, "signals": signals}, f)


        updates = get_sas_updates(grid_points, sinks, signal_t, signals, config)

        updates = updates.reshape((updates.shape[0], grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]))

        sum_update = np.sum(updates, axis=0)

        map += sum_update

        map_abs = np.abs(map)

        vmax = np.max(map_abs)
        vmin = np.min(map_abs)

        fig, ax = plt.subplots()

        img_display = ax.imshow(map_abs[:, :, 0], extent=grid_xy_extent, vmin=vmin, vmax=vmax)
        ax.set_ylabel("X (m)")
        ax.set_xlabel("Y (m)")

        def animate(frame_i):
            img_display.set_data(map_abs[:, :, frame_i])
            ax.set_title(grid_z[0, 0, frame_i])
            return [img_display]

        anim = matplotlib.animation.FuncAnimation(fig, animate, frames=map_abs.shape[-1], interval=1000, blit=True)

        plt.show()


if __name__ == "__main__":
    main()