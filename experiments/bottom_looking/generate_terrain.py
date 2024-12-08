# https://github.com/pvigier/perlin-numpy

import matplotlib.pyplot as plt
import numpy as np
import pickle
from perlin_numpy import (
    generate_fractal_noise_2d
)

np.random.seed(0)

terrain_size_m = 10
terrain_size_px = 32

terrain_height_m = 0.01

terrain_x = (terrain_size_m / terrain_size_px) * np.arange(terrain_size_px) - (terrain_size_m / 2)
terrain_y = (terrain_size_m / terrain_size_px) * np.arange(terrain_size_px) - (terrain_size_m / 2)

terrain_y, terrain_x = np.meshgrid(terrain_y, terrain_x, indexing="ij")

res = 4
octaves = 3

noise = generate_fractal_noise_2d((terrain_size_px, terrain_size_px), (res, res), octaves)

terrain_z = (terrain_height_m / 2) * (noise + 1.0)

plt.figure()
plt.pcolormesh(terrain_x, terrain_y, terrain_z)
plt.colorbar()
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(terrain_x, terrain_y, terrain_z, cmap="viridis")
plt.show()

with open("terrain.pkl", "wb") as fp:
    pickle.dump({
        "terrain_x": terrain_x,
        "terrain_y": terrain_y,
        "terrain_z": terrain_z,
    }, fp)
