import numpy as np
import matplotlib.pyplot as plt
import scipy

# grid parameters
nx = 100
ny = 100
grid_size = 0.1

# array intrinsics
array_spacing = 0.0075
array_n = 8

# array extrinsics
array_start_pos, array_end_pos, array_step = 0.0, 10.0, 0.25

# steering
angles = np.linspace(-np.pi / 2, np.pi / 2, 20, endpoint=True)

# frequencies
C = 1500
fc = 100_000
wc = 2 * np.pi * fc
kc = wc / C
km = 2*np.pi*2 # rad / m

xx = np.arange(nx) * grid_size
yy = np.arange(ny) * grid_size

grid = np.array(np.meshgrid(xx, yy)).T

vals = np.zeros((nx, ny), dtype=np.complex64)

targets = np.array(([3, 5], [4, 6], [2, 2]))
# for i in range(0, 10):
#     ang = i * np.pi / 5
#     targets[i] = np.array([np.cos(ang)*2+5, np.sin(ang)*2 + 5])

array = np.arange(array_n) * array_spacing
array_pos = np.arange(array_start_pos, array_end_pos, array_step)

def gain(looking_dir, steering_dir):
    looking_x = looking_dir[0]
    steering_x = steering_dir[0]

    looking_phases = kc * array * looking_x
    steering_phases = kc * array * steering_x
    p = looking_phases - steering_phases
    return np.sqrt(np.sum(np.sin(p))**2 + np.sum(np.cos(p))**2)

# compute directivities for plotting:
directivities = np.zeros((len(angles), 100))
plot_looking_angles = np.linspace(-np.pi/2, np.pi/2, 100)
for steering_i, steering_angle in enumerate(angles):
    for looking_i, looking_angle in enumerate(plot_looking_angles):
        steering_dir = np.array([np.sin(steering_angle), np.cos(steering_angle)])
        looking_dir = np.array([np.sin(looking_angle), np.cos(looking_angle)])
        directivities[steering_i, looking_i] = gain(looking_dir, steering_dir)

fig = plt.figure()

dist = scipy.stats.norm

T_sample = grid_size / C

for arr_x in array_pos:
    for steering_i, steering_angle in enumerate(angles):
        steering_dir = np.array([np.sin(steering_angle), np.cos(steering_angle)])
        range_intensity = np.zeros((
            int(((nx**2+ny**2)**0.5 * grid_size / C) / T_sample * 2 + 1),
        )) # intensity(t)
        tt = np.arange(len(range_intensity)) * T_sample

        for target in targets:
            arr_target = np.array([target[0] - arr_x, target[1]])
            target_range = np.linalg.norm(arr_target)
            target_dir = arr_target / target_range
            range_intensity += dist.pdf(tt, loc=(2 * target_range / C), scale=3 * T_sample)

        # plt.plot(tt, range_intensity)
        # plt.show()

        for i_x, x in enumerate(xx):
            for i_y, y in enumerate(yy):
                r_m = np.linalg.norm(np.array([x, y]) - np.array([arr_x, 0]))
                range_samples = int((2 * r_m / C) / T_sample)
                psi = range_intensity[range_samples] * np.exp(-2j*km*r_m)
                vals[i_x, i_y] += psi

    # Plot magnitude of vals
    plt.contourf(xx, yy, np.abs(vals.T), levels=50, cmap='viridis')
    plt.colorbar(label='Magnitude')

    # Plot current array location
    plt.scatter(array_pos, np.zeros_like(array_pos), color='red', label='Array Location')

    plt.scatter(targets[:, 0], targets[:, 1], color='black', facecolors='none', label='Target')

    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Array Position: {arr_x}')
    plt.show()