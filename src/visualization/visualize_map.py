import numpy as np
import matplotlib.pyplot as plt
import pickle

# Create a figure for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with open('../../experiments/map.pkl', "rb") as f:
    map = pickle.load(f)

# Generate voxel positions
voxels = np.copy(map.map)
min_val = np.min(voxels)
max_val = np.max(voxels)
voxels = (voxels - min_val) / (max_val - min_val)
x, y, z = np.indices(voxels.shape)

# Set the condition for where to draw voxels
# filled = map.map > 0.1  # Change the threshold to show/hide voxels
thresh = np.quantile(voxels, 0.99)
voxels = voxels > thresh

# Define colors based on the data values (e.g., grayscale)
# colors = np.empty(voxels.shape + (4,), dtype=np.float32)
# colors[..., 0] = voxels  # R channel
# colors[..., 1] = voxels  # G channel
# colors[..., 2] = voxels  # B channel
# colors[..., 3] = voxels  # Alpha channel (transparency)

# Plot voxels

uniform_color = (1, 0, 0, 0.5)  # (R, G, B, Alpha)
ax.voxels(voxels, facecolors=uniform_color, edgecolor='k')

# Set labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Show the plot
plt.show()
