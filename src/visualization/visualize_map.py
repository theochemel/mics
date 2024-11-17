from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pickle

import open3d as o3d


# https://stackoverflow.com/questions/74704897/how-to-convert-binary-voxelgrid-numpy-3d-array-to-open3d-voxel-format
def np_to_voxels(grid: np.ndarray):
    # --> otherwise it will default to 0 and the grid will be invisible
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = 1
    for z in range(grid.shape[2]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                if np.isnan(grid[x, y, z]):
                    continue
                voxel = o3d.geometry.Voxel()
                voxel.color = np.array([0.0, 1.0, 0.0])
                voxel.grid_index = np.array([x, y, z])
                # Add voxel object to grid
                voxel_grid.add_voxel(voxel)
    return voxel_grid


def plot_slices_with_colormap(voxels,
                              coordinates,
                              axis=2, n_slices=5, colormap='viridis',
                              geometry: List[o3d.geometry.TriangleMesh] = None,
                              vehicle_pose = None):
    X, Y, Z = voxels.shape
    pcd = o3d.geometry.PointCloud()

    slice_indices = np.linspace(0, voxels.shape[axis]-1, n_slices).astype(np.int32)

    # Get the colormap from Matplotlib
    cmap = plt.get_cmap(colormap)

    if axis == 0:
        slice_points = coordinates[slice_indices, :, :, :]
        slice_values = voxels[slice_indices, :, :]
    elif axis == 1:
        slice_points = coordinates[:, slice_indices, :, :]
        slice_values = voxels[:, slice_indices, :]
    elif axis == 2:
        slice_points = coordinates[:, :, slice_indices, :]
        slice_values = voxels[:, :, slice_indices]
    else:
        raise RuntimeError

    slice_points = slice_points.cpu().numpy().reshape((-1, 4))[:, :3]
    slice_values = slice_values.reshape((-1,))
    color = cmap(slice_values)

    pcd.points.extend(o3d.utility.Vector3dVector(slice_points))
    pcd.colors.extend(o3d.utility.Vector3dVector(color[:, :3]))

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd] +
                                      geometry +
                                      [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(vehicle_pose)])


class SliceViewer:
    def __init__(self, voxels, axis=2, colormap='viridis'):
        self.voxels = voxels
        self.axis = axis
        self.slice_idx = 0
        self.colormap = plt.get_cmap(colormap)
        self.max_idx = voxels.shape[axis] - 1

        # Initialize the point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.update_slice()

    def update_slice(self):
        # Extract the current slice
        if self.axis == 0:
            slice_data = self.voxels[self.slice_idx, :, :]
            points = np.argwhere(slice_data)
            points = np.insert(points, 0, self.slice_idx, axis=1)
        elif self.axis == 1:
            slice_data = self.voxels[:, self.slice_idx, :]
            points = np.argwhere(slice_data)
            points = np.insert(points, 1, self.slice_idx, axis=1)
        elif self.axis == 2:
            slice_data = self.voxels[:, :, self.slice_idx]
            points = np.argwhere(slice_data)
            points = np.insert(points, 2, self.slice_idx, axis=1)

        # Normalize the slice index for colormap (0 to 1)
        norm_idx = self.slice_idx / self.max_idx
        color = self.colormap(norm_idx)[:3]  # Get RGB values

        # Update the point cloud
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))

    def next_slice(self):
        if self.slice_idx < self.max_idx:
            self.slice_idx += 1
            self.update_slice()

    def prev_slice(self):
        if self.slice_idx > 0:
            self.slice_idx -= 1
            self.update_slice()

def main(voxels, axis=2, colormap='viridis'):
    viewer = SliceViewer(voxels, axis, colormap)

    def update_view(vis):
        vis.update_geometry(viewer.pcd)
        vis.poll_events()
        vis.update_renderer()

    def right_arrow_callback(vis):
        viewer.next_slice()
        update_view(vis)

    def left_arrow_callback(vis):
        viewer.prev_slice()
        update_view(vis)

    # Create Open3D visualization window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(viewer.pcd)

    # Register key callbacks
    vis.register_key_callback(262, right_arrow_callback)  # Right arrow key
    vis.register_key_callback(263, left_arrow_callback)   # Left arrow key

    # Start the visualization loop
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # Create a figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    with open('../../map.pkl', "rb") as f:
        map = pickle.load(f)

    map = np.abs(map)

    # Generate voxel positions
    voxels = np.copy(map)
    min_val = np.min(voxels)
    max_val = np.max(voxels)
    voxels = (voxels - min_val) / (max_val - min_val)
    x, y, z = np.indices(voxels.shape)

    thresh = np.quantile(voxels, 0.95)
    voxels[voxels < thresh] = np.nan


    voxel_grid = np_to_voxels(voxels)

    # Visualize
    o3d.visualization.draw_geometries([voxel_grid])
    # Define colors based on the kdata values (e.g., grayscale)
    # colors = np.empty(voxels.shape + (4,), dtype=np.float32)
    # colors[..., 0] = voxels  # R channel
    # colors[..., 1] = voxels  # G channel
    # colors[..., 2] = voxels  # B channel
    # colors[..., 3] = voxels  # Alpha channel (transparency)

    # Plot voxels

    # uniform_color = (1, 0, 0, 0.5)  # (R, G, B, Alpha)
    # ax.voxels(voxels, facecolors=uniform_color, edgecolor='k')

    # Set labels
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    #
    # Show the plot
    # plt.show()
