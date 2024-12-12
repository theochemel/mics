import numpy as np
from spatialmath import SE3, SO3

from motion.linear_constant_acceleration_trajectory import LinearConstantAccelerationTrajectory
from sonar.occupancy_grid import OccupancyGridMap
from sonar.phased_array import RectangularArray
from sonar.utils import BarkerCode, FMBarker, PMBarker, Chirp
from motion.linear_constant_velocity_trajectory import LinearConstantVelocityTrajectory
from tracer.run_experiment import run_experiment
from pathlib import Path
from tracer.scene import Source, UniformAngularDistribution, SimpleMaterial, Surface, Scene
import open3d as o3d
import argparse

from numpy import pi



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o')
    parser.add_argument('--cubes', action='store_true')
    args = parser.parse_args()

    sources = [
        Source(
            id="source_1",
            pose=SE3(),
            distribution=UniformAngularDistribution(),
        )
    ]

    arr = RectangularArray(2, 2, 1e-2, UniformAngularDistribution())

    sand_material = SimpleMaterial(
        absorption=0.9,
    )

    # bottom_tile_size = 2
    # bottom_nx = 5
    # botton_ny = 5
    #
    # bottom_xx = np.arange(bottom_nx) * bottom_tile_size - (bottom_nx * bottom_tile_size / 2)
    # bottom_yy = np.arange(botton_ny) * bottom_tile_size - (botton_ny * bottom_tile_size / 2)

    surfaces = [
        Surface(
            id=f"bottom1",
            pose=SE3.Trans(0, 0, -1),
            material=sand_material,
            mesh=o3d.io.read_triangle_mesh("assets/cube_10cm.ply"),
        ),
        # Surface(
        #     id=f"bottom2",
        #     pose=SE3.Trans(0, -0.5, -1),
        #     material=sand_material,
        #     mesh=o3d.io.read_triangle_mesh("assets/cube_10cm.ply"),
        # ),
        # Surface(
        #     id=f"bottom3",
        #     pose=SE3.Trans(0, 0.5, -1),
        #     material=sand_material,
        #     mesh=o3d.io.read_triangle_mesh("assets/cube_10cm.ply"),
        # ),
    ]

    scene = Scene(
        sources=sources,
        sinks=arr.sinks,
        surfaces=surfaces,
    )

    trajectory = LinearConstantAccelerationTrajectory(
        keyposes=[
            SE3.Trans(0.0, -0.25, 0),
            SE3.Trans(0.0, 0.25, 0),
            SE3.Trans(0.0, 0.0, 0),
            SE3.Trans(0.0, 0.25, 0),
            SE3.Trans(0.0, -0.25, 0),
            SE3.Trans(0.0, 0.25, 0),
        ],
        max_velocity=0.4,
        acceleration=1.0,
        dt=0.01
    )

    print(f"Trajectory length: {len(trajectory.poses)}")

    T_tx = T_rx = 1e-6 # 1 MHz
    code = Chirp(fc=50e3, bw=50e3, T_sample=T_tx, T_chirp=1e-3)

    result = run_experiment(Path.cwd() / Path(args.o),
                            scene,
                            trajectory,
                            code,
                            T_tx,
                            T_rx,
                            n_rays=100000,
                            array=arr,
                            visualize=False)
