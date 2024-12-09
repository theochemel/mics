import numpy as np
from spatialmath import SE3, SO3

from motion.linear_constant_acceleration_trajectory import LinearConstantAccelerationTrajectory
from sonar.occupancy_grid import OccupancyGridMap
from sonar.phased_array import RectangularArray
from sonar.utils import BarkerCode, FMBarker, PMBarker, Chirp
from motion.linear_constant_velocity_trajectory import LinearConstantVelocityTrajectory
from tracer.run_experiment import run_experiment
from pathlib import Path
from tracer.scene import Source, UniformContinuousAngularDistribution, SimpleMaterial, Surface, Scene
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
            distribution=UniformContinuousAngularDistribution(
                min_az=-pi,
                max_az=pi,
                min_el=0,
                max_el=pi,
            )
        )
    ]

    arr = RectangularArray(6, 6, 1e-2, UniformContinuousAngularDistribution(
        min_az=-pi, max_az=pi, min_el=0, max_el=pi
    ))

    sand_material = SimpleMaterial(
        absorption=0.9,
    )

    bottom_tile_size = 2
    bottom_nx = 5
    botton_ny = 5

    bottom_xx = np.arange(bottom_nx) * bottom_tile_size - (bottom_nx * bottom_tile_size / 2)
    bottom_yy = np.arange(botton_ny) * bottom_tile_size - (botton_ny * bottom_tile_size / 2)

    surfaces = [
        Surface(
            id=f"cube-1-{x}",
            pose=SE3.Trans(x, 3, 0),
            material=sand_material,
            mesh=o3d.io.read_triangle_mesh("assets/cube.ply"),
        )
        # for x in np.linspace(-10, 10, 5)
        for x in [0]
    ]
    # ] + [
    #     Surface(
    #         id=f"cube-2-{x}",
    #         pose=SE3.Trans(x, -1, 0),
    #         material=sand_material,
    #         mesh=o3d.io.read_triangle_mesh("assets/cube.ply"),
    #     )
    #     for x in np.linspace(-10, 10, 3)
    # ]

    # for x in bottom_xx:
    #     for y in bottom_yy:
    #         surfaces.append(Surface(
    #             id=f'bottom-{x}-{y}',
    #             pose=SE3.Rt(SO3(), np.array([x, y, -3.0])),
    #             material=sand_material,
    #             mesh=o3d.io.read_triangle_mesh("assets/lumpy_8x8.ply")
    #         ))
    #
    if args.cubes:
        cube_xx, cube_yy = bottom_xx, bottom_yy
        cube_coords = np.stack(np.meshgrid(cube_xx, cube_yy), axis=2)

        cube_coords += np.random.normal(loc=0, scale=0.6, size=cube_coords.shape)
        cube_coords = cube_coords.reshape(-1, 2)

        for x, y in cube_coords:
            surfaces.append(Surface(
                id=f'cube-{x}-{y}',
                pose=SE3.Rt(SO3(), np.array([x, y, 0])),
                material=sand_material,
                mesh=o3d.io.read_triangle_mesh("assets/cube_10cm.ply")
            ))

    scene = Scene(
        sources=sources,
        sinks=arr.sinks,
        surfaces=surfaces,
    )

    trajectory = LinearConstantAccelerationTrajectory(
        keyposes=[
            SE3.Trans(-0.5, 0.0, 0),
            SE3.Trans(0.5, 0.0, 0),
            # SE3.Trans(0.5, 0.5, 0),
            # SE3.Trans(-0.5, 0.5, 0),
            # SE3.Trans(-0.5, -0.5, 0)
        ],
        max_velocity=0.2,
        acceleration=100,
        dt=0.05
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
                            n_rays=10000,
                            array=arr,
                            visualize=False)
