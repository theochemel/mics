import numpy as np
from spatialmath import SE3, SO3

from sonar.occupancy_grid import OccupancyGridMap
from sonar.phased_array import RectangularArray
from sonar.utils import BarkerCode, FMBarker
from tracer.motion_random_tracer import Trajectory
from tracer.run_experiment import run_experiment
from pathlib import Path
from tracer.scene import Source, UniformContinuousAngularDistribution, SimpleMaterial, Surface, Scene
import open3d as o3d

from numpy import pi

sources = [
    Source(
        id="source_1",
        pose=SE3(),
        distribution=UniformContinuousAngularDistribution(
            min_az=-pi / 4,
            max_az=pi / 4,
            min_el=pi / 2,
            max_el=pi,
        )
    )
]

arr = RectangularArray(10, 10, 0.075, UniformContinuousAngularDistribution(
    min_az=-pi, max_az=pi, min_el=0, max_el=pi/2
))

sand_material = SimpleMaterial(
    absorption=0.9,
)
sand_surfaces = [
    Surface(
        id=f"sand",
        pose=SE3.Rt(SO3(), np.array([0.0, 0.0, 0.0])),
        material=sand_material,
        mesh=o3d.io.read_triangle_mesh("../assets/cube.ply"),
    )
]

scene = Scene(
    sources=sources,
    sinks=arr.sinks,
    surfaces=sand_surfaces,
)


trajectory = Trajectory(Path('experiments/circular_path.csv'))
T_tx = T_rx = 1e-6 # 1 MHz
code = FMBarker(BarkerCode.Sequence.BARKER_7, 100_000, 110_000, T_tx, 100e-6)

result = run_experiment(Path('exp_res.pkl'),
                        scene,
                        trajectory,
                        code.baseband,
                        T_tx,
                        T_rx,
                        n_rays=10000)



