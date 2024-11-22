import numpy as np
from spatialmath import SE3, SO3

from sonar.occupancy_grid import OccupancyGridMap
from sonar.phased_array import RectangularArray
from sonar.utils import BarkerCode, FMBarker, PMBarker, Chirp
from tracer.motion_random_tracer import Trajectory
from tracer.run_experiment import run_experiment
from pathlib import Path
from tracer.scene import Source, UniformContinuousAngularDistribution, SimpleMaterial, Surface, Scene
import open3d as o3d

from numpy import pi
import matplotlib.pyplot as plt

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

arr = RectangularArray(10, 1, 0.0075, UniformContinuousAngularDistribution(
    min_az=-pi, max_az=pi, min_el=0, max_el=pi
))

sand_material = SimpleMaterial(
    absorption=0.9,
)

surfaces = [
    Surface(
        id=f"cube1",
        pose=SE3.Rt(SO3.RPY(0, 0, 0), np.array([0.0, 0.0, 0.0])),
        material=sand_material,
        mesh=o3d.io.read_triangle_mesh("assets/cube.ply"),
    ),
]

scene = Scene(
    sources=sources,
    sinks=arr.sinks,
    surfaces=surfaces,
)


trajectory = Trajectory(Path('experiments/circular_path.csv'))

# code = PMBarker(BarkerCode.Sequence.BARKER_2, 100_000, T_tx, 100e-6)

T_tx = T_rx = 1e-6 # 1 MHz
code = Chirp(f_hi=100e3, f_lo=50e3, T_sample=T_tx, T_chirp=1e-3)
# code = PMBarker(BarkerCode.Sequence.BARKER_13, 100e3, T_rx, 50e-6)

result = run_experiment(Path('exp_res.pkl'),
                        scene,
                        trajectory,
                        code,
                        T_tx,
                        T_rx,
                        n_rays=10000,
                        array=arr,
                        visualize=False)
