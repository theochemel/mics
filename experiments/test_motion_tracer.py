import numpy as np
import open3d as o3d
import copy
from math import pi, cos, sin
import pickle

from matplotlib import pyplot as plt
from scipy.signal import spectrogram
from spatialmath import SE3, SO3

from tracer.run_experiment import run_experiment
from tracer.scene import *
from tracer.random_tracer import *
from tracer.motion_random_tracer import MotionTracer, Trajectory
from pathlib import Path

sources = [
    Source(
        id="source_1",
        pose=SE3(),
        distribution=UniformContinuousAngularDistribution(
            min_az=-pi / 2,
            max_az=pi / 2,
            min_el=0,
            max_el=pi,
        )
    ),
]

c = 1500
f = 200e3
l = c / f

spacing = l / 2
N = 4

array_x = spacing * np.arange(N) - (spacing * (N - 1) / 2)
array_y = spacing * np.arange(N) - (spacing * (N - 1) / 2)

array_x, array_y = np.meshgrid(array_x, array_y, indexing="xy")
array_x = array_x.flatten()
array_y = array_y.flatten()

sinks = [
    Sink(
        id=f"sink_i",
        pose=SE3.Rt(SO3(), np.array([0.0, array_x[i], array_y[i]])),
        distribution=UniformContinuousAngularDistribution(
            min_az=-pi,
            max_az=pi,
            min_el=pi / 2,
            max_el=pi,
        )
    ) for i in range(len(array_x))
]

sand_material = SimpleMaterial(
    absorption=0.9,
)
sand_surfaces = [
    Surface(
        id=f"sand1",
        pose=SE3.Rt(SO3(), np.array([10.0, 0.0, 0.0])),
        material=sand_material,
        mesh=o3d.io.read_triangle_mesh("assets/cube.ply"),
    ),
]

scene = Scene(
    sources=sources,
    sinks=sinks,
    surfaces=sand_surfaces,
)

# scene.visualize()

trajectory = Trajectory(Path('experiments/circular_path.csv'))

T_tx = T_rx = 1e-6 # 1 MHz
duration = 1e-3
tt = np.arange(0, duration, T_tx)
f_min, f_max = 100e3, 200e3

k = (f_max - f_min) / duration  # Sweep rate
instantaneous_frequency = f_min + k * tt
chirp = np.sin(2 * np.pi * (f_min * tt + 0.5 * k * tt**2)).reshape(1, -1)

run_experiment(Path('exp_res.pkl'),
               scene,
               trajectory,
               chirp,
               T_tx)
