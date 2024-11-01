import numpy as np
import open3d as o3d
import copy
from math import pi, cos, sin

from spatialmath import SE3, SO3

from tracer.scene import *
from tracer.random_tracer import *

ARRAY_RADIUS = 0.025
ARRAY_N = 16

sources = [
    Source(
        id="source_1",
        pose=SE3(),
        distribution=UniformContinuousAngularDistribution(
            min_az=0,
            max_az=0,
            min_el=5 * pi / 8,
            max_el=5 * pi / 8,
        )
    ),
    # Source(
    #     id="source_2",
    #     pose=SE3.Tx(-1.0) @ SE3.Rx(pi),
    #     distribution=UniformContinuousAngularDistribution(
    #         min_az=0,
    #         max_az=pi,
    #         min_el=0,
    #         max_el=pi / 4,
    #     )
    # ),
]

sinks = [
    Sink(
        id=f"sink_{angle:.2f}",
        pose=SE3.Rt(SO3(), np.array([ARRAY_RADIUS * cos(angle), ARRAY_RADIUS * sin(angle), 0])),
        distribution=UniformContinuousAngularDistribution(
            min_az=0,
            max_az=2 * pi,
            min_el=pi / 2,
            max_el=pi,
        )
    ) for angle in np.linspace(0, 2 * pi, ARRAY_N + 1)[:-1]
]

sand_material = SimpleMaterial(
    absorption=0.9,
)
sand_surfaces = [
    Surface(
        id=f"sand",
        pose=SE3.Rt(SO3(), np.array([2.0, 0.0, -2.0])),
        material=sand_material,
        mesh=o3d.io.read_triangle_mesh("../assets/cube.ply"),
    )
]

scene = Scene(
    sources=sources,
    sinks=sinks,
    surfaces=sand_surfaces,
)

# scene.visualize()

tracer = Tracer(scene)
tracer.trace(n_rays=100, n_bounces=10)

tracer.visualize()
