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
            min_az=-pi / 4,
            max_az=pi / 4,
            min_el=pi / 2,
            max_el=pi,
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
        id=f"sink_{x:.3f}",
        pose=SE3.Rt(SO3(), np.array([x, 0.0, 0.0])),
        distribution=UniformContinuousAngularDistribution(
            min_az=0,
            max_az=2 * pi,
            min_el=pi / 2,
            max_el=pi,
        )
    ) for x in np.linspace(1.0, 2.0, ARRAY_N)
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
