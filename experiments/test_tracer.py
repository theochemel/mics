import numpy as np
import open3d as o3d
import copy
from math import pi, cos, sin

from spatialmath import SE3, SO3

from tracer.scene import *
from tracer.tracer import *

ARRAY_RADIUS = 0.025
ARRAY_N = 16

source = OmnidirectionalSource(
    id="source",
    pose=SE3(),
    sensitivity=0,
)

sinks = [
    OmnidirectionalSink(
        id=f"sink_{angle:.2f}",
        pose=SE3.Rt(SO3(), np.array([ARRAY_RADIUS * cos(angle), ARRAY_RADIUS * sin(angle), 0])),
        sensitivity=0,
    ) for angle in np.linspace(0, 2 * pi, ARRAY_N + 1)[:-1]
]

sand_material = SimpleMaterial(
    absorption=0.9,
)

# sand_surfaces = [
#     Surface(
#         id=f"sand_{i}",
#         pose=SE3.Tz(-10.0) * SE3.Tx(2 * i),
#         material=sand_material,
#         mesh=o3d.io.read_triangle_mesh("../assets/lumpy_8x8.ply"),
#     )
#     for i in range(2)
# ]

sand_surfaces = [
    Surface(
        id=f"sand",
        pose=SE3.Tz(-10.0),
        material=sand_material,
        mesh=o3d.io.read_triangle_mesh("../assets/checkerboard_10x10.ply"),
    )
]

scene = Scene(
    sources=[source],
    sinks=sinks,
    surfaces=sand_surfaces,
)

# scene.visualize()

tracer = Tracer(scene)
tracer.trace(max_path_length=4)

tracer.visualize()