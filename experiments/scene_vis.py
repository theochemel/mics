from tracer.random_tracer import *

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
        pose=SE3.Rt(SO3(), np.array([array_x[i], array_y[i], 0.0])),
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
surfaces = [
    Surface(
        id=f"bottom",
        pose=SE3.Rt(SO3(), np.array([0.0, 0.0, -2.0])),
        material=sand_material,
        mesh=o3d.io.read_triangle_mesh("assets/lumpy_8x8.ply"),
    ),
    Surface(
        id=f"cube",
        pose=SE3.Rt(SO3(), np.array([0.0, 0.0, -1.0])),
        material=sand_material,
        mesh=o3d.io.read_triangle_mesh("assets/cube_10cm.ply"),
    ),
]

scene = Scene(
    sources=sources,
    sinks=sinks,
    surfaces=surfaces,
)

tracer = Tracer(scene)
tracer.trace(1000, 10)

tracer.visualize()