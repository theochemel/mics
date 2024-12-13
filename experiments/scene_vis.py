from tracer.random_tracer import *
from tracer.motion_random_tracer import *
from motion.linear_constant_acceleration_trajectory import *
from sonar.utils import Chirp

sources = [
    Source(
        id="source_1",
        pose=SE3(),
        distribution=UniformAngularDistribution()
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
        distribution=UniformAngularDistribution(),
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

T_tx = T_rx = 1e-6  # 1 MHz
code = Chirp(fc=50e3, bw=50e3, T_sample=T_tx, T_chirp=1e-3)

trajectory = LinearConstantAccelerationTrajectory(
    keyposes=[
        SE3.Trans(0.0, 0.0, 0),
        SE3.Trans(0.0, 0.01, 0),
    ],
    max_velocity=0.4,
    acceleration=1.0,
    dt=0.05
)

motion_tracer = MotionTracer(scene, n_bounces=4, n_rays=1000)
tx_pattern_raw = np.array([code.baseband] * len(scene.sources))
rx_pattern = motion_tracer.trace_trajectory(trajectory, tx_pattern_raw, T_tx, T_rx, visualize=False)

fig, ax = plt.subplots(figsize=(3, 1))
ax.plot(rx_pattern[0][0])
ax.set_xticks([])
ax.set_yticks([])
plt.show()