import numpy as np
import open3d as o3d
import copy
from math import pi, cos, sin
import pickle

from matplotlib import pyplot as plt
from scipy.signal import spectrogram
from spatialmath import SE3, SO3

from tracer.scene import *
from tracer.random_tracer import *
from tracer.motion_random_tracer import MotionTracer, Trajectory
from pathlib import Path

ARRAY_RADIUS = 0.025
ARRAY_N = 2

sources = [
    Source(
        id="source_1",
        pose=SE3(),
        distribution=UniformContinuousAngularDistribution(
            min_az=-pi,
            max_az=pi,
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
        pose=SE3.Tx(x),
        distribution=UniformContinuousAngularDistribution(
            min_az=-pi,
            max_az=pi,
            min_el=pi / 2,
            max_el=pi,
        )
    ) for x in [-1, 1]
]

sand_material = SimpleMaterial(
    absorption=0.9,
)
sand_surfaces = [
    # Surface(
    #     id=f"sand0",
    #     pose=SE3.Rt(SO3(), np.array([0.0, 0.0, 0.0])),
    #     material=sand_material,
    #     mesh=o3d.io.read_triangle_mesh("assets/cube.ply"),
    # ),
    Surface(
        id=f"sand1",
        pose=SE3.Rt(SO3(), np.array([-15.0, 0.0, -1.0])),
        material=sand_material,
        mesh=o3d.io.read_triangle_mesh("assets/cube.ply"),
    ),
]

scene = Scene(
    sources=sources,
    sinks=sinks,
    surfaces=sand_surfaces,
)


motion_tracer = MotionTracer(scene)
trajectory = Trajectory(Path('experiments/rotation.csv'))

T_tx = T_rx = 1e-6 # 1 MHz
duration = 1e-3
tt = np.arange(0, duration, T_tx)
f_min, f_max = 100e3, 200e3

k = (f_max - f_min) / duration  # Sweep rate
instantaneous_frequency = f_min + k * tt
chirp = np.sin(2 * np.pi * (f_min * tt + 0.5 * k * tt**2)).reshape(1, -1)

wave = motion_tracer.trace_trajectory(trajectory, chirp, T_tx, T_rx)

experiment_result = {
    "n_sinks": ARRAY_N,
    "n_sources": 1,
    "tx_wave": chirp,
    "rx_wave": wave,
    "T_tx": T_tx,
    "T_rx": T_rx,
}

with open("rotation_result.pkl", "wb") as f:
    pickle.dump(experiment_result, f)

#
# fig, axes = plt.subplots(2, 1, figsize=(15, 6))
#
# ax0 = axes[0]
# f, t, S = spectrogram(wave[0], 1 / T_tx)
# pcm0 = ax0.pcolormesh(t, f, 10 * np.log10(S_tx), shading='gouraud', cmap='viridis')
# fig.colorbar(pcm0, ax=ax0, label='Power/Frequency (dB/Hz)')
# ax0.set_ylabel('Frequency (Hz)')
# ax0.set_xlabel('Time (s)')
# ax0.set_title('Rx0')
#
# ax1 = axes[1]
# f, t, Sxx = spectrogram(wave[1], 1 / T_rx)
# pcm1 = ax1.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
# fig.colorbar(pcm1, ax=ax1, label='Power/Frequency (dB/Hz)')
# ax1.set_ylabel('Frequency (Hz)')
# ax1.set_xlabel('Time (s)')
# ax1.set_title('Rx1')
#
# plt.tight_layout()
# plt.show()
#
