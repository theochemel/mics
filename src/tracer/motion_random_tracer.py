import numpy as np
import open3d as o3d
from spatialmath import SE3, Twist3

from pathlib import Path
from typing import List, Tuple

from experiments.if_beamforming import samples
from tracer.scene import *
from tracer.geometry import *
from tracer.random_tracer import Tracer


class Trajectory:

    def __init__(self, trajectory_file: Path):
        self._poses: List[Tuple[float, SE3]] = []

        with open(str(trajectory_file), 'r') as f:
            while line :=  f.readline():
                vals = line.split(',')
                timestamp = float(vals[0])
                pose = SE3.RPY(*vals[4:7]) * SE3.Trans(*vals[1:4])
                self._poses.append((timestamp, pose))

        self._idx = 0

    def __getitem__(self, value) -> Tuple[float, SE3]:
        return self._poses[value]

    def __len__(self) -> int:
        return len(self._poses)

    def __iter__(self) -> Tuple[float, SE3]:
        self._idx = 0
        return self._poses[self._idx]


    def __next__(self) -> Tuple[float, SE3]:
        self._idx += 1
        return self._poses[self._idx]


class MotionTracer:

    def __init__(self, scene: Scene):
        self._scene = scene
        self._N_BOUNCES = 4
        self._N_RAYS = 1000

    def trace_trajectory(self, traj: Trajectory):
        prev_timestamp, prev_pose = traj[0]

        for timestamp, pose in traj:
            frame: Scene = self._get_frame(pose)
            tracer = Tracer(frame)
            tracer.trace(self._N_RAYS, self._N_BOUNCES)

            prev_t_curr = pose * prev_pose.inv()
            twist = prev_t_curr.log() / (timestamp - prev_timestamp)

            source_velocities = MotionTracer.get_elem_velocities(pose, self._scene.source_poses, twist)
            sink_velocities = MotionTracer.get_elem_velocities(pose, self._scene.sink_poses, twist)

            transforms = tracer.get_propagation_transforms(source_velocities, sink_velocities)

            prev_timestamp = timestamp
            prev_pose = pose


    @staticmethod
    def get_elem_velocities(world_t_vehicle: SE3, world_t_elems: List[SE3], v_world: Twist3):
        return [(world_t_vehicle * world_t_elem).inv().Ad() * v_world for world_t_elem in world_t_elems]

    def _get_frame(self, transform: SE3):
        frame = self._scene
        for source in frame.sources.values():
            source.transform(transform)

        for sink in frame.sinks.values():
            sink.transform(transform)

        return frame

    def _propagate_wave(self, wave: np.array, transforms: np.array, T_tx: float, T_rx: [float | None] = None):
        # transforms: [sources, sinks, segments*paths, 3]
        # wave: [sources, n_samples]

        if T_rx == None:
            T_rx = T_tx

        eps = 1e-6
        # Find first and last return delays
        min_delay = np.min(transforms[:, :, :, 1])
        max_delay = np.max(transforms[:, :, :, 1])

        # Find min and max attenuations
        min_attn =  np.min(transforms[:, :, :, 0])

        # Normalize delays and attenuations
        transforms[:, :, :, 1] -= min_delay
        transforms[:, :, :, 0] -= min_attn

        # Allocate the result array
        sources_n_samples = wave.shape[1]
        doppler_margin = 1.5
        sinks_n_samples = (max_delay + sources_n_samples*T_tx*doppler_margin - min_delay) // T_rx
        n_sinks = transforms.shape[1]
        sinks_wave = np.array((n_sinks, sinks_n_samples))

        t_tx = np.arange(sources_n_samples) * T_tx

        # Multiply and add
        for sink_i in range(n_sinks):
            for source_i in range(transforms.shape[0]):
                for transform_i in range(transforms.shape[2]):
                    transform = transforms[source_i, sink_i, transform_i]
                    t_doppler = t_tx * transform[2]  # todo: multiply or divide?
                    t_resampled = np.arange(0, t_doppler[-1], T_rx)
                    sink_signal = np.interp(t_resampled, t_doppler, wave[source_i])
                    start_idx = transform[1] // T_rx
                    sinks_wave[sink_i, start_idx:start_idx+len(sink_signal)] = sink_signal * 10**(transform[0] / 2)

        return sinks_wave
