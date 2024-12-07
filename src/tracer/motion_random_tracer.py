import copy
from locale import normalize
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt

import numpy as np
import torch
from tqdm import tqdm

from tracer.geometry import *
from tracer.random_tracer import Tracer
from tracer.scene import *


class Trajectory:

    def __init__(self, trajectory_file: Path):
        self._poses: List[Tuple[float, SE3]] = []

        with open(str(trajectory_file), 'r') as f:
            while line := f.readline():
                vals = line.split(',')
                vals = list(map(float, vals))
                timestamp = vals[0]
                pose = SE3.Rt(SO3.RPY(vals[4:7]), vals[1:4])
                self._poses.append((timestamp, pose))

        self._idx = 0

    def __getitem__(self, value) -> Tuple[float, SE3]:
        return self._poses[value]

    def __len__(self) -> int:
        return len(self._poses)

    def __iter__(self):
        return self._poses.__iter__()


class MotionTracer:

    def __init__(self, scene: Scene, n_bounces=1, n_rays=5000):
        self._scene = scene
        self._N_BOUNCES = n_bounces
        self._N_RAYS = n_rays

    def trace_trajectory(self,
                         traj: Trajectory,
                         tx_pattern: np.array,
                         T_tx: float,
                         T_rx: [float | None] = None,
                         visualize = False) -> List[np.array]:
        prev_timestamp, prev_pose = traj._poses[0]

        rx_pattern = []

        for timestamp, pose in tqdm(traj._poses[1:]):
            transformed_scene: Scene = self._get_frame(pose)

            tracer = Tracer(transformed_scene)
            tracer.trace(self._N_RAYS, self._N_BOUNCES)

            if visualize:
                tracer.visualize()

            dt = timestamp - prev_timestamp
            source_velocities = MotionTracer.get_elem_velocities(prev_pose, pose, self._scene.source_poses, dt)
            sink_velocities = MotionTracer.get_elem_velocities(prev_pose, pose, self._scene.sink_poses, dt)

            attenuations, delays, doppler_coeffs = tracer.get_propagation_transforms(source_velocities, sink_velocities)

            sinks_wave = self._propagate_wave(tx_pattern, attenuations, delays, doppler_coeffs, T_tx, T_rx)
            rx_pattern.append(sinks_wave)

            prev_timestamp = timestamp
            prev_pose = pose

        return rx_pattern

    @staticmethod
    def get_elem_velocities(vehicle_prev: SE3, vehicle: SE3, vehicle_t_elems: SE3, dt: float):
        result = []
        for vehicle_t_elem in vehicle_t_elems:
            dp = (vehicle_t_elem * vehicle).t - (vehicle_t_elem * vehicle_prev).t
            result.append(dp / dt)
        return result

    def _get_frame(self, transform: SE3):
        frame = copy.deepcopy(self._scene)  # todo: only copy sources and sinks
        for source in frame.sources:
            source.get_tf_from_world(transform)

        for sink in frame.sinks:
            sink.get_tf_from_world(transform)

        return frame

    def _propagate_wave(self,
                        wave: np.array,
                        attenuations,
                        delays,
                        doppler_coeffs,
                        T_tx: float,
                        T_rx: [float | None] = None,
                        device=None) -> np.ndarray:
        # transforms: [sources, sinks, segments*paths, 3]
        # wave: [sources, n_samples]

        if device is None:
            device = torch.device('cuda')

        n_sources = len(self._scene.sources)
        n_sinks = len(self._scene.sinks)

        if T_rx is None:
            T_rx = T_tx

        # Find first and last return delays
        max_delay = max(np.max(path_delays) for path_delays in delays.values())

        # Allocate the result array
        sources_n_samples = wave.shape[1]
        sinks_n_samples = int((max_delay + sources_n_samples * T_tx + 1e-3) // T_rx)
        sinks_wave = torch.zeros((n_sinks, sinks_n_samples), dtype=np.float64, device=device)

        t_tx = torch.arange(sources_n_samples, dtype=torch.float32, device=device) * T_tx

        # Multiply and add
        for source in range(n_sources):
            path_attenuations = torch.tensor(attenuations[source], dtype=torch.float32, device=device)
            path_delays = torch.tensor(delays[source], dtype=torch.float32, device=device)

            for sink in range(n_sinks):
                sink_attenuations = path_attenuations[:, sink]
                sink_delays = path_delays[:, sink]

                t_rx = np.repeat(
                    np.arange(0, t_tx[-1], T_rx)[np.newaxis], repeats=len(sink_delays), axis=0
                ) + (sink_delays % T_rx)[:, np.newaxis]
                start_idx = np.floor(sink_delays // T_rx).astype(int)

                sink_signal = np.interp(t_rx, t_tx, wave[source])

                for path in range(len(sink_attenuations)):
                    sinks_wave[sink, start_idx[path]:start_idx[path] + len(sink_signal[path])] += sink_signal[path] * db_to_amplitude(sink_attenuations[path])

        return sinks_wave
