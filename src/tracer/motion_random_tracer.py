import copy
from locale import normalize
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt

import numpy as np

from motion.trajectory import Trajectory

# try:
#     import cupy as cp
# except ModuleNotFoundError:
#     import numpy as cp
#
import numpy as cp
from tqdm import tqdm

from tracer.geometry import *
from tracer.random_tracer import Tracer
from tracer.scene import *


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
        prev_timestamp, prev_pose = traj.time[0], traj.poses[0]

        rx_pattern = []

        for idx in tqdm(range(1, len(traj.poses))):
            pose = traj.poses[idx]
            timestamp = traj.time[idx]

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
                        T_rx: [float | None] = None) -> np.ndarray:
        # transforms: [sources, sinks, segments*paths, 3]
        # wave: [sources, n_samples]

        n_sources = len(self._scene.sources)
        n_sinks = len(self._scene.sinks)

        if T_rx is None:
            T_rx = T_tx

        # Find first and last return delays
        max_delay = max(np.max(path_delays) for path_delays in delays.values())

        # Allocate the result array
        sources_n_samples = wave.shape[1]
        sinks_n_samples = int((max_delay + sources_n_samples * T_tx + 1e-3) // T_rx)
        sinks_wave = np.zeros((n_sinks, sinks_n_samples), dtype=np.float64)

        t_tx = cp.arange(sources_n_samples) * T_tx

        wave = cp.asarray(wave)

        # Multiply and add
        for source in range(n_sources):
            path_attenuations = cp.asarray(attenuations[source])
            path_delays = cp.asarray(delays[source])

            n_paths, _ = path_attenuations.shape
            path_idx = cp.arange(n_paths)

            for sink in range(n_sinks):
                sink_attenuations = path_attenuations[:, sink]
                sink_delays = path_delays[:, sink]

                t_rx = cp.repeat(cp.arange(0, t_tx[-1], T_rx)[cp.newaxis], repeats=len(sink_delays), axis=0) + (sink_delays % T_rx)[:, cp.newaxis]

                start_idx = cp.floor(sink_delays // T_rx).astype(int)
                sink_signal = cp.interp(t_rx, t_tx, wave[source])

                attenuated_signal = sink_signal * db_to_amplitude(sink_attenuations[:, cp.newaxis])

                for path in range(len(sink_attenuations)):
                    sinks_wave[sink, start_idx[path]:start_idx[path] + len(sink_signal[path])] += attenuated_signal[path]

        return sinks_wave
