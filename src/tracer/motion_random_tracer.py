import copy
from locale import normalize
from pathlib import Path
from typing import Tuple

import numpy as np
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
                timestamp = float(vals[0])
                pose = SE3.RPY(*map(float, vals[4:7])) * SE3.Trans(*map(float, vals[1:4]))
                self._poses.append((timestamp, pose))

        self._idx = 0

    def __getitem__(self, value) -> Tuple[float, SE3]:
        return self._poses[value]

    def __len__(self) -> int:
        return len(self._poses)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[float, SE3]:
        self._idx += 1
        if self._idx == len(self):
            raise StopIteration
        else:
            return self._poses[self._idx]


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
        prev_timestamp, prev_pose = traj[0]

        rx_pattern = []

        for timestamp, pose in traj:
            transformed_scene: Scene = self._get_frame(pose)
            print(pose)

            tracer = Tracer(transformed_scene)
            tracer.trace(self._N_RAYS, self._N_BOUNCES)

            if visualize:
                tracer.visualize()

            dt = timestamp - prev_timestamp
            source_velocities = MotionTracer.get_elem_velocities(prev_pose, pose, self._scene.source_poses, dt)
            sink_velocities = MotionTracer.get_elem_velocities(prev_pose, pose, self._scene.sink_poses, dt)

            transforms = tracer.get_propagation_transforms(source_velocities, sink_velocities)

            sinks_wave = self._propagate_wave(tx_pattern, transforms, T_tx, T_rx)
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
                        transforms: Dict[Tuple[int, int], np.array],
                        T_tx: float,
                        T_rx: [float | None] = None) -> np.ndarray:
        # transforms: [sources, sinks, segments*paths, 3]
        # wave: [sources, n_samples]

        n_sources = len(self._scene.sources)
        n_sinks = len(self._scene.sinks)

        if T_rx is None:
            T_rx = T_tx

        # Find first and last return delays
        max_delay = max([np.max(transforms_source_sink[:, 1]) for transforms_source_sink in transforms.values()])

        # Allocate the result array
        sources_n_samples = wave.shape[1]
        doppler_margin = 1.2
        sinks_n_samples = int((max_delay + sources_n_samples * T_tx * doppler_margin) // T_rx)
        sinks_wave = np.zeros((n_sinks, sinks_n_samples), dtype=np.float64)

        t_tx = np.arange(sources_n_samples) * T_tx

        # Multiply and add
        for sink in range(n_sinks):
            for source in range(n_sources):
                print(f"Source {source}, sink{sink}")
                for transform_i in tqdm(range(len(transforms[(source, sink)]))):  # todo: vectorize
                    transform = transforms[(source, sink)][transform_i]
                    if np.isneginf(transform[0]):
                        continue
                    t_doppler = t_tx * transform[2]  # todo: multiply or divide?
                    t_resampled = np.arange(0, t_doppler[-1], T_rx)
                    sink_signal = np.interp(t_resampled, t_doppler, wave[source])
                    start_idx = int(transform[1] // T_rx)
                    sinks_wave[sink, start_idx:start_idx + len(sink_signal)] += sink_signal * 10 ** (transform[0] / 2)

        return sinks_wave
