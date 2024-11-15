import pickle
from pathlib import Path

from tracer.motion_random_tracer import MotionTracer, Trajectory
from tracer.random_tracer import *


def run_experiment(output_file: Path,
                   scene: Scene,
                   trajectory: Trajectory,
                   tx_pattern: np.array,
                   T_tx: float,
                   T_rx: [float | None] = None,
                   n_bounces: int = 1,
                   n_rays: int = 10000,
                   visualize: bool = False):
    if T_rx is None:
        T_rx = T_tx

    motion_tracer = MotionTracer(scene, n_bounces, n_rays)
    rx_pattern = motion_tracer.trace_trajectory(trajectory, tx_pattern, T_tx, T_rx, visualize=visualize)

    experiment_result = {
        "n_sinks": len(scene.sinks),
        "n_sources": len(scene.sources),
        "tx_pattern": tx_pattern,
        "rx_pattern": rx_pattern,
        "T_tx": T_tx,
        "T_rx": T_rx,
    }

    with open(output_file, "wb") as f:
        pickle.dump(experiment_result, f)

    return experiment_result