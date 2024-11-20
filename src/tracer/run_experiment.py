import pickle
from pathlib import Path

from sonar.phased_array import RectangularArray
from sonar.utils import BarkerCode, Chirp
from tracer.motion_random_tracer import MotionTracer, Trajectory
from tracer.random_tracer import *


def run_experiment(output_file: Path,
                   scene: Scene,
                   trajectory: Trajectory,
                   tx_pattern: [np.ndarray | BarkerCode],
                   T_tx: float,
                   T_rx: [float | None] = None,
                   n_bounces: int = 1,
                   n_rays: int = 10000,
                   array: RectangularArray = None,
                   visualize: bool = False):

    if T_rx is None:
        T_rx = T_tx

    if isinstance(tx_pattern, BarkerCode) or isinstance(tx_pattern, Chirp):
        tx_pattern_raw = np.array([tx_pattern.baseband] * len(scene.sources))
    elif isinstance(tx_pattern, np.ndarray):
        tx_pattern_raw = tx_pattern
    else:
        raise RuntimeError()

    motion_tracer = MotionTracer(scene, n_bounces, n_rays)
    rx_pattern = motion_tracer.trace_trajectory(trajectory, tx_pattern_raw, T_tx, T_rx, visualize=visualize)

    experiment_result = {
        'n_sinks': len(scene.sinks),
        'n_sources': len(scene.sources),
        'tx_pattern': tx_pattern,
        'rx_pattern': rx_pattern,
        'T_tx': T_tx,
        'T_rx': T_rx,
        'C': 1500,
        'trajectory': trajectory,
        'array': array
    }

    with open(output_file, 'wb') as f:
        pickle.dump(experiment_result, f)

    return experiment_result
