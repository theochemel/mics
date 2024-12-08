import numpy as np
import scipy as sp
from typing import Tuple
from math import ceil

from config import Config


def cosine_envelope(t: np.array, config: Config):
    return np.where(
        (t >= -config.chirp_duration / 2) & (t <= config.chirp_duration / 2),
        np.cos(np.pi * t / config.chirp_duration) ** 2,
        0
    )


def chirp(t: np.array, config: Config):
    print(config.chirp_K)
    return cosine_envelope(t, config) * np.exp(2.0j * np.pi * config.chirp_fc * t + 1.0j * np.pi * config.chirp_K * t ** 2)


def reference_chirp(t: np.array, config: Config):
    return cosine_envelope(t, config) * np.exp(1.0j * np.pi * config.chirp_K * t ** 2)


def get_demod_signals(sinks: np.array, targets: np.array, config: Config) -> Tuple[float, np.array]:
    # sinks is [n, 3]
    # Targets is [m, 3]

    distances = np.linalg.norm(sinks[:, np.newaxis, :] - targets[np.newaxis, :, :], axis=-1) # [n x m]

    rtts = (2 * distances) / config.c

    min_rtt = max(np.min(rtts) - config.chirp_duration, 0)
    max_rtt = min(np.max(rtts) + config.chirp_duration, config.max_rt_t)

    n_samples = ceil(max(max_rtt - min_rtt, 0) / config.Ts)

    start_time = min_rtt

    signal_t = start_time + config.Ts * np.arange(n_samples)

    # Sum chirps along targets axis
    signals = np.sum(chirp(signal_t[np.newaxis, np.newaxis, :] - rtts[:, :, np.newaxis], config), axis=1)

    demod_signals = signals * np.exp(-2.0j * np.pi * config.chirp_fc * signal_t)[np.newaxis, :]

    return signal_t, demod_signals


def demod_signal(signal_t: np.array, signal: np.array, config: Config) -> np.array:
    # signal is a REAL SIGNAL
    iq_signal = sp.signal.hilbert(signal, axis=-1)

    return iq_signal * np.exp(-2.0j * np.pi * config.chirp_fc * signal_t)[np.newaxis, :]


def pulse_compress_signals(signal: np.array, config: Config):
    reference_signal_t = config.Ts * np.arange(int(config.chirp_duration / config.Ts)) - (config.chirp_duration / 2)
    reference_signal = reference_chirp(reference_signal_t, config)

    correlation = np.empty_like(signal)

    for i in range(signal.shape[0]):
        correlation[i] = sp.signal.correlate(signal[i], reference_signal, mode="same")

    return correlation


def main():
    import matplotlib.pyplot as plt

    config = Config()

    targets = np.array([
        [2.0, 0.0, 0.0],
    ])

    sinks = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    signal_t, raw_signals = get_demod_signals(sinks, targets, config)

    plt.plot(signal_t, np.real(raw_signals[0]))
    plt.plot(signal_t, np.imag(raw_signals[0]))
    plt.plot(signal_t, np.real(raw_signals[1]))
    plt.plot(signal_t, np.imag(raw_signals[1]))
    plt.show()

    signals = pulse_compress_signals(raw_signals, config)

    plt.plot(signal_t, np.real(signals[0]))
    plt.plot(signal_t, np.imag(signals[0]))
    plt.plot(signal_t, np.real(signals[1]))
    plt.plot(signal_t, np.imag(signals[1]))
    plt.show()


if __name__ == "__main__":
    main()