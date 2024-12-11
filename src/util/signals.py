import numpy as np
import scipy as sp

from util.config import Config


def modulated_chirp(t: np.ndarray, config: Config):
    return (cosine_envelope(t, config) *
            np.exp(2.0j * np.pi * config.chirp_fc * t +
                   1.0j * np.pi * config.chirp_K * t ** 2))


def cosine_envelope(t: np.ndarray, config: Config):
    return np.where(
        (t >= -config.chirp_duration / 2) & (t <= config.chirp_duration / 2),
        np.cos(np.pi * t / config.chirp_duration) ** 2,
        0
    )


def baseband_chirp(t: np.ndarray, config: Config):
    return cosine_envelope(t, config) * np.exp(1.0j * np.pi * config.chirp_K * t ** 2)


def pulse_compress(signal: np.ndarray, c: Config):
    reference_signal_t = c.Ts * np.arange(int(c.chirp_duration / c.Ts)) - (c.chirp_duration / 2)
    reference_signal = baseband_chirp(reference_signal_t, c)

    correlation = sp.signal.correlate(signal, reference_signal, mode="same")

    return correlation
