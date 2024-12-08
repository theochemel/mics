import numpy as np

from config import Config


def get_sas_updates(points: np.array, sinks: np.array, signal_t: np.array, signals: np.array, config: Config) -> np.array:
    # points is [n, 3]
    # sinks is [s, 3]
    # signal_t is [d]
    # signals is [s, d]

    n = points.shape[0]
    s = sinks.shape[0]
    d = len(signal_t)

    # [n, s]
    range = np.linalg.norm(points[:, np.newaxis, :] - sinks[np.newaxis, :, :], axis=-1)
    rtt = (2 * range) / config.c

    # [n, s]
    d_interp = (rtt - signal_t[0]) / config.Ts
    d_i = np.floor(d_interp).astype(int)
    d_i_plus_1 = d_i + 1
    d_a = d_interp - d_i

    # [n, s]
    s_i = np.repeat(np.arange(s)[np.newaxis, :], n, axis=0)

    # [n, s]
    valid = (0 <= d_i) & (d_i_plus_1 < d)

    # [v]
    interp_pulse = (1 - d_a[valid]) * signals[s_i[valid], d_i[valid]] + d_a[valid] * signals[s_i[valid], d_i_plus_1[valid]]

    # [n, s]
    updates = np.zeros((n, s), dtype=np.complex128)

    # [n, s]
    updates[valid] = interp_pulse * np.exp(2.0j * np.pi * config.spatial_f * rtt[valid])

    # [s, n]
    updates = np.transpose(updates)

    return updates

