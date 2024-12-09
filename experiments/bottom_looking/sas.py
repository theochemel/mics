import numpy as np
import torch

from config import Config


def get_sas_updates(points: np.array,
                    sinks: np.array,
                    source: np.array,
                    signal_t: np.array,
                    signals: np.array,
                    config: Config) -> np.array:
    # points is [n, 3]
    # sinks is [s, 3]
    # source is [3]
    # signal_t is [d]
    # signals is [s, d]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        points = torch.tensor(points).to(device)
        sinks = torch.tensor(sinks).to(device)
        source = torch.tensor(source).to(device)
        signal_t = torch.tensor(signal_t).to(device)
        signals = torch.tensor(signals, dtype=torch.complex128).to(device)

        n = points.shape[0]
        s = sinks.shape[0]
        d = len(signal_t)

        # [n, s]
        updates = torch.zeros((n, s), dtype=torch.complex128, device=device)

        for sink in range(s):
            # [n, 3]
            delta_sink = points - sinks[sink]

            # [n, 3]
            delta_source = points - source.unsqueeze(0)

            d_sink = torch.linalg.norm(delta_sink, axis=-1)
            d_source = torch.linalg.norm(delta_source, axis=-1)

            # [n]
            radius = torch.sqrt(delta_sink[:, 0] ** 2 + delta_sink[:, 1] ** 2)
            angle = torch.arctan(radius / -delta_sink[:, 2])

            # [n]
            r = torch.linalg.norm(delta_sink, axis=-1) + torch.linalg.norm(delta_source, axis=-1)
            rtt = r / config.c

            # [n]
            d_interp = (rtt - signal_t[0]) / config.Ts
            d_i = torch.floor(d_interp).to(torch.int64)
            d_i_plus_1 = d_i + 1
            d_a = d_interp - d_i

            # [n]
            valid = (0 <= d_i) & (d_i_plus_1 < d) # & (torch.abs(angle) < config.fov / 2)

            # [v]
            interp_pulse = (1 - d_a[valid]) * signals[sink, d_i[valid]] + d_a[valid] * signals[sink, d_i_plus_1[valid]]

            # [n, s]
            updates[valid, sink] = interp_pulse * torch.exp(2.0j * np.pi * config.chirp_fc * rtt[valid])
            # updates[:, sink] *= (d_source ** 2) * (d_sink ** 2)
            # updates[:, sink] *= (1 / torch.cos(angle))

    # [s, n]
    updates = torch.transpose(updates, dim0=0, dim1=1)

    updates = updates.cpu().numpy()

    return updates

def get_sas_weights(points: np.array,
                    sinks: np.array,
                    source: np.array,
                    signal_t: np.array,
                    signals: np.array,
                    config: Config) -> np.array:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        points = torch.tensor(points).to(device)
        sinks = torch.tensor(sinks).to(device)
        source = torch.tensor(source).to(device)
        signal_t = torch.tensor(signal_t).to(device)
        signals = torch.tensor(signals, dtype=torch.complex128).to(device)

        n = points.shape[0]
        s = sinks.shape[0]
        d = len(signal_t)

        # [n, s]
        weights = torch.zeros((n, s), dtype=torch.float64, device=device)

        for sink in range(s):
            # [n, 3]
            delta_sink = points - sinks[sink]

            # [n, 3]
            delta_source = points - source.unsqueeze(0)

            d_sink = torch.linalg.norm(delta_sink, axis=-1)
            d_source = torch.linalg.norm(delta_source, axis=-1)

            # [n]
            radius = torch.sqrt(delta_sink[:, 0] ** 2 + delta_sink[:, 1] ** 2)
            angle = torch.arctan(radius / -delta_sink[:, 2])

            # [n]
            r = torch.linalg.norm(delta_sink, axis=-1) + torch.linalg.norm(delta_source, axis=-1)
            rtt = r / config.c

            # [n]
            d_interp = (rtt - signal_t[0]) / config.Ts
            d_i = torch.floor(d_interp).to(torch.int64)
            d_i_plus_1 = d_i + 1
            d_a = d_interp - d_i

            # [n]
            valid = (0 <= d_i) & (d_i_plus_1 < d) # & (torch.abs(angle) < config.fov / 2)

            # # [v]
            # interp_pulse = (1 - d_a[valid]) * signals[sink, d_i[valid]] + d_a[valid] * signals[sink, d_i_plus_1[valid]]

            # [n, s]
            # weights[valid, sink] += 1
            weights[valid, sink] += torch.cos(angle[valid]) / ((d_source ** 2) * (d_sink ** 2))
            # updates[:, sink] *= (d_source ** 2) # * (d_sink ** 2)
            # updates[:, sink] *= (1 / torch.cos(angle))

    # [s, n]
    weights = torch.transpose(weights, dim0=0, dim1=1)

    weights = weights.cpu().numpy()

    return weights
