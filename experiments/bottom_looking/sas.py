import numpy as np
import torch

from config import Config


def get_sas_updates(points: np.array, sinks: np.array, signal_t: np.array, signals: np.array, config: Config) -> np.array:
    # points is [n, 3]
    # sinks is [s, 3]
    # signal_t is [d]
    # signals is [s, d]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        points = torch.tensor(points).to(device)
        sinks = torch.tensor(sinks).to(device)
        signal_t = torch.tensor(signal_t).to(device)
        signals = torch.tensor(signals, dtype=torch.complex128).to(device)

        n = points.shape[0]
        s = sinks.shape[0]
        d = len(signal_t)

        # [n, s, 3]
        delta = points.unsqueeze(1) - sinks.unsqueeze(0)

        # [n, s]
        r = torch.sqrt(delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2)
        angle = torch.arctan2(r, -delta[:, :, 2])

        # [n, s]
        range = torch.linalg.norm(delta, axis=-1)
        rtt = (2 * range) / config.c

        # [n, s]
        d_interp = (rtt - signal_t[0]) / config.Ts
        d_i = torch.floor(d_interp).to(torch.int64)
        d_i_plus_1 = d_i + 1
        d_a = d_interp - d_i

        # [n, s]
        s_i = torch.arange(s).unsqueeze(0).repeat((n, 1))

        # [n, s]
        valid = (0 <= d_i) & (d_i_plus_1 < d)#  & (torch.abs(angle) < config.fov / 2)

        # [v]
        interp_pulse = (1 - d_a[valid]) * signals[s_i[valid], d_i[valid]] + d_a[valid] * signals[s_i[valid], d_i_plus_1[valid]]

        # [n, s]
        updates = torch.zeros((n, s), dtype=torch.complex128)

        # [n, s]
        updates[valid] = interp_pulse * torch.exp(2.0j * np.pi * config.spatial_f * rtt[valid])

        # [s, n]
        updates = torch.transpose(updates, dim0=0, dim1=1)

    updates = updates.cpu().numpy()

    return updates

