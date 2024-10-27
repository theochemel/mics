import numpy as np

def transform_rays(a_t_bs: np.array, rays_b: np.array) -> np.array:
    ray_positions_b = rays_b[:, :3]
    ray_directions_b = rays_b[:, 3:]

    ray_positions_b_h = np.concatenate((
        ray_positions_b,
        np.ones((rays_b.shape[0], 1)),
    ), axis=-1)

    ray_directions_b_h = np.concatenate((
        ray_directions_b,
        np.zeros((rays_b.shape[0], 1)),
    ), axis=-1)

    ray_positions_a_h = (a_t_bs @ ray_positions_b_h[:, :, np.newaxis]).squeeze(-1)
    ray_directions_a_h = (a_t_bs @ ray_directions_b_h[:, :, np.newaxis]).squeeze(-1)

    rays_a = np.concatenate((
        ray_positions_a_h[:, :3],
        ray_directions_a_h[:, :3],
    ), axis=-1)

    return rays_a


def az_el_to_direction(az_el: np.array) -> np.array:
    az = az_el[:, 0]
    el = az_el[:, 1]

    return np.stack((
        np.cos(az) * np.sin(el),
        np.sin(az) * np.sin(el),
        np.cos(el),
    ), axis=-1)
