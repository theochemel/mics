import numpy as np


def amplitude_to_db(x: np.array) -> np.array:
    return 20.0 * np.log10(x)


def db_to_amplitude(x: np.array) -> np.array:
    return 10.0 ** (x / 20.0)


def power_to_db(x: np.array) -> np.array:
    return 10.0 * np.log10(x)


def db_to_power(x: np.array) -> np.array:
    return 10.0 ** (x / 10.0)


def wrap_angle(x: np.array) -> np.array:
    return (x + np.pi) % (2 * np.pi) - np.pi


def transform_points(a_t_bs: np.array, points_b: np.array) -> np.array:
    points_b_h = np.concatenate((
        points_b,
        np.ones((points_b.shape[0], 1)),
    ), axis=-1)

    points_a_h = (a_t_bs @ points_b_h[:, :, np.newaxis]).squeeze(-1)

    points_a = points_a_h[:, :3]

    return points_a


def transform_vectors(a_t_bs: np.array, vectors_b: np.array) -> np.array:
    vectors_b_h = np.concatenate((
        vectors_b,
        np.zeros((vectors_b.shape[0], 1))
    ), axis=-1)

    vectors_a_h = (a_t_bs @ vectors_b_h[:, :, np.newaxis]).squeeze(-1)

    vectors_a = vectors_a_h[:, :3]

    return vectors_a


def transform_rays(a_t_bs: np.array, rays_b: np.array) -> np.array:
    rays_a = np.concatenate((
        transform_points(a_t_bs, rays_b[:, :3]),
        transform_vectors(a_t_bs, rays_b[:,3:])
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


def az_el_to_direction_grid(az, el):
    grid = np.transpose(np.meshgrid(az, el)).reshape(-1, 2)
    x = np.cos(grid[:, 0]) * np.sin(grid[:, 1])
    y = np.sin(grid[:, 0]) * np.sin(grid[:, 1])
    z = np.cos(grid[:, 1])
    return np.transpose((x, y, z)).reshape((len(az), len(el), 3))


def direction_to_az_el(direction: np.array) -> np.array:
    x = direction[:, 0]
    y = direction[:, 1]
    z = direction[:, 2]

    return np.stack((
        np.arctan2(y, x),
        (np.pi / 2) - np.arctan2(z, np.sqrt(x**2 + y**2))
    ), axis=-1)

