from abc import ABC, abstractmethod, abstractproperty
from functools import cache
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from spatialmath import SE3, SO3

from tracer.scene import Source, Sink, ContinuousAngularDistribution, UniformContinuousAngularDistribution
from dataclasses import dataclass


class RectangularArray:

    def __init__(self, nx: int, ny: int, spacing: float, ang_dist: ContinuousAngularDistribution):
        size_x = spacing * (nx - 1)
        size_y = spacing * (ny - 1)
        t_x = np.arange(nx) * spacing - size_x / 2
        t_y = np.arange(ny) * spacing - size_y / 2

        self._elem_t = np.empty((nx, ny, 3))
        self._elem_t[:, :, :2] = np.array(np.meshgrid(t_x, t_y)).transpose((1, 2, 0))
        self._elem_t[:, :, 2] = 0

        self._ang_dist = ang_dist
        self._nx = nx
        self._ny = ny
        self._spacing = spacing

    @property
    @cache
    def sinks(self) -> List[Sink]:
        """
        :return: List of all sinks in the array
        """
        sinks: List[Sink] = []
        for ix in range(self._nx):
            for iy in range(self._ny):
                sinks.append(Sink(
                    f'sink_{ix}_{iy}',
                    SE3.Rt(SO3(), [self._elem_t[ix][iy][0], self._elem_t[ix][iy][1], 0]),
                    self._ang_dist
                ))

        return sinks

    def get_gain(self, steering_dir: np.array, looking_dir: np.array, k: np.array) -> np.array:
        """
        Get array directivity in given direction(s) for given steering angle(s)
        :param steering_dir: N x 3 (steering unit vector in the array coordinate frame)
        :param looking_dir: M x 3 (looking angle unit vector in the array coordinate frame)
        :param k: K (angular wavenumber rad/m)
        :return: N x M x K (gain relative to isotropic source in dB)
        """

        # TODO: use angular distribution

        n_steering = len(steering_dir)
        n_theta = len(looking_dir)
        n_k = len(k)

        # calculate phase delays for steering
        steering_phases = self.steer(k, steering_dir)
        steering_phases = steering_phases.reshape(*steering_phases.shape[:-2], -1) # n_k x n_steering x n_elems

        # calculate phase delays for looking angle
        theta_phase_delays = self.steer(k, looking_dir)
        theta_phase_delays = theta_phase_delays.reshape(*theta_phase_delays.shape[:-2], -1) # n_k x n_theta x n_elems

        # calculate phases for looking angle
        theta_phases = theta_phase_delays[:, :, np.newaxis, :] - steering_phases[:, np.newaxis, :, :]

        A = np.sqrt(
            np.sum(np.sin(theta_phases), axis=-1)**2 + np.sum(np.cos(theta_phases), axis=-1)**2
        )

        return 20*np.log10(A)

    def steer(self, k: np.array, steering_dir: np.array) -> np.array:
        """
        Compute phase delays for given steering angle(s)
        :param k: (n_k, ) angular wavenumbers
        :param steering_dir: (n_steering, 3) steering unit vector in the array coordinate frame
        :return: (n_k, n_steering, nx, ny) phase delays for each element relative to the centerpoint of the array
        """

        offsets = np.einsum('ni,xyi->nxy', steering_dir, self._elem_t)
        steering_phases = np.einsum('k,nxy->knxy', k, offsets)

        return steering_phases

    def beamform_receive(self, k: np.array, steering: np.array, rx_pattern: np.array, T: float) -> np.array:
        """
        Delay-and-sum individual element signals steering the array in given direction(s)
        :param k: (n_k, ) angular wavenumbers
        :param steering: (n_steering, 3) steering unit vector in the array coordinate frame
        :param rx_pattern: (nx, ny, [t/T_rx]) wave for each sink
        :param T: sample period
        :return (n_k, n_steering, [t'/T_rx])
        """

        shifts = self.steer(k, steering)
        shifts_samples = np.round(shifts / T).astype(np.int64)

        front_padding = np.min(shifts_samples)
        back_padding = np.max(shifts_samples)

        t = rx_pattern.shape[2]
        n_k = len(k)
        n_steering = len(steering)

        result = np.zeros((n_k, n_steering, front_padding + t + back_padding))
        for i_k, k_val in enumerate(k): # TODO vectorize
            for i_steering, steering_val in enumerate(steering):
                for x in range(self._nx):
                    for y in range(self._ny):
                        d = front_padding - shifts_samples[i_k, i_steering, x, y]
                        result[n_k, n_steering, ] += rx_pattern[x, y, d:d+t]

        return result

if __name__ == '__main__':
    arr = RectangularArray(8, 8, 0.0075, UniformContinuousAngularDistribution)
    f = 100_000
    c = 1500
    w = 2*pi*f
    wavelength = c / f
    k = 2*pi / wavelength

    azimuth = np.linspace(0, 2 * np.pi, 360)
    elevation = np.linspace(0, np.pi / 2, 200)
    azimuth, elevation = np.meshgrid(azimuth, elevation)
    azimuth = azimuth.reshape(-1)
    elevation = elevation.reshape(-1)

    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)

    theta = np.transpose([x, y, z])

    steering_az = pi / 4
    steering_el = pi / 4

    x = np.cos(steering_el) * np.cos(steering_az)
    y = np.cos(steering_el) * np.sin(steering_az)
    z = np.sin(steering_el)
    steering = np.array([[x, y, z]])

    dir = arr.get_gain(steering, theta, np.array([k]))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    print(np.max(dir))
    dir -= np.min(dir)
    theta *= dir[0]

    theta = theta.reshape((200, 360, 3))

    surface = ax.plot_surface(theta[:,:, 0], theta[:,:, 1], theta[:,:, 2], cmap='viridis', edgecolor='k', alpha=0.8)
    fig.colorbar(surface, shrink=0.5, aspect=10, label='Magnitude')
    plt.show()
