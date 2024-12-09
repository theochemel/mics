from abc import ABC, abstractmethod
from typing import List

import numpy as np
from numpy import pi
from spatialmath import SE3

from tracer.geometry import amplitude_to_db, az_el_to_direction_grid, db_to_amplitude, direction_to_az_el
from tracer.scene import Sink, ContinuousAngularDistribution, UniformContinuousAngularDistribution


class Array(ABC):

    @property
    @abstractmethod
    def sinks(self) -> List[Sink]:
        pass

    @property
    @abstractmethod
    def positions(self) -> np.array:
        pass


class RectangularArray(Array):

    def __init__(self, nx: int, ny: int, spacing: float, ang_dist: ContinuousAngularDistribution):
        size_x = spacing * (nx - 1)
        size_y = spacing * (ny - 1)

        t_x = np.arange(nx) * spacing - size_x / 2
        t_y = np.arange(ny) * spacing - size_y / 2

        t_x, t_y = np.meshgrid(t_x, t_y, indexing="xy")
        t_x = t_x.flatten()
        t_y = t_y.flatten()

        self._positions = np.stack((
            t_x, np.zeros_like(t_y), t_y,
        ), axis=-1)

        self._ang_dist = ang_dist

    @property
    def sinks(self) -> List[Sink]:
        """
        :return: List of all sinks in the array
        """
        sinks: List[Sink] = []

        for i in range(self._positions.shape[0]):
            sinks.append(Sink(
                f'sink_{i}',
                SE3.Trans(self._positions[i]),
                self._ang_dist
            ))

        return sinks

    @property
    def positions(self) -> np.array:
        """
        :return: [n_elem, 3] positions of elements in array coordinate frame
        """
        return self._positions


class DASBeamformer:

    def __init__(self, array: Array, C: float):
        self._array = array
        self._C = C

    def get_gain(self, steering_dir: np.array, looking_dir: np.array, k: float) -> np.array:
        """
        Get array directivity in given direction(s) for given steering angle(s)
        :param steering_dir: [n_steering, 3] (steering unit vector in the array coordinate frame)
        :param looking_dir: [n_looking, 3] (looking angle unit vector in the array coordinate frame)
        :param k (angular wavenumber rad/m)
        :return: [n_steering, n_looking] (gain relative to isotropic source in dB)
        """

        # [n_steering, n_elems]
        steering_delays = self.steer(steering_dir, k)

        # [n_looking, n_elems]
        looking_delays = self.steer(looking_dir, k)

        # [n_steering, n_looking, n_elem]
        steering_delays_grid = np.repeat(steering_delays[:, np.newaxis, :], looking_delays.shape[0], axis=1)

        # [n_steering, n_looking, n_elem]
        looking_delays_grid = np.repeat(looking_delays[np.newaxis, :, :], steering_delays.shape[0], axis=0)

        # calculate phases for looking angle
        delta_delays = steering_delays_grid - looking_delays_grid

        # [n_steering, n_looking]
        A = np.sqrt(
            np.sum(np.sin(delta_delays), axis=-1) ** 2 + np.sum(np.cos(delta_delays), axis=-1) ** 2
        ) / delta_delays.shape[-1]

        return amplitude_to_db(A)

    def steer(self, steering_dir: np.array, k: float) -> np.array:
        """
        Compute phase delays for given steering angle(s)
        :param steering_dir: (n_steering, 3) steering unit vector in the array coordinate frame
        :param k: angular wavenumber
        :return: (n_steering, n_elems) phase delays for each element relative to the centerpoint of the array
        """
        positions = self._array.positions

        steering_dir_grid = np.repeat(steering_dir[:, np.newaxis, :], positions.shape[0], axis=1)
        position_grid = np.repeat(positions[np.newaxis, :, :], steering_dir.shape[0], axis=0)

        delays = -np.sum(steering_dir_grid * position_grid, axis=-1) * k

        return delays

    def beamform_receive(self, steering_dir: np.array, rx_pattern: np.array, T: float, k: float) -> np.array:
        """
        Delay-and-sum individual element signals steering the array in given direction(s)
        :param steering_dir: (n_steering, 3) steering unit vector in the array coordinate frame
        :param rx_pattern: (n_elem, [t/T_rx]) wave for each sink
        :param T: sample period
        :param k: angular wavenumber
        :return (n_steering, [t'/T_rx]) beamformed signals for each steering direction
        """

        n_elems, n_samples = rx_pattern.shape
        n_steering = steering_dir.shape[0]

        delays = self.steer(steering_dir, k)
        time_delays = delays / (k * self._C)

        sample_delays = np.round(time_delays / T).astype(int)

        back_padding = np.max(sample_delays) + 1

        result = np.zeros((n_steering, n_samples + back_padding))

        for steering_i in range(n_steering):
            for elem_i in range(n_elems):
                d = sample_delays[steering_i, elem_i]

                input_start = 0
                input_end = n_samples
                output_start = d
                output_end = d + n_samples

                if output_start < 0:
                    input_start += -output_start
                    output_start = 0

                if output_end >= result.shape[1]:
                    input_end -= (output_end - result.shape[1] + 1)
                    output_end = result.shape[1] - 1

                result[steering_i, output_start:output_end] += rx_pattern[elem_i, input_start:input_end]

        return result

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    f = 100e3
    C = 1500
    l = C / f

    array = RectangularArray(10, 5, l / 2, UniformContinuousAngularDistribution(
        min_az=-pi, max_az=pi, min_el=0, max_el=pi
    ))

    beamformer = DASBeamformer(array, C)

    steering_az = np.linspace(-pi, pi, 16, endpoint=False)
    steering_el = np.linspace(0, pi / 2, 4, endpoint=False)  # elevation from x-y plane toward +z
    steering_dir = az_el_to_direction_grid(steering_az, steering_el)
    steering_dir = steering_dir.reshape(-1, 3)

    looking_res_deg = 1
    looking_az = np.linspace(-pi, pi, 360 // looking_res_deg)
    looking_el = np.linspace(0, pi / 2, 90 // looking_res_deg)  # elevation from x-y plane toward +z
    looking_dir = az_el_to_direction_grid(looking_az, looking_el)

    k = 2 * pi * f / C

    gain_db = beamformer.get_gain(steering_dir, looking_dir.reshape(-1, 3), k)
    gain_db = gain_db.reshape((steering_dir.shape[0], looking_dir.shape[0], looking_dir.shape[1]))

    for steering_i in range(steering_dir.shape[0]):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        c = ax.pcolormesh(looking_az, looking_el, gain_db[steering_i].T, shading='auto', vmin=-80, vmax=0, cmap='viridis')
        plt.colorbar(c)
        steering_az_el = direction_to_az_el(steering_dir)
        ax.scatter(steering_az_el[steering_i, 0], steering_az_el[steering_i, 1], c="r")

        plt.show()
