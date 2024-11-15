from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from numpy import pi
from scipy.signal import correlate



class BarkerCode(ABC):

    class Sequence:
        BARKER_2 = np.array([+1, -1])
        BARKER_3 = np.array([+1, +1, -1])
        # ...
        BARKER_7 = np.array([+1, +1, +1, -1, -1, +1, -1])
        # ...

    @abstractmethod
    def correlate(self, signal: np.array) -> np.array:
        ...


class FMBarker:

    def __init__(self, code: np.array, f_low, f_high, T_sample, T_bit, continuous_phase = False):
        # TODO: make cont phase work
        result_samples = int(T_bit * len(code) / T_sample)
        bit_samples = int(T_bit / T_sample)
        baseband = np.zeros((result_samples,))
        baseband_low = np.zeros_like(baseband)
        baseband_high = np.zeros_like(baseband)

        tt = np.arange(0, bit_samples) * T_sample
        omega_high, omega_low = 2 * pi * f_high, 2 * pi * f_low
        bit_high = np.cos(tt * omega_high)
        bit_low = np.cos(tt * omega_low)



        for i, bit in enumerate(code):
            shift = i * bit_samples
            baseband[shift:shift + bit_samples] = bit_high if bit == 1 else bit_low
            if bit == 1:
                baseband_high[shift:shift + bit_samples] = bit_high
            else:
                baseband_low[shift:shift + bit_samples] = bit_low

        self._code = code
        self._f_low = f_low
        self._f_high = f_high
        self._T_sample = T_sample
        self._T_bit = T_bit
        self._baseband = baseband
        self._baseband_low = baseband_low
        self._baseband_high = baseband_high

    def correlate(self, signal: np.array) -> np.array:
        """

        :param signal: (2, t)
        :return: (t', )
        """

        corr_low = correlate(signal[0], self._baseband_low, mode='valid')
        corr_high = correlate(signal[1], self._baseband_high, mode='valid')

        return corr_low + corr_high

    @property
    def baseband(self) -> np.array:
        return self._baseband
