from abc import ABC, abstractmethod, abstractproperty
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

    @property
    @abstractmethod
    def carrier(self) -> float:
        ...

    @property
    @abstractmethod
    def baseband(self) -> np.array:
        ...


class PMBarker(BarkerCode):

    def __init__(self, code: np.array, f, T_sample, T_bit):
        result_samples = int(T_bit * len(code) / T_sample)
        bit_samples = int(T_bit / T_sample)
        baseband = np.zeros((result_samples,))
        digital = np.zeros((result_samples,))
        tt = np.arange(0, bit_samples) * T_sample
        omega = 2 * pi * f
        bit_high = np.cos(tt * omega)
        bit_low = np.sin(tt * omega)
        for i, bit in enumerate(code):
            shift = i * bit_samples
            baseband[shift:shift + bit_samples] = bit_high if bit == 1 else bit_low
            digital[shift:shift + bit_samples] = 1.0 if bit == 1 else -1.0
        self._baseband = baseband
        self._digital = digital
        self._carrier = f


    def correlate(self, signal: np.array) -> np.array:
        return correlate(signal, self._baseband, mode='valid')

    @property
    def carrier(self) -> float:
        return self._carrier

    @property
    def baseband(self) -> np.array:
        return self._baseband


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

        for i, bit in code:
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



class Chirp:

    def __init__(self, f_hi: float, f_lo: float, T_sample, T_chirp):
        self._f_hi = f_hi
        self._f_lo = f_lo
        self._T_sample = T_sample
        self._T_chirp = T_chirp

        t = T_sample * np.arange(T_chirp / T_sample)
        self._baseband_t = t
        self._baseband = np.sin(2 * np.pi * (f_hi - f_lo) / (2 * T_chirp) * t ** 2 + f_lo * t)


    @property
    def baseband(self) -> np.array:
        return self._baseband


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.signal import correlate

    chirp = Chirp(100e3, 50e3, 1e-6, 1e-3)

    plt.plot(chirp._baseband_t, chirp._baseband)
    plt.show()

    rx_t = chirp._T_sample * np.arange(10000)
    rx_signal = np.zeros_like(rx_t)

    rx_signal[2000:2000 + len(chirp._baseband)] += chirp._baseband

    correlation = correlate(rx_signal, chirp.baseband, mode="same")

    plt.plot(rx_t, correlation / np.max(correlation))
    plt.plot(rx_t, np.sin(2 * pi * chirp._f_lo * rx_t))
    plt.show()
