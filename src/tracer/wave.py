import numpy as np


class Wave:

    def __init__(self, waveform: np.array):
        self._waveform: np.array = waveform
        self._time_shift: float = 0

    def frequency_shift(self, c: float):
        xp = np.linspace(0, 1, self._waveform.shape[0])
        fp = np.linspace(0, c, self._waveform.shape[0])
        return np.interp(self._waveform, fp, xp)

    def time_shift(self, c):
        self._time_shift += c
