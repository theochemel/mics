import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


class OccupancyGridMap:

    def __init__(self, x: int, y: int, z: int):
        self._grid = np.zeros((x,y,z), dtype=np.float64)

