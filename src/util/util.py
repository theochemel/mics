import numpy as np


def wrap2pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi
