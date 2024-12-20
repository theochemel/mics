from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple
from spatialmath import SE3


class Trajectory(ABC):

    @property
    @abstractmethod
    def keyposes(self) -> List[SE3]:
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        pass

    @property
    @abstractmethod
    def time(self) -> np.array:
        pass

    @property
    @abstractmethod
    def poses(self) -> List[SE3]:
        pass

    @property
    @abstractmethod
    def position_world(self) -> np.array:
        pass

    @property
    @abstractmethod
    def velocity_world(self) -> np.array:
        pass

    @property
    @abstractmethod
    def acceleration_world(self) -> np.array:
        pass

    @property
    @abstractmethod
    def orientation_rpy_world(self) -> np.array:
        pass

    @property
    @abstractmethod
    def angular_velocity_world(self) -> np.array:
        pass

    @abstractmethod
    def __getitem__(self, value) -> Tuple[float, SE3]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self):
        pass
