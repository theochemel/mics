import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict
from spatialmath import SE3
import open3d as o3d



class ContinuousAngularDistribution(ABC):

    @abstractmethod
    def pdf(self, az: np.array, el: np.array) -> np.array:
        pass

    @abstractmethod
    def cdf_az(self, az: np.array) -> np.array:
        pass

    @abstractmethod
    def cdf_el_given_az(self, el: np.array, az: np.array) -> np.array:
        pass

    @abstractmethod
    def cdf_inv_az(self, p: np.array) -> np.array:
        pass

    @abstractmethod
    def cdf_inv_el_given_az(self, p: np.array, az: np.array) -> np.array:
        pass

    def sample(self, n_samples: int):
        p_az = np.random.uniform(low=0, high=1, size=n_samples)
        p_el = np.random.uniform(low=0, high=1, size=n_samples)

        az = self.cdf_inv_az(p_az)
        el = self.cdf_inv_el_given_az(p_el, az)

        return np.stack((az, el), axis=-1)


class UniformContinuousAngularDistribution(ContinuousAngularDistribution):

    def __init__(self, min_az: float, max_az: float, min_el: float, max_el: float):
        self._min_az = min_az
        self._max_az = max_az
        self._min_el = min_el
        self._max_el = max_el

        self._area = (max_az - min_az) * (max_el - min_el)

    def pdf(self, az: np.array, el: np.array) -> np.array:
        return np.full_like(az, fill_value=1 / self._area)

    def cdf_az(self, az: np.array) -> np.array:
        return np.clip((az - self._min_az) / (self._max_az - self._min_az), 0, 1)

    def cdf_el_given_az(self, el: np.array, az: np.array) -> np.array:
        return np.clip((el - self._min_el) / (self._max_el - self._min_el), 0, 1)

    def cdf_inv_az(self, p: np.array) -> np.array:
        return (self._max_az - self._min_az) * p + self._min_az

    def cdf_inv_el_given_az(self, p: np.array, az: np.array) -> np.array:
        return (self._max_el - self._min_el) * p + self._min_el


class Source:
    id: str
    pose: SE3

    distribution: ContinuousAngularDistribution

    def __init__(self, id, pose, distribution):
        self.id = id
        self.pose = pose
        self.distribution = distribution


class Sink(ABC):
    id: str
    pose: SE3

    distribution: ContinuousAngularDistribution

    def __init__(self, id, pose, distribution):
        self.id = id
        self.pose = pose
        self.distribution = distribution


class Material(ABC):

    @abstractmethod
    def absorption(self, frequency: float) -> float:
        pass

    @abstractmethod
    def scattering(self, angle: float) -> float:
        pass


class SimpleMaterial(Material):

    def __init__(self, absorption: float):
        self._absorption = absorption

    def absorption(self, frequency: float) -> float:
        return self._absorption

    def scattering(self, angle: float) -> float:
        return 1


class Surface:
    id: str
    pose: SE3

    material: Material
    mesh: o3d.geometry.TriangleMesh

    def __init__(self, id, pose, material, mesh):
        self.id = id
        self.pose = pose
        self.material = material
        self.mesh = mesh.transform(np.array(pose))


class Scene:
    sources: Dict[str, Source]
    sinks: Dict[str, Sink]
    surfaces: Dict[str, Surface]

    def __init__(self, sources: [Source], sinks: [Sink], surfaces: [Surface]):
        self.sources = {source.id: source for source in sources}
        self.sinks = {sink.id: sink for sink in sinks}
        self.surfaces = {surface.id: surface for surface in surfaces}

    def visualization_geometry(self):
        source_geometries = [
            o3d.geometry.TriangleMesh.create_sphere(radius=0.001).translate(source.pose.t) for source in self.sources.values()
        ]

        for g in source_geometries:
            g.paint_uniform_color([1, 0, 0])

        sink_geometries = [
            o3d.geometry.TriangleMesh.create_sphere(radius=0.001).translate(sink.pose.t) for sink in self.sinks.values()
        ]

        for g in sink_geometries:
            g.paint_uniform_color([0, 0, 1])

        surface_geometries = [
            surface.mesh for surface in self.surfaces.values()
        ]

        geometries = source_geometries + sink_geometries + surface_geometries

        for geometry in geometries:
            geometry.compute_vertex_normals()

        return geometries

    def visualize(self):
        o3d.visualization.draw_geometries(self.visualization_geometry())

    def save(self):
        pass

    def load(self):
        pass