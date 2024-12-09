import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict, List
from spatialmath import SE3, SO3
import open3d as o3d

from .geometry import wrap_angle, amplitude_to_db, power_to_db


class ContinuousAngularDistribution(ABC):

    @abstractmethod
    def pdf(self, az: np.array, el: np.array) -> np.array:
        pass

    @abstractmethod
    def pdf_max(self) -> float:
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

    def attenuation_db(self, az: np.array, el: np.array) -> np.array:
        return power_to_db(self.pdf(az, el) / self.pdf_max())


class UniformAngularDistribution(ContinuousAngularDistribution):

    def pdf(self, az: np.array, el: np.array) -> np.array:
        return np.sin(el) / (4 * np.pi)

    def pdf_max(self) -> float:
        return 1 / (4 * np.pi)

    def cdf_az(self, az: np.array) -> np.array:
        return (az + np.pi) / (2 * np.pi)

    def cdf_el_given_az(self, el: np.array, az: np.array) -> np.array:
        return (1 - np.cos(el)) / 2

    def cdf_inv_az(self, p: np.array) -> np.array:
        return (2 * np.pi * p) - np.pi

    def cdf_inv_el_given_az(self, p: np.array, az: np.array) -> np.array:
        return np.arccos(1 - 2 * p)


class BidirectionalReflectanceDistribution(ABC):

    @abstractmethod
    def pdf(self, az: np.array, el: np.array, incident_az_el: np.array) -> np.array:
        pass

    @abstractmethod
    def pdf_max(self, incident_az_el: np.array) -> np.array:
        pass

    @abstractmethod
    def cdf_az(self, az: np.array, incident_az_el: np.array) -> np.array:
        pass

    @abstractmethod
    def cdf_el_given_az(self, el: np.array, az: np.array, incident_az_el: np.array) -> np.array:
        pass

    @abstractmethod
    def cdf_inv_az(self, p: np.array, incident_az_el: np.array) -> np.array:
        pass

    @abstractmethod
    def cdf_inv_el_given_az(self, p: np.array, az: np.array, incident_az_el: np.array) -> np.array:
        pass

    def sample(self, n_samples: int, incident_az_el: np.array):
        p_az = np.random.uniform(low=0, high=1, size=n_samples)
        p_el = np.random.uniform(low=0, high=1, size=n_samples)

        az = self.cdf_inv_az(p_az, incident_az_el)
        el = self.cdf_inv_el_given_az(p_el, az, incident_az_el)

        return np.stack((az, el), axis=-1)

    def attenuation_db(self, az: np.array, el: np.array, incident_az_el: np.array) -> np.array:
        return power_to_db(self.pdf(az, el, incident_az_el) / self.pdf_max(incident_az_el))


class DiffuseBidirectionalReflectanceDistribution(BidirectionalReflectanceDistribution):

    def pdf(self, az: np.array, el: np.array, incident_az_el: np.array) -> np.array:
        return np.full_like(az, fill_value=1 / (2 * np.pi))

    def pdf_max(self, incident_az_el: np.array) -> np.array:
        return np.full(incident_az_el.shape[0], fill_value=1.0 / (2.0 * np.pi))

    def cdf_az(self, az: np.array, incident_az_el: np.array) -> np.array:
        return np.clip((az + np.pi) / (2 * np.pi), 0, 1)

    def cdf_el_given_az(self, el: np.array, az: np.array, incident_az_el: np.array) -> np.array:
        return np.clip(el / (np.pi / 2), 0, 1)

    def cdf_inv_az(self, p: np.array, incident_az_el: np.array) -> np.array:
        return (2 * np.pi) * p - np.pi

    def cdf_inv_el_given_az(self, p: np.array, az: np.array, incident_az_el: np.array) -> np.array:
        return (np.pi / 2) * p


class SpecularBidirectionalReflectanceDistribution(BidirectionalReflectanceDistribution):

    def pdf(self, az: np.array, el: np.array, incident_az_el: np.array) -> np.array:
        raise NotImplemented

    def pdf_max(self, incident_az_el: np.array) -> np.array:
        return np.full(incident_az_el.shape[0], fill_value=np.inf)

    def cdf_az(self, az: np.array, incident_az_el: np.array) -> np.array:
        raise NotImplemented

    def cdf_el_given_az(self, el: np.array, az: np.array, incident_az_el: np.array) -> np.array:
        raise NotImplemented

    def cdf_inv_az(self, p: np.array, incident_az_el: np.array) -> np.array:
        return wrap_angle(incident_az_el[:, 0] + np.pi)

    def cdf_inv_el_given_az(self, p: np.array, az: np.array, incident_az_el: np.array) -> np.array:
        return incident_az_el[:, 1]



class Source:
    id: str
    pose: SE3

    distribution: ContinuousAngularDistribution

    def __init__(self, id, pose, distribution):
        self.id = id
        self.pose = pose
        self.distribution = distribution

    def get_tf_from_world(self, world_t_vehicle: SE3):
        self.pose = world_t_vehicle * self.pose


class Sink(ABC):
    id: str
    pose: SE3

    distribution: ContinuousAngularDistribution

    def __init__(self, id, pose, distribution):
        self.id = id
        self.pose = pose
        self.distribution = distribution

    def get_tf_from_world(self, world_t_vehicle: SE3):
        self.pose = world_t_vehicle * self.pose

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
    sources: List[Source]
    sinks: List[Sink]
    surfaces: Dict[str, Surface]

    def __init__(self, sources: [Source], sinks: [Sink], surfaces: [Surface]):
        self.sources = sources
        self.sinks = sinks
        self.surfaces = {surface.id: surface for surface in surfaces}  # todo: change to list for consistency


    def visualization_geometry(self):
        source_geometries = [
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(source.pose) for source in self.sources
        ]

        sink_geometries = [
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(sink.pose) for sink in self.sinks
        ]

        surface_geometries = [
            surface.mesh for surface in self.surfaces.values()
        ]

        for surface_geometry in surface_geometries:
            surface_geometry.paint_uniform_color((1.0, 0.0, 0.0))

        geometries = source_geometries + sink_geometries + surface_geometries

        for geometry in geometries:
            geometry.compute_vertex_normals()

        return geometries

    def visualize(self):
        o3d.visualization.draw_geometries(self.visualization_geometry())

    @property
    def sink_poses(self):
        return SE3([sink.pose for sink in self.sinks])

    @property
    def source_poses(self):
        return SE3([source.pose for source in self.sources])
