import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict
from spatialmath import SE3
import open3d as o3d


class Source(ABC):
    id: str
    pose: SE3

    @abstractmethod
    def sensitivity(self, yaw: float, pitch: float) -> float:
        pass


class OmnidirectionalSource(Source):

    def __init__(self, id: str, pose: SE3, sensitivity: float):
        self.id = id
        self.pose = pose
        self._sensitivity = sensitivity

    def sensitivity(self, yaw: float, pitch: float) -> float:
        return self._sensitivity


class Sink(ABC):
    id: str
    pose: SE3

    @abstractmethod
    def sensitivity(self, yaw: float, pitch: float) -> float:
        pass


class OmnidirectionalSink(Source):

    def __init__(self, id: str, pose: SE3, sensitivity: float):
        self.id = id
        self.pose = pose
        self._sensitivity = sensitivity

    def sensitivity(self, yaw: float, pitch: float) -> float:
        return self._sensitivity


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