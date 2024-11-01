import numpy as np
from math import pi

from tracer.scene import *
from tracer.geometry import *

MIN_BOUNCE = 1e-3
RAY_START_SHIFT = 1e-3


class Path:
    def __init__(self):
        self._positions = []

    def add_bounce(self, position):
        self._positions.append(position)


class Tracer:

    def __init__(self, scene: Scene):
        self._scene = scene

        self._n_sources = len(scene.sources)

        # self._scattering_distribution = UniformContinuousAngularDistribution(
        #     min_az=-np.deg2rad(10),
        #     max_az=np.deg2rad(10),
        #     min_el=pi / 2 - np.deg2rad(10),
        #     max_el=pi / 2 + np.deg2rad(10),
        # )

        self._scattering_distribution = SpecularBidirectionalReflectanceDistribution()


    def trace(self, n_rays: int, n_bounces: int):
        self._build_raycast_scene()

        self._generate_source_rays(n_rays)

        for i in range(n_bounces):
            self._propagate_rays()

            pass

        # Now we have all the information we need in self._rays, self._ray_path_ids, and self._path_source_ids
        # We can reconstruct a path from each of those

        self._reconstruct_paths()

    def _build_raycast_scene(self):
        self._raycast_scene = o3d.t.geometry.RaycastingScene()

        for surface in self._scene.surfaces.values():
            self._raycast_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(surface.mesh))

    def _generate_source_rays(self, n_rays: int):
        ray_sets = []
        path_source_id_sets = []

        for i, source in enumerate(self._scene.sources.values()):
            # Pose for every ray is the same
            world_t_sources = np.array([
                np.array(source.pose) for _ in range(n_rays)
            ])

            rays = self._generate_rays(world_t_sources, source.distribution)

            ray_sets.append(rays)

            path_source_id_sets.append(np.full(rays.shape[0], i))

        rays = np.concatenate(ray_sets, axis=0)
        self._path_source_ids = np.concatenate(path_source_id_sets, axis=0)

        self._active_rays = rays
        self._rays = rays

        ray_path_ids = np.arange(rays.shape[0])

        self._n_paths = len(ray_path_ids)
        self._active_ray_path_ids = ray_path_ids
        self._ray_path_ids = ray_path_ids

    def _generate_rays(self, world_t_locals: np.array, distribution: ContinuousAngularDistribution):
        az_els = distribution.sample(world_t_locals.shape[0])

        directions_local = az_el_to_direction(az_els)
        positions_local = np.zeros_like(directions_local)

        rays_local = np.concatenate((positions_local, directions_local), axis=-1)

        rays = transform_rays(world_t_locals, rays_local)

        return rays


    def _generate_scatter_rays(self, world_t_locals: np.array, incident_az_els: np.array, distribution: BidirectionalReflectanceDistribution):
        az_els = distribution.sample(world_t_locals.shape[0], incident_az_els)

        directions_local = az_el_to_direction(az_els)
        positions_local = np.zeros_like(directions_local)

        rays_local = np.concatenate((positions_local, directions_local), axis=-1)

        rays = transform_rays(world_t_locals, rays_local)

        return rays


    def _propagate_rays(self):
        rays = self._active_rays

        raycasts = self._raycast_scene.cast_rays(
            o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        )

        t_hit = raycasts["t_hit"].numpy()
        primitive_normals = raycasts["primitive_normals"].numpy()
        geometry_ids = raycasts["geometry_ids"].numpy()

        hit = np.isfinite(t_hit) & (t_hit > MIN_BOUNCE)

        valid_rays = rays[hit]

        new_ray_path_ids = self._active_ray_path_ids[hit]

        hit_positions = valid_rays[:, :3] + t_hit[hit][:, np.newaxis] * valid_rays[:, 3:]

        normals = primitive_normals[hit]

        incident_directions_world = valid_rays[:, 3:]

        local_x = incident_directions_world - (np.sum(incident_directions_world * normals, axis=-1)[:, np.newaxis] * normals)
        local_z = normals
        local_y = np.cross(local_z, local_x, axis=-1)

        world_t_locals = np.zeros((hit_positions.shape[0], 4, 4))

        world_t_locals[:, :3, :] = np.stack((
            local_x, local_y, local_z, hit_positions
        ), axis=-1)

        world_t_locals[:, 3, :] = np.array([0, 0, 0, 1])

        incident_directions_local = -transform_vectors(np.linalg.inv(world_t_locals), incident_directions_world)

        incident_az_el = direction_to_az_el(incident_directions_local)

        new_rays = self._generate_scatter_rays(world_t_locals, incident_az_el, self._scattering_distribution)

        # Move start point along ray to prevent intersection with same surface
        new_rays[:, 0:3] += RAY_START_SHIFT * new_rays[:, 3:6]

        self._active_rays = new_rays
        self._rays = np.concatenate((self._rays, new_rays), axis=0)

        self._active_ray_path_ids = new_ray_path_ids
        self._ray_path_ids = np.concatenate((self._ray_path_ids, new_ray_path_ids), axis=0)

    def _reconstruct_paths(self):
        self._paths = []

        for path_i in range(self._n_paths):
            path = Path()

            rays = self._rays[self._ray_path_ids == path_i]

            for ray in rays:
                path.add_bounce(ray[0:3])

            path.add_bounce(rays[-1, 0:3] + 0.1 * rays[-1, 3:6])

            self._paths.append(path)

    def visualize(self):
        scene_geometries = self._scene.visualization_geometry()

        path_geometries = []

        for path in self._paths:
            path_geometries.append(o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.stack(path._positions, axis=0)),
                lines=o3d.utility.Vector2iVector(np.stack((np.arange(len(path._positions) - 1), np.arange(1, len(path._positions))), axis=1)),
            ))

        geometries = scene_geometries + path_geometries

        o3d.visualization.draw_geometries(geometries)

