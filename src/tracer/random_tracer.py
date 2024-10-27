import numpy as np
from math import pi

from tracer.scene import *
from tracer.geometry import *

MIN_BOUNCE = 1e-2


class Tracer:

    def __init__(self, scene: Scene):
        self._scene = scene

        self._n_sources = len(scene.sources)

        self._scattering_distribution = UniformContinuousAngularDistribution(
            min_az=-pi / 8,
            max_az=pi / 8,
            min_el=pi / 3,
            max_el=2 * pi / 3,
        )


    def trace(self, n_rays: int, n_bounces: int):
        self._build_raycast_scene()

        self._generate_source_rays(n_rays)

        for i in range(n_bounces):
            self._propagate_rays()

        # Now we have all the information we need in self._rays, self._ray_path_ids, and self._path_source_ids
        # We can reconstruct a path from each of those

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

        self._active_ray_path_ids = ray_path_ids
        self._ray_path_ids = ray_path_ids

    def _generate_rays(self, world_t_locals: np.array, distribution: ContinuousAngularDistribution):
        az_els = distribution.sample(world_t_locals.shape[0])

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

        incident_directions = valid_rays[:, 3:]

        reflected_directions = incident_directions - 2 * (np.sum(incident_directions * normals, axis=-1)[:, np.newaxis]) * normals

        reflected_directions_proj = reflected_directions - (np.sum(reflected_directions * normals, axis=-1)[:, np.newaxis] * normals)

        reflected_x = reflected_directions
        reflected_y = np.cross(reflected_directions_proj, normals, axis=-1)
        reflected_z = np.cross(reflected_x, reflected_y, axis=-1)

        reflected_poses = np.zeros((hit_positions.shape[0], 4, 4))

        reflected_poses[:, :3, :] = np.stack((
            reflected_x, reflected_y, reflected_z, hit_positions
        ), axis=-1)

        reflected_poses[:, 3, :] = np.array([0, 0, 0, 1])

        new_rays = self._generate_rays(reflected_poses, self._scattering_distribution)

        self._active_rays = new_rays
        self._rays = np.concatenate((self._rays, new_rays), axis=0)

        self._active_ray_path_ids = new_ray_path_ids
        self._ray_path_ids = np.concatenate((self._ray_path_ids, new_ray_path_ids), axis=0)

        pass

    def visualize(self):
        scene_geometries = self._scene.visualization_geometry()

        n_rays = len(self._rays)

        ray_points = np.concatenate((
            self._rays[:, :3],
            self._rays[:, :3] + self._rays[:, 3:],
        ), axis=0)

        ray_lines = np.stack((
            np.arange(0, n_rays),
            n_rays + np.arange(0, n_rays),
        ), axis=-1)

        edge_geometries = [
            o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(ray_points),
                lines=o3d.utility.Vector2iVector(ray_lines),
            )
        ]

        geometries = scene_geometries + edge_geometries

        o3d.visualization.draw_geometries(geometries)

