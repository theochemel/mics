import numpy as np
from math import pi

from tracer.scene import *
from tracer.geometry import *

MIN_BOUNCE = 1e-3
RAY_START_SHIFT = 1e-3


class Path:


    def __init__(self):
        self._source_id: str = None
        self._source_position: np.array = None
        self._sink_id: str = None

        self._segment_delays: [float] = []

        self._segment_hit_positions: [np.array] = []
        self._segment_hit_attenuations: [float] = []

        self._segment_sink_visibles: [np.array] = [] # [n_segments, n_sinks]
        self._segment_sink_delays: [np.array] = []
        self._segment_sink_attenuations: [np.array] = []
        self._segment_sink_positions: [np.array] = []

    def add_source(self, source_id: str, source_position: np.array):
        self._source_id = source_id
        self._source_position = source_position

    def add_segment(self, delay: float, hit_position: np.array, hit_attenuation: float, sink_visible: np.array, sink_positions: np.array, sink_delays: np.array, sink_attenuations: np.array):
        self._segment_delays.append(delay)
        self._segment_hit_positions.append(hit_position)
        self._segment_hit_attenuations.append(hit_attenuation)
        self._segment_sink_visibles.append(sink_visible)
        self._segment_sink_delays.append(sink_delays)
        self._segment_sink_attenuations.append(sink_attenuations)
        self._segment_sink_positions.append(sink_positions)

    def visualization_geometry(self):
        if len(self._segment_hit_positions) == 0:
            return []

        path_points = np.concatenate((
            self._source_position[np.newaxis, :],
            self._segment_hit_positions,
            self._segment_sink_positions[0],
        ), axis=0)

        n_sinks = len(self._segment_sink_positions[0])

        path_lines = []

        path_lines.append(
            np.stack((
                np.arange(len(self._segment_hit_positions)),
                np.arange(1, len(self._segment_hit_positions) + 1),
            ), axis=-1)
        )

        for segment_i in range(len(self._segment_sink_visibles)):
            path_lines.append(
                np.stack((
                    np.full(np.sum(self._segment_sink_visibles[segment_i]), 1 + segment_i), # Segment end position
                    1 + len(self._segment_hit_positions) + np.arange(n_sinks)[self._segment_sink_visibles[segment_i]], # Sink position
                ), axis=-1)
            )

            pass

        path_lines = np.concatenate(path_lines, axis=0)

        path_geometry = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(path_points),
            lines=o3d.utility.Vector2iVector(path_lines),
        )

        return [path_geometry]


class Tracer:

    def __init__(self, scene: Scene):
        self._scene = scene

        self._n_sources = len(scene.sources)

        self._scattering_distribution = DiffuseBidirectionalReflectanceDistribution()
        self._material_attenuation = 0
        self._c = 1500


    def trace(self, n_rays: int, n_bounces: int):
        self._build_raycast_scene()

        self._generate_source_rays(n_rays)

        for i in range(n_bounces):
            self._propagate_rays()


    def _build_raycast_scene(self):
        self._raycast_scene = o3d.t.geometry.RaycastingScene()

        for surface in self._scene.surfaces.values():
            self._raycast_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(surface.mesh))

    def _generate_source_rays(self, n_rays: int):
        ray_sets = []
        paths = []

        for source_i, source in enumerate(self._scene.sources.values()):
            # Pose for every ray is the same
            world_t_sources = np.array([
                np.array(source.pose) for _ in range(n_rays)
            ])

            rays = self._generate_rays(world_t_sources, source.distribution)

            ray_sets.append(rays)

            for ray_i in range(len(rays)):
                path = Path()
                path.add_source(
                    source_id=source.id,
                    source_position=rays[ray_i, :3],
                )

                paths.append(path)

        rays = np.concatenate(ray_sets, axis=0)

        self._active_rays = rays
        self._rays = rays
        self._paths = paths

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

        hit_attenuations = self._material_attenuation + power_to_db(
            np.sum(
                incident_directions_local *
                np.repeat(
                    np.array([[0.0, 0.0, 1.0]]),
                    incident_directions_local.shape[0],
                    axis=0
                ),
                axis=-1
            )
        )

        # Move start point along ray to prevent intersection with same surface
        new_rays[:, 0:3] += RAY_START_SHIFT * new_rays[:, 3:6]

        self._active_rays = new_rays
        self._rays = np.concatenate((self._rays, new_rays), axis=0)

        self._active_ray_path_ids = new_ray_path_ids
        self._ray_path_ids = np.concatenate((self._ray_path_ids, new_ray_path_ids), axis=0)

        for ray_id, ray_path_id in enumerate(new_ray_path_ids):
            start_position = valid_rays[ray_id, :3]
            end_position = new_rays[ray_id, :3]

            sink_positions = self._scene.sink_poses[:, 0:3, 3]

            sink_directions = sink_positions - end_position[np.newaxis, :]
            sink_distances = np.linalg.norm(sink_directions, axis=-1)
            sink_directions = sink_directions / sink_distances[:, np.newaxis]

            sink_rays = np.concatenate((
                np.repeat(end_position[np.newaxis, :], sink_directions.shape[0], axis=0),
                sink_directions,
            ), axis=-1)

            sink_directions_incoming = transform_vectors(np.linalg.inv(self._scene.sink_poses), sink_directions)

            sink_az_el_outgoing = direction_to_az_el(sink_directions)
            sink_az_el_incoming = direction_to_az_el(sink_directions_incoming)

            sink_attenuation_outgoing = self._scattering_distribution.attenuation_db(sink_az_el_outgoing[:, 0], sink_az_el_outgoing[:, 1], np.repeat(incident_az_el[ray_id][np.newaxis, :], sink_az_el_outgoing.shape[0], axis=0))
            sink_attenuation_incoming = np.array([sink.distribution.attenuation_db(sink_az_el_incoming[i, 0], sink_az_el_incoming[i, 1]) for i, sink in enumerate(self._scene.sinks.values())])

            raycasts = self._raycast_scene.cast_rays(
                o3d.core.Tensor(sink_rays, dtype=o3d.core.Dtype.Float32)
            )

            t_hit = raycasts["t_hit"].numpy()

            visible = t_hit >= sink_distances

            self._paths[ray_path_id].add_segment(
                delay=np.linalg.norm(start_position - end_position) / self._c,
                hit_position=end_position,
                hit_attenuation=hit_attenuations[ray_id],
                sink_visible=visible,
                sink_positions=sink_positions,
                sink_delays=sink_distances / self._c,
                sink_attenuations=np.where(visible, sink_attenuation_outgoing + sink_attenuation_incoming, -np.inf), # TODO: Evaluate sink PDF here in the visible case
            )

    def visualize(self):
        scene_geometries = self._scene.visualization_geometry()

        geometries = scene_geometries

        for path in self._paths:
            geometries += path.visualization_geometry()

        o3d.visualization.draw_geometries(geometries)
