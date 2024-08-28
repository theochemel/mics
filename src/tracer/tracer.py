import open3d as o3d
import numpy as np

from tracer.scene import Scene


def triangle_side_test(triangle_vertices, point):
    v0, v1, v2 = triangle_vertices

    edge1 = v1 - v0
    edge2 = v2 - v0

    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

    vector_to_point = point - v0

    dot_product = np.dot(normal, vector_to_point)

    if dot_product > 0:
        return True
    else:
        return False


class Tracer:

    def __init__(self, scene: Scene):
        self._scene = scene


    def trace(self, max_path_length: int):
        raycasting_scene = o3d.t.geometry.RaycastingScene()

        n_nodes = 0
        node_positions = []

        source_nodes = {}

        for i, source in enumerate(self._scene.sources.values()):
            source_nodes[n_nodes + i] = source.id

            node_positions.append(source.pose.t)

        n_nodes += len(source_nodes)

        sink_nodes = {}

        for i, sink in enumerate(self._scene.sinks.values()):
            sink_nodes[n_nodes + i] = sink.id

            node_positions.append(sink.pose.t)

        n_nodes += len(sink_nodes)

        surface_start_nodes = {}

        raycast_id_to_start_node = {}
        node_to_start_node = {}
        node_to_face = {}

        surface_vertices = {}

        for surface in self._scene.surfaces.values():
            n_faces = len(surface.mesh.triangles)

            node = n_nodes
            n_nodes += n_faces

            surface_start_nodes[node] = surface.id

            raycast_id_to_start_node[raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(surface.mesh))] = node

            node_to_face.update({ node + i: i for i in range(len(surface.mesh.triangles)) })
            node_to_start_node.update({ node + i: node for i in range(len(surface.mesh.triangles)) })

            vertices = np.asarray(surface.mesh.vertices)
            triangles = np.asarray(surface.mesh.triangles)

            triangle_vertices = vertices[triangles]

            triangle_centers = np.mean(triangle_vertices, axis=1)

            node_positions.extend(triangle_centers)

            surface_vertices[surface.id] = np.asarray(surface.mesh.vertices)

        node_positions = np.array(node_positions)

        # if it's not a source node or a sink node, it's a face node
        # find smallest surface_start_node s.t. surface_start_node <= node
        # then node is face node - surface_start_node of surface surface_start_node

        edges = []

        for start_node in range(n_nodes):
            for end_node in range(start_node + 1, n_nodes):
                # Ignore intra-geometry bounces
                if start_node in node_to_start_node \
                    and end_node in node_to_start_node \
                        and node_to_start_node[start_node] == node_to_start_node[end_node]:
                    continue

                if start_node in source_nodes and end_node in source_nodes:
                    continue

                if start_node in sink_nodes and end_node in sink_nodes:
                    continue

                start_position = node_positions[start_node]
                end_position = node_positions[end_node]

                epsilon = 1e-4

                delta = end_position - start_position
                unit_delta = delta / np.linalg.norm(delta)

                rays = o3d.core.Tensor(
                    [
                        np.concatenate((start_position + epsilon * unit_delta, unit_delta)),
                        np.concatenate((end_position + epsilon * -unit_delta, -unit_delta)),
                    ],
                    dtype=o3d.core.Dtype.Float32
                )

                ray_results = raycasting_scene.cast_rays(rays)

                # Problem: t_hit is 0, 0 if checking line-of-sight between two faces
                # It just hits the same two faces. We need to start a little ways

                line_of_sight = not ( # Either ray collides
                    ( # Ray one collides
                        not (ray_results["t_hit"][0].isinf() or ray_results["t_hit"][0] == 0) # We hit something
                        and (
                            end_node in node_to_start_node and (
                                # That something belongs to different geometry
                                raycast_id_to_start_node[ray_results["geometry_ids"][0].item()] != node_to_start_node[end_node] # we need the start node for this end node
                                or
                                # That something belongs to the same geometry, but is a different face
                                ray_results["primitive_ids"][0].item() != node_to_face[end_node]
                            )
                        )
                    )
                    or
                    ( # Ray two colides
                        not (ray_results["t_hit"][1].isinf() or ray_results["t_hit"][1] == 0)
                        and (
                            start_node in node_to_start_node
                            and (
                                raycast_id_to_start_node[ray_results["geometry_ids"][1].item()] != node_to_start_node[start_node]
                                or
                                ray_results["primitive_ids"][1].item() != node_to_face[start_node]
                            )
                        )
                    )
                )

                if not line_of_sight:
                    continue

                if start_node in node_to_start_node:
                    surface_id = surface_start_nodes[node_to_start_node[start_node]]
                    face = node_to_face[start_node]

                    surface = self._scene.surfaces[surface_id]

                    triangle_vertices = surface_vertices[surface_id][surface.mesh.triangles[face]]

                    if not triangle_side_test(triangle_vertices, end_position):
                        continue

                if end_node in node_to_start_node:
                    surface_id = surface_start_nodes[node_to_start_node[end_node]]
                    face = node_to_face[end_node]

                    surface = self._scene.surfaces[surface_id]

                    triangle_vertices = surface_vertices[surface_id][surface.mesh.triangles[face]]

                    if not triangle_side_test(triangle_vertices, start_position):
                        continue

                edges.append((start_node, end_node))

        self._source_nodes = source_nodes
        self._sink_nodes = sink_nodes
        self._surface_start_nodes = surface_start_nodes
        self._node_positions = node_positions
        self._edges = edges

    def save(self):
        pass

    def load(self):
        pass

    def visualize(self):
        scene_geometries = self._scene.visualization_geometry()

        edge_geometries = [
            o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(self._node_positions),
                lines=o3d.utility.Vector2iVector(np.array(self._edges)),
            )
        ]

        geometries = scene_geometries + edge_geometries

        o3d.visualization.draw_geometries(geometries)

