import open3d as o3d

armadillo_mesh = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)

mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])

pass