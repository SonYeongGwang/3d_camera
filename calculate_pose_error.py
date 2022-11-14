import open3d as o3d
import numpy as np
import yaml
import copy

with open('./shape_info/shape_meta_info.yml') as f:
    shape_meta_info = yaml.load(f, Loader=yaml.FullLoader)

pcd_ground = o3d.geometry.PointCloud()
pcd_estimated = o3d.geometry.PointCloud()

ground_truth_pose = shape_meta_info['cam2object']

##########################################################################################################
#                                        Insert Estimated Pose                                           #
##########################################################################################################

## box
estimated_pose = np.array( [[0.026538657796817928, -0.40825098072701255, 0.9124838828043919, 0.010389156644171735],
                            [0.006552881672655901, 0.9127147251329524, 0.40854484486683396, 0.036839897624263755],
                            [-0.9996263098699092, 0.01682163074479789, -0.021547003386970593, 0.7966939787391589],
                            [0.0, 0.0, 0.0, 1.0]])

# cylinder
# estimated_pose = np.array( [[ 0.0585681 , -0.89255204,  0.44712485, -0.04206563],
#                             [ 0.00830552,  0.44831386,  0.89383762,  0.01316391],
#                             [-0.99824886, -0.04863677,  0.03367001,  0.89330201],
#                             [ 0.        ,  0.        ,  0.        ,  1.        ]]
#                             )

# estimated_pose = np.array( [[0.9004471023526769, -0.013047008766512038, 0.4347698143005266, -0.05115167670731399],
#                            [-0.43387593962150756, 0.04376366988502975, 0.8999091121973101, 0.01843055440646978],
#                            [-0.03076824470492633, -0.998956714157547, 0.03374608654847372, 0.9016077081764418],
#                            [0.0, 0.0, 0.0, 1.0]]
#                             )

##########################################################################################################
if shape_meta_info['shape'] == 'Box':
    mesh_ground = o3d.geometry.TriangleMesh.create_box(width=shape_meta_info['shape_parameters']['width'], height=shape_meta_info['shape_parameters']['height'], depth=shape_meta_info['shape_parameters']['depth'])
    mesh_estimated = o3d.geometry.TriangleMesh.create_box(width=shape_meta_info['shape_parameters']['width'], height=shape_meta_info['shape_parameters']['height'], depth=shape_meta_info['shape_parameters']['depth'])
    # mesh_estimated = o3d.geometry.TriangleMesh.create_box(width=0.060154107799408774, height=0.085678466, depth=0.13982427)

elif shape_meta_info['shape'] == 'Cylinder':
    mesh_ground = o3d.geometry.TriangleMesh.create_cylinder(radius=shape_meta_info['shape_parameters']['radius'], height=shape_meta_info['shape_parameters']['height'], resolution=40)
    # mesh_estimated = o3d.geometry.TriangleMesh.create_cylinder(radius=shape_meta_info['shape_parameters']['radius'], height=shape_meta_info['shape_parameters']['height'], resolution=40)
    mesh_estimated = o3d.geometry.TriangleMesh.create_cylinder(radius=0.027600782545470052, height=0.16509402879018856, resolution=40)
elif shape_meta_info['shape'] == 'Sphere':
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=shape_meta_info['shape_parameters']['radius'], resolution=20)

scene_pcd = o3d.io.read_point_cloud('./shape_info/scene.pcd')

# pcd_ground.points = mesh_ground.vertices
# pcd_estimated.points = mesh_estimated.vertices

# pcd_ground = mesh_ground.sample_points_uniformly(number_of_points=40)
# pcd_estimated = mesh_estimated.sample_points_uniformly(number_of_points=40)

pcd_ground = mesh_ground.sample_points_poisson_disk(number_of_points=5000)
pcd_estimated = mesh_estimated.sample_points_poisson_disk(number_of_points=5000)

pcd_ground.paint_uniform_color([0, 0, 1])
pcd_estimated.paint_uniform_color([0, 1, 0])

pcd_ground.transform(ground_truth_pose)
pcd_estimated.transform(estimated_pose)

a = pcd_estimated.get_center()
b = pcd_ground.get_center()

print(np.linalg.norm(a-b))

# pcd = mesh.sample_points_poisson_disk(number_of_points=8)

o3d.visualization.draw_geometries([pcd_ground, pcd_estimated])

pcd_ground_kdtree = o3d.geometry.KDTreeFlann(pcd_ground)
points = np.asarray(pcd_estimated.points)
point_len = len(points)

neighbor_list = []
distance_list = []
for p in points:
    _, idx, _ = pcd_ground_kdtree.search_knn_vector_3d(p, 1)
    neighbor_list.append(idx.pop())

    matching_point = np.asarray(pcd_ground.points[neighbor_list[-1]])
    dis = np.linalg.norm(p-matching_point)
    distance_list.append(dis)

    # print("query point", p)
    # print("matching point", matching_point)
    # print("distance bwt two points", dis)
    # print("stored distance information", distance_list[-1])
    # print()

line_points = np.concatenate((np.asarray(pcd_estimated.points), np.asarray(pcd_ground.points)), axis=0)
lines = np.asarray([[i, neighbor_list[i]+point_len] for i in range(point_len)])
colors = [[0, 0, 1] for i in range(len(lines))]
line_sets = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines),
    )

line_sets.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd_ground, pcd_estimated, line_sets])

## calculate pose error
pose_error = (np.sum(distance_list)/point_len)*100
print("-"*50)
print("POSE ERROR:{}cm".format(pose_error))
print("-"*50)

if shape_meta_info['shape'] == 'Box':
    vertices = np.asarray(mesh_ground.vertices)
    vertices = vertices - np.array([shape_meta_info['shape_parameters']['width']/2, shape_meta_info['shape_parameters']['height']/2, shape_meta_info['shape_parameters']['depth']/2])
    mesh_ground.vertices = o3d.utility.Vector3dVector(vertices)

    vertices = np.asarray(mesh_estimated.vertices)
    vertices = vertices - np.array([shape_meta_info['shape_parameters']['width']/2, shape_meta_info['shape_parameters']['height']/2, shape_meta_info['shape_parameters']['depth']/2])
    mesh_estimated.vertices = o3d.utility.Vector3dVector(vertices)

mesh_ground.compute_vertex_normals()
mesh_estimated.compute_vertex_normals()

mesh_ground.paint_uniform_color([0, 0, 1])
mesh_estimated.paint_uniform_color([0, 1, 0])

mesh_ground.transform(ground_truth_pose)
mesh_estimated.transform(estimated_pose)

o3d.visualization.draw_geometries([mesh_ground, mesh_estimated, scene_pcd])
