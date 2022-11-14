import open3d as o3d
import numpy as np
import yaml
import time
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
estimated_pose = np.array( [[0.10154734433626042, -0.02947658472078168, 0.9943939198383378, 0.01],
                           [-0.2427660263381235, 0.9686082989586079, 0.05350345451011805, 0.0],
                           [-0.9647553022991141, -0.24683819425226175, 0.09120368711862581, 0.7999999999999999],
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
# 19, 6.8
if shape_meta_info['shape'] == 'Box':
    mesh_ground = o3d.geometry.TriangleMesh.create_box(width=shape_meta_info['shape_parameters']['width'], height=shape_meta_info['shape_parameters']['height'], depth=shape_meta_info['shape_parameters']['depth'])
    mesh_estimated = o3d.geometry.TriangleMesh.create_box(width=shape_meta_info['shape_parameters']['width'], height=shape_meta_info['shape_parameters']['height'], depth=shape_meta_info['shape_parameters']['depth'])
    # mesh_estimated = o3d.geometry.TriangleMesh.create_box(width=0.060154107799408774, height=0.085678466, depth=0.13982427)
elif shape_meta_info['shape'] == 'Cylinder':
    mesh_ground = o3d.geometry.TriangleMesh.create_cylinder(radius=shape_meta_info['shape_parameters']['radius'], height=shape_meta_info['shape_parameters']['height'], resolution=40)
    mesh_estimated = o3d.geometry.TriangleMesh.create_cylinder(radius=shape_meta_info['shape_parameters']['radius'], height=shape_meta_info['shape_parameters']['height'], resolution=40)
    # mesh_estimated = o3d.geometry.TriangleMesh.create_cylinder(radius=0.027600782545470052, height=0.16509402879018856, resolution=40)
elif shape_meta_info['shape'] == 'Sphere':
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=shape_meta_info['shape_parameters']['radius'], resolution=20)

pcd_ground.points = mesh_ground.vertices
pcd_estimated.points = mesh_estimated.vertices

# pcd_ground = mesh_ground.sample_points_uniformly(number_of_points=5000)
# pcd_estimated = mesh_estimated.sample_points_uniformly(number_of_points=5000)

# pcd_ground = mesh_ground.sample_points_poisson_disk(number_of_points=4000)
# pcd_estimated = copy.deepcopy(pcd_ground)

pcd_ground.paint_uniform_color([0, 0, 1])
pcd_estimated.paint_uniform_color([0, 1, 0])

pcd_ground.transform(ground_truth_pose)
pcd_estimated.transform(estimated_pose)

a = pcd_estimated.get_center()
b = pcd_ground.get_center()

print(np.linalg.norm(a-b))

o3d.visualization.draw_geometries([pcd_ground, pcd_estimated])

pcd_ground_kdtree = o3d.geometry.KDTreeFlann(pcd_ground)
points = np.asarray(pcd_estimated.points)
point_len = len(points)

ground_point_list = []
distance_list = []

pcd_ground_points = np.asarray(pcd_ground.points)

for p in points:
    _, idx, d = pcd_ground_kdtree.search_knn_vector_3d(p, 1)
    idx = idx.pop()
    ground_point_list.append(pcd_ground_points[idx])
    distance_list.append(d.pop())

    pcd_ground_points = np.delete(pcd_ground_points, idx, 0)
    pcd_ground.points = o3d.utility.Vector3dVector(pcd_ground_points)
    pcd_ground_kdtree = o3d.geometry.KDTreeFlann(pcd_ground)

ground_point_list = np.asarray(ground_point_list)

line_points = np.concatenate((np.asarray(pcd_estimated.points), ground_point_list), axis=0)
lines = np.asarray([[i, i+point_len] for i in range(point_len)])
colors = [[0, 0, 1] for i in range(len(lines))]
line_sets = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines),
    )

pcd_ground.points = o3d.utility.Vector3dVector(ground_point_list)

line_sets.colors = o3d.utility.Vector3dVector(colors)

a = pcd_estimated.get_center()
b = pcd_ground.get_center()
print(np.linalg.norm(a-b))

o3d.visualization.draw_geometries([pcd_ground, pcd_estimated, line_sets])

## calculate pose error
pose_error = (np.sum(distance_list)/point_len)*100
print("-"*50)
print("POSE ERROR:{}cm".format(pose_error))
print("-"*50)