import open3d as o3d
import numpy as np
import yaml

with open('./shape_info/shape_meta_info.yml') as f:
    shape_meta_info = yaml.load(f, Loader=yaml.FullLoader)

pcd_ground = o3d.geometry.PointCloud()
pcd_estimated = o3d.geometry.PointCloud()

ground_truth_pose = shape_meta_info['cam2object']

##########################################################################################################
#                                        Insert Estimated Pose                                           #
##########################################################################################################

## box
# estimated_pose = np.array( [[-0.04023262,  0.88360023, -0.46651039, -0.04865666],
#                             [ 0.0291515 , -0.46565166, -0.88448781,  0.01145382],
#                             [-0.99876499, -0.04918474, -0.00702388,  0.87835401],
#                             [ 0.        ,  0.        ,  0.        ,  1.        ]])

# cylinder
estimated_pose = np.array( [[ 0.0585681 , -0.89255204,  0.44712485, -0.04206563],
                            [ 0.00830552,  0.44831386,  0.89383762,  0.01316391],
                            [-0.99824886, -0.04863677,  0.03367001,  0.89330201],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]]
                            )

##########################################################################################################
# 19, 6.8
if shape_meta_info['shape'] == 'Box':
    mesh_ground = o3d.geometry.TriangleMesh.create_box(width=shape_meta_info['shape_parameters']['width'], height=shape_meta_info['shape_parameters']['height'], depth=shape_meta_info['shape_parameters']['depth'])
    mesh_estimated = o3d.geometry.TriangleMesh.create_box(width=0.060154107799408774, height=0.085678466, depth=0.13982427)
elif shape_meta_info['shape'] == 'Cylinder':
    mesh_ground = o3d.geometry.TriangleMesh.create_cylinder(radius=shape_meta_info['shape_parameters']['radius'], height=shape_meta_info['shape_parameters']['height'], resolution=40)
    mesh_estimated = o3d.geometry.TriangleMesh.create_cylinder(radius=0.027600782545470052, height=0.16509402879018856, resolution=40)
elif shape_meta_info['shape'] == 'Sphere':
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=shape_meta_info['shape_parameters']['radius'], resolution=20)

pcd_ground.points = mesh_ground.vertices
pcd_estimated.points = mesh_estimated.vertices

pcd_ground.paint_uniform_color([0, 0, 1])
pcd_estimated.paint_uniform_color([0, 1, 0])

pcd_ground.transform(ground_truth_pose)
pcd_estimated.transform(estimated_pose)

# pcd = mesh.sample_points_poisson_disk(number_of_points=8)

o3d.visualization.draw_geometries([pcd_ground, pcd_estimated])

pcd_ground_kdtree = o3d.geometry.KDTreeFlann(pcd_ground)
points = np.asarray(pcd_estimated.points)
point_len = len(points)

neighbor_list = []
distance_list = []
for p in points:
    _, idx, d = pcd_ground_kdtree.search_knn_vector_3d(p, 1)
    neighbor_list.append(idx.pop())
    distance_list.append(d.pop())

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