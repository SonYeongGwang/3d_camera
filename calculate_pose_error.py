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

estimated_pose = np.array( [[ 0.01395671,  0.06192953,  0.99798294,  0.00391953],
                            [-0.06277135,  0.99616575, -0.06093892,  0.035     ],
                            [-0.99793034, -0.06179423,  0.0177906 ,  0.79992816],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]])

##########################################################################################################

if shape_meta_info['shape'] == 'Box':
    mesh = o3d.geometry.TriangleMesh.create_box(width=shape_meta_info['shape_parameters']['width'], height=shape_meta_info['shape_parameters']['height'], depth=shape_meta_info['shape_parameters']['depth'])
elif shape_meta_info['shape'] == 'Cylinder':
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=shape_meta_info['shape_parameters']['radius'], height=shape_meta_info['shape_parameters']['height'], resolution=20)
elif shape_meta_info['shape'] == 'Sphere':
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=shape_meta_info['shape_parameters']['radius'], resolution=20)

pcd_ground.points = mesh.vertices
pcd_estimated.points = mesh.vertices

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
pose_error = (np.sum(distance_list)/point_len)*10
print("-"*50)
print("POSE ERROR:{}cm".format(pose_error))
print("-"*50)