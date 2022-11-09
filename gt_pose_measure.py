import open3d as o3d
import numpy as np
import yaml
import cv2
from camera import IntelCamera

def _transform(x=0, y=0, z=0, rx=0, ry=0, rz=0):
    if x != 0:
        T = np.array([[    1,                         0,                         0,                          x],
                      [    0,                         np.math.cos(rx),           -np.math.sin(rx),           y],
                      [    0,                         np.math.sin(rx),            np.math.cos(rx),           z],
                      [    0,                         0,                         0,                          1]])
    elif y != 0:                 
        T = np.array([[    np.math.cos(ry),           0,                         np.math.sin(ry),            x],
                      [    0,                         1,                         0,                          y],
                      [    -np.math.sin(ry),          0,                         np.math.cos(ry),            z],
                      [    0,                         0,                         0,                          1]])
    elif z != 0:                 
        T = np.array([[    np.math.cos(rz),           -np.math.sin(rz),          0,                          x],
                      [    np.math.sin(rz),           np.math.cos(rz),           0,                          y],
                      [    0,                         0,                         1,                          z],
                      [    0,                         0,                         0,                          1]])

    elif rx != 0:
        T = np.array([[    1,                         0,                         0,                          x],
                      [    0,                         np.math.cos(rx),           -np.math.sin(rx),           y],
                      [    0,                         np.math.sin(rx),            np.math.cos(rx),           z],
                      [    0,                         0,                         0,                          1]])
    elif ry != 0:                 
        T = np.array([[    np.math.cos(ry),           0,                         np.math.sin(ry),            x],
                      [    0,                         1,                         0,                          y],
                      [    -np.math.sin(ry),          0,                         np.math.cos(ry),            z],
                      [    0,                         0,                         0,                          1]])
    elif rz != 0:                 
        T = np.array([[    np.math.cos(rz),           -np.math.sin(rz),          0,                          x],
                      [    np.math.sin(rz),           np.math.cos(rz),           0,                          y],
                      [    0,                         0,                         1,                          z],
                      [    0,                         0,                         0,                          1]])

    return T

print("-"*50)
print("select primitive shape\n")
shape = input("1: Sphere       2: Cylinder       3: Box\n")
shape_list = ["Shpere", "Cylinder", "Box"]
shape_param = {}

if shape == str(1):
    print("select shape parameters\n")
    radius = float(input("radius: "))
    shape_param['radius'] = radius
    primitive_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=80)
elif shape == str(2):
    print("select shape parameters\n")
    radius = float(input("radius: "))
    height = float(input("height: "))
    shape_param['radius'] = radius
    shape_param['height'] = height
    primitive_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=80)
elif shape == str(3):
    print("select shape parameters\n")
    width = float(input("x (width): "))
    height = float(input("y (height): "))
    depth = float(input("z (depth): "))
    shape_param['width'] = width
    shape_param['height'] = height
    shape_param['depth'] = depth
    
    primitive_mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
else:
    raise ValueError("select only 1, 2 or 3!!")

camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
primitive_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

primitive_mesh.compute_vertex_normals()

T = np.eye(4, 4)

T_past_mesh = np.eye(4, 4)
T_update_mesh = np.eye(4, 4)
T_mesh_exclusive = np.eye(4, 4)

T_past_frame = np.eye(4, 4)
T_update_frame = np.eye(4, 4)

if shape == str(3):
    vertices = np.asarray(primitive_mesh.vertices)
    vertices = vertices - np.array([width/2, height/2, depth/2])
    primitive_mesh.vertices = o3d.utility.Vector3dVector(vertices)

cam = IntelCamera(cfg=[])

cam.z_min = 0.01
transform_resolution = 0.1

vis = o3d.visualization.Visualizer()
vis.create_window('Measurement Display', width=848, height=480)
pcd = o3d.geometry.PointCloud()
added = False

for _ in range (20):
    rgb_img, depth_img = cam.stream()
    xyz = cam.generate(depth_img)
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if added == False:
        vis.add_geometry(pcd)
        added = True
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

added = False
while 1:

    win = cv2.namedWindow("key")
    key = cv2.waitKey(1)

    if key == ord('q'):
        T = _transform(x=transform_resolution, y=0, z=0, rx=0, ry=0, rz=0)

    elif key == ord('w'):
        T = _transform(x=0, y=transform_resolution, z=0, rx=0, ry=0, rz=0)

    elif key == ord('e'):
        T = _transform(x=0, y=0, z=transform_resolution, rx=0, ry=0, rz=0)

    elif key == ord('a'):
        T = _transform(x=-transform_resolution, y=0, z=0, rx=0, ry=0, rz=0)

    elif key == ord('s'):
        T = _transform(x=0, y=-transform_resolution, z=0, rx=0, ry=0, rz=0)

    elif key == ord('d'):
        T = _transform(x=0, y=0, z=-transform_resolution, rx=0, ry=0, rz=0)

    elif key == ord('i'):
        T =_transform(x=0, y=0, z=0, rx=transform_resolution, ry=0, rz=0)

    elif key == ord('o'):
        T =_transform(x=0, y=0, z=0, rx=0, ry=transform_resolution, rz=0)

    elif key == ord('p'):
        T =_transform(x=0, y=0, z=0, rx=0, ry=0, rz=transform_resolution)

    elif key == ord('j'):
        T = _transform(x=0, y=0, z=0, rx=-transform_resolution, ry=0, rz=0)

    elif key == ord('k'):
        T = _transform(x=0, y=0, z=0, rx=0, ry=-transform_resolution, rz=0)

    elif key == ord('l'):
        T = _transform(x=0, y=0, z=0, rx=0, ry=0, rz=-transform_resolution)

    elif key == ord('y'):
        print("-"*50)
        print('T_measured\n', repr(T_update_frame))

        shape_meta_info = {}

        with open('./shape_info/shape_meta_info.yml', 'w') as f:
            shape_meta_info['cam2object'] = T_update_frame.tolist()
            shape_meta_info['shape'] = shape_list[int(shape)-1]
            shape_meta_info['shape_parameters'] = shape_param

            yaml.dump(shape_meta_info, f, default_flow_style=None)
        print("Generated shape meta information!")
        print("-"*50)

    elif key == ord('r'):
        print("select trasnsform resolution level (currnet:{})".format(transform_resolution))
        transform_resolution = float(input())

    T_update_mesh = np.dot(T_past_mesh, T)
    # T_update_mesh = np.dot(T_mesh_exclusive, T_update_mesh)

    T_update_frame = np.dot(T_past_frame, T)

    primitive_mesh.transform(np.linalg.inv(T_past_mesh))
    primitive_mesh.transform(T_update_mesh)
    primitive_frame.transform(np.linalg.inv(T_past_frame))
    primitive_frame.transform(T_update_frame)

    if added == False:
        vis.add_geometry(camera_frame)
        vis.add_geometry(primitive_mesh)
        vis.add_geometry(primitive_frame)
        added = True
    vis.update_geometry(camera_frame)
    vis.update_geometry(primitive_mesh)
    vis.update_geometry(primitive_frame)
    vis.poll_events()
    vis.update_renderer()

    T_past_mesh = T_update_mesh
    T_past_frame = T_update_frame
    T = np.eye(4, 4)
    # T_mesh_exclusive = np.eye(4, 4)