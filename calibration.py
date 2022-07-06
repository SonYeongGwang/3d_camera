import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import open3d as o3d

# pose1
b2g_pose1 = np.asarray([[ 0.98433682, -0.17312578,  0.03329385,  0.30252],
            [ 0.16712995,  0.8562494,  -0.48877862, -0.70223   ],
            [ 0.05611234,  0.48668719,  0.87177232,  0.47824   ],
            [ 0.    ,      0.   ,      0.    ,      1.        ]])

c2m_pose1 = np.asarray([[ 0.23390079,  0.85964598 , 0.45420173 ,-0.22724335],
            [ 0.97150724, -0.22503364, -0.07438776, -0.08731337],
            [ 0.03826353 , 0.45865963, -0.88778784,  0.49130207],
            [ 0.       ,   0.    ,      0.      ,    1.        ]])

#pose2
b2g_pose2 = np.asarray([[ 9.60926383e-01, -1.88095651e-01 ,-2.03077604e-01 , 2.94270000e-01],
            [ 1.83995806e-01  ,9.82150700e-01, -3.90582241e-02 ,-7.01480000e-01],
            [ 2.06799493e-01 , 1.66650536e-04 , 9.78383331e-01 , 5.87510000e-01],
            [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00  ,1.00000000e+00]])

c2m_pose2 = np.asarray([[ 0.24148553 , 0.96887413, 0.05447624 ,-0.24011779],
            [ 0.93762508, -0.24742795,  0.24421018, -0.07021107],
            [ 0.25008787 ,-0.00789494 ,-0.96819096 , 0.395606  ],
            [ 0.       ,   0.  ,       0.    ,     1.      ,  ]])

#pose3
b2g_pose3 = np.asarray([[ 0.89338572, -0.14105918 , 0.42657269  ,0.2186    ],
 [ 0.37635776 , 0.75351418 ,-0.53904658 ,-0.73591  , ],
 [-0.2453911  , 0.64212046 , 0.72626753 , 0.49209   ],
 [ 0.       ,   0.     ,     0. ,         1.  ,      ]])
c2m_pose3 = np.asarray([[ 0.31986957 , 0.80236196,  0.50388367, -0.19527102],
 [ 0.89191362 ,-0.0755793 , -0.44584511, -0.00895836],
 [-0.31964598 , 0.59203299, -0.73981307,  0.5082621 ],
 [ 0.   ,       0.      ,    0.,          1.        ]])



b2e_pose = np.asarray([[-0.35152884,  0.93617705,  0.,          0.0226    ],
 [-0.93617705, -0.35152884,  0.       ,  -0.77496   ],
 [ 0.        ,  0.        ,  1.       ,   0.70203   ],
 [ 0.        ,  0.        ,  0.       ,   1.        ]])

c2m_pose = np.asarray([[ 0.93926384,  0.3405783,   0.04230661, -0.20959253],
 [ 0.33821253, -0.93949638,  0.0543952 ,  0.03389935],
 [ 0.05827273, -0.03678282, -0.99762283,  0.30283646],
 [ 0.        ,  0.        ,  0.        ,  1.        ]])

m2e_pose = np.asarray([[1,     0,     0,     0.242/2], 
                       [0,     1,     0,     0.172/2], 
                       [0,     0,     1,     -0.003 ], 
                       [0,     0,     0,     1     ]])

c2b = np.dot(np.dot(c2m_pose, m2e_pose), np.linalg.inv(b2e_pose))
b2c = np.linalg.inv(c2b)
print(b2c)

# c2b = np.dot(c2g_pose, np.linalg.inv(b2g_pose))
# b2c = np.linalg.inv(c2b)

# c2m = np.asarray([[-0.01245396, -0.99990607,  0.00572225, -0.22365874],
#  [-0.9999106  , 0.01248147 , 0.0047971  , 0.29990225],
#  [-0.00486808 ,-0.00566199 ,-0.99997212 , 0.98127839],
#  [ 0.         , 0.         , 0.         , 1.        ]])

# m2b = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0.62], [0, 0, 1, -0.02], [0, 0, 0, 1]])

# c2b = np.dot(c2m, m2b)
# b2c = np.linalg.inv(c2b)
# print("b2c")
# print(b2c)
# print()

class Cam2gripper:
    def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):

        if  eye_to_hand:
            # change coordinates from gripper2base to base2gripper
            R_base2gripper, t_base2gripper = [], []


            # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            # T = np.eye(4)
            # T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
            # T[:3, -1] = [1,1,1]
            # print(T)
      

            for R, t in zip(R_gripper2base, t_gripper2base):
                R_b2g = R.T
                t_b2g = -R_b2g @t
                R_base2gripper.append(R_b2g)
                t_base2gripper.append(t_b2g)
                # change parameters values
                R_gripper2base = R_base2gripper
                t_gripper2base = t_base2gripper

            # calibrate
            R, t = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            )

            return R, t

if __name__ == '__main__':
    import copy
    def toM(list):

        return np.asarray(list)/1000

   
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T = np.eye(4)
    T[:3, :3] = _temp = mesh.get_rotation_matrix_from_xyz((0, 0, -1.93))
    T[:3, -1] = toM([22.6, -774.96, 702.03])
    # print(_temp)
    # print(T)

    c2g = Cam2gripper
    R_gripper2base = [b2g_pose1[:3,:3],b2g_pose2[:3,:3],b2g_pose3[:3,:3]]
    t_gripper2base = [b2g_pose1[:3,3],b2g_pose2[:3,3],b2g_pose3[:3,3]]
    R_target2cam = [c2m_pose1[:3,:3],c2m_pose2[:3,:3],c2m_pose3[:3,:3]]
    t_target2cam = [c2m_pose1[:3,3],c2m_pose2[:3,3],c2m_pose3[:3,3]]
    
    R ,t = c2g.calibrate_eye_hand(R_gripper2base,t_gripper2base,R_target2cam,t_target2cam)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    mesh_trans = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)
    mesh_trans = copy.deepcopy(mesh_trans)
    mesh_trans.transform(c2b)
    o3d.visualization.draw_geometries([mesh, mesh_trans])