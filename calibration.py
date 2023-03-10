import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
########################################################이거 쓰시면 댑니다 ^^v
 #[[-0.09088078 -0.99574393  0.01532044  0.07322322]
 #[-0.99564129  0.09052619 -0.02243697 -0.98057482]
 #[ 0.02095458 -0.01729274 -0.99963086  0.87000959]
 #[ 0.          0.          0.          1.        ]]
#####################################################





c2m_pose = np.asarray([[-0.06087859, -0.99813828,  0.00371062,  0.07778141],
                       [-0.99795552,  0.06079414, -0.0197195 ,  0.11545451],
                       [ 0.0194572 , -0.00490353, -0.99979867,  0.35118238],
                       [ 0.        ,  0.        ,  0.        ,  1.        ],]

)
m2mc_pose = np.asarray([[1,  0,  0,  0.242/2], 
                        [0,  1,  0,  0.172/2], 
                        [0,  0,  1,  -0.003 ], 
                        [0,  0,  0,  1     ]])

# for rb10
mc2e_pose = np.asarray([[1,  0,  0,  0],
                        [0,  0,  1,  0],
                        [0, -1,  0,  0],
                        [0,  0,  0,  1]])

# mc2e_pose = np.asarray([[ 0, 0, 1, 0],
#                         [ 0, 1, 0, 0],
#                         [-1, 0, 0, 0],
#                         [ 0, 0, 0, 1]])


#for ur10
# mc2e_pose = np.eye(4,4)

b2e_pose = np.asarray([[ 9.99873876e-01,  0.00000000e+00, -1.58818285e-02,  2.16470000e-01],
                       [ 1.58803568e-02, -1.36131477e-02,  9.99781224e-01, -8.23140000e-01],
                       [-2.16201676e-04, -9.99907337e-01, -1.36114307e-02,  5.26640000e-01],
                       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],]
)

# b2e_pose_R = b2e_pose[:3, :3]
# b2e_pose_t = b2e_pose[:3, 3]

# b2e_pose_inv = np.zeros_like(b2e_pose)
# b2e_pose_inv[:3, :3] = b2e_pose_R.T
# b2e_pose_inv[:3, 3] = -b2e_pose_R.T @ b2e_pose_t
# b2e_pose_inv[-1,-1] = 1

# c2e = np.dot(np.dot(c2m_pose, m2mc_pose), mc2e_pose)
# print(c2e)

# print("A\n", b2e_pose_inv)
# print("B\n",np.linalg.inv(b2e_pose))

c2b = np.dot(np.dot(np.dot(c2m_pose, m2mc_pose), mc2e_pose), np.linalg.inv(b2e_pose))
b2c = np.linalg.inv(c2b)
print("b2c^^v\n",b2c)

# c2b = np.dot(c2g_pose, np.linalg.inv(b2g_pose))
# b2c = np.linalg.inv(c2b)

# c2m = np.asarray([[-0.01245926, -0.99905093, -0.04173723, -0.08563992],
#  [-0.99800642,  0.00984184,  0.06234035,  0.03953801],
#  [-0.06187041,  0.04243074, -0.99718187,  1.00480634],
#  [ 0.        ,  0.        ,  0.        ,  1.        ]])

# m2b = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0.97], [0, 0, 1, -0.02], [0, 0, 0, 1]])

# c2b = np.dot(c2m, m2b)
# b2c = np.linalg.inv(c2b)
# print("b2c")
# print(b2c)
# print()

# class Cam2gripper:
#     def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):

#         if  eye_to_hand:
#             # change coordinates from gripper2base to base2gripper
#             R_base2gripper, t_base2gripper = [], []


#             # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
#             # T = np.eye(4)
#             # T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
#             # T[:3, -1] = [1,1,1]
#             # print(T)
      

#             # for R, t in zip(R_gripper2base, t_gripper2base):
#             #     R_b2g = R.T
#             #     t_b2g = -R_b2g @t
#             #     R_base2gripper.append(R_b2g)
#             #     t_base2gripper.append(t_b2g)
#             #     # change parameters values
#             #     R_gripper2base = R_base2gripper
#             #     t_gripper2base = t_base2gripper

#             # # calibrate
#             # R, t = cv2.calibrateHandEye(
#             # R_gripper2base=R_gripper2base,
#             # t_gripper2base=t_gripper2base,
#             # R_target2cam=R_target2cam,
#             # t_target2cam=t_target2cam,
#             # )

#             return R, t

if __name__ == '__main__':
    import copy

    def toM(list):

        return np.asarray(list)/1000

    def D2R(list):
        return np.asarray(list)*np.math.pi/180

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T = np.eye(4)
    # _temp = R.from_euler('xyz', [-90, 0, 90], degrees=True).as_quat()
    # _temp = R.from_euler('zyx', [90, 0, -90], degrees=True).as_quat()
    # r = R.from_quat(_temp)
    # print()
    # print(r.as_matrix())
    # T[:3, :3] = r.as_matrix()

    # z: -180(x), 90(x), -90(x), 0(x)

    T[:3, :3] = _temp2 = mesh.get_rotation_matrix_from_xyz(D2R([-90.0, 0, 0]))
    T[:3, -1] = toM([216.47, -823.14, 526.64])
    # print(_temp2)
    # print()
    print(T)

    # c2g = Cam2gripper
    # R_gripper2base = [b2g_pose1[:3,:3],b2g_pose2[:3,:3],b2g_pose3[:3,:3]]
    # t_gripper2base = [b2g_pose1[:3,3],b2g_pose2[:3,3],b2g_pose3[:3,3]]
    # R_target2cam = [c2m_pose1[:3,:3],c2m_pose2[:3,:3],c2m_pose3[:3,:3]]
    # t_target2cam = [c2m_pose1[:3,3],c2m_pose2[:3,3],c2m_pose3[:3,3]]
    
    # R ,t = c2g.calibrate_eye_hand(R_gripper2base,t_gripper2base,R_target2cam,t_target2cam)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    mesh_trans = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)
    mesh_trans = copy.deepcopy(mesh_trans)
    mesh_trans.transform(c2b)
    o3d.visualization.draw_geometries([mesh, mesh_trans])