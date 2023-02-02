##############################################################
#   camera.py
#   version: 3.1.1 (edited in 2023.02.02)
##############################################################
import sys
import os
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import yaml

import pyrealsense2 as rs
import numpy as np
import open3d as o3d

from cv2 import aruco
import copy

class IntelCamera:
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.context = rs.context()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.spatial_filter = rs.spatial_filter()
        self.hole_filling_filter = rs.hole_filling_filter(0)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

        print(self.device_product_line + " is ready")
        self.device_name = device.get_info(rs.camera_info.name).replace(" ", "_")
        self.device_name = self.device_name + "_" + device.get_info(rs.camera_info.serial_number)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.color_intrinsic = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsic = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.fx = self.color_intrinsic.fx
        self.fy = self.color_intrinsic.fy
        self.ppx = self.color_intrinsic.ppx
        self.ppy = self.color_intrinsic.ppy

        if self.device_product_line == 'L500':
            self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(960, 540, self.fx, self.fy, self.ppx, self.ppy)
        
        else:
            self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(640, 480, self.fx, self.fy, self.ppx, self.ppy)

        self.camera_mat = np.array([[self.fx, 0, self.ppx], [0, self.fy, self.ppy], [0, 0, 1]], dtype=np.float)

        self.dist_coeffs = np.zeros(4)
        self.colorizer = rs.colorizer(color_scheme = 2)

        self.saw_yaml = False
        self.saw_aruco = False
        self.saw_charuco = False
        self.aruco_marker_size = 0.054
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.aruco_dict_ch = aruco.Dictionary_get(aruco.DICT_4X4_250)

        self.z_min = -0.05
    
    def stream(self):
        
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        ## filter depth frame
        # depth_frame = self.spatial_filter.process(depth_frame)
        # depth_frame = self.hole_filling_filter.process(depth_frame)

        colored_depth_frame = self.colorizer.colorize(depth_frame)

        self.color_image = np.asanyarray(color_frame.get_data())
        self.colored_depth_image = np.asanyarray(colored_depth_frame.get_data())
        self.depth_image = np.asanyarray(depth_frame.get_data())

        return self.color_image, self.depth_image

    def generate(self, depth, downsample=True):
        depth_o3d = o3d.geometry.Image(depth)
        if self.device_product_line == 'L500':
            self.pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, self.intrinsic_o3d, depth_scale=4000.0)
        else:
            self.pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, self.intrinsic_o3d, depth_scale=1000.0)
        if downsample:
            self.pcd = self.pcd.voxel_down_sample(voxel_size = 0.006)
        self.xyz = np.asarray(self.pcd.points)
        return self.xyz

    def draw_workspace(self):
        self.orig_stored_cam2marker = copy.deepcopy(self.stored_cam2marker)
        marker_frame_center = self.stored_cam2marker[:3, 3]
        marker_frame_center[0] = marker_frame_center[0] + self.origin_to_corner_x
        marker_frame_center[1] = marker_frame_center[1] - self.origin_to_corner_y
        pixel = (np.dot(self.camera_mat, marker_frame_center)/marker_frame_center[-1])[:2]
        pixel = pixel.astype(np.int64)
        pixel = np.reshape(pixel, (2,))
        pixel = tuple(pixel)
        cv2.circle(img=self.color_image, center=pixel, radius=4, color=(0, 0, 255), thickness=-1)

        width_end_point = np.reshape([self.W, 0, 0, 1], (4, 1))
        width_end_point_from_cam = np.dot(self.stored_cam2marker, width_end_point)[:3]
        pixel_width = (np.dot(self.camera_mat, width_end_point_from_cam)/width_end_point_from_cam[-1])[:2]
        pixel_width = pixel_width.astype(np.int64)
        pixel_width = np.reshape(pixel_width, (2,))
        pixel_width = tuple(pixel_width)
        cv2.circle(img=self.color_image, center=pixel_width, radius=4, color=(0, 0, 255), thickness=-1)

        height_end_point = np.reshape([0, self.H, 0, 1], (4, 1))
        height_end_point_from_cam = np.dot(self.stored_cam2marker, height_end_point)[:3]
        pixel_length = (np.dot(self.camera_mat, height_end_point_from_cam)/height_end_point_from_cam[-1])[:2]
        pixel_length = pixel_length.astype(np.int64)
        pixel_length = np.reshape(pixel_length, (2,))
        pixel_length = tuple(pixel_length)
        cv2.circle(img=self.color_image, center=pixel_length, radius=4, color=(0, 0, 255), thickness=-1)

        vector2corner4th = np.reshape([self.W, self.H, 0, 1], (4, 1))
        vector2corner4th_from_cam = np.dot(self.stored_cam2marker, vector2corner4th)[:3]
        pixel_4th = (np.dot(self.camera_mat, vector2corner4th_from_cam)/vector2corner4th_from_cam[-1])[:2]
        pixel_4th = pixel_4th.astype(np.int64)
        pixel_4th = np.reshape(pixel_4th, (2,))
        pixel_4th = tuple(pixel_4th)
        cv2.circle(img=self.color_image, center=pixel_4th, radius=4, color=(0, 0, 255), thickness=-1)

        cv2.line(img=self.color_image, pt1=pixel, pt2=pixel_width, color=(0, 0, 255), thickness=2)
        cv2.line(img=self.color_image, pt1=pixel, pt2=pixel_length, color=(0, 0, 255), thickness=2)
        cv2.line(img=self.color_image, pt1=pixel_4th, pt2=pixel_width, color=(0, 0, 255), thickness=2)
        cv2.line(img=self.color_image, pt1=pixel_4th, pt2=pixel_length, color=(0, 0, 255), thickness=2)

        self.stored_cam2marker = self.orig_stored_cam2marker

    def detectAruco(self):
        if self.saw_aruco != True:
            self.parameters = aruco.DetectorParameters_create()
            self.saw_aruco = True

        gray_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray_img, self.aruco_dict, parameters=self.parameters)
        frame_markers = aruco.drawDetectedMarkers(self.color_image, corners, ids)
        if np.shape(corners)[0] > 0:
                for i in range(np.shape(corners)[0]):
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], self.aruco_marker_size, cameraMatrix=cam.camera_mat, distCoeffs=cam.dist_coeffs)
                    frame_markers = cv2.drawFrameAxes(frame_markers, cameraMatrix=cam.camera_mat, distCoeffs=cam.dist_coeffs, rvec=rvecs, tvec=tvecs, length=0.050, thickness=2)
                    ## for SE3 trasnformation matrix (marker with respect to the camera)
                    R, _ = cv2.Rodrigues(rvecs)
                    tvecs = np.reshape(tvecs, (3, 1))
                    self.cam2marker = np.concatenate((R, tvecs), axis = 1)
                    ## add [0, 0, 0, 1] to make it SE3 format
                    self.cam2marker = np.concatenate((self.cam2marker, np.array([[0, 0, 0, 1]])), axis = 0)

    def detectCharuco(self):
        if self.saw_charuco != True:
            self.board = aruco.CharucoBoard_create(7, 5, 0.035, 0.025, self.aruco_dict_ch)
            self.params = aruco.DetectorParameters_create()
            self.saw_charuco = True

        gray_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_points  =  aruco.detectMarkers(gray_img, self.aruco_dict_ch, parameters=self.params)
        if np.shape(corners)[0] <= 0:
            print("INFO: No Marker Detected")
        
        else:
            print("INFO: Marker Detected:",len(corners))
            aruco.refineDetectedMarkers(gray_img, self.board, corners, ids, rejected_points)
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray_img, self.board, self.camera_mat, self.dist_coeffs)
            if len(corners) > 10:
                _, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, self.camera_mat, self.dist_coeffs)
                R, _ = cv2.Rodrigues(rvec)
                tvec = np.reshape(tvec, (3, 1))
                self.cam2marker = np.concatenate((R, tvec), axis = 1)
                self.cam2marker = np.concatenate((self.cam2marker, np.array([[0, 0, 0, 1]])), axis = 0)
                aruco.drawAxis(self.color_image, self.camera_mat, self.dist_coeffs, rvec, tvec, 0.07)
                aruco.drawDetectedCornersCharuco(self.color_image, charuco_corners, charuco_ids, (255, 0, 0))

    def create_aruco_marker(self, dict):
        if not isinstance(dict, cv2.aruco_Dictionary):
            dict = aruco.Dictionary_get(dict)
        marker_generated = aruco.drawMarker(dict, 0, 600, 1)
        print(marker_generated.shape)
        print("ArUco Dictionary:", dict)
        print("ArUco Marker is created!")

        cv2.imshow("AruCo Marker", marker_generated) # Show image file
        cv2.imwrite("AruCo Marker.png", marker_generated) # Save image file
        cv2.waitKey() # Maintain until keyboard input

    def define_workspace(self):
        if self.saw_yaml != True:
            self.contents = 0

            home_path = os.path.expanduser('~')
            cfg_path = home_path+'/catkin_ws/src/suction_net_ros/config/workspace.yml'

            with open(cfg_path) as f:   
                self.workspace_cfg = yaml.load(f, Loader=yaml.FullLoader)
                self.saw_yaml = True
                self.W = self.workspace_cfg["width"]
                self.H = self.workspace_cfg["height"]
                self.origin_to_corner_x = self.workspace_cfg['origin_to_corner_x']
                self.origin_to_corner_y = self.workspace_cfg['origin_to_corner_y']
                self.stored_cam2marker = self.workspace_cfg['cam2marker']
                self.stored_cam2marker = np.reshape(self.stored_cam2marker, (4, 4))

        self.draw_workspace()
                    
    def crop_points(self):

        self.define_workspace()

        R = self.stored_cam2marker[:3, :3]
        self.tvecs = self.stored_cam2marker[:3, 3]
        R_inv = np.transpose(R)
        t_inv = -1 * np.dot(R_inv, self.tvecs)
        t_inv = np.reshape(t_inv, (3, 1))
        H_inv = np.concatenate((R_inv, t_inv), axis = 1)
        H_inv = np.concatenate((H_inv, np.array([[0, 0, 0, 1]])), axis = 0)
        self.pcd.transform(H_inv)
        self.xyz = np.asarray(self.pcd.points)
        valid_idx = np.where(((self.xyz[:, 0] > -self.origin_to_corner_x) & (self.xyz[:, 0] < (self.W - self.origin_to_corner_x))) & ((self.xyz[:, 1] > -self.origin_to_corner_y) & (self.xyz[:, 1] < (self.H-self.origin_to_corner_y))) & (self.xyz[:, 2] > self.z_min) & (self.xyz[:, 2] < 0.3))[0]

        # self.pcd = self.select_by_index(self.pcd, valid_idx)
        self.pcd = self.pcd.select_by_index(valid_idx)

        ## transform point cloud to original frame (camera frame)
        self.pcd.transform(self.stored_cam2marker)
        self.xyz = np.asarray(self.pcd.points)
        return self.xyz

    @staticmethod
    def select_by_index(pcd, index=[]): 
        
        xyz = o3d.geometry.PointCloud()

        if pcd.has_points():
            points = np.asarray(pcd.points)
            xyz.points = o3d.utility.Vector3dVector(points[index])

        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            xyz.normals = o3d.utility.Vector3dVector(normals[index])

        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            xyz.colors = o3d.utility.Vector3dVector(colors[index])

        return xyz

class KinectCamera(IntelCamera):
    def __init__(self, cfg):
        self.fx = 607.4124755859375
        self.fy = 607.33538818359375
        self.ppx = 637.793212890625
        self.ppy = 365.12252807617188
        self.depth_scale = 0.001 # mm to m scale

        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(1280, 720, self.fx, self.fy, self.ppx, self.ppy)
        self.camera_mat = np.array([[self.fx, 0, self.ppx], [0, self.fy, self.ppy], [0, 0, 1]], dtype=np.float)
        self.dist_coeffs = np.zeros(4)

        self.saw_yaml = False
        self.saw_charuco = False
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.aruco_dict_ch = aruco.Dictionary_get(aruco.DICT_4X4_250)
        self.cfg = cfg

        self.config = o3d.io.AzureKinectSensorConfig()
        self.sensor = o3d.io.AzureKinectSensor(self.config)
        self.device = 0

        if not self.sensor.connect(self.device):
            raise RuntimeError('Failed to connect to sensor')
        
        else:
            print('MicroSoft AzureKinect' + " is ready")
            self.device_product_line = 'AzureKinect'

        self._dummy_frame = 0

    def stream(self):
        
        align_depth_to_color = True

        rgbd = self.sensor.capture_frame(align_depth_to_color)

        while rgbd is None:
            rgbd = self.sensor.capture_frame(align_depth_to_color)
            
        rgb = np.asarray(rgbd.color)
        self.color_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.depth_image = np.asarray(rgbd.depth)

        return self.color_image, self.depth_image

if __name__ == '__main__':

    # cam = IntelCamera(cfg=[])
    # cam.create_aruco_marker(aruco.DICT_6X6_50)
    import os
    ref_path = os.getcwd()

    # with open(ref_path+"/core/config/suction_config.yml") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    #     cfg['TF']['end2cam'] = np.reshape(cfg['TF']['end2cam'], (4, 4))
    cfg = []
    # cam = KinectCamera(cfg)
    cam = IntelCamera(cfg)
    print(cam.depth_scale)

    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Clouds", width=848, height=480)
    added = True

    while 1:
        rgb_img, depth_img = cam.stream()
        cam.detectAruco()

        # print(cam.cam2marker)
        # print(np.average(depth_img*0.00025))
        # xyz = cam.generate(depth_img)
        # cam.detectCharuco()
        # xyz = cam.cropPoints()
        # pcd.points = o3d.utility.Vector3dVector(xyz)

        ## visualize rgb and depth image
        cv2.imshow("rgb", rgb_img)
        # cv2.imshow("depth", depth_img)
        cv2.waitKey(1)

        # visualize point cloud caculated from the depth image
        # if added == True:
            # vis.add_geometry(pcd)
            # added = False
        # vis.update_geometry(pcd)
        # vis.poll_events()
        # vis.update_renderer()