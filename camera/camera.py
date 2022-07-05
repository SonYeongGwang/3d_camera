import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d

from cv2 import aruco

class Camera:
    def __init__(self):
        self.context = rs.context()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.spatial_filter = rs.spatial_filter()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

        print(self.device_product_line + " is ready")

        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(self.depth_scale)

        self.color_intrinsic = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsic = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.fx = self.color_intrinsic.fx
        self.fy = self.color_intrinsic.fy
        self.ppx = self.color_intrinsic.ppx
        self.ppy = self.color_intrinsic.ppy

        self.camera_mat = np.array([[self.fx, 0, self.ppx], [0, self.fy, self.ppy], [0 ,0 ,1]], dtype=np.float)
        self.distCoeffs = np.zeros(4)
        self.colorizer = rs.colorizer(color_scheme = 2)
    
    def stream(self, colored_depth = False):
        
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        ## filter depth frame
        # depth_frame = self.self.spatial_filter.process(depth_frame)

        colored_depth_frame = self.colorizer.colorize(depth_frame)

        self.color_image = np.asanyarray(color_frame.get_data())

        if colored_depth == True:
            self.depth_image = np.asanyarray(colored_depth_frame.get_data())
        else: 
            self.depth_image = np.asanyarray(depth_frame.get_data())

        return self.color_image, self.depth_image

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


    def generate_pcd(self, depth):
        self.pcd = o3d.geometry.PointCloud()
        ## return raw point cloud xyz from intensity-aligned depth map
        ## using color intrinsics and vectorized computation programing technique
        w                = np.shape(depth)[1]
        h                = np.shape(depth)[0]
        z                = depth * self.depth_scale  # raw distance
        u                = np.arange(0, w)
        v                = np.arange(0, h)
        mesh_u, mesh_v   = np.meshgrid(u, v)
        mesh_x           = (mesh_u - self.ppx) * z / self.fx
        mesh_y           = (mesh_v - self.ppy) * z / self.fy
        ## remove zeros and NaN values from x, y, z  this is valid regardless of if depth image is filtered or not
        if np.any(z == 0) or np.isnan(z).any():
            z = z[np.nonzero(z)]
            z = z[~ np.isnan(z)]
            mesh_x = mesh_x[np.nonzero(mesh_x)]
            mesh_x = mesh_x[~ np.isnan(mesh_x)]
            mesh_y = mesh_y[np.nonzero(mesh_y)]
            mesh_y = mesh_y[~ np.isnan(mesh_y)]
        ## raw point cloud in numpy format
        xyz         = np.zeros((np.size(mesh_x), 3))
        xyz[:, 0]   = np.reshape(mesh_x, -1)
        xyz[:, 1]   = np.reshape(mesh_y, -1)
        xyz[:, 2]   = np.reshape(z,      -1)
        ## raw point cloud in o3d format
        self.pcd.points  = o3d.utility.Vector3dVector(xyz)
        self.pcd = self.pcd.voxel_down_sample(voxel_size = 0.005)
        xyz         = np.asarray(self.pcd.points)
        return xyz

if __name__ == '__main__':
    cam = Camera()

    while 1:
        rgb_img, depth_img = cam.stream(colored_depth=False)

        cam.detectCharuco()
        print(cam.cam2marker)

        ## visualize rgb and depth image
        cv2.imshow("rgb", rgb_img)
        cv2.imshow("depth", depth_img)
        cv2.waitKey(1)
