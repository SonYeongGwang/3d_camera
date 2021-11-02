import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from numpy import linalg as LA
from multiprocessing import Pool
from functools import partial
from mpl_toolkits import mplot3d
from numpy.linalg import norm
import imutils
import sys


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
        #depth_frame = self.self.spatial_filter.process(depth_frame)

        colored_depth_frame = self.colorizer.colorize(depth_frame)

        self.color_image = np.asanyarray(color_frame.get_data())

        if colored_depth == True:
            self.depth_image = np.asanyarray(colored_depth_frame.get_data())
        else:
            self.depth_image = np.asanyarray(depth_frame.get_data())

        return self.color_image, self.depth_image

    def generate_pcd(self, depth):

        ## return raw point cloud xyz from intensity-aligned depth map
        ## using color intrinsics and vectorized computation programing technique

        w                = np.shape(depth)[1]
        h                = np.shape(depth)[0]
        z                = depth * self.depth_scale  # raw distance
        #print(type(depth),'depth')
        #print(type(self.depth_scale),'depth_scale')
        u                = np.arange(0, w)
        v                = np.arange(0, h)
        mesh_u, mesh_v   = np.meshgrid(u, v)
        mesh_x           = (mesh_u - self.ppx) * z / self.fx
        #print(self.ppx,'ppx')
        #print(self.fx,'fx')
        mesh_y           = (mesh_v - self.ppy) * z / self.fy
        #print(self.ppy,'ppy')
        #print(self.fy,'fy')
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
        #print(xyz,'xyz')
        self.pcd = o3d.geometry.PointCloud()

        self.pcd.points  = o3d.utility.Vector3dVector(xyz)
        self.pcd = self.pcd.voxel_down_sample(voxel_size = 0.005)
        #print(pcd,'pcd')
        xyz         = np.asarray(self.pcd.points)
        #print(xyz,'xyz1')
        return xyz

    def FilterNeighbors(self, xyz, neighbors,):
        #neighbors filtering
        tree = KDTree(xyz, leaf_size = 200)
        dist, ind = tree.query(xyz, k =neighbors)
        #compute meandist
        meandist = np.mean(dist, axis=1)
        #computes the standard deviation of mean distance space
        mdstd = np.std(meandist)
        #print(mdstd,'std')
        #computes mean of mean distance space
        mdmean = np.mean(meandist)
        #print(mdmean,'mean')

        #compute the min and max range for the filter
        alpha = 1
        minpxyz = (mdmean - alpha * mdstd)
        maxpxyz = (mdmean + alpha * mdstd)

        #filter the PC with meandist
        inliers  = np.where((meandist > minpxyz ) & (meandist < maxpxyz))
        #print(inliers,'inliers')

        #Matching of index to pcd
        xyz = xyz[inliers]

        return xyz

    def FilterRadius(self,xyz,radius,minpoints):
        tree = KDTree(xyz, leaf_size = 200)
        ind = tree.query_radius(xyz, r=radius)
        no_pts_in_given_radius = np.array([len(ind[i]) for i in range(ind.shape[0])])
        #print(no_pts_in_given_radius,'nopts')

        inliers_radius  = np.where((no_pts_in_given_radius > minpoints))

        xyz = xyz[inliers_radius]

        #self.pcd = o3d.geometry.PointCloud()

        #self.pcd.points  = o3d.utility.Vector3dVector(xyz)

        #self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        #o3d.visualization.draw_geometries( [self.pcd], point_show_normal=True)

        return xyz

    def SurfaceNormalEstimation(self, xyz, neighbors,   test):

        tree = KDTree(xyz)
        dist, ind = tree.query(xyz, k = neighbors)

        Xi = xyz[ind]

        normals = p.map(cam.Normals,Xi)

        if test == True:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points  = o3d.utility.Vector3dVector(xyz)
            self.pcd.normals = o3d.utility.Vector3dVector(normals)
            o3d.visualization.draw_geometries( [self.pcd], point_show_normal=True)

        return normals


    @staticmethod
    def Normals(Xi):

        xm = np.mean(Xi,axis = 0)
        neighbors = len(Xi)

        cov = (1/neighbors)*( np.dot(np.transpose(Xi-xm), (Xi - xm)))

        w,v = LA.eig(cov)

        idx = w.argsort()
        w = w[idx]

        v = v[:,idx]

        normals = v[:,0]
        if normals.dot(xm)>0:
            normals = -normals

        return  normals

    def Gaussian_image(self, normals):

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        normals = np.array(normals)

        #print(type(normals),'normals')
        #print(normals[0],'normals')
        #print(normals[0,0],'normals1')

        ax.scatter3D(normals[:,0],normals[:,1],normals[:,2], c=normals[:,2], cmap='Greens')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

        return

    def Crease_removal(self,xyz,normals,neighbors):

        tree = KDTree(xyz)
        dist, ind = tree.query(xyz, k = neighbors)
        Xi = xyz[ind]

        #print(ind[0],'creaseind')
        normals= np.array(normals)
        #print(len(normals),'normals1')
        Cosine_Similarity =  np.zeros((len(normals), 30))

        not_creaseind = np.zeros(len(normals))

        for i in range(len(normals)):
            anorm = np.array(normals[ind[i]])
            bnorm = np.array(normals[i])

            Cosine_Similarity[i] = np.dot(anorm, bnorm)

            temp_index = np.where(Cosine_Similarity[i] > 0.75)
            #print(len(temp_index[0]))
            if len(temp_index[0]) >  29:
                not_creaseind[i] = int(i)


        xyz = xyz[not_creaseind.astype(int)]

        return xyz

    def Segmentate(self, xyz, normals):

        db = DBSCAN(eps=0.028, min_samples=50).fit(xyz)
        #np.set_printoptions(threshold=np.inf)
        #print(db,'db')
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        #print(labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(n_clusters, 'clusters')
        n_noise = list(labels).count(-1)
        print(n_noise,'noise')

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter3D(xyz[:,0],xyz[:,1],xyz[:,2], c=db.labels_, cmap='jet')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

        #exit()
        return

    def marker_detection(self, rgb_img):

        type_arucodict = cv2.aruco.DICT_ARUCO_ORIGINAL
        arucoDict = cv2.aruco.Dictionary_get(type_arucodict)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(rgb_img, arucoDict,
        	parameters=arucoParams)


        draw = False
        if draw == True:
            if len(corners) > 0:
                ids = ids.flatten()

                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned in

                    # top-left, top-right, bottom-right, and bottom-left order)
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    #print(topLeft,'topLeft')
                    #print(topRight,'topRight')
                    #print(bottomRight,'bottomRight')
                    #print(bottomLeft,'bottomLeft')

                    #drawline around marker
                    cv2.line(rgb_img,topLeft,topRight,(255,0,255),2)
                    cv2.line(rgb_img,topRight,bottomRight,(0,0,255),2)
                    cv2.line(rgb_img,bottomRight,bottomLeft,(255,0,0),2)
                    cv2.line(rgb_img,bottomLeft,topLeft,(0,255,0),2)
                    cv2.imshow("Image",rgb_img)
                    cv2.waitKey(0)


        #calculate pose
        ArUCoSize = (2,2)
        #ArUCoSize = (7,7)#chess


        objp = np.array([[0,0,0],[0,0.1,0],[0.1,0.1,0],[0.1,0,0]])
        #objp = np.zeros((ArUCoSize[0] * ArUCoSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:ArUCoSize[0], 0:ArUCoSize[1]].T.reshape(-1,2)
        #axis = np.float32([3,0,0], [0,3,0, [0,0,-3]]).reshape(-1,3)
        axisbox = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                                [0,0,-1], [0,1,-1],[1,1,-1], [1,0,-1]])


        #img = cv2.imread(rgb_img)
        #gray = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)

        #print(np.array([corners]),'cornersAruco')
        #ret, corners = cv2.findChessboardCorners(gray, (7,7),None)

        #print(ret,'ret')
        #print(corners,'cornerschess')

        #corners = np.array([[topLeft[0],topLeft[1],[topRight[0],topRight[1]],[bottomRight[0], bottomRight[1]],
        #            [bottomLeft[0],bottomLeft[1]]]])

        corners = np.array(corners)
        print(corners,'corners')

        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        #print(corners2,'corners2')

        ret, rvecs ,tvecs = cv2.solvePnP(objp, corners, self.camera_mat, self.distCoeffs)
        imgpts, jac = cv2.projectPoints(axisbox, rvecs, tvecs, self.camera_mat, self.distCoeffs)

        #print(imgpts,'imgpts')
        #print(jac,'jac')

        print(imgpts,'imgpts')

        imgpts = np.int32(imgpts).reshape(-1,2)

        print(imgpts,'imgpts2')

        Draw = True
        if Draw == True:
            img = cv2.drawContours(rgb_img, [imgpts[:4]],-1,(0,255,0),-3)

            for i,j in zip(range(4),range(4,8)):
                img = cv2.line(rgb_img, tuple(imgpts[i]),tuple(imgpts[j]),(255),3)

            img = cv2.drawContours(rgb_img, [imgpts[:4]],-1,(0,0,255),3)
            cv2.imshow('img',img)
            cv2.waitKey(0)




        exit()
        input()

        return


if __name__ == '__main__':
    cam = Camera()
    p = Pool()

    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()

    #vis.create_window("Point Clouds", width=640, height=480)
    added = True

    neighbors = 30
    radius = 0.02
    minpoints = 30
    test_surface_normals = False

    while 1:
        rgb_img, depth_img = cam.stream(colored_depth=False)

        cam.marker_detection(rgb_img)

        xyz = cam.generate_pcd(depth_img)

        FNxyz = cam.FilterNeighbors(xyz, neighbors)

        FRxyz = cam.FilterRadius(FNxyz, radius, minpoints)

        normals = cam.SurfaceNormalEstimation(FRxyz, neighbors, test_surface_normals)

        #cam.Gaussian_image(normals)

        CRxyz = cam.Crease_removal(FRxyz,normals,neighbors)

        normals = cam.SurfaceNormalEstimation(CRxyz, neighbors, test_surface_normals)

        cam.Segmentate(CRxyz,normals)

        pcd.points = o3d.utility.Vector3dVector(CRxyz)

        rgb_img, depth_img = cam.stream(colored_depth=True)

        ## visualize rgb and depth image
        #cv2.imshow("rgb", rgb_img)
        #cv2.imshow("depth", depth_img)
        #cv2.waitKey(1)

        ## visualize point cloud caculated from the depth image
        if added == True:
            vis.add_geometry(pcd)
            added = False
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
