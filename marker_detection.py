import cv2
from cv2 import aruco
import numpy as np
from camera import Camera

cam = Camera()
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
marker_size = 0.068 # actual marker size in meters

while True:
    rgb, depth = cam.stream()
    gray_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(rgb.copy(), corners, ids)
    if np.shape(corners)[0] > 0:
            for i in range(np.shape(corners)[0]):
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], marker_size, cameraMatrix=cam.camera_mat, distCoeffs=cam.distCoeffs)
                frame_markers = cv2.drawFrameAxes(frame_markers, cameraMatrix=cam.camera_mat, distCoeffs=cam.distCoeffs, rvec=rvecs, tvec=tvecs, length=0.050, thickness=2)
                ## for SE3 trasnformation matrix (marker with respect to the camera)
                R, _ = cv2.Rodrigues(rvecs)
                tvecs = np.reshape(tvecs, (3, 1))
                cam2marker = np.concatenate((R, tvecs), axis = 1)
                ## add [0, 0, 0, 1] to make it SE3 format
                cam2marker = np.concatenate((cam2marker, np.array([[0, 0, 0, 1]])), axis = 0)

    cv2.imshow("res", frame_markers)
    cv2.waitKey(1)