import numpy as np
import os
import sys
import yaml
import numpy as np
import open3d as o3d
import cv2
from camera import IntelCamera, KinectCamera

cam = IntelCamera(cfg=[])


with open('/home/robot/3d_camera/config/workspace.yml') as f:
    workspace_cfg = yaml.load(f, Loader=yaml.FullLoader)

pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window("Point Clouds", width=848, height=480)
added = True

while 1:
    rgb, depth = cam.stream()
    xyz = cam.generate(depth, downsample=True)
    cam.detectCharuco()
    xyz_crop = cam.crop_points()
    try:
        cam2marker = cam.cam2marker
    except:
        pass

    pcd.points = o3d.utility.Vector3dVector(xyz_crop)

    cv2.imshow("res", rgb)
    key = cv2.waitKey(1)

    if key == ord('r'):
        with open('/home/robot/3d_camera/config/workspace.yml', 'w') as f:
            workspace_cfg['cam2marker'] = cam2marker.tolist()
            yaml.dump(workspace_cfg, f, default_flow_style=None)
        
        print("DONE!")
        break
    if added == True:
        vis.add_geometry(pcd)
        added = False
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()