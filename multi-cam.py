import pyrealsense2 as rs
import cv2
import numpy as np

context = rs.context()

devices = context.query_devices()

dev_1 = devices[0]
dev_2 = devices[1]

dev_1_serial = dev_1.get_info(rs.camera_info.serial_number)
dev_2_serial = dev_2.get_info(rs.camera_info.serial_number)

pipeline_1 = rs.pipeline()
pipeline_2 = rs.pipeline()

config_1 = rs.config()
config_2 = rs.config()

# for d in devices:
#     print("Device Name:", d.get_info(rs.camera_info.name))
#     print("Serial Number:", d.get_info(rs.camera_info.serial_number))
#     print("Firmware Version:", d.get_info(rs.camera_info.firmware_version))
#     print("Product ID:", d.get_info(rs.camera_info.product_id))
#     print("-" * 30)

config_1.enable_device(dev_1_serial)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config_2.enable_device(dev_2_serial)
config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

print("device_1", dev_1_serial)
print("device_2", dev_2_serial)

# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)

colorizer = rs.colorizer(color_scheme = 2)

try:
    while True:

        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        colored_depth_frame_1 = colorizer.colorize(depth_frame_1)
        if not depth_frame_1 or not color_frame_1:
            continue
        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        colored_depth_1 = np.asanyarray(colored_depth_frame_1.get_data())

        # Camera 2
        # Wait for a coherent pair of frames: depth and color
        frames_2 = pipeline_2.wait_for_frames()
        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()
        colored_depth_frame_2 = colorizer.colorize(depth_frame_2)
        if not depth_frame_2 or not color_frame_2:
            continue
        # Convert images to numpy arrays
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        colored_depth_2 = np.asanyarray(colored_depth_frame_2.get_data())

        # Stack all images horizontally
        images = np.hstack((color_image_1, colored_depth_1,color_image_2, colored_depth_2))

        # Show images from both cameras
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        # Save images and depth maps from both cameras by pressing 's'
        ch = cv2.waitKey(25)
        if ch==115:
            cv2.imwrite("my_image_1.jpg",color_image_1)
            cv2.imwrite("my_depth_1.jpg",depth_colormap_1)
            cv2.imwrite("my_image_2.jpg",color_image_2)
            cv2.imwrite("my_depth_2.jpg",depth_colormap_2)

finally:

    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()