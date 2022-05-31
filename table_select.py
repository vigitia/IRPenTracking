from .surface_selector import SurfaceSelector
from realsense_d435 import ...
import cv2

realsense_d435_camera = RealsenseD435Camera()
realsense_d435_camera.init_video_capture()
realsense_d435_camera.start()

current_frame = realsense_d435_camera.left_ir_image  # we get this from realsense

calibration_finished = surface_selector.select_surface(current_frame)
if calibration_finished:
    print("[Surface Selector Node]: Calibration Finished")
    exit()

cv2.waitKey(1)

