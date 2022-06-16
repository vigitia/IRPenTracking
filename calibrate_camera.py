import sys

import cv2

from realsense_d435 import RealsenseD435Camera
from calibration_image_capture_service import CalibrationImageCaptureService
from camera_calibration_service import CameraCalibrationService

NUM_IMAGES_TARGET = 20


class CalibrateCamera:

    num_collected_calibration_frames = {}

    def __init__(self):
        camera_names = ['ir_full']

        for camera_name in camera_names:
            self.num_collected_calibration_frames[camera_name] = 0

        self.calibration_image_capture_service = CalibrationImageCaptureService(camera_names)
        self.camera_calibration_service = CameraCalibrationService()

        self.realsense_d435_camera = RealsenseD435Camera(extract_projection_area=False)
        self.realsense_d435_camera.init_video_capture()
        self.realsense_d435_camera.start()

        self.loop()

    def loop(self):
        while True:
            ir_image_table = self.realsense_d435_camera.get_ir_image()

            if ir_image_table is not None:
                camera_name = 'ir_full'
                self.collect_calibration_images(ir_image_table, camera_name)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)

    def collect_calibration_images(self, ir_image_table, camera_name):
        if self.num_collected_calibration_frames[camera_name] < NUM_IMAGES_TARGET:
            self.num_collected_calibration_frames[camera_name] = self.calibration_image_capture_service.collect_calibration_image(
                ir_image_table, camera_name)

            cv2.imshow(camera_name, ir_image_table)

        if self.check_all_frames_collected():
            self.calibrate_cameras()

    def calibrate_cameras(self):
        self.camera_calibration_service.calibrate_cameras(self.num_collected_calibration_frames.keys())
        print('Calibration Finished')

        cv2.destroyAllWindows()
        sys.exit(0)

    def check_all_frames_collected(self):
        for key in self.num_collected_calibration_frames.keys():
            if self.num_collected_calibration_frames[key] < NUM_IMAGES_TARGET:
                # print('Still missing some frames:')
                print(self.num_collected_calibration_frames)
                return False

        print('ALL frames collected')
        return True


if __name__ == '__main__':
    run = CalibrateCamera()
