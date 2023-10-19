import sys
import threading

import cv2


from TipTrack.cameras.flir_blackfly_s import FlirBlackflyS
# from realsense_d435 import RealsenseD435Camera

from TipTrack.intrinsic_camera_calibration.calibration_image_capture_service import CalibrationImageCaptureService
from TipTrack.intrinsic_camera_calibration.camera_calibration_service import CameraCalibrationService

NUM_IMAGES_TARGET = 10

CAM_EXPOSURE_TIME_MICROSECONDS = 100000

MANUAL_MODE = True


class IntrinsicCameraCalibration:

    # num_collected_calibration_frames = {}
    num_collected_calibration_frames = None

    collection_finished = False

    frames = []
    camera_serial_numbers = []

    def __init__(self):

        self.camera_calibration_service = CameraCalibrationService()

        self.flir_blackfly_s = FlirBlackflyS(cam_exposure=CAM_EXPOSURE_TIME_MICROSECONDS, subscriber=self)

        thread = threading.Thread(target=self.main_thread)
        thread.start()

    def main_thread(self):
        while True:
            # time.sleep(1)

            for i, frame in enumerate(self.frames):
                cv2.imshow(self.camera_serial_numbers[i], frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == 32 and MANUAL_MODE:  # Space
                for i, frame in enumerate(self.frames):
                    camera_serial_number = self.camera_serial_numbers[i]
                    if self.num_collected_calibration_frames[camera_serial_number] < NUM_IMAGES_TARGET:
                        self.num_collected_calibration_frames[camera_serial_number] = self.calibration_image_capture_service.save_image(frame, camera_serial_number)

                    if self.check_all_frames_collected():
                        self.calibrate_cameras()

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):
        if len(frames) > 0 and not self.collection_finished:

            self.frames = frames

            if len(self.camera_serial_numbers) == 0:
                self.camera_serial_numbers = camera_serial_numbers

            if self.num_collected_calibration_frames is None:
                self.num_collected_calibration_frames = {}
                for camera_serial_number in camera_serial_numbers:
                    self.num_collected_calibration_frames[camera_serial_number] = 0
                self.calibration_image_capture_service = CalibrationImageCaptureService(camera_serial_numbers)

            if not MANUAL_MODE:
                for i, frame in enumerate(frames):
                    self.collect_calibration_images(frame, camera_serial_numbers[i])



    # def loop(self):
    #     while True:
    #         # ir_image_table = self.realsense_d435_camera.get_ir_image()
    #         frames, matrices = self.flir_blackfly_s.get_camera_frames()
    #
    #         if len(frames) > 0:
    #             for i, frame in enumerate(frames):
    #                 self.collect_calibration_images(frame, 'FlirBlackflyS {}'.format(i))
    #             # self.collect_calibration_images(frames[0], 'FlirBlackflyS 0')
    #
    #         key = cv2.waitKey(1)
    #         if key == 27:  # ESC
    #             cv2.destroyAllWindows()
    #             sys.exit(0)

    def collect_calibration_images(self, frame, camera_serial_number):
        if self.num_collected_calibration_frames[camera_serial_number] < NUM_IMAGES_TARGET:
            self.num_collected_calibration_frames[camera_serial_number] = self.calibration_image_capture_service.collect_calibration_image(frame, camera_serial_number)
            # self.num_collected_calibration_frames[camera_serial_number] = self.calibration_image_capture_service.save_image(frame, camera_serial_number)

            # cv2.imshow(camera_serial_number, frame)

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
                # print(self.num_collected_calibration_frames)
                return False

        print('ALL frames collected')
        self.collection_finished = True
        return True


if __name__ == '__main__':
    intrinsic_camera_calibration = IntrinsicCameraCalibration()
