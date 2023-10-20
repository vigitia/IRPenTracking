import sys
import threading
import time

import cv2


from TipTrack.cameras.flir_blackfly_s import FlirBlackflyS
# from realsense_d435 import RealsenseD435Camera

from TipTrack.intrinsic_camera_calibration.calibration_image_capture_service import CalibrationImageCaptureService
from TipTrack.intrinsic_camera_calibration.camera_calibration_service import CameraCalibrationService

NUM_IMAGES_TARGET = 20

CAM_EXPOSURE_TIME_MICROSECONDS = 100000

CHESSBOARD_SQUARES = (11, 7)

# Termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

MANUAL_MODE = True


class IntrinsicCameraCalibration:

    # num_collected_calibration_frames = {}
    num_collected_calibration_frames = None

    collection_finished = False

    frames = []
    camera_serial_numbers = []

    last_preview_timestamp = 0

    def __init__(self):

        self.camera_calibration_service = CameraCalibrationService(CHESSBOARD_SQUARES, CRITERIA)

        self.flir_blackfly_s = FlirBlackflyS(cam_exposure=CAM_EXPOSURE_TIME_MICROSECONDS, subscriber=self)

        thread = threading.Thread(target=self.main_thread)
        thread.start()

    def main_thread(self):
        while True:
            # time.sleep(1)

            now = time.time()

            corners_found = []


            #if now - self.last_preview_timestamp > 0.1:
            #    self.last_preview_timestamp = now
            for i, frame in enumerate(self.frames):
                corners_found.append(False)
                # Speed improvements by using cv2.CALIB_CB_NORMALIZE_IMAGE
                ret, corners = cv2.findChessboardCorners(frame, CHESSBOARD_SQUARES, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)

                if ret:
                    corners_found[i] = True
                    corners = cv2.cornerSubPix(frame, corners, (11, 11), (-1, -1), CRITERIA)
                    colored_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    cv2.drawChessboardCorners(colored_frame, CHESSBOARD_SQUARES, corners, ret)
                    cv2.imshow(self.camera_serial_numbers[i], colored_frame)
                else:
                    cv2.imshow(self.camera_serial_numbers[i], frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == 32 and MANUAL_MODE:  # Space
                if len(corners_found) == len(self.frames) and False not in corners_found:
                    print('Chessboard detected in all frames. Saving them now')
                    for i, frame in enumerate(self.frames):
                        camera_serial_number = self.camera_serial_numbers[i]
                        if self.num_collected_calibration_frames[camera_serial_number] < NUM_IMAGES_TARGET:
                            self.num_collected_calibration_frames[camera_serial_number] = self.calibration_image_capture_service.save_image(frame, camera_serial_number)

                        if self.check_all_frames_collected():
                            self.calibrate_cameras()
                else:
                    print('Chessboard not detected in all frames. Not saving them')

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

        self.flir_blackfly_s.end_camera_capture()
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
