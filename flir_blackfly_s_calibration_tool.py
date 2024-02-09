
import time
import threading
import cv2
import numpy as np

import Constants
from TipTrack.cameras.flir_blackfly_s import FlirBlackflyS
from TipTrack.utility.surface_selector import SurfaceSelector

EXTRACT_PROJECTION_AREA = False
PREVIEW_MODE = False  # Em

OUTPUT_RESOLUTION = (Constants.OUTPUT_WINDOW_HEIGHT, Constants.OUTPUT_WINDOW_WIDTH)


class FlirBlackFlySCalibrationTool:
    """ Flir BlackFly S Calibration Tool

        Use this script to select the projection surface in each cameras preview frame.

    """

    frames = []
    camera_serial_numbers = []

    windows_initialized = False

    finished_camera_calibrations = []

    def __init__(self):

        cv2.namedWindow('screen', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Create white frame used to light up the entire projection area
        self.calibration_target = np.zeros(OUTPUT_RESOLUTION, np.uint8)
        self.calibration_target.fill(255)

        if PREVIEW_MODE:
            self.flir_blackfly_s = FlirBlackflyS(subscriber=self)
        else:
            self.surface_selector = SurfaceSelector()
            self.flir_blackfly_s = FlirBlackflyS(cam_exposure=Constants.CAM_EXPOSURE_FOR_CALIBRATION, subscriber=self, gain=Constants.CAM_GAIN_FOR_CALIBRATION)

        self.main_loop()

    def main_loop(self):
        while True:

            extracted_frames = []

            if PREVIEW_MODE:
                if not self.windows_initialized and len(self.camera_serial_numbers) > 0:
                    self.__init_preview_windows(self.frames, self.camera_serial_numbers)

            cv2.imshow('screen', self.calibration_target)

            if len(self.frames) > 0:
                for i, frame in enumerate(self.frames):

                    window_name = 'Flir Blackfly S {}'.format(self.camera_serial_numbers[i])

                    if PREVIEW_MODE:

                        window_name_extracted = 'Flir Camera {} Extracted'.format(self.camera_serial_numbers[i])

                        cv2.imshow(window_name, frame)

                        # if EXTRACT_PROJECTION_AREA:
                        #     global surface_extractor
                        #     extracted_frame = surface_extractor.extract_table_area(frame, window_name)
                        #     extracted_frame = cv2.resize(extracted_frame, (3840, 2160))
                        #     extracted_frames.append(extracted_frame)
                        #     cv2.imshow(window_name_extracted, extracted_frame)

                    else:  # When in calibration mode
                        calibration_finished = self.surface_selector.select_surface(frame, window_name)

                        if calibration_finished:
                            if window_name not in self.finished_camera_calibrations:
                                self.finished_camera_calibrations.append(window_name)
                            if len(self.finished_camera_calibrations) == len(self.frames):
                                self.__close_calibration_tool()
                                return

                self.frames = []

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                self.__close_calibration_tool()
                return

    def __close_calibration_tool(self):
        cv2.destroyAllWindows()
        self.flir_blackfly_s.end_camera_capture()

    def __init_preview_windows(self, frames, camera_serial_numbers):
        for camera_serial_number in camera_serial_numbers:

            frame_width = frames[0].shape[1]
            frame_height = frames[0].shape[0]

            cv2.namedWindow('Flir Blackfly S {}'.format(camera_serial_number), cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Flir Blackfly S {}'.format(camera_serial_number), frame_width, frame_height)
            # cv2.resizeWindow('Flir Blackfly S {}'.format(camera_serial_number), 480, 480)

            if EXTRACT_PROJECTION_AREA:
                cv2.namedWindow('Flir Camera {} Extracted'.format(camera_serial_number), cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Flir Camera {} Extracted'.format(camera_serial_number), frame_width, frame_height)
                # cv2.resizeWindow('Flir Camera {} Extracted'.format(camera_serial_number), 480, 480)

        self.windows_initialized = True

    # callback function for camera. New frames will arrive here
    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):
        if len(self.camera_serial_numbers) == 0:
            self.camera_serial_numbers = camera_serial_numbers

        self.frames = frames


if __name__ == '__main__':
    FlirBlackFlySCalibrationTool()
