
import sys
import time
import threading
import cv2

from flir_blackfly_s import FlirBlackflyS
from surface_selector import SurfaceSelector

DEBUG_MODE = True
EXTRACT_PROJECTION_AREA = False
CALIBRATION_MODE = True

CAM_EXPOSURE_FOR_CALIBRATION = 200000  # Increase Brightness to better see the corners


class FlirBlackFlySCalibrationTool:

    frames = []
    camera_serial_numbers = []
    frame_width = 0
    frame_height = 0

    windows_initialized = False

    def __init__(self):
        thread = threading.Thread(target=self.debug_mode_thread)
        thread.start()

        if CALIBRATION_MODE:
            self.surface_selector = SurfaceSelector()

            self.flir_blackfly_s = FlirBlackflyS(cam_exposure=CAM_EXPOSURE_FOR_CALIBRATION, subscriber=self)
        else:
            self.flir_blackfly_s = FlirBlackflyS(subscriber=self)

        # Keep thread alive
        # TODO: Improve this
        time.sleep(86400)  # Wait 24h

    def debug_mode_thread(self):
        while True:

            extracted_frames = []

            if not self.windows_initialized:
                if len(self.camera_serial_numbers) > 0:
                    for camera_serial_number in self.camera_serial_numbers:
                        cv2.namedWindow('Flir Blackfly S {}'.format(camera_serial_number), cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Flir Blackfly S {}'.format(camera_serial_number), self.frame_width, self.frame_height)
                        # cv2.resizeWindow('Flir Blackfly S {}'.format(camera_serial_number), 480, 480)

                        if EXTRACT_PROJECTION_AREA:
                            cv2.namedWindow('Flir Camera {} Extracted'.format(camera_serial_number), cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('Flir Camera {} Extracted'.format(camera_serial_number), self.frame_width, self.frame_height)
                            # cv2.resizeWindow('Flir Camera {} Extracted'.format(camera_serial_number), 480, 480)

                            # cv2.namedWindow('Flir Camera Frames combined', cv2.WND_PROP_FULLSCREEN)
                            # cv2.setWindowProperty('Flir Camera Frames combined', cv2.WND_PROP_FULLSCREEN,
                            #                       cv2.WINDOW_FULLSCREEN)
                    self.windows_initialized = True
                else:
                    continue

            if len(self.frames) > 0:
                for i, frame in enumerate(self.frames):

                    window_name = 'Flir Blackfly S {}'.format(self.camera_serial_numbers[i])
                    window_name_extracted = 'Flir Camera {} Extracted'.format(self.camera_serial_numbers[i])

                    # pen_event_roi, brightest, (x, y) = self.ir_pen.crop_image(frame)

                    if DEBUG_MODE:
                        cv2.imshow(window_name, frame)
                        # cv2.imshow(window_name, pen_event_roi)

                    if EXTRACT_PROJECTION_AREA:
                        global surface_extractor
                        extracted_frame = surface_extractor.extract_table_area(frame, window_name)
                        extracted_frame = cv2.resize(extracted_frame, (3840, 2160))
                        extracted_frames.append(extracted_frame)
                        cv2.imshow(window_name_extracted, extracted_frame)

                    if CALIBRATION_MODE:
                        calibration_finished = self.surface_selector.select_surface(frame, window_name)

                        if calibration_finished:
                            print('[Surface Selector Node]: Calibration finished for {}'.format(window_name))

                self.frames = []

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                self.flir_blackfly_s.end_camera_capture()
                sys.exit(0)

    # callback function for camera. New frames will arrive here
    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):
        if len(self.camera_serial_numbers) == 0:
            self.camera_serial_numbers = camera_serial_numbers
            self.frame_width = frames[0].shape[1]
            self.frame_height = frames[0].shape[0]

        self.frames = frames


if __name__ == '__main__':
    FlirBlackFlySCalibrationTool()
