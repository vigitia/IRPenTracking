
import os
import sys

import cv2
from TipTrack.cameras.flir_blackfly_s import FlirBlackflyS


EXPOSURE_TIME_MICROSECONDS = 200  # μs -> must be small because we calibrate without filters

GRAYCODE_PATTERN_PATH = 'projector_camera_calibration/graycode_patterns'
OUTPUT_PATH = 'projector_camera_calibration/captures'

# Increment after changing each camera pose. TODO: Automate progressing to next step
CURRENT_CAPTURE_ID = 1


class ProCamCalibration:

    frames = []

    def __init__(self):

        self.flir_blackfly_s = FlirBlackflyS(cam_exposure=EXPOSURE_TIME_MICROSECONDS, subscriber=self)

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
            os.makedirs(os.path.join(OUTPUT_PATH, 'capture_' + str(CURRENT_CAPTURE_ID)))

        self.greycode_patterns = self.load_greycode_patterns()

        # Create OpenCV fullscreen window
        cv2.namedWindow('Greycode patterns', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Greycode patterns', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow('Greycode patterns', self.greycode_patterns[-1])  # Show black image at the start

        self.show_greycode_patterns()

    def load_greycode_patterns(self):
        # Based on: https://github.com/illuminant-ai/procam-calibration/blob/main/cap_chessboard.py
        # Read the graycode pattern pngs.
        patterns = [None] * len(os.listdir(GRAYCODE_PATTERN_PATH))
        for filename in os.listdir(GRAYCODE_PATTERN_PATH):
            if filename.endswith(".png"):
                image = cv2.imread(os.path.join(GRAYCODE_PATTERN_PATH, filename))
                # Extract the index from filename "pattern_<index>.png"
                position = int(filename[8:10])
                patterns[position] = image
            else:
                continue

        return patterns

    def show_greycode_patterns(self):

        cv2.waitKey(1000)

        for i, pattern in enumerate(self.greycode_patterns):
            cv2.imshow('Greycode patterns', pattern)

            key = cv2.waitKey(30)

            file_name_index = str(i)
            if len(file_name_index) == 1:
                file_name_index = '0' + file_name_index

            filename = 'graycode_' + file_name_index + '.png'

            cv2.imwrite(os.path.join(OUTPUT_PATH, 'camera0', filename), self.frames[0])
            cv2.imwrite(os.path.join(OUTPUT_PATH, 'camera1', filename), self.frames[1])
            print('{}/{} frames captured'.format(i + 1, len(self.greycode_patterns)))

            # if key == 27:  # ESC

        cv2.destroyAllWindows()
        sys.exit(0)

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):

        # create o folder for each camera
        for camera_serial_number in camera_serial_numbers:
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'capture_' + str(CURRENT_CAPTURE_ID), camera_serial_number)):
                os.makedirs(os.path.join(OUTPUT_PATH, 'capture_' + str(CURRENT_CAPTURE_ID), camera_serial_number))

        if len(frames) > 0:
            self.frames = frames


if __name__ == '__main__':
    procam_calibration = ProCamCalibration()
