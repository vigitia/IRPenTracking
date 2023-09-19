import time
from datetime import datetime

import cv2

from TipTrack.cameras.flir_blackfly_s import FlirBlackflyS


class TipTrackTester:

    frames = []
    camera_serial_numbers = []

    frame_counter = 0
    start_time = datetime.now()

    def __init__(self):
        self.flir_blackfly_s = FlirBlackflyS(subscriber=self)

        self.main_loop()

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):
        # print('new frames')
        self.frames = frames
        self.camera_serial_numbers = camera_serial_numbers

        self.frame_counter += 1
        if self.frame_counter == 158:
            self.frame_counter = 0
            end_time = datetime.now()
            run_time = (end_time - self.start_time).microseconds / 1000.0
            print('It took {}ms to capture 158 frames (target should be 1000ms)'.format(run_time))
            self.start_time = datetime.now()

    def main_loop(self):
        while True:
            self.preview_raw_frames(self.frames, self.camera_serial_numbers)



    def preview_raw_frames(self, frames, camera_serial_numbers):
        if len(frames) > 0:
            for i, frame in enumerate(frames):
                cv2.imshow(camera_serial_numbers[i], frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            self.flir_blackfly_s.end_camera_capture()


if __name__ == '__main__':
    TipTrackTester()