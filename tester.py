import time
from datetime import datetime

import cv2

from TipTrack.cameras.flir_blackfly_s import FlirBlackflyS
from TipTrack.pen_events.ir_pen import IRPen


class TipTrackTester:

    frames = []
    camera_serial_numbers = []
    transform_matrices = []

    frame_counter = 0
    start_time = datetime.now()

    def __init__(self):
        self.flir_blackfly_s = FlirBlackflyS(subscriber=self)
        self.ir_pen = IRPen(debug_mode=True)

        self.main_loop()

    def on_new_frame_group(self, frames, camera_serial_numbers, transform_matrices):
        # print('new frames')
        self.frames = frames
        self.camera_serial_numbers = camera_serial_numbers
        self.transform_matrices = transform_matrices

        self.frame_counter += 1
        if self.frame_counter == 158:
            self.frame_counter = 0
            end_time = datetime.now()
            run_time = (end_time - self.start_time).microseconds / 1000.0
            # print('It took {}ms to capture 158 frames (target should be 1000ms)'.format(run_time))
            self.start_time = datetime.now()

    def main_loop(self):
        while True:
            if len(self.frames) > 0:
                active_pen_events, stored_lines, pen_events_to_remove = self.ir_pen.get_ir_pen_events(self.frames, self.transform_matrices)

                self.preview_raw_frames(self.frames, self.camera_serial_numbers)

                # for i, frame in enumerate(frames):
                # rois_new, roi_coords_new, max_brightness_values = self.ir_pen.get_all_rois(frame)
                # self.rois = rois_new
                # for pen_event_roi in rois_new:
                #     prediction, confidence = self.ir_pen.ir_pen_cnn.get_prediction(pen_event_roi)
                #     if prediction == 'undefined':
                #         self.rois = [pen_event_roi]
                #     print(max_brightness_values)

                #new_pen_data = self.ir_pen.get_new_pen_data(self.frames, self.transform_matrices)
                #print(new_pen_data)
                #for entry in new_pen_data:
                #    print(entry.prediction, entry.max_brightness, entry.transformed_coords)

                self.frames = []

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                self.flir_blackfly_s.end_camera_capture()

    def preview_raw_frames(self, frames, camera_serial_numbers):
        if len(frames) > 0:
            for i, frame in enumerate(frames):
                cv2.imshow(camera_serial_numbers[i], frame)


if __name__ == '__main__':
    TipTrackTester()
