
import sys
import cv2
import datetime

import numpy as np

from realsense_d435 import RealsenseD435Camera
from ir_pen import IRPen

from draw_shape import ShapeCreator

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

LINE_THICKNESS = 2
PEN_COLOR = (255, 255, 255)
LINE_COLOR = (80, 80, 80)


def timeit(prefix):
    def timeit_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            # print("I " + prefix + "> " + str(start_time))
            retval = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            run_time = (end_time - start_time).microseconds / 1000.0
            # print("O " + prefix + "> " + str(end_time) + " (" + str(run_time) + " ms)")
            print(prefix + "> " + str(run_time) + " ms", flush=True)
            return retval
        return wrapper
    return timeit_decorator


class Run:

    def __init__(self):

        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.ir_pen = IRPen()

        shape_creator = ShapeCreator(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.base_image = shape_creator.draw_shape('shapes/wave.svg', (800, 800), 1000, LINE_THICKNESS, ShapeCreator.DASH, LINE_COLOR)

        self.realsense_d435_camera = RealsenseD435Camera()
        self.realsense_d435_camera.init_video_capture()
        self.realsense_d435_camera.start()

        self.loop()

    def loop(self):

        while True:
            self.process_frame()

    @timeit("Process Frame")
    def process_frame(self):
        ir_image_table = self.realsense_d435_camera.get_ir_image()

        if ir_image_table is not None:
            #cv2.imshow('ir frame', ir_image_table)
            active_pen_events, stored_lines, new_lines, pen_events_to_remove = self.ir_pen.get_ir_pen_events(
                ir_image_table)

            new_frame = self.base_image.copy()

            for active_pen_event in active_pen_events:
                print(active_pen_event)
                if active_pen_event.state.value == 3:  # HOVER
                    cv2.circle(new_frame, active_pen_event.get_coordinates(), 5, (0, 255, 0))
                else:
                    line = np.array(active_pen_event.history, np.int32)
                    cv2.polylines(new_frame, [line], isClosed=False, color=PEN_COLOR, thickness=LINE_THICKNESS)

            for line in stored_lines:
                line = np.array(line, np.int32)
                cv2.polylines(new_frame, [line], isClosed=False, color=PEN_COLOR, thickness=LINE_THICKNESS)

            # print(ir_image_table.shape)

            cv2.imshow('window', new_frame)
        else:
            cv2.imshow('window', self.base_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)


if __name__ == '__main__':
    run = Run()
