import sys
import cv2
import datetime

import numpy as np

from realsense_d435 import RealsenseD435Camera
from ir_pen import IRPen, State

from pen_hid import InputSimulator

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

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
        self.ir_pen = IRPen()

        self.input_device = InputSimulator(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.realsense_d435_camera = RealsenseD435Camera()
        self.realsense_d435_camera.init_video_capture()
        self.realsense_d435_camera.start()

        self.loop()

    def loop(self):
        while True:
            self.process_frame()

    def process_frame(self):
        ir_image_table = self.realsense_d435_camera.get_ir_image()

        if ir_image_table is not None:
            active_pen_events, stored_lines, new_lines, pen_events_to_remove = self.ir_pen.get_ir_pen_events(
                ir_image_table)

            state = 'hover'
            x = 0
            y = 0
            for active_pen_event in active_pen_events:
                x = active_pen_event.x
                y = active_pen_event.y
                if active_pen_event.state == State.DRAG:
                    state = 'draw'

            self.input_device.input_event(x, y, state)

if __name__ == '__main__':
    run = Run()
