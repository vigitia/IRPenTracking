# Run this command first if you get the error message
# "evdev.uinput.UInputError: "/dev/uinput" cannot be opened for writing"

# sudo chmod +0666 /dev/uinput

import datetime
import time

import cv2

# from ir_pen import IRPen, State
from pen_hid import InputSimulator

from flir_blackfly_s import FlirBlackflyS
# from realsense_d435 import RealsenseD435Camera


WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

MAX_MOVEMENT_PX = 5
LONG_CLICK_TIME = 1.75  # 0.35

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
        #self.ir_pen = IRPen()

        # self.input_device = InputSimulator(WINDOW_WIDTH, WINDOW_HEIGHT)

        # self.realsense_d435_camera = RealsenseD435Camera()
        # self.realsense_d435_camera.init_video_capture()
        # self.realsense_d435_camera.start()

        # self.flir_blackfly_s = FlirBlackflyS()
        # self.flir_blackfly_s.start()

        self.flir_blackfly_s = FlirBlackflyS(subscriber=self)

        # cv2.namedWindow("preview", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # self.loop()

    def loop(self):
        while True:
            self.process_frames()

    last_state = 'hover'
    draw_start_timestamp = 0
    logging_draw_events = True
    last_draw_coords = []

    # current_click_type = 'right'
    current_click_type = 'left'

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):

        if len(frames) > 0:
            # self.frame_counter += 1
            # print('Received {} new frames from Flir Blackfly S'.format(len(frames)))
            print('run', frames[0].shape)

            # _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(frames[0])
            # if brightest > 100:
            #     self.input_device.click_event(self.current_click_type, 'draw')
            # else:
            #     self.input_device.click_event(self.current_click_type, 'hover')

    def process_frames(self):
        # ir_image_table = self.realsense_d435_camera.get_ir_image()
        # if ir_image_table is not None:
        new_frames, matrices = self.flir_blackfly_s.get_camera_frames()

        if len(new_frames) > 0:

            _, brightest, _, (max_x, max_y) = cv2.minMaxLoc(new_frames[0])
            if brightest > 100:
                self.input_device.click_event(self.current_click_type, 'draw')
            else:
                self.input_device.click_event(self.current_click_type, 'hover')

            return

            active_pen_events, stored_lines, _, _, debug_distances = self.ir_pen.get_ir_pen_events_multicam(new_frames, matrices)

            # cv2.waitKey(1)
            # return

            state = 'hover'
            draw_just_started = False
            draw_just_ended = False

            x = 0
            y = 0
            # for active_pen_event in active_pen_events:
            if len(active_pen_events) > 0:
                active_pen_event = active_pen_events[0]

                x = active_pen_event.x
                y = active_pen_event.y
                if active_pen_event.state == State.DRAG:
                    state = 'draw'

                if self.last_state == 'hover' and state == 'draw':
                    self.draw_start_timestamp = time.time()
                    # print('Hover to draw')
                    draw_just_started = True
                    self.logging_draw_events = True
                elif self.last_state == 'draw' and state == 'hover':
                    # print('Draw to hover')
                    draw_just_ended = True
                    self.current_click_type = 'left'

                if self.logging_draw_events:
                    if state == 'draw':
                        self.last_draw_coords.append((x, y))

                        min_x = min(self.last_draw_coords, key=lambda t: t[0])[0]
                        min_y = min(self.last_draw_coords, key=lambda t: t[1])[1]
                        max_x = max(self.last_draw_coords, key=lambda t: t[1])[0]
                        max_y = max(self.last_draw_coords, key=lambda t: t[1])[1]

                        # If the event is not stationary
                        if (max_x - min_x) > MAX_MOVEMENT_PX and (max_y - min_y) > MAX_MOVEMENT_PX:
                            # Reset
                            self.current_click_type = 'left'
                            self.logging_draw_events = False
                            self.last_draw_coords = []
                        else:
                            now = time.time()
                            # print(now, self.draw_start_timestamp, now - self.draw_start_timestamp)
                            if now - self.draw_start_timestamp >= LONG_CLICK_TIME:
                                self.current_click_type = 'right'
                                draw_just_started = True
                                # print('RIGHT CLICK')

                                self.logging_draw_events = False
                                self.last_draw_coords = []

                    else:
                        # Reset
                        self.logging_draw_events = False
                        self.last_draw_coords = []

            self.last_state = state

            # Check if the event happens within the projection area
            # if 0 <= x <= WINDOW_WIDTH and 0 <= y <= WINDOW_HEIGHT:
            if 0 < x <= WINDOW_WIDTH - 300 and 0 < y <= WINDOW_HEIGHT:
                print(x, y)

                self.input_device.move_event(int(x), int(y))

                # if self.logging_draw_events == False:
                # if self.current_click_type == 'draw':

                if draw_just_started:
                    self.input_device.click_event(self.current_click_type, 'draw')
                elif draw_just_ended:
                    # print(self.current_click_type, 'CLICK')
                    #self.input_device.click_event(self.current_click_type, 'draw')
                    self.input_device.click_event(self.current_click_type, 'hover')
                    self.current_click_type = 'left'
                else:
                    print(self.current_click_type, state)
            else:
                # print(x, y, ' -> Outside screen borders')
                pass

            # if self.STATE == 'AWAY':
            # elif self.STATE == 'HOVER':
            # elif self.STATE == 'WAIT_FOR_LEFTCLICK':
            # elif self.STATE == 'DRAW':


if __name__ == '__main__':
    run = Run()
