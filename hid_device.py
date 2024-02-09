
""" TipTrack HID Device

    This script will allow you to use a TipTrack Pen as a Human Interface Device.

    Briefly touch the surface -> Left click
    Touch the surface and hold -> Right click

    It currently only works with one Pen!

    If you get the error message "evdev.uinput.UInputError: "/dev/uinput" cannot be opened for writing"
    -> Run this command first in your terminal (Linux)

    sudo chmod +0666 /dev/uinput

"""

import time
import threading
from TipTrack.pen_events.pen_state import PenState
from TipTrack.pen_events.ir_pen import IRPen
from TipTrack.cameras.flir_blackfly_s import FlirBlackflyS
from TipTrack.utility.mouse_input_generator import MouseInputGenerator


WINDOW_WIDTH = Constants.OUTPUT_WINDOW_WIDTH
WINDOW_HEIGHT = Constants.OUTPUT_WINDOW_HEIGHT

MAX_MOVEMENT_PX = 5
LONG_CLICK_TIME_SEC = 0.35


class HIDDevice:

    last_state = 'hover'
    draw_start_timestamp = 0
    logging_draw_events = True
    last_draw_coords = []

    current_click_type = 'left'
    def __init__(self):

        self.ir_pen = IRPen()
        self.flir_blackfly_s = FlirBlackflyS(subscriber=self)
        self.mouse_input_generator = MouseInputGenerator(WINDOW_WIDTH, WINDOW_HEIGHT)

        # Workaround so that this script does not terminate itself
        thread = threading.Thread(target=self.debug_mode_thread)
        thread.start()

    def debug_mode_thread(self):
        while True:
            time.sleep(1)

    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):
        """ When new frames are available from the primary camera(s).

        This method will be called automatically every time new frames are available from the primary camera(s) used
        for detecting the pen events on the surface.


        """
        if len(frames) > 0:
            active_pen_events, stored_lines, pen_events_to_remove = self.ir_pen.get_ir_pen_events(frames, matrices)

            self.process_events(active_pen_events)

    def process_events(self, active_pen_events):
        state = 'hover'
        draw_just_started = False
        draw_just_ended = False

        x = 0
        y = 0

        # TODO: Deal with multiple pens
        active_pen_event = active_pen_events[0]

        x = active_pen_event.x
        y = active_pen_event.y
        if active_pen_event.state == PenState.DRAG:
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

                # If the event is not stationary
                if self.is_click_stationary():
                    # Reset
                    self.current_click_type = 'left'
                    self.logging_draw_events = False
                    self.last_draw_coords = []
                else:
                    now = time.time()
                    # print(now, self.draw_start_timestamp, now - self.draw_start_timestamp)
                    if now - self.draw_start_timestamp >= LONG_CLICK_TIME_SEC:
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
        if 0 <= x <= WINDOW_WIDTH and 0 <= y <= WINDOW_HEIGHT:
            # print(int(x), int(y))

            self.mouse_input_generator.move_event(int(x), int(y))

            if draw_just_started:
                self.mouse_input_generator.click_event(self.current_click_type, 'draw')
            elif draw_just_ended:
                self.mouse_input_generator.click_event(self.current_click_type, 'hover')
                self.current_click_type = 'left'
            else:
                # print(self.current_click_type, state)
                pass

            self.mouse_input_generator.sync_event()
        else:
            # print(x, y, ' -> Outside screen borders')
            pass

    def is_click_stationary(self):
        min_x = min(self.last_draw_coords, key=lambda t: t[0])[0]
        min_y = min(self.last_draw_coords, key=lambda t: t[1])[1]
        max_x = max(self.last_draw_coords, key=lambda t: t[1])[0]
        max_y = max(self.last_draw_coords, key=lambda t: t[1])[1]

        # If the event is not stationary
        if (max_x - min_x) > MAX_MOVEMENT_PX and (max_y - min_y) > MAX_MOVEMENT_PX:
            return True
        return False


if __name__ == '__main__':
    hid_device = HIDDevice()
