#!/usr/bin/python3
import sys
import time
import socket
import threading
import queue

import pynput.keyboard as keyboard

from enum import Enum
from random import Random

from TipTrack.pen_events.pen_state import PenState
from TipTrack.pen_events.pen_event import PenEvent
from TipTrack.pen_events.ir_pen import IRPen
#from main import *
from pen_color_detection.pen_color_detector import PenColorDetector

from widget import Widget
from palette import Palette
SEND_DATA_USING_UNIX_SOCKET = True

UNIX_SOCK_NAME = '/tmp/uds_test'

USE_SDL_FRONTEND = True
if USE_SDL_FRONTEND:
    import subprocess
    subprocess.Popen("cd sdl_frontend && ./sdl_frontend '/tmp/uds_test' 100", shell=True)

    time.sleep(2)

RESOLUTION = "1080P"
SCALE_RES= 1

if RESOLUTION == "4K":
    SCALE_RES = 1
elif RESOLUTION == "1080P":
    SCALE_RES = 0.5

ERASE_RADIUS_SMALL = SCALE_RES * 10 
ERASE_RADIUS_BIG = SCALE_RES * 50

PALETTE_FILE_PATH = "assets/big_palette_expanded.png"
PALETTE_POS_X = SCALE_RES * 840
PALETTE_POS_Y = SCALE_RES * 0
PALETTE_WIDTH = int(SCALE_RES * 1800)
PALETTE_HEIGHT = int(SCALE_RES * 150)


class TestFrontend:
    """ Wrapper for Main class that allows for 
    """
    #mainclass = Main()
    print("STARTING...")
    unix_socket = None
    message_queue = queue.Queue()
    

    class Tool(Enum):
        TOOL_DRAW = "draw"
        TOOL_ERASE = "erase"
        TOOL_CLEAR = "clear"

    def __init__(self):

        self.__init_unix_socket()
        # Start main loop in its own thread

        time.sleep(2) #give backend time to load a renderer
        self.tool = self.Tool.TOOL_DRAW
        self.widgets = []
        self.init_palette()

        self.erase_radius = ERASE_RADIUS_SMALL

        
        key_listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        key_listener.start()

        message_thread = threading.Thread(target=self.main_loop)
        message_thread.start()
        time.sleep(5)
        self.test_palette()


    def init_palette(self):
        
        self.draw_color = (255,255,255)
        colors = [
            ( -2,  -2,  -2),
            ( -1,  -1, ERASE_RADIUS_BIG),
            ( -1,  -1, ERASE_RADIUS_SMALL),
            (255,  51, 255),
            (255,  51,  51),
            (255, 149,   0),
            (255, 255,  17),
            ( 51, 255,  51),
            ( 51, 238, 238),
            ( 76,  76, 255),
            (128, 128, 128),
            (255, 255, 255)]
        self.palette_id = 25101881

        
        palette = Palette(self.palette_id, PALETTE_POS_X, PALETTE_POS_Y, colors, PALETTE_HEIGHT, callback=self.choose_color_or_tool)
        
        message = "u {} {} {} {} {} {} {}".format(self.palette_id, 1, PALETTE_POS_X, PALETTE_POS_Y, PALETTE_WIDTH, PALETTE_HEIGHT, PALETTE_FILE_PATH)
        self.send_message(message)


        self.indicator_id = 9553487
        indicator_pos_x = PALETTE_POS_X + 11 * PALETTE_HEIGHT
        indicator_pos_y = PALETTE_POS_Y
        indicator_width = indicator_height = PALETTE_HEIGHT
        indicator_filepath = "assets/palette_indicator.png"

        indicator_message = "u {} {} {} {} {} {} {}".format(self.indicator_id, 1, indicator_pos_x, indicator_pos_y, indicator_width, indicator_height, indicator_filepath)
        self.send_message(indicator_message)
        
        palette.set_function_shift_indicator(self.move_indicator)

        self.widgets.append(palette)


    def test_eraser (self):
        self.draw_grid(5, 5,25,100, 100)
        self.erase_in_line(50,50, 700, 700, 300)
        self.draw_grid(5, 5,25,700, 700)

    def test_palette (self):
        self.draw_grid(4, 4, 26, 250, 250)

        for i in range(3,11):
            time.sleep(1)
            click_event = PenEvent(PALETTE_POS_X + PALETTE_HEIGHT/2 + i * PALETTE_HEIGHT,PALETTE_HEIGHT/2, PenState.DRAG)
            for widget in self.widgets:
                if widget.is_point_on_widget(*click_event.get_coordinates()):
                    widget.on_click(click_event)
                #else:
                #    print("ClickEvent missed")
            time.sleep(1)
            self.draw_grid(4,4,25, 250, 350 + 100 * i)
        
        click_on_erase_event = PenEvent(PALETTE_POS_X + PALETTE_HEIGHT/2 + 1 * PALETTE_HEIGHT, PALETTE_HEIGHT / 2, PenState.DRAG)

        for widget in self.widgets:
            if widget.is_point_on_widget(*click_on_erase_event.get_coordinates()):
                widget.on_click(click_on_erase_event)

        for i in range(0,100):
            self.draw_random_line(200,200,1000,700,500)

        self.erase_in_line(250,400,1400,1100, 1500)

        for i in range(0,20):
            self.draw_random_line(1200,200,1400,700,500)

    def test_new_erasers(self):
        for i in range (0,250):
            self.draw_random_line(200,200, 1600, 750, 100)
        
        select_big_one_event = PenEvent(PALETTE_POS_X + 2 * PALETTE_HEIGHT - PALETTE_HEIGHT / 2, PALETTE_POS_Y + PALETTE_HEIGHT / 2, PenState.DRAG)
        
        for widget in self.widgets:
            if widget.is_point_on_widget(*select_big_one_event.get_coordinates()):
                widget.on_click(select_big_one_event)
        self.erase_in_line(1500, 300, 300, 900, 1000)
        time.sleep(1)
        erase_event = PenEvent(900, 500, PenState.DRAG)
        self.erase_at_point(erase_event)
        time.sleep(1)

        select_small_one_event = PenEvent(PALETTE_POS_X + 3 * PALETTE_HEIGHT- PALETTE_HEIGHT / 2, PALETTE_POS_Y + PALETTE_HEIGHT / 2, PenState.DRAG)
        for widget in self.widgets:
            if widget.is_point_on_widget(*select_small_one_event.get_coordinates()):
                widget.on_click(select_small_one_event)
        self.erase_in_line(300, 300, 1500, 900, 1000)
        
        time.sleep(1)
        erase_event = PenEvent(900, 800, PenState.DRAG)
        self.erase_at_point(erase_event)
        time.sleep(1)

        select_clear_event = PenEvent(PALETTE_POS_X + PALETTE_HEIGHT / 2, PALETTE_POS_Y + PALETTE_HEIGHT / 2, PenState.DRAG)
        
        for widget in self.widgets:
            if widget.is_point_on_widget(*select_clear_event.get_coordinates()):
                widget.on_click(select_clear_event)

        
        

    def draw_grid(self, size_x, size_y, spacing, offset_x, offset_y):
        if not self.tool == self.Tool.TOOL_DRAW:
            return

        x = 0
        y = 0
        for i in range(0, size_x):
            for j in range(0, size_y):
                x = i * spacing + offset_x
                y = j * spacing + offset_y
                sim_pen_event = PenEvent(x,y,PenState.DRAG)
                self.add_new_line_point(sim_pen_event)
                time.sleep(0.002)
            end_pen_event = PenEvent(x,y, PenState.HOVER)
            self.finish_line(end_pen_event)
        
        for i in range(0, size_x):
            for j in range(0, size_y):
                y = i * spacing + offset_y
                x = j * spacing + offset_x
                sim_pen_event = PenEvent(x,y,PenState.DRAG)
                self.add_new_line_point(sim_pen_event)
                time.sleep(0.002)
            end_pen_event = PenEvent(x,y, PenState.HOVER)
            self.finish_line(end_pen_event)

    def draw_random_line(self,min_x, min_y, max_x, max_y, steps):
        start_x = x = Random().randrange(min_x, max_x)
        end_x = Random().randrange(min_x, max_x)
        start_y = y = Random().randrange(min_y, max_y)
        end_y = Random().randrange(min_x, max_x)

        for i in range(0,steps):
            x += ((end_x - start_x) / steps)
            y += ((end_y - start_y) / steps)

            sim_pen_event = PenEvent(x, y, PenState.DRAG)
            self.add_new_line_point(sim_pen_event)
            time.sleep(0.0001)
        end_pen_event = PenEvent(end_x, end_y, PenState.DRAG)
        self.finish_line(end_pen_event)
        

    def erase_in_line(self, start_x, start_y, end_x, end_y, steps):
        if not self.tool == self.Tool.TOOL_ERASE:
            return
        x = start_x
        y = start_y

        for i in range(0,steps):
            x += ((end_x - start_x) / steps)
            y += ((end_y - start_y) / steps)

            sim_pen_event = PenEvent(x, y, PenState.DRAG)
            self.erase_at_point(sim_pen_event)
            time.sleep(0.01)
        
        eraser_finish_event = PenEvent(x, y, PenState.HOVER)
        self.finish_erasing(eraser_finish_event)

        x = start_x
        y = start_y

        for i in range(0,steps):
            x += ((end_x - start_x) / steps)
            y += ((end_y - start_y) / steps)

            sim_pen_event = PenEvent(x, y, PenState.DRAG)
            self.add_new_line_point(sim_pen_event)
            time.sleep(0.001)
        self.finish_line(PenEvent(x,y, PenState.DRAG))

    def choose_color_or_tool(self, action, color):
        if action == "COLOR":
            #print(f"updating color to {color}")
            self.tool = self.Tool.TOOL_DRAW
            self.draw_color = color
        elif action == "ERASE":
            self.tool = self.Tool.TOOL_ERASE
            self.erase_radius = color[2]
        elif action == "CLEAR":
            self.tool = self.Tool.TOOL_CLEAR
            self.clear_all()
        
        #print(f"You have now selected the {self.tool} tool")

    def move_indicator(self,new_x, new_y):
        message = "u {} {} {} {}".format(self.indicator_id, 1, new_x, new_y)
        #message = "u {} {} {} {} {} {} {}".format(self.indicator_id, 1, new_x, new_y, 180, 180, "assets/palette_indicator.png")
        self.send_message(message)
    


    def __init_unix_socket(self):
        self.unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        print('Connecting to UNIX Socket %s' % UNIX_SOCK_NAME)
        try:
            self.unix_socket.connect(UNIX_SOCK_NAME)
        except socket.error as error:
            print('Error while connecting to UNIX Socket:', error)
            print('Make sure that the frontend is already running before starting this python script')
            sys.exit(1)
        self.uds_initialized = True

    def main_loop(self):
        while True:
            self.__process_message_queue()


    def finish_line(self, pen_event_to_remove):
        """

        """
        message = 'f {}'.format(pen_event_to_remove.id)

        self.send_message(message)

        
    def finish_erasing(self, pen_event_to_remove):

        message = 'v {}'.format(pen_event_to_remove.id)
        self.send_message(message)

    last_timestamp = 0

    def add_new_line_point(self, active_pen_event):
        #r = g = b = 255
        r,g,b = self.draw_color

        message = 'l {} {} {} {} {} {} {}'.format(active_pen_event.id, r, g, b,
                                                  int(active_pen_event.x),
                                                  int(active_pen_event.y),
                                                  0 if active_pen_event.state == PenState.HOVER else 1)

        # now = time.time_ns()
        # diff = now - self.last_timestamp
        # print(diff)
        #
        # self.last_timestamp = now

        self.send_message(message)
    

    # Sends message to frontend to erase all points in a radius around the current position of the pen.
    def erase_at_point(self, active_pen_event):
        radius = self.erase_radius
        message = 'd {} {} {} {} {}'.format(active_pen_event.id, float(active_pen_event.x), float(active_pen_event.y), float(radius),  0 if active_pen_event.state == PenState.HOVER else 1)
        self.send_message(message)

    def send_message(self, message):
        self.message_queue.put(message)

    
    def __process_message_queue(self):
        # This sleep seems necessary. Otherwise, this loop will block everything else
        time.sleep(0.0001)
        if SEND_DATA_USING_UNIX_SOCKET:
            try:
                message = self.message_queue.get(block=False)
            except queue.Empty:
                # No message in the queue
                return

            if not self.uds_initialized:
                raise Exception('Unix Socket not initialized')
            else:
                try:
                    msg_encoded = bytearray(message, 'ascii')
                    size = len(msg_encoded)
                    # print('size', size)

                    MAX_MESSAGE_SIZE = 500

                    if size > MAX_MESSAGE_SIZE:
                        raise Exception('Unix Message way larger than expected. Seems fishy...')

                    self.unix_socket.send(size.to_bytes(4, 'big'))
                    self.unix_socket.send(msg_encoded, socket.MSG_NOSIGNAL)
                except Exception as e:
                    print('---------')
                    print(e)
                    print('ERROR: Broken Pipe!')
                    # print('size', size)

                    # Restart the Unix unix_socket after a short amount of time
                    time.sleep(5000)
                    self.__init_unix_socket()

    def on_key_press(self, key):
        if key == keyboard.Key.shift:
            for widget in self.widgets:
                widget.set_visibility(not widget.is_visible)
            self.toggle_hide_ui()

    def on_key_release(self, key):
        pass

    def toggle_hide_ui(self):
        message = "h"
        self.send_message(message)

    def clear_all (self):
        message = "x"
        self.send_message(message)
    

if __name__ == '__main__':
    main=TestFrontend()

