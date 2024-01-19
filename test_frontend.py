#!/usr/bin/python3
import sys
import time
import socket
import threading
import queue

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

class TestFrontend:
    """ Wrapper for Main class that allows for 
    """
    #mainclass = Main()
    print("STARTING...")
    unix_socket = None
    message_queue = queue.Queue()

    def __init__(self):

        self.__init_unix_socket()
        # Start main loop in its own thread

        self.draw_color = (255,255,255)
        self.widgets = []
        colors = [
            ( -1,  -1,  -1),
            (  0,   0, 255),
            (  0, 255,   0),
            (255,   0,   0),
            (255, 255, 255)]
        self.widgets.append(Palette(300353, 0,0,colors, 200, callback=self.choose_color_or_tool))

        message_thread = threading.Thread(target=self.main_loop)
        message_thread.start()
        time.sleep(1)
        self.test_palette()

    
    def test_eraser (self):
        self.draw_grid(5, 5,25,100, 100)
        self.erase_in_line(50,50, 700, 700, 30)
        self.draw_grid(5, 5,25,700, 700)

    def test_palette (self):
        self.draw_grid(4, 4, 25, 250, 250)

        for i in range(0,4):
            click_event = PenEvent(300 + i * 200,100, PenState.DRAG)
            for widget in self.widgets:
                if widget.is_point_on_widget(*click_event.get_coordinates()):
                    widget.on_click(click_event)
                else:
                    print("ClickEvent missed")
            self.draw_grid(4,4,25, 250, 350 + 100 * i)
        

    def draw_grid(self, size_x, size_y, spacing, offset_x, offset_y):
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

    def erase_in_line(self, start_x, start_y, end_x, end_y, steps):
        x = start_x
        y = start_y

        for i in range(0,steps):
            x += ((end_x - start_x) / steps)
            y += ((end_y - start_y) / steps)

            print(f"Eraser at {x,y}")

            sim_pen_event = PenEvent(x, y, PenState.DRAG)
            self.erase_at_point(sim_pen_event)
            time.sleep(1)

        x = start_x
        y = start_y

        for i in range(0,steps):
            x += ((end_x - start_x) / steps)
            y += ((end_y - start_y) / steps)

            sim_pen_event = PenEvent(x, y, PenState.DRAG)
            self.add_new_line_point(sim_pen_event)
            time.sleep(0.01)
        self.finish_line(PenEvent(x,y, PenState.DRAG))

    def choose_color_or_tool(self, action, color):
        if action == "COLOR":
            print(f"updating color to {color}")
            self.draw_color = color
        elif action == "ERASE":
            print("TODO: IMPLEMENT ERASING THINGS")

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
    # currently only there to define the syntax for the UNIX Socket message.
    def erase_at_point(self, active_pen_event):
        radius = 50 #TODO: make this a constant
        message = 'd {} {} {} {}'.format(active_pen_event.id, float(active_pen_event.x), float(active_pen_event.y), float(radius))
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

if __name__ == '__main__':
    main=TestFrontend()

