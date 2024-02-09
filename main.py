#!/usr/bin/python3

import sys
import time
import socket
import threading
import queue
from enum import Enum

import pynput.keyboard as keyboard

from TipTrack.pen_events.pen_state import PenState
from TipTrack.pen_events.pen_event import PenEvent
from TipTrack.pen_events.ir_pen import IRPen
from TipTrack.cameras.flir_blackfly_s import FlirBlackflyS
from TipTrack.utility.surface_extractor import SurfaceExtractor

from Constants import *

#from pen_color_detection.pen_color_detector import PenColorDetector

from palette import Palette

TRAINING_DATA_COLLECTION_MODE = False  # Enable if ROIs should be saved to disk
DEBUG_MODE = False  # Enable for Debug print statements and preview windows

# Select Frontend and other target applications here:

SEND_DATA_USING_UNIX_SOCKET = True  # Enable if points should be forwarded using a Unix Socket

USE_SDL_FRONTEND = True
if USE_SDL_FRONTEND:
    import subprocess
    subprocess.Popen("cd sdl_frontend && ./sdl_frontend '../uds_test' 100", shell=True)

    time.sleep(2)

DOCUMENTS_DEMO = False
if DOCUMENTS_DEMO:
    from demo_applications.documents_demo.AnalogueDigitalDocumentsDemo import AnalogueDigitalDocumentsDemo

#RESOLUTION = "4K"
SCALE_RES= 1

if OUTPUT_WINDOW_WIDTH == 3840 and OUTPUT_WINDOW_HEIGHT == 2160:
    SCALE_RES = 1
elif OUTPUT_WINDOW_WIDTH == 1920 and OUTPUT_WINDOW_HEIGHT == 1080:
    SCALE_RES = 0.5


ERASE_RADIUS_SMALL = SCALE_RES * ERASER_SIZE_SMALL
ERASE_RADIUS_BIG = SCALE_RES * ERASER_SIZE_BIG

PALETTE_WIDTH = int(SCALE_RES * UNSCALED_PALETTE_WIDTH)
PALETTE_HEIGHT = int(SCALE_RES * UNSCALED_PALETTE_HEIGHT)
PALETTE_POS_X = (OUTPUT_WINDOW_WIDTH - PALETTE_WIDTH) / 2
PALETTE_POS_Y = SCALE_RES * UNSCALED_PALETTE_Y_POS


class Main:
    """ Entry point to the TipTrack Software


    """

    
    class Tool(Enum):
        TOOL_DRAW = "draw"
        TOOL_ERASE = "erase"
        TOOL_CLEAR = "clear"

    uds_initialized = False
    unix_socket = None
    message_queue = queue.Queue()
    last_color_frame = None  # Temporary store last color frame here to be used if needed
    color_id_assignments = {}  # This dict will contain mappings between pen event IDs and their assigned color if available
    known_pens = []

    def __init__(self):

        if SEND_DATA_USING_UNIX_SOCKET:
            self.__init_unix_socket()

        self.surface_extractor = SurfaceExtractor()
        self.ir_pen = IRPen()
        self.flir_blackfly_s = FlirBlackflyS(subscriber=self)

        self.widgets = []
        self.tool = self.Tool.TOOL_DRAW

        self.init_palette()

        self.erase_radius = ERASE_RADIUS_SMALL

        if TRAINING_DATA_COLLECTION_MODE:
            from TipTrack.utility.training_images_collector import TrainingImagesCollector
            exposure = self.flir_blackfly_s.get_exposure_time()
            gain = self.flir_blackfly_s.get_gain()
            self.training_images_collector = TrainingImagesCollector(self.ir_pen, exposure, gain)
        if DOCUMENTS_DEMO:
            self.analogue_digital_document = AnalogueDigitalDocumentsDemo()

        # setup key event listener
        key_listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        key_listener.start()

        # Start main loop in its own thread
        message_thread = threading.Thread(target=self.main_loop)
        message_thread.start()

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

    def init_palette(self):
        
        self.draw_color = (255,255,255)
        colors = [
            ( -2,  -2,  -2), # == Clear everything
            ( -1,  -1, ERASE_RADIUS_BIG),# == Eraser, bigger Radius
            ( -1,  -1, ERASE_RADIUS_SMALL), # == Eraser, smaller Radius
            (255,  51, 255), # == an RGB color value (as everything below)
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
        indicator_pos_x = PALETTE_POS_X + POSITION_WHITE * PALETTE_HEIGHT
        indicator_pos_y = PALETTE_POS_Y
        indicator_width = indicator_height = PALETTE_HEIGHT

        indicator_message = "u {} {} {} {} {} {} {}".format(self.indicator_id, 1, indicator_pos_x, indicator_pos_y, indicator_width, indicator_height, INDICATOR_FILEPATH)
        self.send_message(indicator_message)
        
        palette.set_function_shift_indicator(self.move_indicator)

        self.widgets.append(palette)



    def main_loop(self):
        while True:
            self.__process_message_queue()

    # ----------------------------------------------------------------------------------------------------------------

    # Only for documents demo
    def send_heartbeat(self, document_found):
        message = f's {int(document_found)}'

        self.send_message(message)

    # Only for documents demo
    def send_matrix(self, matrix):
        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
        flat = [item for sublist in matrix for item in sublist]

        message = f'm'

        for i in flat:
            message += f' {i}'

        self.send_message(message)

    # Only for documents demo
    def send_corner_points(self, converted_document_corner_points):
        if not self.uds_initialized:
            raise Exception('Unix Socket not initialized')

        message = f'k'
        if len(converted_document_corner_points) == 4:

            for point in converted_document_corner_points:
                message += f'{int(point[0])} {int(point[1])} '
            message += ' 1'  # document exists
        else:

            for i in range(9):
                message += ' 0'

        self.send_message(message)

    # For documents demo
    # id: id of the rect (writing to an existing id should move the rect)
    # state: alive = 1, dead = 0; use it to remove unused rects!
    # coords list: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] -- use exactly four entries and make sure they are sorted!
    def send_rect(self, rect_id, state, coord_list):

        message = f'r {rect_id}'
        for coords in coord_list:
            # message += f'{coords[0]} {coords[1]}'
            message += f' {coords}'
        message += f' {state}'

        self.send_message(message)

        return 1

    # Only for documents demo
    def delete_line(self, line_id):
        message = f'd {line_id}'

        self.send_message(message)

    # Only for documents demo
    def clear_rects(self):
        if not self.uds_initialized:
            raise Exception('Unix Socket not initialized')

        if SEND_DATA_USING_UNIX_SOCKET:
            try:
                print('CLEAR RECTS')
                self.unix_socket.sendall('c |'.encode())
                return 1
            except Exception as e:
                print(e)
                print('---------')
                print('Broken Pipe in clear_rects()')
                self.__init_unix_socket()

    # ----------------------------------------------------------------------------------------------------------------

    # callback function that reacts to the option chosen on the palette.
    def choose_color_or_tool(self, action, color):
        if action == "COLOR":
            self.tool = self.Tool.TOOL_DRAW
            self.draw_color = color
        elif action == "ERASE":
            self.tool = self.Tool.TOOL_ERASE
            self.erase_radius = color[2]
        elif action == "CLEAR":
            self.tool = self.Tool.TOOL_CLEAR
            self.clear_all()
            #simulate click on white field
            sim_pen_event = PenEvent(PALETTE_POS_X + int((POSITION_WHITE+0.5) * PALETTE_HEIGHT), PALETTE_POS_Y + int(0.5 * PALETTE_HEIGHT), PenState.DRAG)

            for widget in self.widgets:
                widget.on_click(sim_pen_event)
            self.move_indicator(PALETTE_POS_X + int(POSITION_WHITE * PALETTE_HEIGHT),  PALETTE_POS_Y)
            time.sleep(0.01)

        #print(f"You have now selected the {self.tool} tool")

    def finish_line(self, pen_event_to_remove):
        """

        """
        message = 'f {}'.format(pen_event_to_remove.id)

        #print('Finish line', pen_event_to_remove.id)

        self.send_message(message)

    last_timestamp = 0

    def add_new_line_point(self, active_pen_event):
        r, g, b = self.draw_color

        # Create test colors
        if DEBUG_MODE:
            if active_pen_event.id % 6 == 0:
                r = 255
            if active_pen_event.id % 6 == 1:
                g = 255
            if active_pen_event.id % 6 == 2:
                b = 255
            if active_pen_event.id % 6 == 3:
                r = 255
                g = 255
            if active_pen_event.id % 6 == 4:
                r = 255
                b = 255
            if active_pen_event.id % 6 == 5:
                g = 255
                b = 255

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
    
    # callback function. Used by the palette to set the position of the indicator
    def move_indicator(self,new_x, new_y):
        message = "u {} {} {} {}".format(self.indicator_id, 1, new_x, new_y)
        #message = "u {} {} {} {} {} {} {}".format(self.indicator_id, 1, new_x, new_y, 180, 180, "assets/palette_indicator.png")
        self.send_message(message)
    

    # Sends message to frontend to erase all points in a radius around the current position of the pen.
    def erase_at_point(self, active_pen_event):
        radius = self.erase_radius
        message = 'd {} {} {} {} {}'.format(active_pen_event.id, int(active_pen_event.x), int(active_pen_event.y), radius, 0 if active_pen_event.state == PenState.HOVER else 1)
        self.send_message(message)
        
    # Sends message to frontend to pause the erase process (a.k.a. stop showing the eraser indicator)
    def finish_erasing(self, pen_event_to_remove):
        message = 'v {}'.format(pen_event_to_remove.id)
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


    # def append_line(self, line_id, line):
    #
    #     print('append', line_id, line)
    #
    #     if not self.uds_initialized:
    #         return 0
    #
    #     line_color = '255 255 255'
    #
    #     message = f'a {line_id} {line_color} '
    #
    #     for i, point in enumerate(line):
    #         message += f'{int(point[0])},{int(point[1])}'
    #         if i < len(line) - 1:
    #             message += ';'
    #
    #     message += ' - |'  # Append message end
    #
    #     print('message append', len(message), message)
    #
    #     if ENABLE_FIFO_PIPE:
    #         os.write(self.pipeout, bytes(message, 'utf8'))
    #     if ENABLE_UNIX_SOCKET:
    #         try:
    #             self.sock.send(message.encode())
    #         except Exception as e:
    #             print(e)
    #             print('---------')
    #             print('Broken Pipe in append_line')
    #             self.init_unix_socket()


    # @timeit('assign_color_to_pen()')
    def assign_color_to_pen(self, active_pen_events):
        relevant_pen_events = []

        for pen_event in active_pen_events:
            if pen_event.state == PenState.DRAG:
                if pen_event.id not in self.known_pens:
                    relevant_pen_events.append(pen_event)
                    self.known_pens.append(pen_event.id)

        for i in self.known_pens:
            if i not in [pen_event.id for pen_event in active_pen_events]:
                del self.known_pens[self.known_pens.index(i)]

        if len(relevant_pen_events) > 0:
            ids_and_points = [[pen_event.id, pen_event.x, pen_event.y] for pen_event in relevant_pen_events]

            # TODO Vitus: Don't use extracted frame here
            extracted_frame = self.surface_extractor.extract_table_area(self.last_color_frame, 'Logitech Brio')
            # current_time = time.process_time()

            if extracted_frame is not None:
                ids_and_colors = self.pen_detector.detect(extracted_frame, ids_and_points)
                # print("TIME PenColorDetector", time.process_time() - current_time)
                for k, v in ids_and_colors.items():
                    self.color_id_assignments = {key: val for key, val in self.color_id_assignments.items() if val != v["color"]}
                    self.color_id_assignments[k] = v["color"]

        # self.color_id_assignments = {self.known_pens[-1]: "red"}

    #@timeit('on_new_frame_group')
    def on_new_frame_group(self, frames, camera_serial_numbers, matrices):
        """ When new frames are available from the primary camera(s).

        This method will be called automatically every time new frames are available from the primary camera(s) used
        for detecting the pen events on the surface.

        """
        if len(frames) > 0:
            # When in TRAINING_DATA_COLLECTION_MODE, we will only write the received frames to the hard drive
            if TRAINING_DATA_COLLECTION_MODE:
                self.training_images_collector.save_training_images(frames)
            # otherwise, the frames will be used to detect new pen events in the following steps
            else:
                self.__process_new_frames(frames, matrices)

    def __process_new_frames(self, frames, matrices):
        active_pen_events, stored_lines, pen_events_to_remove = self.ir_pen.get_ir_pen_events(frames, matrices)

        if JUERGEN_MODE:
            self.assign_color_to_pen(active_pen_events)

        if SEND_DATA_USING_UNIX_SOCKET:

            for active_pen_event in active_pen_events:
                is_touch_on_widget = False
                for widget in self.widgets: #check first if the pen collides with a widget
                    if widget.is_point_on_widget(*active_pen_event.get_coordinates()) and widget.is_visible:
                        widget.on_click(active_pen_event) #if that's the case, let the widget decide what happens on a click
                        is_touch_on_widget = True

                if not is_touch_on_widget: #else, manipulate the canvas with the selected tool
                    if self.tool == self.Tool.TOOL_DRAW:
                        self.add_new_line_point(active_pen_event)
                    elif self.tool == self.Tool.TOOL_ERASE:
                        self.erase_at_point(active_pen_event)

            for pen_event in pen_events_to_remove:
                if self.tool == self.Tool.TOOL_DRAW:
                    self.finish_line(pen_event)
                elif self.tool == self.Tool.TOOL_ERASE:
                    self.finish_erasing(pen_event)


        if DOCUMENTS_DEMO:
            self.analogue_digital_document.on_new_finished_lines(stored_lines)

    def on_new_color_frame(self, frame, homography_matrix):
        """ Receive a new color frame

        This function will be automatically called everytime a new color frame is available. Color frames can be used
        in addition to the monochrome/infrared frames from the primary cameras.

        """
        # print(frame.shape)

        self.last_color_frame = frame

        if DOCUMENTS_DEMO:
            document_found, highlight_dict, document_changed, document_removed, document_moved, \
                converted_document_corner_points, document_moved_matrix = self.analogue_digital_document.get_highlight_rectangles(frame, homography_matrix)

            self.send_heartbeat(document_found)

            if document_moved and not document_removed:
                self.send_corner_points(converted_document_corner_points)  # , int(not document_removed) Andi was here
                try:
                    for highlight_id, rectangle in highlight_dict.items():
                        self.send_rect(highlight_id, 1, rectangle)
                except:
                    print('error sending highlights')

            if document_removed:
                self.send_corner_points([])

            if document_found:
                if len(document_moved_matrix) > 0:
                    self.send_matrix(document_moved_matrix)

            if document_changed or document_removed:
                self.clear_rects()

    #TODO: remove. Let the frontend handle everything with key inputs.
    def on_key_press(self, key):
        #print("Pressed key {}".format(key))
        if key == keyboard.Key.page_down:
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

    #TODO (BIG): shift most of the functionality for choosing tools, handling widgets etc. to the frontend.
    


if __name__ == '__main__':
    main = Main()
