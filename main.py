#!/usr/bin/python3

import sys
import time
import socket
import threading
import queue

from TipTrack.pen_events.pen_state import PenState
from TipTrack.pen_events.ir_pen import IRPen
from TipTrack.cameras.flir_blackfly_s import FlirBlackflyS
from TipTrack.utility.surface_extractor import SurfaceExtractor

UNIX_SOCK_NAME = 'uds_test'
TRAINING_DATA_COLLECTION_MODE = False  # Enable if ROIs should be saved to disk
DEBUG_MODE = False  # Enable for Debug print statements and preview windows

# Select Frontend and other target applications here:

SEND_DATA_USING_UNIX_SOCKET = True  # Enable if points should be forwarded using a Unix Socket

USE_SDL_FRONTEND = True
if USE_SDL_FRONTEND:
    import subprocess
    subprocess.Popen("cd sdl_frontend && ./sdl_frontend '../uds_test' 100", shell=True)

    time.sleep(2)


class Main:
    """ Entry point to the TipTrack Software


    """

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

        if TRAINING_DATA_COLLECTION_MODE:
            from TipTrack.utility.training_images_collector import TrainingImagesCollector
            exposure = self.flir_blackfly_s.get_exposure_time()
            gain = self.flir_blackfly_s.get_gain()
            self.training_images_collector = TrainingImagesCollector(self.ir_pen, exposure, gain)

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

    def main_loop(self):
        while True:
            self.__process_message_queue()

    def finish_line(self, pen_event_to_remove):
        """

        """
        message = 'f {}'.format(pen_event_to_remove.id)

        print('Finish line', pen_event_to_remove.id)

        self.send_message(message)

    last_timestamp = 0

    def add_new_line_point(self, active_pen_event):
        r = g = b = 255

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

        if SEND_DATA_USING_UNIX_SOCKET:
            for active_pen_event in active_pen_events:
                self.add_new_line_point(active_pen_event)

            for pen_event in pen_events_to_remove:
                self.finish_line(pen_event)

    def on_new_color_frame(self, frame, homography_matrix):
        """ Receive a new color frame

            This function will be automatically called everytime a new color frame is available. Color frames can be
            used in addition to the monochrome/infrared frames from the primary cameras.

        """

        self.last_color_frame = frame


if __name__ == '__main__':
    main = Main()
